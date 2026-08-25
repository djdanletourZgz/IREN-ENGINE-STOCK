from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from iren_engine.config import CFG
from iren_engine.market_data import load_daily, load_intraday
from iren_engine.probability import (
    build_daily_state,
    daily_probabilities,
    intraday_probability,
    target_touch_probability,
)
from iren_engine.v2a1 import build_v2a1_state, current_v2a1_context, v2a1_probabilities
from iren_engine.options_orats import fetch_live_chain, compute_gamma_metrics
from iren_engine.storage import save_snapshot

st.set_page_config(page_title="IREN Tactical Radar", page_icon="📡", layout="wide")
st.title("📡 IREN Tactical Radar — V2-A.1")
st.caption(
    "Dirección experimental: logística regularizada + residual IREN/IA + transición de régimen. "
    "Recorrido/touch permanece en el motor empírico V1. Gamma sigue fuera de P↑/P↓ hasta validación histórica."
)
st.warning(
    "V2-A.1 acaba de cambiar de arquitectura y todavía NO está validada. "
    "Las probabilidades direccionales se muestran para someterlas al siguiente walk-forward, no para asumir que ya existe edge."
)


def pct(v):
    return "—" if v is None or not np.isfinite(v) else f"{float(v):.0%}"


def pct1(v):
    return "—" if v is None or not np.isfinite(v) else f"{float(v):+.1%}"


def money(v):
    return "—" if v is None or not np.isfinite(v) else f"${float(v):.2f}"


@st.cache_data(ttl=180)
def get_daily():
    tickers = list(dict.fromkeys([
        CFG.ticker, CFG.qqq, CFG.vix, CFG.btc, CFG.nvda, CFG.eurusd, *CFG.ai_basket
    ]))
    return load_daily(tickers, CFG.daily_period)


@st.cache_data(ttl=120)
def get_intraday():
    return load_intraday([CFG.ticker], CFG.intraday_period, CFG.intraday_interval)[CFG.ticker]


with st.sidebar:
    st.header("Control")
    target = st.number_input("Nivel que quieres vigilar ($)", min_value=1.0, max_value=500.0, value=40.0, step=0.5)
    with st.expander("Ajustes avanzados"):
        k_range = st.slider("Vecinos motor recorrido V1", 60, 220, CFG.daily_neighbors, 10)
        k_intra = st.slider("Vecinos intradía provisional", 60, 300, CFG.intraday_neighbors, 10)
        max_dte = st.slider("Gamma: DTE máximo", 7, 90, CFG.options_max_dte, 1)
        token_default = ""
        try:
            token_default = st.secrets.get("ORATS_TOKEN", "")
        except Exception:
            token_default = os.getenv("ORATS_TOKEN", "")
        token = st.text_input("ORATS token (opcional)", value=token_default, type="password")
    st.caption("La fiabilidad numérica vuelve sólo cuando demuestre monotonicidad fuera de muestra.")

try:
    daily = get_daily()
    intra = get_intraday()
except Exception as e:
    st.error(f"No se pudieron descargar datos de mercado: {e}")
    st.stop()

iren = daily.get(CFG.ticker, pd.DataFrame())
if iren.empty:
    st.error("No hay datos de IREN.")
    st.stop()

spot = float(iren["Close"].iloc[-1])
if intra is not None and not intra.empty:
    spot = float(intra["Close"].iloc[-1])

market = {k: v for k, v in daily.items() if k != CFG.ticker}

# MOTOR A: DIRECCIÓN V2-A.1
state_direction = build_v2a1_state(
    iren,
    market,
    ai_tickers=CFG.ai_basket,
    beta_window=CFG.v2a1_beta_window,
    beta_half_life=CFG.v2a1_beta_half_life,
    beta_min_obs=CFG.v2a1_beta_min_obs,
)
direction_res = v2a1_probabilities(
    state_direction,
    model_start=CFG.v2a1_model_start,
    lookback_rows=CFG.v2a1_model_lookback,
    min_train=CFG.v2a1_min_train,
    c_value=CFG.v2a1_logistic_c,
)
context = current_v2a1_context(state_direction)

# MOTOR B: RECORRIDO V1. Se conserva deliberadamente separado.
try:
    state_range = build_daily_state(
        iren,
        daily[CFG.btc],
        daily[CFG.qqq],
        daily.get(CFG.nvda),
    )
    range_res = daily_probabilities(state_range, k=k_range)
except Exception:
    state_range = pd.DataFrame()
    range_res = []

# HOY sigue provisional hasta construir/validar el consensus lead intradía.
intra_res = intraday_probability(intra, k=k_intra) if intra is not None and not intra.empty else None

# Conversión orientativa USD→EUR.
eur_price = None
fx = daily.get(CFG.eurusd)
if fx is not None and not fx.empty:
    eurusd = float(fx["Close"].dropna().iloc[-1])
    if eurusd > 0:
        eur_price = spot / eurusd

# Gamma live: visible, pero sin peso en dirección todavía.
gamma = None
gex_by = None
gamma_error = None
if "token" not in locals():
    token = ""
    max_dte = CFG.options_max_dte
if token:
    try:
        chain = fetch_live_chain(CFG.ticker, token=token)
        gamma, gex_by = compute_gamma_metrics(chain, spot=spot, max_dte=max_dte)
    except Exception as e:
        gamma_error = str(e)

ai5 = context.get("ai_ret5")
beta = context.get("ai_beta")
c1, c2, c3, c4 = st.columns(4)
c1.metric("IREN (USD)", f"${spot:.2f}")
c2.metric("Equivalente EUR", "—" if eur_price is None else f"€{eur_price:.2f}")
c3.metric("AI factor 5D", pct1(ai5))
c4.metric("Beta IREN/IA", "—" if beta is None else f"{beta:.2f}x")


def status_for(p_up: float | None, ood: bool = False):
    if p_up is None or not np.isfinite(p_up):
        return "⚪ SIN DATOS"
    if ood:
        return "⚪ SIN VENTAJA · OOD"
    if p_up >= 0.60:
        return "🟢 ALCISTA*"
    if p_up <= 0.40:
        return "🔴 BAJISTA*"
    if p_up >= 0.53:
        return "🟡 LEVE ALCISTA*"
    if p_up <= 0.47:
        return "🟡 LEVE BAJISTA*"
    return "⚪ SIN VENTAJA"


st.subheader("🔮 Dirección + recorrido")
dirmap = {r.horizon: r for r in direction_res}
rangemap = {r.horizon: r for r in range_res}
rows = []

if intra_res is not None:
    p = intra_res.probabilities.get("close_up")
    rows.append({
        "Horizonte": "HOY†",
        "P ↑": pct(p),
        "P ↓": pct(None if p is None else 1 - p),
        "Fiabilidad": "NO VALIDADA",
        "Estado": "🟡 INTRADÍA V1 PROVISIONAL",
        "Rango probable": (
            f"{money(spot*(1+intra_res.q10_return))} – {money(spot*(1+intra_res.q90_return))}"
            if intra_res.q10_return is not None and intra_res.q90_return is not None else "—"
        ),
    })

for h in ["1D", "3D", "5D"]:
    d = dirmap.get(h)
    r = rangemap.get(h)
    p = d.probabilities.get("close_up") if d else None
    ood = bool(getattr(d, "ood", False)) if d else False
    rows.append({
        "Horizonte": h,
        "P ↑": pct(p),
        "P ↓": pct(None if p is None else 1 - p),
        "Fiabilidad": "PENDIENTE BACKTEST" + (" · OOD" if ood else ""),
        "Estado": status_for(p, ood),
        "Rango probable": (
            f"{money(spot*(1+r.q10_return))} – {money(spot*(1+r.q90_return))}"
            if r and r.q10_return is not None and r.q90_return is not None else "—"
        ),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "* Dirección V2-A.1 experimental hasta pasar el nuevo walk-forward. "
    "† HOY sigue con el motor intradía V1: el siguiente experimento intradía será el consensus lead/cascada del universo IA."
)

# 3D queda pre-registrado como target primario del siguiente experimento.
primary = dirmap.get("3D")
if primary is not None:
    p3 = primary.probabilities.get("close_up")
    state3 = status_for(p3, bool(getattr(primary, "ood", False)))
    msg = (
        f"TARGET PRIMARIO 3D · {state3} · P↑ {pct(p3)} / P↓ {pct(1-p3 if p3 is not None else None)}. "
        f"{context.get('transition', '—')} · {context.get('residual_label', '—')}."
    )
    if state3.startswith("🟢"):
        st.success(msg)
    elif state3.startswith("🔴"):
        st.error(msg)
    else:
        st.warning(msg)

st.subheader(f"🎯 Motor de recorrido V1 — P(tocar ${target:.2f})")
if state_range is not None and not state_range.empty:
    tcols = st.columns(3)
    for col, h in zip(tcols, [1, 3, 5]):
        p, n = target_touch_probability(state_range, target, h, k=k_range)
        col.metric(f"En ≤ {h} sesión{'es' if h > 1 else ''}", pct(p), f"n={n}")
else:
    st.info("Motor de recorrido no disponible en esta carga.")

with st.expander("🔬 Diagnóstico interno — residual / transición / drivers"):
    d3 = dirmap.get("3D")
    diag = {
        "Transición IA": context.get("transition", "—"),
        "AI factor 5D": pct1(context.get("ai_ret5")),
        "Breadth IA hoy": pct(context.get("ai_breadth1")),
        "Beta IREN/IA (causal)": "—" if context.get("ai_beta") is None else f"{context['ai_beta']:.3f}",
        "Residual IREN 1D": pct1(context.get("resid_1")),
        "Residual z60": "—" if context.get("resid_z60") is None else f"{context['resid_z60']:+.2f}σ",
        "OOD score 3D": "—" if d3 is None else f"{getattr(d3, 'ood_score', np.nan):.2f}",
        "Drivers 3D": [] if d3 is None else getattr(d3, "top_drivers", []),
    }
    st.json(diag)

    plot = iren.tail(260).copy()
    plot["EMA21"] = plot["Close"].ewm(span=21, adjust=False).mean()
    plot["EMA50"] = plot["Close"].ewm(span=50, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=plot.index, open=plot.Open, high=plot.High, low=plot.Low, close=plot.Close, name="IREN"))
    fig.add_trace(go.Scatter(x=plot.index, y=plot.EMA21, name="EMA21"))
    fig.add_trace(go.Scatter(x=plot.index, y=plot.EMA50, name="EMA50"))
    if gamma:
        for val, name in [
            (gamma.call_wall, "Call Wall"),
            (gamma.put_wall, "Put Wall"),
            (gamma.gamma_flip_proxy, "Gamma Flip proxy"),
        ]:
            if val:
                fig.add_hline(y=val, line_dash="dot", annotation_text=name)
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Gamma / opciones (NO integrado todavía en P↑/P↓)**")
    if gamma:
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Net GEX proxy", f"{gamma.net_gex:,.0f}")
        g2.metric("Call Wall", money(gamma.call_wall))
        g3.metric("Put Wall", money(gamma.put_wall))
        g4.metric("Gamma Flip proxy", money(gamma.gamma_flip_proxy))
        st.caption("OI no revela por sí solo quién está largo/corto. Gamma recibe peso sólo tras demostrar valor incremental con histórico.")
        if gex_by is not None and not gex_by.empty:
            near = gex_by[(gex_by.strike >= spot * 0.75) & (gex_by.strike <= spot * 1.25)]
            gf = go.Figure()
            gf.add_bar(x=near.strike, y=near.call_gex, name="Call GEX")
            gf.add_bar(x=near.strike, y=near.put_gex, name="Put GEX")
            gf.update_layout(barmode="relative", height=380)
            st.plotly_chart(gf, use_container_width=True)
    elif gamma_error:
        st.warning(f"ORATS no respondió: {gamma_error}")
    else:
        st.info("Gamma desactivado. Puedes añadir ORATS_TOKEN en ajustes avanzados.")

try:
    payload = {
        "model_direction": "V2-A.1 logistic",
        "model_range": "V1 nearest-neighbors",
        "direction": [
            {
                "horizon": r.horizon,
                "n_train": r.n,
                "p_close_up": r.probabilities.get("close_up"),
                "base_rate": getattr(r, "base_rate", None),
                "ood": getattr(r, "ood", None),
                "ood_score": getattr(r, "ood_score", None),
                "drivers": getattr(r, "top_drivers", []),
            }
            for r in direction_res
        ],
        "range": [
            {
                "horizon": r.horizon,
                "n": r.n,
                "q10": r.q10_return,
                "q90": r.q90_return,
                "touch": r.probabilities,
            }
            for r in range_res
        ],
        "intraday": ({
            "n": intra_res.n,
            "probabilities": intra_res.probabilities,
            "confidence": intra_res.confidence,
        } if intra_res else None),
        "context": context,
        "gamma": gamma.__dict__ if gamma else None,
    }
    save_snapshot("data/iren_engine.db", CFG.ticker, spot, payload)
except Exception:
    pass

with st.expander("Qué ha cambiado en V2-A.1"):
    st.markdown(
        """
- **Dirección y recorrido ya son dos motores distintos.** V2-A.1 decide P↑/P↓; V1 conserva rango y probabilidades de tocar niveles.
- **Residual IREN/IA:** estima con datos anteriores la beta de IREN al factor IA y mide si IREN está siendo más fuerte o más débil de lo esperado.
- **Transición:** mira si el risk-on/risk-off está acelerando, estabilizándose o agotándose; no interpreta simplemente rojo=malo / verde=bueno.
- **Logística regularizada:** sustituye nearest-neighbors como motor direccional para reducir sensibilidad a vecindarios históricos inestables.
- **Fiabilidad:** hemos eliminado el antiguo quality score. Hasta demostrar una fiabilidad real fuera de muestra, la app muestra **PENDIENTE BACKTEST**.
- **OOD:** es sólo un veto de “estado demasiado raro”, no una probabilidad de acierto.
- **3D es el target primario pre-registrado** del próximo backtest; 1D y 5D serán confirmatorios.
- **Lead-lag/consensus lead no está aún dentro.** Se estudiará aparte con barras intradía para no confundir una buena intuición con una feature no validada.
"""
    )
