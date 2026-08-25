from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from iren_engine.config import CFG
from iren_engine.market_data import load_daily, load_intraday
from iren_engine.probability import intraday_probability
from iren_engine.v2a import build_v2a_state, current_context, v2a_probabilities, v2a_target_touch_probability
from iren_engine.options_orats import fetch_live_chain, compute_gamma_metrics
from iren_engine.storage import save_snapshot

st.set_page_config(page_title="IREN Tactical Radar", page_icon="📡", layout="wide")
st.title("📡 IREN Tactical Radar — V2-A")
st.caption(
    "Oráculo táctico experimental: dirección + recorrido usando IREN, universo IA, fuerza relativa, "
    "régimen de mercado y recencia. Gamma se muestra aparte hasta validarlo con histórico."
)


def pct(v):
    return "—" if v is None or not np.isfinite(v) else f"{float(v):.0%}"


def money(v):
    return "—" if v is None or not np.isfinite(v) else f"${float(v):.2f}"


@st.cache_data(ttl=180)
def get_daily():
    tickers = list(dict.fromkeys([
        CFG.ticker, CFG.qqq, CFG.vix, CFG.btc, CFG.eurusd, *CFG.ai_basket
    ]))
    return load_daily(tickers, CFG.daily_period)


@st.cache_data(ttl=120)
def get_intraday():
    return load_intraday([CFG.ticker], CFG.intraday_period, CFG.intraday_interval)[CFG.ticker]


with st.sidebar:
    st.header("Control")
    reliability_threshold = st.slider("Umbral mínimo de fiabilidad", 45, 85, 65, 5)
    target = st.number_input("Nivel que quieres vigilar ($)", min_value=1.0, max_value=500.0, value=40.0, step=0.5)
    with st.expander("Ajustes avanzados"):
        k_daily = st.slider("Vecinos históricos V2-A", 60, 220, CFG.daily_neighbors, 10)
        k_intra = st.slider("Vecinos intradía provisional", 60, 300, CFG.intraday_neighbors, 10)
        max_dte = st.slider("Gamma: DTE máximo", 7, 90, CFG.options_max_dte, 1)
        token_default = ""
        try:
            token_default = st.secrets.get("ORATS_TOKEN", "")
        except Exception:
            token_default = os.getenv("ORATS_TOKEN", "")
        token = st.text_input("ORATS token (opcional)", value=token_default, type="password")
    st.caption("La fiabilidad V2-A todavía es PROVISIONAL hasta que el nuevo backtest la calibre.")

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
state = build_v2a_state(iren, market, ai_tickers=CFG.ai_basket)
daily_res = v2a_probabilities(
    state,
    k=k_daily,
    lookback_rows=CFG.v2_lookback_rows,
    recency_half_life=CFG.v2_recency_half_life,
)
intra_res = intraday_probability(intra, k=k_intra) if intra is not None and not intra.empty else None
context = current_context(state)

# Conversión orientativa USD->EUR desde el último EURUSD disponible.
eur_price = None
fx = daily.get(CFG.eurusd)
if fx is not None and not fx.empty:
    eurusd = float(fx["Close"].dropna().iloc[-1])
    if eurusd > 0:
        eur_price = spot / eurusd

# Gamma live, aún fuera de la probabilidad hasta disponer de validación histórica.
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("IREN (USD)", f"${spot:.2f}")
c2.metric("Equivalente EUR", "—" if eur_price is None else f"€{eur_price:.2f}")
c3.metric("Universo IA", context.get("ai_label", "—"))
c4.metric("IREN vs IA", context.get("relative_label", "—").replace("🟢 ", "").replace("🔴 ", "").replace("🟡 ", ""))

st.subheader("🔮 Dirección")


def status_for(p_up: float | None, quality: float | None):
    if p_up is None or not np.isfinite(p_up):
        return "⚪ SIN DATOS"
    if quality is not None and np.isfinite(quality) and quality < reliability_threshold:
        return "⚪ SIN VENTAJA"
    if p_up >= 0.60:
        return "🟢 ALCISTA"
    if p_up <= 0.40:
        return "🔴 BAJISTA"
    if p_up >= 0.53:
        return "🟡 LEVE ALCISTA"
    if p_up <= 0.47:
        return "🟡 LEVE BAJISTA"
    return "⚪ SIN VENTAJA"


rows = []
if intra_res is not None:
    p = intra_res.probabilities.get("close_up")
    rows.append({
        "Horizonte": "HOY*",
        "P ↑": pct(p),
        "P ↓": pct(None if p is None else 1 - p),
        "Fiabilidad": f"{intra_res.confidence}*",
        "Estado": "🟡 PROVISIONAL INTRADÍA",
        "Rango probable": f"{money(spot*(1+intra_res.q10_return))} – {money(spot*(1+intra_res.q90_return))}" if intra_res.q10_return is not None and intra_res.q90_return is not None else "—",
    })

for r in daily_res:
    p = r.probabilities.get("close_up")
    q = getattr(r, "quality_score", np.nan)
    rows.append({
        "Horizonte": r.horizon,
        "P ↑": pct(p),
        "P ↓": pct(None if p is None else 1 - p),
        "Fiabilidad": "—" if not np.isfinite(q) else f"{q:.0f}%",
        "Estado": status_for(p, q),
        "Rango probable": f"{money(spot*(1+r.q10_return))} – {money(spot*(1+r.q90_return))}" if r.q10_return is not None and r.q90_return is not None else "—",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("* HOY todavía usa el motor intradía V1 y queda marcado como provisional. V2-A valida primero 1D/3D/5D.")

valid = [r for r in daily_res if np.isfinite(getattr(r, "quality_score", np.nan))]
if valid:
    best = max(valid, key=lambda r: getattr(r, "quality_score", 0.0))
    pbest = best.probabilities.get("close_up", 0.5)
    qbest = getattr(best, "quality_score", np.nan)
    state_label = status_for(pbest, qbest)
    reasons = [context.get("ai_label", ""), context.get("relative_label", ""), context.get("volume_label", "")]
    reasons = [x for x in reasons if x]
    msg = f"{state_label} en {best.horizon}: P↑ {pct(pbest)}, P↓ {pct(1-pbest)}, fiabilidad provisional {qbest:.0f}%. " + " · ".join(reasons)
    if state_label.startswith("🟢"):
        st.success(msg)
    elif state_label.startswith("🔴"):
        st.error(msg)
    else:
        st.warning(msg)

st.subheader(f"🎯 Probabilidad de tocar ${target:.2f}")
tcols = st.columns(3)
for col, h in zip(tcols, [1, 3, 5]):
    p, n = v2a_target_touch_probability(
        state,
        target,
        h,
        k=k_daily,
        lookback_rows=CFG.v2_lookback_rows,
        recency_half_life=CFG.v2_recency_half_life,
    )
    col.metric(f"En ≤ {h} sesión{'es' if h > 1 else ''}", pct(p), f"n={n}")

with st.expander("🔬 Diagnóstico interno — no necesitas mirarlo para operar"):
    st.markdown("**Contexto V2-A**")
    diag = {
        "IA 5D": pct(context.get("ai_ret5")),
        "Breadth IA hoy": pct(context.get("ai_breadth1")),
        "IREN 5D": pct(context.get("ire_ret5")),
        "IREN menos IA 5D": pct(context.get("rel_ai_5")),
        "Volumen": context.get("volume_label", "—"),
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
        for val, name in [(gamma.call_wall, "Call Wall"), (gamma.put_wall, "Put Wall"), (gamma.gamma_flip_proxy, "Gamma Flip proxy")]:
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
        st.caption("OI no revela por sí solo quién está largo/corto. Gamma entra en el predictor sólo cuando tengamos histórico suficiente para validarlo.")
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
        "model": "V2-A",
        "daily": [
            {
                "horizon": r.horizon,
                "n": r.n,
                "probabilities": r.probabilities,
                "expected_return": r.expected_return,
                "q10": r.q10_return,
                "q90": r.q90_return,
                "quality_score": getattr(r, "quality_score", None),
                "confidence": r.confidence,
            }
            for r in daily_res
        ],
        "intraday": ({
            "n": intra_res.n,
            "probabilities": intra_res.probabilities,
            "expected_return": intra_res.expected_return,
            "confidence": intra_res.confidence,
        } if intra_res else None),
        "context": context,
        "gamma": gamma.__dict__ if gamma else None,
    }
    save_snapshot("data/iren_engine.db", CFG.ticker, spot, payload)
except Exception:
    pass

with st.expander("Qué significa 'fiabilidad'"):
    st.markdown(
        """
- **P↑ / P↓** son probabilidades empíricas ponderadas de estados históricos parecidos.
- **Fiabilidad V2-A** es, por ahora, una medida provisional de calidad de muestra + similitud + claridad de la señal. **No es todavía un hit-rate histórico.**
- El backtest V2-A sirve precisamente para convertir esta medida en una fiabilidad calibrada y comprobar si 60%, 70%, etc. significan lo que deberían.
- **SIN VENTAJA** no oculta la predicción: verás siempre P↑ y P↓, pero el motor avisa cuando la evidencia no alcanza tu umbral.
- El universo IA se pondera dinámicamente según su correlación reciente con IREN; así MU/MRVL/NVDA/etc. pueden ganar o perder importancia con el tiempo.
- V2-A todavía NO mete noticias ni Gamma dentro de la probabilidad. Es deliberado: cada capa debe demostrar que mejora el resultado antes de recibir peso.
"""
    )
