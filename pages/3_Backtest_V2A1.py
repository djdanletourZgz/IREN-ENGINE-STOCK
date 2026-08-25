from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from iren_engine.config import CFG
from iren_engine.market_data import load_daily
from iren_engine.v2a1 import (
    build_v2a1_state,
    walk_forward_v2a1,
    direction_metrics,
    calibration_bins_v2a1,
    performance_by_period_v2a1,
    performance_by_volatility_v2a1,
    performance_by_transition_v2a1,
    residual_deciles_v2a1,
    attach_empirical_reliability_v2a1,
    reliability_summary_v2a1,
)

BACKTEST_VERSION = "2.1-A"

st.set_page_config(page_title="IREN Backtest V2-A.1", page_icon="🧪", layout="wide")
st.title("🧪 IREN Engine — Backtest V2-A.1")
st.caption(
    "Prueba walk-forward del NUEVO motor direccional: logística regularizada + residual IREN/IA + transición. "
    "3D es el target primario pre-registrado; 1D y 5D son confirmatorios."
)

with st.sidebar:
    st.header("Test fijado")
    test_window = st.slider("Últimas fechas", 200, 500, 350, 25)
    min_train = st.slider("Mínimo entrenamiento", 60, 160, CFG.v2a1_min_train, 10)
    st.caption(
        "Parámetros no tuneados con este resultado: "
        f"beta EWLS/Huber {CFG.v2a1_beta_half_life:g}d · lookback {CFG.v2a1_model_lookback} · C={CFG.v2a1_logistic_c}."
    )


@st.cache_data(ttl=3600)
def get_state():
    tickers = list(dict.fromkeys([
        CFG.ticker, CFG.qqq, CFG.vix, CFG.btc, CFG.nvda, *CFG.ai_basket
    ]))
    daily = load_daily(tickers, CFG.daily_period)
    iren = daily[CFG.ticker]
    market = {k: v for k, v in daily.items() if k != CFG.ticker}
    return build_v2a1_state(
        iren,
        market,
        ai_tickers=CFG.ai_basket,
        beta_window=CFG.v2a1_beta_window,
        beta_half_life=CFG.v2a1_beta_half_life,
        beta_min_obs=CFG.v2a1_beta_min_obs,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def run_bt(state: pd.DataFrame, test_window: int, min_train: int):
    return walk_forward_v2a1(
        state,
        test_window=test_window,
        model_start=CFG.v2a1_model_start,
        lookback_rows=CFG.v2a1_model_lookback,
        min_train=min_train,
        c_value=CFG.v2a1_logistic_c,
    )


def build_zip(bt, metrics, bins, periods, vols, transitions, residuals, relsum, params):
    meta = {
        "backtest_version": BACKTEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": CFG.ticker,
        "primary_target": "3D direction",
        "direction_model": "L2 logistic regression",
        "model_start": CFG.v2a1_model_start,
        "beta_method": "causal EWLS weights + Huber robust regression",
        "beta_window": CFG.v2a1_beta_window,
        "beta_half_life": CFG.v2a1_beta_half_life,
        "model_lookback_rows": CFG.v2a1_model_lookback,
        "logistic_C": CFG.v2a1_logistic_c,
        "test_window": params[0],
        "min_train": params[1],
        "gamma_in_probability": False,
        "lead_lag_in_probability": False,
        "macro_liquidity_in_probability": False,
        "news_in_probability": False,
        "range_model_changed": False,
        "note": "V1 range/touch remains separate and is not retested here.",
    }
    readme = f"""IREN ENGINE — BACKTEST V{BACKTEST_VERSION}

OBJETIVO
Validar exclusivamente el nuevo motor DIRECCIONAL V2-A.1.

TARGET PRIMARIO PRE-REGISTRADO
3D.
1D y 5D son confirmatorios.

CAMBIOS
- nearest-neighbors deja de decidir dirección;
- logística L2 regularizada;
- beta causal IREN vs AI factor;
- residual IREN/AI;
- variables continuas de transición risk-on/risk-off;
- antiguo quality score eliminado;
- fiabilidad empírica usa sólo predicciones OOS anteriores ya resueltas.

NO INCLUIDO TODAVÍA
- lead-lag intradía / consensus lead;
- Gamma/VEX dentro de P(up/down);
- M2/liquidez;
- noticias.

ARCHIVOS
- predictions_v2a1.csv: todas las predicciones walk-forward.
- direction_metrics.csv: Brier skill, señales fuertes y retorno firmado.
- calibration_bins.csv: calibración P(up).
- performance_by_period.csv: T1/T2/T3.
- performance_by_volatility.csv: diagnóstico del posible cambio de vol.
- performance_by_transition.csv: comportamiento por transición IA.
- residual_deciles.csv: monotonicidad del residual.
- reliability_summary.csv: si la fiabilidad empírica es monotónica o no.
- metadata.json
- README.txt
"""
    raw = bt.copy()
    if "date" in raw:
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions_v2a1.csv", raw.to_csv(index=False))
        z.writestr("direction_metrics.csv", metrics.to_csv(index=False))
        z.writestr("calibration_bins.csv", bins.to_csv(index=False))
        z.writestr("performance_by_period.csv", periods.to_csv(index=False))
        z.writestr("performance_by_volatility.csv", vols.to_csv(index=False))
        z.writestr("performance_by_transition.csv", transitions.to_csv(index=False))
        z.writestr("residual_deciles.csv", residuals.to_csv(index=False))
        z.writestr("reliability_summary.csv", relsum.to_csv(index=False))
        z.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        z.writestr("README.txt", readme)
    return buffer.getvalue()


try:
    state = get_state()
except Exception as e:
    st.error(f"No se pudieron cargar datos: {e}")
    st.stop()

if st.button("▶️ Ejecutar V2-A.1", type="primary"):
    with st.spinner("Entrenando cada fecha sólo con información que ya existía entonces..."):
        st.session_state["bt_v2a1"] = run_bt(state, test_window, min_train)
        st.session_state["bt_v2a1_params"] = (test_window, min_train)

if "bt_v2a1" not in st.session_state:
    st.info("Pulsa **Ejecutar V2-A.1**. Este test ya no vuelve a ejecutar V1 y V2-A completos, para no hacerte esperar 10 minutos sin necesidad.")
    st.stop()

bt = st.session_state["bt_v2a1"]
if bt.empty:
    st.error("No hay muestra suficiente con estos parámetros.")
    st.stop()

bt_rel = attach_empirical_reliability_v2a1(bt)
metrics = direction_metrics(bt_rel)
bins = calibration_bins_v2a1(bt_rel)
periods = performance_by_period_v2a1(bt_rel)
vols = performance_by_volatility_v2a1(bt_rel)
transitions = performance_by_transition_v2a1(bt_rel)
residuals = residual_deciles_v2a1(bt_rel)
relsum = reliability_summary_v2a1(bt_rel)

st.subheader("🏁 Resultado principal")
show = metrics.copy()
for c in ["Brier", "Brier benchmark", "Brier skill", "Cobertura", "Acierto señales", "Retorno firmado medio", "OOD"]:
    if c in show:
        show[c] = show[c].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show, use_container_width=True, hide_index=True)

m3 = metrics[metrics["Horizonte"] == "3D"]
if not m3.empty:
    r = m3.iloc[0]
    skill = r["Brier skill"]
    hit = r["Acierto señales"]
    signed = r["Retorno firmado medio"]
    if pd.notna(skill) and skill > 0.02 and pd.notna(hit) and hit >= 0.55 and pd.notna(signed) and signed > 0:
        st.success(f"3D PASA el primer filtro: Brier skill {skill:+.1%} · señales {hit:.1%} · retorno firmado {signed:+.1%}. Falta estabilidad T3 y fiabilidad.")
    else:
        st.warning(f"3D todavía NO demuestra edge limpio: Brier skill {skill:+.1%} · señales {hit:.1%} · retorno firmado {signed:+.1%}.")

params = st.session_state.get("bt_v2a1_params", (test_window, min_train))
feedback_zip = build_zip(bt_rel, metrics, bins, periods, vols, transitions, residuals, relsum, params)
st.download_button(
    "⬇️ DESCARGAR V2-A.1 COMPLETO PARA FEEDBACK (.ZIP)",
    data=feedback_zip,
    file_name=f"IREN_backtest_feedback_v{BACKTEST_VERSION}.zip",
    mime="application/zip",
    type="primary",
)

st.subheader("¿Se arregla el problema T3?")
show_periods = periods.copy()
for c in ["Brier", "Brier benchmark", "Brier skill", "Cobertura", "Acierto señales", "Retorno firmado medio", "OOD"]:
    if c in show_periods:
        show_periods[c] = show_periods[c].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show_periods, use_container_width=True, hide_index=True)

st.subheader("¿Era cambio de volatilidad y no sólo concept drift?")
show_vol = vols.copy()
for c in ["Brier", "Brier benchmark", "Brier skill", "Cobertura", "Acierto señales", "Retorno firmado medio", "OOD"]:
    if c in show_vol:
        show_vol[c] = show_vol[c].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show_vol, use_container_width=True, hide_index=True)

st.subheader("Transiciones IA")
show_trans = transitions.copy()
for c in ["Brier", "Brier benchmark", "Brier skill", "Cobertura", "Acierto señales", "Retorno firmado medio", "OOD"]:
    if c in show_trans:
        show_trans[c] = show_trans[c].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show_trans, use_container_width=True, hide_index=True)

st.subheader("Residual IREN/IA — ¿hay monotonicidad?")
show_resid = residuals.copy()
for c in ["Retorno forward medio", "P cerrar arriba real"]:
    if c in show_resid:
        show_resid[c] = show_resid[c].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show_resid, use_container_width=True, hide_index=True)

st.subheader("Fiabilidad REAL provisional")
st.caption(
    "Cada predicción sólo mira llamadas OOS anteriores cuyo resultado ya se conocía entonces. "
    "Si los buckets altos no aciertan más que los bajos, esta fiabilidad se descarta."
)
show_rel = relsum.copy()
for c in ["Fiabilidad media", "Acierto real", "IC95 inferior medio"]:
    if c in show_rel:
        show_rel[c] = show_rel[c].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
st.dataframe(show_rel, use_container_width=True, hide_index=True)

st.subheader("Calibración P↑")
show_bins = bins.copy()
for c in ["Prob. media", "Ocurrió", "Diferencia"]:
    if c in show_bins:
        show_bins[c] = show_bins[c].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show_bins, use_container_width=True, hide_index=True)

with st.expander("Criterios que NO vamos a mover después de ver el resultado"):
    st.markdown(
        """
- **Target primario: 3D.** No cambiaremos a 1D o 5D sólo porque salga más bonito.
- Queremos **Brier skill positivo**, señales fuertes con acierto útil y retorno firmado positivo.
- T3 debe mejorar claramente frente al colapso anterior; un buen promedio con T3 malo no basta.
- El residual debe mostrar relación estable/monótona, no un único decil milagroso.
- La fiabilidad sólo se acepta si mayor fiabilidad declarada implica mayor acierto posterior fuera de muestra.
- OOD es un veto de distribución, no un confidence score.
- Si V2-A.1 no mejora de forma clara, no añadimos cinco capas para maquillarlo: revisamos la hipótesis.
"""
    )
