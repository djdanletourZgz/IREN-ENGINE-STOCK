from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from iren_engine.config import CFG
from iren_engine.market_data import load_daily
from iren_engine.probability import build_daily_state, daily_probabilities
from iren_engine.v2a import build_v2a_state, v2a_probabilities
from iren_engine.backtest import (
    walk_forward_backtest,
    calibration_summary,
    directional_summary,
    calibration_bins,
    performance_by_period,
    performance_by_ai_regime,
    model_comparison,
    human_verdict,
)

BACKTEST_VERSION = "2.0-A"

st.set_page_config(page_title="IREN Backtest V2-A", page_icon="🧪", layout="wide")
st.title("🧪 IREN Engine — V1 vs V2-A")
st.caption(
    "Misma prueba walk-forward para saber si añadir universo IA + fuerza relativa + recencia "
    "mejora de verdad la predicción direccional. Sin mirar datos futuros."
)

with st.sidebar:
    st.header("Configuración del test")
    k = st.slider("Vecinos históricos", 60, 200, CFG.daily_neighbors, 10)
    test_window = st.slider("Últimas fechas de test", 150, 600, 350, 25)
    min_train = st.slider("Mínimo de días previos", 150, 350, 220, 10)
    st.caption("Ejecuta primero con 120 vecinos / 350 fechas / 220 días para comparar con V1.1.")


@st.cache_data(ttl=3600)
def get_states():
    tickers = list(dict.fromkeys([
        CFG.ticker, CFG.btc, CFG.qqq, CFG.nvda, CFG.vix, *CFG.ai_basket
    ]))
    daily = load_daily(tickers, CFG.daily_period)
    v1 = build_daily_state(
        daily[CFG.ticker], daily[CFG.btc], daily[CFG.qqq], daily.get(CFG.nvda)
    )
    market = {k: v for k, v in daily.items() if k != CFG.ticker}
    v2 = build_v2a_state(daily[CFG.ticker], market, ai_tickers=CFG.ai_basket)
    return v1, v2


@st.cache_data(ttl=3600, show_spinner=False)
def run_both(v1: pd.DataFrame, v2: pd.DataFrame, k: int, min_train: int, test_window: int):
    bt1 = walk_forward_backtest(
        v1,
        k=k,
        min_train=min_train,
        test_window=test_window,
        predictor=daily_probabilities,
        model_name="V1",
    )
    bt2 = walk_forward_backtest(
        v2,
        k=k,
        min_train=min_train,
        test_window=test_window,
        predictor=v2a_probabilities,
        model_name="V2-A",
        predictor_kwargs={
            "lookback_rows": CFG.v2_lookback_rows,
            "recency_half_life": CFG.v2_recency_half_life,
        },
    )
    return pd.concat([bt1, bt2], ignore_index=True)


def build_feedback_zip(bt, cal, dirsum, bins, periods, regimes, comparison, params, verdict_v1, verdict_v2):
    k0, min_train0, test_window0 = params
    meta = {
        "backtest_version": BACKTEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": CFG.ticker,
        "models": {
            "V1": ["IREN técnicos/precio", "BTC", "QQQ", "NVDA"],
            "V2-A": [
                "IREN técnicos/precio",
                "AI factor dinámico (SOXX/NVDA/MU/MRVL/AVGO/AMD/VRT/CRWV/NBIS)",
                "fuerza relativa IREN vs IA",
                "breadth/dispersion/correlación IA",
                "QQQ/VIX/BTC contexto",
                "ponderación por recencia",
            ],
        },
        "gamma_included_in_probability": False,
        "news_included": False,
        "k_neighbors": int(k0),
        "min_train_days": int(min_train0),
        "test_window_dates": int(test_window0),
        "v2_lookback_rows": CFG.v2_lookback_rows,
        "v2_recency_half_life": CFG.v2_recency_half_life,
        "verdict_v1": verdict_v1,
        "verdict_v2a": verdict_v2,
    }
    readme = f"""IREN ENGINE — FEEDBACK BACKTEST V{BACKTEST_VERSION}

OBJETIVO
Comparar V1 contra V2-A sin cambiar la vara de medir.

V2-A añade universo IA, fuerza relativa y recencia. Todavía NO integra Gamma ni noticias.

ARCHIVOS
- predictions_all.csv: todas las predicciones y realidad de ambos modelos.
- model_comparison.csv: comparación directa V1 vs V2-A.
- calibration.csv: Brier/Brier skill por modelo/horizonte/evento.
- directional_signals.csv: señales >=60% o <=40%.
- calibration_bins.csv: calibración por rangos de probabilidad.
- performance_by_period.csv: estabilidad cronológica (3 tramos).
- performance_by_ai_regime.csv: comportamiento por RISK-ON / MIXTO / RISK-OFF IA.
- metadata.json
- README.txt

VEREDICTO V1
{verdict_v1}

VEREDICTO V2-A
{verdict_v2}
"""
    raw = bt.copy()
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions_all.csv", raw.to_csv(index=False))
        z.writestr("model_comparison.csv", comparison.to_csv(index=False))
        z.writestr("calibration.csv", cal.to_csv(index=False))
        z.writestr("directional_signals.csv", dirsum.to_csv(index=False))
        z.writestr("calibration_bins.csv", bins.to_csv(index=False))
        z.writestr("performance_by_period.csv", periods.to_csv(index=False))
        z.writestr("performance_by_ai_regime.csv", regimes.to_csv(index=False))
        z.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        z.writestr("README.txt", readme)
    return buffer.getvalue()


try:
    state_v1, state_v2 = get_states()
except Exception as e:
    st.error(f"No se pudieron descargar los datos históricos: {e}")
    st.stop()

if st.button("▶️ Ejecutar V1 vs V2-A", type="primary"):
    with st.spinner("Reproduciendo V1 y V2-A fecha por fecha sin mirar el futuro..."):
        st.session_state["bt_v2a"] = run_both(state_v1, state_v2, k, min_train, test_window)
        st.session_state["bt_v2a_params"] = (k, min_train, test_window)

if "bt_v2a" not in st.session_state:
    st.info("Pulsa **Ejecutar V1 vs V2-A**. Después descarga el ZIP completo y envíamelo.")
    st.stop()

bt = st.session_state["bt_v2a"]
if bt.empty:
    st.error("No hay muestra suficiente.")
    st.stop()

cal = calibration_summary(bt)
dirsum = directional_summary(bt)
bins = calibration_bins(bt)
periods = performance_by_period(bt)
regimes = performance_by_ai_regime(bt)
comparison = model_comparison(bt)
verdict_v1 = human_verdict(cal, dirsum, model="V1")
verdict_v2 = human_verdict(cal, dirsum, model="V2-A")

st.subheader("🏁 Comparación directa")
show_cmp = comparison.copy()
for c in ["Brier skill dirección", "Acierto señales fuertes", "Cobertura señales fuertes", "Retorno firmado medio"]:
    if c in show_cmp:
        show_cmp[c] = show_cmp[c].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show_cmp, use_container_width=True, hide_index=True)

v2row = comparison[comparison["Modelo"] == "V2-A"]
v1row = comparison[comparison["Modelo"] == "V1"]
if not v2row.empty and not v1row.empty:
    dskill = float(v2row.iloc[0]["Brier skill dirección"] - v1row.iloc[0]["Brier skill dirección"])
    dhit = float(v2row.iloc[0]["Acierto señales fuertes"] - v1row.iloc[0]["Acierto señales fuertes"])
    if dskill > 0 and dhit > 0:
        st.success(f"V2-A mejora V1 en ambos tests principales: Δ Brier skill {dskill:+.1%} · Δ acierto señales {dhit:+.1%}.")
    else:
        st.warning(f"V2-A todavía NO mejora limpiamente V1: Δ Brier skill {dskill:+.1%} · Δ acierto señales {dhit:+.1%}.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**V1**")
    st.write(verdict_v1)
with c2:
    st.markdown("**V2-A**")
    st.write(verdict_v2)

params = st.session_state.get("bt_v2a_params", (k, min_train, test_window))
feedback_zip = build_feedback_zip(bt, cal, dirsum, bins, periods, regimes, comparison, params, verdict_v1, verdict_v2)
st.download_button(
    "⬇️ DESCARGAR TODO V1 vs V2-A PARA FEEDBACK (.ZIP)",
    data=feedback_zip,
    file_name=f"IREN_backtest_feedback_v{BACKTEST_VERSION}.zip",
    mime="application/zip",
    type="primary",
)
st.caption("Mándame este único ZIP. Incluye resultados completos, periodos y regímenes IA.")

st.subheader("¿Cuando se moja, acierta?")
show_dir = dirsum.copy()
for col in ["Cobertura", "Acierto", "Retorno medio real", "Calidad media"]:
    if col in show_dir:
        show_dir[col] = show_dir[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}" if col != "Calidad media" else f"{x:.0f}%")
st.dataframe(show_dir, use_container_width=True, hide_index=True)

st.subheader("Calibración dirección")
core_cal = cal[cal["Evento"] == "Cerrar arriba"].copy()
for col in ["Prob. media", "Ocurrió", "Error calibración", "Brier", "Benchmark prob. media", "Brier benchmark rolling", "Brier skill"]:
    if col in core_cal:
        core_cal[col] = core_cal[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(core_cal, use_container_width=True, hide_index=True)

st.subheader("¿Sigue funcionando con el tiempo?")
show_periods = periods.copy()
for col in ["Brier skill", "Acierto señales", "Cobertura"]:
    if col in show_periods:
        show_periods[col] = show_periods[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show_periods, use_container_width=True, hide_index=True)

st.subheader("¿Qué pasa según el régimen del universo IA?")
if regimes.empty:
    st.info("No hay muestra suficiente por régimen.")
else:
    show_reg = regimes.copy()
    for col in ["Brier skill", "Acierto señales", "Cobertura", "Retorno medio IREN"]:
        if col in show_reg:
            show_reg[col] = show_reg[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
    st.dataframe(show_reg, use_container_width=True, hide_index=True)

with st.expander("Calibración por bins y últimas predicciones"):
    show_bins = bins.copy()
    for col in ["Prob. media", "Ocurrió", "Diferencia"]:
        if col in show_bins:
            show_bins[col] = show_bins[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
    st.dataframe(show_bins, use_container_width=True, hide_index=True)

    recent = bt.sort_values(["date", "model", "horizon"], ascending=[False, True, True]).head(80).copy()
    recent["P↑"] = recent["p_close_up"].map(lambda x: f"{x:.0%}")
    recent["Real"] = recent["actual_close_up"].map(lambda x: "↑" if x else "↓")
    recent["Retorno"] = recent["actual_return"].map(lambda x: f"{x:+.1%}")
    recent["date"] = pd.to_datetime(recent["date"]).dt.date
    cols = ["date", "model", "horizon", "P↑", "Real", "Retorno", "confidence", "quality_score", "ai_ret5", "rel_ai_5"]
    st.dataframe(recent[[c for c in cols if c in recent.columns]], use_container_width=True, hide_index=True)

with st.expander("Qué estamos validando"):
    st.markdown(
        """
- **V1** = técnicos/precio IREN + BTC + QQQ + NVDA.
- **V2-A** = técnicos/precio IREN + factor IA dinámico + fuerza relativa + breadth/dispersion + QQQ/VIX/BTC + recencia.
- El factor IA pondera cada componente según su correlación reciente con IREN; no fijamos para siempre que NVDA/MU/MRVL pesen igual.
- **Gamma y noticias siguen fuera**: no entrarán en P↑/P↓ hasta demostrar que añaden edge.
- `performance_by_period.csv` sirve para detectar concept drift: una mejora que sólo existe en un tramo no nos vale.
- `performance_by_ai_regime.csv` comprueba si el modelo funciona distinto cuando el universo IA está risk-on, mixto o risk-off.
"""
    )
