from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from iren_engine.config import CFG
from iren_engine.market_data import load_daily
from iren_engine.probability import build_daily_state
from iren_engine.backtest import (
    walk_forward_backtest,
    calibration_summary,
    directional_summary,
    calibration_bins,
    human_verdict,
)

BACKTEST_VERSION = "1.1"

st.set_page_config(page_title="IREN Backtest V1.1", page_icon="🧪", layout="wide")
st.title("🧪 IREN Engine — Backtest walk-forward")
st.caption(
    "Simula qué habría dicho el motor en cada fecha histórica usando únicamente "
    "información disponible hasta ese día, y después lo compara con lo que ocurrió realmente."
)

with st.sidebar:
    st.header("Configuración del test")
    k = st.slider("Vecinos históricos", 60, 200, CFG.daily_neighbors, 10)
    test_window = st.slider("Últimas fechas de test", 150, 600, 350, 25)
    min_train = st.slider("Mínimo de días previos", 150, 350, 220, 10)
    st.caption("Más fechas = test más robusto pero tarda más. 350 es un buen punto de partida.")


@st.cache_data(ttl=3600)
def get_state():
    tickers = [CFG.ticker, CFG.btc, CFG.qqq, CFG.nvda]
    daily = load_daily(tickers, CFG.daily_period)
    return build_daily_state(
        daily[CFG.ticker],
        daily[CFG.btc],
        daily[CFG.qqq],
        daily.get(CFG.nvda),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def run_bt(state: pd.DataFrame, k: int, min_train: int, test_window: int):
    return walk_forward_backtest(
        state,
        k=k,
        min_train=min_train,
        test_window=test_window,
    )


def build_feedback_zip(
    bt: pd.DataFrame,
    cal: pd.DataFrame,
    dirsum: pd.DataFrame,
    bins: pd.DataFrame,
    params: tuple[int, int, int],
    verdict: str,
) -> bytes:
    """Paquete único para enviar el backtest completo como feedback."""
    k0, min_train0, test_window0 = params
    unique_dates = int(bt["date"].nunique()) if not bt.empty else 0

    meta = {
        "backtest_version": BACKTEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": CFG.ticker,
        "features": ["IREN técnicos/precio", "BTC", "QQQ", "NVDA"],
        "gamma_included": False,
        "news_included": False,
        "k_neighbors": int(k0),
        "min_train_days": int(min_train0),
        "test_window_dates": int(test_window0),
        "dates_simulated": unique_dates,
        "horizons": sorted(bt["horizon"].dropna().astype(int).unique().tolist()),
        "verdict": verdict,
        "benchmark": (
            "rolling base-rate: en cada fecha sólo usa resultados históricos "
            "cuyo horizonte ya había terminado"
        ),
    }

    readme = f"""IREN ENGINE — PAQUETE DE FEEDBACK BACKTEST V{BACKTEST_VERSION}

Este ZIP está pensado para enviarlo completo como feedback.

ARCHIVOS
- predictions_all.csv: TODAS las predicciones walk-forward y resultados reales.
- calibration.csv: Brier, benchmark rolling y Brier skill por horizonte/evento.
- directional_signals.csv: rendimiento cuando el modelo da señal fuerte.
- calibration_bins.csv: qué ocurre realmente en cada rango de P(cerrar arriba).
- metadata.json: parámetros exactos del test y versión.
- README.txt: este fichero.

IMPORTANTE
- Gamma histórico: NO incluido.
- Noticias: NO incluidas.
- El benchmark rolling NO mira la frecuencia futura del periodo de prueba.
- Esto valida información predictiva; todavía no es un backtest monetario con
  reglas de compra/venta, spread, slippage, comisiones e impuestos.

VEREDICTO
{verdict}
"""

    raw = bt.copy()
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions_all.csv", raw.to_csv(index=False))
        z.writestr("calibration.csv", cal.to_csv(index=False))
        z.writestr("directional_signals.csv", dirsum.to_csv(index=False))
        z.writestr("calibration_bins.csv", bins.to_csv(index=False))
        z.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
        z.writestr("README.txt", readme)
    return buffer.getvalue()


try:
    state = get_state()
except Exception as e:
    st.error(f"No se pudieron descargar los datos históricos: {e}")
    st.stop()

if st.button("▶️ Ejecutar backtest", type="primary"):
    with st.spinner("Reproduciendo el pasado sin mirar el futuro..."):
        st.session_state["bt_v1"] = run_bt(state, k, min_train, test_window)
        st.session_state["bt_params"] = (k, min_train, test_window)

if "bt_v1" not in st.session_state:
    st.info(
        "Pulsa **Ejecutar backtest**. El test diario V1.1 evalúa técnicos + "
        "comportamiento de IREN + BTC + QQQ + NVDA. Todavía NO incluye Gamma histórico ni noticias."
    )
    st.stop()

bt = st.session_state["bt_v1"]
if bt.empty:
    st.error("No hay muestra suficiente para ejecutar el backtest.")
    st.stop()

cal = calibration_summary(bt)
dirsum = directional_summary(bt)
bins = calibration_bins(bt)
verdict = human_verdict(cal, dirsum)

st.subheader("Conclusión rápida")
if verdict.startswith("🟢"):
    st.success(verdict)
elif verdict.startswith("🟡"):
    st.warning(verdict)
else:
    st.error(verdict)

unique_dates = bt["date"].nunique()
core = cal[cal["Evento"] == "Cerrar arriba"].copy()
mean_skill = core["Brier skill"].dropna().mean() if not core.empty else float("nan")
mean_hit = dirsum["Acierto"].dropna().mean() if not dirsum.empty else float("nan")
coverage = dirsum["Cobertura"].dropna().mean() if not dirsum.empty else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fechas simuladas", f"{unique_dates}")
c2.metric(
    "Brier skill medio",
    "—" if pd.isna(mean_skill) else f"{mean_skill:+.1%}",
    help=(
        ">0 significa que mejora a un benchmark rolling que en cada fecha "
        "sólo conoce el pasado. Cuanto mayor, mejor."
    ),
)
c3.metric(
    "Acierto señales fuertes",
    "—" if pd.isna(mean_hit) else f"{mean_hit:.1%}",
    help="Sólo cuando P(subir) ≥60% o ≤40%.",
)
c4.metric("% días con señal fuerte", "—" if pd.isna(coverage) else f"{coverage:.1%}")

params = st.session_state.get("bt_params", (k, min_train, test_window))
feedback_zip = build_feedback_zip(bt, cal, dirsum, bins, params, verdict)
st.download_button(
    "⬇️ DESCARGAR TODO EL TEST PARA FEEDBACK (.ZIP)",
    data=feedback_zip,
    file_name=f"IREN_backtest_feedback_v{BACKTEST_VERSION}.zip",
    mime="application/zip",
    type="primary",
    help=(
        "Un solo archivo con todas las predicciones, calibración, señales, "
        "parámetros y resultados. Mándame directamente este ZIP."
    ),
)
st.caption(
    "Este botón descarga el test completo, no sólo las 60 filas visibles. "
    "Es el archivo ideal para mandármelo después de cada iteración."
)

st.subheader("¿Cuando el modelo se moja, acierta?")
show_dir = dirsum.copy()
for col in ["Cobertura", "Acierto", "Retorno medio real"]:
    show_dir[col] = show_dir[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
st.dataframe(show_dir, use_container_width=True, hide_index=True)

st.subheader("Calibración de probabilidades")
st.caption(
    "El Brier skill compara ahora contra un benchmark rolling que NO conoce el futuro "
    "del periodo de test. >0 significa que el motor añade información frente a esa referencia."
)
show_cal = cal.copy()
for col in [
    "Prob. media",
    "Ocurrió",
    "Error calibración",
    "Brier",
    "Benchmark prob. media",
    "Brier benchmark rolling",
    "Brier skill",
]:
    if col in show_cal.columns:
        show_cal[col] = show_cal[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
st.dataframe(show_cal, use_container_width=True, hide_index=True)

st.subheader("Calibración de P(cerrar arriba)")
st.caption(
    "Esto ayuda a detectar inversión de señal: por ejemplo, si el motor dice ≥60% "
    "pero históricamente sube sólo el 40%, tenemos una pista clara de qué está fallando."
)
show_bins = bins.copy()
for col in ["Prob. media", "Ocurrió", "Diferencia"]:
    if col in show_bins.columns:
        show_bins[col] = show_bins[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.1%}")
st.dataframe(show_bins, use_container_width=True, hide_index=True)

st.subheader("Predicción histórica vs realidad")
st.caption(
    "Aquí ves las últimas 60 filas. El ZIP de feedback incluye TODAS las filas del test."
)
recent = bt.sort_values(["date", "horizon"], ascending=[False, True]).head(60).copy()
recent["P cerrar arriba"] = recent["p_close_up"].map(lambda x: f"{x:.0%}")
recent["Benchmark subir"] = recent["base_p_close_up"].map(
    lambda x: "—" if pd.isna(x) else f"{x:.0%}"
)
recent["¿Cerró arriba?"] = recent["actual_close_up"].map(lambda x: "✅" if x else "❌")
recent["P tocar +5%"] = recent["p_touch_up_5"].map(lambda x: f"{x:.0%}")
recent["¿Tocó +5%?"] = recent["actual_touch_up_5"].map(lambda x: "✅" if x else "❌")
recent["P tocar -5%"] = recent["p_touch_down_5"].map(lambda x: f"{x:.0%}")
recent["¿Tocó -5%?"] = recent["actual_touch_down_5"].map(lambda x: "✅" if x else "❌")
recent["Retorno real"] = recent["actual_return"].map(lambda x: f"{x:+.1%}")
recent["date"] = pd.to_datetime(recent["date"]).dt.date
st.dataframe(
    recent[
        [
            "date",
            "horizon",
            "P cerrar arriba",
            "Benchmark subir",
            "¿Cerró arriba?",
            "P tocar +5%",
            "¿Tocó +5%?",
            "P tocar -5%",
            "¿Tocó -5%?",
            "Retorno real",
            "confidence",
            "n_neighbors",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

with st.expander("Cómo leer el test"):
    st.markdown(
        """
- **Walk-forward:** cada fecha histórica se trata como si fuese 'hoy'. Los datos posteriores quedan ocultos hasta después de emitir la predicción.
- **Benchmark rolling:** para cada fecha y horizonte usa sólo resultados que ya habían terminado entonces. No conoce el futuro del periodo de test.
- **Brier skill > 0:** el motor mejora al benchmark rolling. < 0 es mala señal.
- **Acierto señales fuertes:** porcentaje de aciertos cuando el modelo dice ≥60% de subir o ≤40% de subir.
- **Cobertura:** qué porcentaje de días se atreve a dar una señal fuerte.
- **Calibración por rangos:** comprueba si un 60% del modelo se comporta realmente como un ~60%.
- **Este test NO valida Gamma:** para Gamma necesitamos histórico de cadenas de opciones/GEX.
- **Tampoco es aún un backtest de dinero real:** faltan reglas exactas de entrada/salida, spread, slippage, comisiones e impuestos.
"""
    )
