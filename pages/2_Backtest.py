from __future__ import annotations

import pandas as pd
import streamlit as st

from iren_engine.config import CFG
from iren_engine.market_data import load_daily
from iren_engine.probability import build_daily_state
from iren_engine.backtest import walk_forward_backtest, calibration_summary, directional_summary, human_verdict

st.set_page_config(page_title="IREN Backtest V1", page_icon="🧪", layout="wide")
st.title("🧪 IREN Engine — Backtest walk-forward")
st.caption("Simula qué habría dicho el motor en cada fecha histórica usando únicamente información disponible hasta ese día, y después lo compara con lo que ocurrió realmente.")

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
    return build_daily_state(daily[CFG.ticker], daily[CFG.btc], daily[CFG.qqq], daily.get(CFG.nvda))

@st.cache_data(ttl=3600, show_spinner=False)
def run_bt(state: pd.DataFrame, k: int, min_train: int, test_window: int):
    return walk_forward_backtest(state, k=k, min_train=min_train, test_window=test_window)

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
    st.info("Pulsa **Ejecutar backtest**. El test diario V1 evalúa técnicos + comportamiento de IREN + BTC + QQQ + NVDA. Todavía NO incluye Gamma histórico ni noticias.")
    st.stop()

bt = st.session_state["bt_v1"]
if bt.empty:
    st.error("No hay muestra suficiente para ejecutar el backtest.")
    st.stop()

cal = calibration_summary(bt)
dirsum = directional_summary(bt)
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
c2.metric("Brier skill medio", "—" if pd.isna(mean_skill) else f"{mean_skill:+.1%}", help=">0 significa que mejora al benchmark base-rate. Cuanto mayor, mejor.")
c3.metric("Acierto señales fuertes", "—" if pd.isna(mean_hit) else f"{mean_hit:.1%}", help="Sólo cuando P(subir) ≥60% o ≤40%.")
c4.metric("% días con señal fuerte", "—" if pd.isna(coverage) else f"{coverage:.1%}")

st.subheader("¿Cuando el modelo se moja, acierta?")
show_dir = dirsum.copy()
for col in ["Cobertura", "Acierto", "Retorno medio real"]:
    show_dir[col] = show_dir[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
st.dataframe(show_dir, use_container_width=True, hide_index=True)

st.subheader("Calibración de probabilidades")
st.caption("Ejemplo: si el motor dice 60% muchas veces, aproximadamente 6 de cada 10 deberían ocurrir. El Brier skill > 0 indica mejora frente a usar siempre la frecuencia histórica del evento.")
show_cal = cal.copy()
for col in ["Prob. media", "Ocurrió", "Error calibración", "Brier", "Brier base-rate", "Brier skill"]:
    show_cal[col] = show_cal[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
st.dataframe(show_cal, use_container_width=True, hide_index=True)

st.subheader("Predicción histórica vs realidad")
st.caption("Aquí puedes coger días concretos y comprobar literalmente qué probabilidad habría mostrado la V1 y qué sucedió después.")
recent = bt.sort_values(["date", "horizon"], ascending=[False, True]).head(60).copy()
recent["P cerrar arriba"] = recent["p_close_up"].map(lambda x: f"{x:.0%}")
recent["¿Cerró arriba?"] = recent["actual_close_up"].map(lambda x: "✅" if x else "❌")
recent["P tocar +5%"] = recent["p_touch_up_5"].map(lambda x: f"{x:.0%}")
recent["¿Tocó +5%?"] = recent["actual_touch_up_5"].map(lambda x: "✅" if x else "❌")
recent["P tocar -5%"] = recent["p_touch_down_5"].map(lambda x: f"{x:.0%}")
recent["¿Tocó -5%?"] = recent["actual_touch_down_5"].map(lambda x: "✅" if x else "❌")
recent["Retorno real"] = recent["actual_return"].map(lambda x: f"{x:+.1%}")
recent["date"] = pd.to_datetime(recent["date"]).dt.date
st.dataframe(
    recent[["date", "horizon", "P cerrar arriba", "¿Cerró arriba?", "P tocar +5%", "¿Tocó +5%?", "P tocar -5%", "¿Tocó -5%?", "Retorno real", "confidence", "n_neighbors"]],
    use_container_width=True,
    hide_index=True,
)

with st.expander("Cómo leer el test"):
    st.markdown("""
- **Walk-forward:** cada fecha histórica se trata como si fuese 'hoy'. Los datos posteriores quedan ocultos hasta después de emitir la predicción.
- **Brier skill > 0:** las probabilidades aportan más información que limitarse a usar la frecuencia histórica del evento. < 0 es mala señal.
- **Acierto señales fuertes:** porcentaje de aciertos cuando el modelo dice ≥60% de subir o ≤40% de subir.
- **Cobertura:** qué porcentaje de días se atreve a dar una señal fuerte. Un 80% de acierto sobre 3 días no vale gran cosa; por eso enseñamos también la muestra.
- **Este test NO valida Gamma:** la V1 todavía separa Gamma del modelo probabilístico. Para validarlo necesitamos histórico de cadenas de opciones/GEX.
- **Tampoco es aún un backtest de dinero real:** faltan reglas exactas de entrada/salida, spread, slippage, comisiones e impuestos. Primero estamos comprobando si las probabilidades contienen información.
""")
