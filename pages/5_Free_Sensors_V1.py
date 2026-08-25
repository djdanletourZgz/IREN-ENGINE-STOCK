from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from iren_engine.config import CFG
from iren_engine.market_data import load_daily, load_intraday
from iren_engine.free_sensors import (
    FREE_CFG,
    add_cross_day_returns,
    attach_prior_finra_to_divergence,
    build_divergence_events_5m,
    build_price_volume_flow,
    combine_divergence_and_flow,
    detect_flow_events,
    divergence_life_summary,
    divergence_stability,
    fetch_finra_short_volume,
    finra_features,
    finra_short_summary,
    flow_summary,
    hourly_divergence_validation,
)

VERSION = "FREE-SENSORS-V1"
PEERS = ("SOXX", "NVDA", "MU", "MRVL", "AVGO", "AMD", "VRT")

st.set_page_config(page_title="IREN Free Sensors V1", page_icon="🛰️", layout="wide")
st.title("🛰️ IREN — FREE SENSORS V1")
st.caption(
    "Cero datos de pago. No busca cientos de patrones: valida el interruptor de divergencia y añade dos sensores "
    "independientes y baratos — flujo OHLCV y FINRA short-sale volume. Nada entra todavía en el oráculo principal."
)

st.info(
    "PROTOCOLO FIJO: no hay sliders en esta primera ronda. La señal de divergencia queda pre-registrada con "
    "30m · AI ≥1σ · gap ≥0,75σ · breadth ≥70% · cooldown 120m. Ahora comprobamos cuánto dura y si otros sensores aportan valor incremental."
)


@st.cache_data(ttl=1800)
def get_5m_market():
    return load_intraday([CFG.ticker, *PEERS], period="60d", interval="5m")


@st.cache_data(ttl=3600)
def get_hourly_market():
    # yfinance permite una historia intradía mucho mayor a 1h que a 5m.
    return load_intraday([CFG.ticker, *PEERS], period="2y", interval="1h")


@st.cache_data(ttl=3600)
def get_daily_market():
    return load_daily([CFG.ticker, *PEERS], period="5y")


@st.cache_data(ttl=21600, show_spinner=False)
def get_finra():
    return fetch_finra_short_volume(CFG.ticker, calendar_days=FREE_CFG.finra_calendar_days, workers=8)


def pct(x):
    return "—" if x is None or pd.isna(x) else f"{float(x):.1%}"


def sigma(x):
    return "—" if x is None or pd.isna(x) else f"{float(x):.2f}σ"


def pretty_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    z = df.copy()
    for c in ["Acierto direccion", "Wilson low", "Wilson high", "Ret firmado medio", "Ret firmado mediano"]:
        if c in z:
            z[c] = z[c].map(pct)
    return z


def build_zip(payload: dict[str, pd.DataFrame], metadata: dict) -> bytes:
    buf = io.BytesIO()
    readme = """IREN ENGINE — FREE SENSORS V1

OBJETIVO
1) Pre-registrar y volver a testear la señal de persistencia de divergencia IREN/AI descubierta post-hoc en Consensus Lead V1.
2) Medir la vida de la señal: 30m, 60m, 120m, cierre, +1D, +2D, +3D.
3) Validarla con barras 1h sobre más historia.
4) Probar un sensor gratuito de flujo/participación construido sólo con OHLCV.
5) Añadir FINRA Consolidated NMS short-sale volume como contexto diario conocido sólo DESPUÉS del cierre.
6) Medir sensores solos y combinados. Ningún resultado se despliega automáticamente.

IMPORTANTE FINRA
Short-sale volume NO es short interest. FINRA indica que el fichero cubre operaciones short públicamente difundidas y reportadas a sus TRFs/ADF, no una posición neta de cortos ni toda la actividad de exchanges.

CRITERIO
No aceptar un único porcentaje bonito. Exigir estabilidad temporal, horizontes vecinos, N razonable y mejora incremental al combinar sensores.
"""
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in payload.items():
            raw = df.copy() if df is not None else pd.DataFrame()
            for c in raw.columns:
                if "timestamp" in c.lower() or c == "date":
                    try:
                        raw[c] = pd.to_datetime(raw[c]).astype(str)
                    except Exception:
                        pass
            zf.writestr(f"{name}.csv", raw.to_csv(index=False))
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
        zf.writestr("README.txt", readme)
    return buf.getvalue()


if st.button("▶️ EJECUTAR FREE SENSORS V1", type="primary"):
    with st.spinner("Descargando datos gratuitos y ejecutando los tres sensores..."):
        try:
            m5 = get_5m_market()
            mh = get_hourly_market()
            md = get_daily_market()

            close5, div = build_divergence_events_5m(m5, CFG.ticker, PEERS)
            div = add_cross_day_returns(div, md.get(CFG.ticker, pd.DataFrame()))
            div_life = divergence_life_summary(div)
            div_stability = divergence_stability(div)

            flow_frame = build_price_volume_flow(m5.get(CFG.ticker, pd.DataFrame()))
            flow_events = detect_flow_events(flow_frame)
            flow_stats = flow_summary(flow_events)
            div_flow_events, combo_stats = combine_divergence_and_flow(div, flow_frame)

            hourly_events, hourly_stats = hourly_divergence_validation(mh, CFG.ticker, PEERS)

            finra_raw = get_finra()
            finra_feat = finra_features(finra_raw, md.get(CFG.ticker, pd.DataFrame()), z_window=FREE_CFG.finra_z_window)
            finra_stats = finra_short_summary(finra_feat)
            div_finra, div_finra_stats = attach_prior_finra_to_divergence(div, finra_feat)

            st.session_state["free_sensors_result"] = {
                "divergence_events": div,
                "divergence_life": div_life,
                "divergence_stability": div_stability,
                "flow_events": flow_events,
                "flow_summary": flow_stats,
                "divergence_flow_events": div_flow_events,
                "combination_summary": combo_stats,
                "hourly_events": hourly_events,
                "hourly_summary": hourly_stats,
                "finra_raw": finra_raw,
                "finra_features": finra_feat,
                "finra_summary": finra_stats,
                "divergence_finra_events": div_finra,
                "divergence_finra_summary": div_finra_stats,
            }
            st.session_state["free_sensors_meta"] = {
                "version": VERSION,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "ticker": CFG.ticker,
                "peers": list(PEERS),
                "five_minute_period": "60d",
                "hourly_period": "2y requested from yfinance; actual aligned sample reported in CSV",
                "finra_calendar_days_requested": FREE_CFG.finra_calendar_days,
                "divergence_definition": "AI >=1 sigma; breadth >=70%; directional IREN lag >=0.75 sigma; prediction follows IREN residual, opposite AI factor; 120m cooldown",
                "flow_definition": "30m signed-volume pressure + VWAP deviation; abs score >=0.35 and RVOL >=1.20; 120m cooldown",
                "finra_source": "Official FINRA Consolidated NMS Daily Short Sale Volume static files",
                "finra_note": "Short-sale volume is NOT short interest and is not a complete consolidated exchange short-volume measure.",
            }
        except Exception as exc:
            st.error(f"Falló FREE SENSORS V1: {exc}")
            st.stop()

if "free_sensors_result" not in st.session_state:
    st.info("Pulsa **EJECUTAR FREE SENSORS V1**. La primera carga de FINRA puede tardar algo porque descarga varios ficheros oficiales y luego queda cacheada.")
    st.stop()

R = st.session_state["free_sensors_result"]
M = st.session_state["free_sensors_meta"]

# -----------------------------------------------------------------------------
st.subheader("1️⃣ INTERRUPTOR #1 — ¿Cuánto dura la divergencia?")
div_life = R["divergence_life"]
if div_life.empty:
    st.error("No hay episodios de divergencia suficientes con el protocolo pre-registrado.")
else:
    c1, c2, c3 = st.columns(3)
    e = R["divergence_events"]
    c1.metric("Episodios 5m", str(len(e)))
    r120 = div_life[div_life["Horizonte"] == "120M"]
    r1d = div_life[div_life["Horizonte"] == "1D"]
    c2.metric("Acierto 120m", "—" if r120.empty else pct(r120.iloc[0]["Acierto direccion"]))
    c3.metric("Acierto +1D", "—" if r1d.empty else pct(r1d.iloc[0]["Acierto direccion"]))
    st.dataframe(pretty_summary(div_life), use_container_width=True, hide_index=True)
    st.caption("La dirección de esta señal ya está congelada ANTES de este test: seguimos la fuerza/debilidad propia de IREN (residual), no intentamos que IREN alcance al sector.")

st.markdown("**Estabilidad 5m — primera vs segunda mitad**")
if R["divergence_stability"].empty:
    st.info("Muestra insuficiente para una división útil.")
else:
    st.dataframe(pretty_summary(R["divergence_stability"]), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
st.subheader("2️⃣ VALIDACIÓN MÁS LARGA — barras 1h")
if R["hourly_summary"].empty:
    st.warning("Yahoo no devolvió suficientes barras 1h alineadas para validar esta capa.")
else:
    st.dataframe(pretty_summary(R["hourly_summary"]), use_container_width=True, hide_index=True)
    st.caption("Esto NO es una fuente independiente de Yahoo. Sirve para comprobar si la estructura sobrevive a otra granularidad y a una historia intradía mayor.")

# -----------------------------------------------------------------------------
st.subheader("3️⃣ INTERRUPTOR #2 — flujo/precio-volumen gratis")
if R["flow_summary"].empty:
    st.warning("No hubo suficientes eventos del sensor de flujo con los umbrales fijados.")
else:
    c1, c2 = st.columns(2)
    c1.metric("Eventos FLOW", str(len(R["flow_events"])))
    r = R["flow_summary"][R["flow_summary"]["Horizonte"] == "120M"]
    c2.metric("FLOW acierto 120m", "—" if r.empty else pct(r.iloc[0]["Acierto direccion"]))
    st.dataframe(pretty_summary(R["flow_summary"]), use_container_width=True, hide_index=True)
    st.caption("FLOW es un proxy gratuito: presión de volumen firmado + posición respecto a VWAP + volumen relativo. NO es order flow real bid/ask.")

st.markdown("**¿Divergencia + FLOW mejora a divergencia sola?**")
if R["combination_summary"].empty:
    st.info("No hubo intersección suficiente.")
else:
    st.dataframe(pretty_summary(R["combination_summary"]), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
st.subheader("4️⃣ INTERRUPTOR #3 — FINRA short-sale volume")
finra_raw = R["finra_raw"]
if finra_raw.empty:
    st.warning("FINRA no devolvió datos. El resto del experimento sigue siendo válido.")
else:
    f1, f2, f3 = st.columns(3)
    f1.metric("Sesiones FINRA", str(len(finra_raw)))
    f2.metric("Último dato", str(pd.to_datetime(finra_raw["date"].max()).date()))
    f3.metric("Último short/total", pct(finra_raw.iloc[-1]["short_ratio"]))
    st.warning(
        "FINRA short-sale volume ≠ short interest. Es volumen de ventas short públicamente difundidas y reportadas a FINRA. "
        "No debe interpretarse como 'X% del mercado está apostando a la baja'."
    )
    if not R["finra_summary"].empty:
        fs = R["finra_summary"].copy()
        for c in ["P(sube)", "Ret medio", "Spearman global"]:
            if c in fs:
                fs[c] = fs[c].map(pct)
        if "Short z medio" in fs:
            fs["Short z medio"] = fs["Short z medio"].map(sigma)
        st.markdown("**FINRA solo — ¿short-volume anormal contiene información para 1–3D?**")
        st.dataframe(fs, use_container_width=True, hide_index=True)

    if not R["divergence_finra_summary"].empty:
        st.markdown("**Divergencia intradía condicionada por FINRA conocido ANTES de abrir**")
        st.dataframe(pretty_summary(R["divergence_finra_summary"]), use_container_width=True, hide_index=True)
        st.caption("Para una señal intradía del día D usamos como máximo FINRA de D-1: el fichero de D no se publica hasta después del cierre.")

# -----------------------------------------------------------------------------
st.subheader("🏁 Qué queremos ver antes de promover nada")
st.markdown(
    """
- **Divergencia:** que el signo siga siendo útil en horizontes vecinos y no muera al salir de los 120 minutos.
- **1h:** que la misma lógica sobreviva razonablemente en más historia; si se invierte, el 72% de 5m pierde credibilidad.
- **FLOW:** que añadir confirmación mejore a la divergencia sola, no sólo que FLOW tenga un porcentaje bonito aislado.
- **FINRA:** primero descubrir si la relación es alcista, bajista o nula; no imponemos el signo porque short-volume no equivale a presión bajista neta.
- **Nada entra en producción** sólo por ganar en esta muestra. Si aparece una combinación buena, la congelamos y la validamos en observaciones futuras.
"""
)

zip_bytes = build_zip(R, M)
st.download_button(
    "⬇️ DESCARGAR FREE SENSORS V1 PARA FEEDBACK (.ZIP)",
    data=zip_bytes,
    file_name="IREN_free_sensors_v1_feedback.zip",
    mime="application/zip",
    type="primary",
)
st.caption("Mándame el ZIP tal cual. No vamos a optimizar parámetros mirando esta misma muestra.")
