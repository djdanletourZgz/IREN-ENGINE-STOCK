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
from iren_engine.consensus_lead import (
    ConsensusLeadConfig,
    align_intraday,
    build_consensus_frame,
    detect_consensus_events,
    detect_consensus_controls,
    horizon_summary,
    direction_summary,
    stability_summary,
    gap_bins,
    compare_with_controls,
    t3_market_audit,
)

VERSION = "Consensus-Lead-V1"
PEERS = ("SOXX", "NVDA", "MU", "MRVL", "AVGO", "AMD", "VRT")

st.set_page_config(page_title="IREN Consensus Lead Test", page_icon="🔥", layout="wide")
st.title("🔥 IREN — Consensus Lead Test V1")
st.caption(
    "Pregunta única: cuando el universo IA ya se ha movido de forma coordinada y IREN todavía va retrasada, "
    "¿IREN tiende a alcanzarlo durante los siguientes 15/30/60/120 minutos?"
)

with st.sidebar:
    st.header("Protocolo")
    st.success("PRIMERA EJECUCIÓN: usa los valores pre-registrados y no los optimices mirando el resultado.")
    lead_minutes = st.selectbox("Ventana que ya se ha movido", [15, 30, 60], index=1)
    factor_z = st.slider("Movimiento mínimo AI (sigma)", 0.5, 2.0, 1.0, 0.25)
    gap_z = st.slider("Gap mínimo IREN (sigma)", 0.25, 2.0, 0.75, 0.25)
    breadth = st.slider("Consenso mínimo", 0.55, 1.0, 0.70, 0.05)
    cooldown_minutes = st.selectbox("Separación entre episodios", [60, 90, 120, 180], index=2)
    st.caption("Defaults pre-registrados: 30m · AI ≥1σ · gap ≥0,75σ · breadth ≥70% · cooldown 120m.")

lead_bars = max(1, int(lead_minutes // 5))
cooldown_bars = max(1, int(cooldown_minutes // 5))
cfg = ConsensusLeadConfig(
    lead_window_bars=lead_bars,
    beta_window_bars=390,
    beta_min_obs=156,
    vol_window_bars=390,
    factor_z_threshold=float(factor_z),
    gap_z_threshold=float(gap_z),
    breadth_threshold=float(breadth),
    cooldown_bars=cooldown_bars,
)


@st.cache_data(ttl=1800)
def get_intraday_market():
    names = [CFG.ticker, *PEERS]
    return load_intraday(names, period="60d", interval="5m")


@st.cache_data(ttl=3600)
def get_daily_market():
    names = [CFG.ticker, *PEERS]
    return load_daily(names, period=CFG.daily_period)


@st.cache_data(ttl=1800, show_spinner=False)
def run_test(intraday_market: dict, daily_market: dict, cfg_dict: dict):
    local_cfg = ConsensusLeadConfig(**cfg_dict)
    close = align_intraday(intraday_market, CFG.ticker, PEERS)
    frame = build_consensus_frame(close, CFG.ticker, PEERS, local_cfg)
    events = detect_consensus_events(frame, local_cfg)
    controls = detect_consensus_controls(frame, local_cfg)
    summary = horizon_summary(events)
    dirs = direction_summary(events)
    stability = stability_summary(events)
    bins = gap_bins(events)
    comparison = compare_with_controls(events, controls)
    audit = t3_market_audit(daily_market, CFG.ticker, PEERS, test_window=350)
    return close, frame, events, controls, summary, dirs, stability, bins, comparison, audit


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"


def build_zip(events, controls, summary, dirs, stability, bins, comparison, audit, cfg_dict, n_bars, n_days):
    meta = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": CFG.ticker,
        "peers": list(PEERS),
        "intraday_period": "60d",
        "interval": "5m",
        "aligned_bars": int(n_bars),
        "aligned_sessions": int(n_days),
        "parameters": cfg_dict,
        "primary_question": "Does a coordinated AI move with a material IREN lag predict catch-up in IREN over 15/30/60/120 minutes?",
        "notes": [
            "Events are de-duplicated with a cooldown so one cascade is not counted as many independent bars.",
            "Rolling beta and volatility are shifted one bar: current bar never sets its own beta/threshold.",
            "Control group = strong AI consensus move without material IREN lag.",
            "No Gamma, options, news or macro inputs are used.",
        ],
    }
    readme = f"""IREN ENGINE — {VERSION}\n\nOBJECTIVE\nTest one narrow hypothesis: coordinated AI move happens first, IREN materially lags, then IREN catches up.\n\nPRIMARY PRE-REGISTERED DEFAULTS\n- lead window: 30 minutes\n- AI move: >= 1.0 sigma\n- breadth: >= 70%\n- IREN directional gap: >= 0.75 sigma\n- episode cooldown: 120 minutes\n\nFILES\n- events.csv: independent consensus-gap episodes\n- controls.csv: strong AI consensus episodes without material IREN gap\n- horizon_summary.csv: follow rate + Wilson interval + signed return\n- direction_summary.csv: bullish vs bearish cascades\n- stability_summary.csv: first vs second half\n- gap_bins.csv: whether larger gaps produce stronger subsequent catch-up\n- control_comparison.csv: incremental value of the gap vs consensus alone\n- t3_market_audit.csv: volatility / internal AI correlation / extreme-day concentration across 3 chronological periods\n- metadata.json\n\nPASS SHOULD NOT MEAN 'one pretty horizon'. We want neighboring horizons, stability, gap monotonicity and improvement over control.\n"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in [
            ("events.csv", events),
            ("controls.csv", controls),
            ("horizon_summary.csv", summary),
            ("direction_summary.csv", dirs),
            ("stability_summary.csv", stability),
            ("gap_bins.csv", bins),
            ("control_comparison.csv", comparison),
            ("t3_market_audit.csv", audit),
        ]:
            raw = df.copy()
            if "timestamp" in raw.columns:
                raw["timestamp"] = pd.to_datetime(raw["timestamp"]).astype(str)
            z.writestr(name, raw.to_csv(index=False))
        z.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        z.writestr("README.txt", readme)
    return buffer.getvalue()


if st.button("▶️ EJECUTAR CONSENSUS LEAD V1", type="primary"):
    with st.spinner("Descargando 5m y buscando cascadas independientes..."):
        try:
            intraday_market = get_intraday_market()
            daily_market = get_daily_market()
            cfg_dict = cfg.__dict__.copy()
            result = run_test(intraday_market, daily_market, cfg_dict)
            st.session_state["consensus_lead_result"] = result
            st.session_state["consensus_lead_cfg"] = cfg_dict
        except Exception as e:
            st.error(f"No se pudo ejecutar el test: {e}")
            st.stop()

if "consensus_lead_result" not in st.session_state:
    st.info("Pulsa **EJECUTAR CONSENSUS LEAD V1** con los defaults. Este test NO modifica todavía el oráculo principal.")
    st.stop()

close, frame, events, controls, summary, dirs, stability, bins, comparison, audit = st.session_state["consensus_lead_result"]
used_cfg = st.session_state.get("consensus_lead_cfg", cfg.__dict__.copy())

if close.empty or frame.empty:
    st.error("No hubo suficientes barras 5m alineadas entre IREN y el basket IA.")
    st.stop()

n_days = len(set(close.index.date))

st.subheader("🏁 Resultado principal")
if events.empty:
    st.error("Con estos umbrales no aparecen episodios suficientes. NO bajes umbrales sólo para fabricar resultado; mándame igualmente el ZIP/resultado.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodios gap", str(len(events)))
    c2.metric("Sesiones alineadas", str(n_days))
    row30 = summary[summary["Horizonte"] == "30M"] if not summary.empty else pd.DataFrame()
    row60 = summary[summary["Horizonte"] == "60M"] if not summary.empty else pd.DataFrame()
    c3.metric("Follow 30m", "—" if row30.empty else fmt_pct(row30.iloc[0]["Follow rate"]))
    c4.metric("Follow 60m", "—" if row60.empty else fmt_pct(row60.iloc[0]["Follow rate"]))

    show = summary.copy()
    for col in ["Follow rate", "Wilson low", "Wilson high", "Retorno firmado medio", "Retorno firmado mediano", "Breadth medio"]:
        if col in show:
            show[col] = show[col].map(fmt_pct)
    if "Gap z medio" in show:
        show["Gap z medio"] = show["Gap z medio"].map(lambda x: f"{x:.2f}σ")
    st.dataframe(show, use_container_width=True, hide_index=True)

st.subheader("🆚 ¿El GAP aporta algo o sólo estamos siguiendo momentum AI?")
if comparison.empty:
    st.info("No hay control suficiente.")
else:
    cmp = comparison.copy()
    for col in ["Follow gap", "Follow control", "Delta follow", "Ret firmado gap", "Ret firmado control", "Delta ret firmado"]:
        if col in cmp:
            cmp[col] = cmp[col].map(fmt_pct)
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    st.caption("El control tiene un movimiento AI fuerte y coordinado, pero IREN NO está materialmente retrasada. Si GAP no mejora al control, no hemos encontrado lead-lag: sólo momentum compartido.")

st.subheader("↕️ Cascadas alcistas vs bajistas")
if not dirs.empty:
    d = dirs.copy()
    for col in ["Follow rate", "Wilson low", "Wilson high", "Retorno firmado medio", "Retorno firmado mediano", "Breadth medio"]:
        if col in d:
            d[col] = d[col].map(fmt_pct)
    if "Gap z medio" in d:
        d["Gap z medio"] = d["Gap z medio"].map(lambda x: f"{x:.2f}σ")
    st.dataframe(d, use_container_width=True, hide_index=True)

st.subheader("🧱 Estabilidad: ¿vive en las dos mitades?")
if stability.empty:
    st.info("Muestra insuficiente para partirla con sentido.")
else:
    s = stability.copy()
    for col in ["Follow rate", "Wilson low", "Wilson high", "Retorno firmado medio", "Retorno firmado mediano", "Breadth medio"]:
        if col in s:
            s[col] = s[col].map(fmt_pct)
    if "Gap z medio" in s:
        s["Gap z medio"] = s["Gap z medio"].map(lambda x: f"{x:.2f}σ")
    st.dataframe(s, use_container_width=True, hide_index=True)

st.subheader("📏 ¿GAP más grande = más catch-up?")
if bins.empty:
    st.info("Muestra insuficiente para terciles de gap.")
else:
    b = bins.copy()
    for col in ["Follow rate", "Retorno firmado medio"]:
        b[col] = b[col].map(fmt_pct)
    b["Gap z medio"] = b["Gap z medio"].map(lambda x: f"{x:.2f}σ")
    st.dataframe(b, use_container_width=True, hide_index=True)

st.subheader("🔬 Auditoría T1/T2/T3 del mercado — sin reinterpretar todavía como concept drift")
if audit.empty:
    st.info("No hay datos diarios suficientes para la auditoría.")
else:
    a = audit.copy()
    for col in ["Vol IREN diaria", "Abs move IREN medio", "Dispersion AI media", "Correlación interna AI", "Top3 share abs move IREN", "Peor día IREN", "Mejor día IREN"]:
        if col in a:
            a[col] = a[col].map(fmt_pct)
    st.dataframe(a, use_container_width=True, hide_index=True)
    st.caption("Esto comprueba alternativas aburridas pero importantes: cambio de volatilidad, cambio de correlación interna del basket y si 3 días extremos concentran una parte anormal del movimiento. No prueba por sí solo por qué falló V2-A.1.")

zip_bytes = build_zip(events, controls, summary, dirs, stability, bins, comparison, audit, used_cfg, len(close), n_days)
st.download_button(
    "⬇️ DESCARGAR CONSENSUS LEAD V1 PARA FEEDBACK (.ZIP)",
    data=zip_bytes,
    file_name="IREN_consensus_lead_v1_feedback.zip",
    mime="application/zip",
    type="primary",
)
st.caption("Mándame este ZIP tal cual. No optimices sliders después de mirar el resultado antes de que evaluemos la primera ejecución.")

with st.expander("Cómo leer este experimento"):
    st.markdown(
        """
- **Factor AI:** mediana robusta del movimiento 5m de SOXX, NVDA, MU, MRVL, AVGO, AMD y VRT.
- **Beta IREN:** calculada con barras PASADAS y desplazada una barra para que el presente no estime su propio benchmark.
- **GAP:** cuánto debería haberse movido IREN por su beta menos cuánto se ha movido realmente.
- **Consensus gap event:** AI se ha movido ≥ umbral, suficiente breadth va en la misma dirección e IREN lleva un retraso ≥ umbral.
- **Cooldown:** una cascada de 2 horas cuenta como UN episodio, no como 24 observaciones de 5m.
- **Follow rate:** porcentaje de episodios donde IREN se mueve después en la dirección que AI ya había tomado.
- **Control:** mismo tipo de movimiento AI fuerte pero sin retraso material de IREN. Es la comparación clave contra simple momentum compartido.
- **PASS real:** no basta un 57% aislado. Queremos efecto en horizontes vecinos, segunda mitad, gaps mayores y mejora frente al control.
- **NO incluye:** Gamma, opciones, noticias, M2 ni el motor V2-A.1 de dirección.
"""
    )
