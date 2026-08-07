from __future__ import annotations
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from iren_engine.config import CFG
from iren_engine.market_data import load_daily, load_intraday
from iren_engine.probability import build_daily_state, daily_probabilities, intraday_probability, target_touch_probability
from iren_engine.options_orats import fetch_live_chain, compute_gamma_metrics
from iren_engine.narrative import plain_summary, pct
from iren_engine.storage import save_snapshot

st.set_page_config(page_title="IREN Tactical Radar", page_icon="📡", layout="wide")
st.title("📡 IREN Tactical Radar — V1")
st.caption("Probabilidades empíricas + técnicos + BTC/QQQ + Gamma opcional. No inventa porcentajes: muestra muestra y confianza.")

@st.cache_data(ttl=180)
def get_daily():
    t=[CFG.ticker,CFG.btc,CFG.qqq,CFG.nvda]
    return load_daily(t, CFG.daily_period)

@st.cache_data(ttl=120)
def get_intraday():
    return load_intraday([CFG.ticker], CFG.intraday_period, CFG.intraday_interval)[CFG.ticker]

with st.sidebar:
    st.header("Control")
    st.write("Ticker fijo V1: **IREN**")
    k_daily=st.slider("Vecinos históricos (diario)",60,250,CFG.daily_neighbors,10)
    k_intra=st.slider("Vecinos históricos (intradía)",60,300,CFG.intraday_neighbors,10)
    max_dte=st.slider("Gamma: DTE máximo",7,90,CFG.options_max_dte,1)
    token_default=""
    try:
        token_default=st.secrets.get("ORATS_TOKEN","")
    except Exception:
        token_default=os.getenv("ORATS_TOKEN","")
    token=st.text_input("ORATS token (opcional)",value=token_default,type="password")
    st.caption("Sin ORATS, la parte Gamma queda desactivada; las probabilidades de precio/técnicos sí funcionan.")
    target=st.number_input("Nivel que quieres vigilar ($)",min_value=1.0,max_value=500.0,value=40.0,step=0.5)

try:
    daily=get_daily()
    intra=get_intraday()
except Exception as e:
    st.error(f"No se pudieron descargar datos de mercado: {e}")
    st.stop()

iren=daily[CFG.ticker]
if iren.empty:
    st.error("No hay datos de IREN.")
    st.stop()
spot=float(iren["Close"].iloc[-1])
if not intra.empty:
    spot=float(intra["Close"].iloc[-1])

state=build_daily_state(iren,daily[CFG.btc],daily[CFG.qqq],daily.get(CFG.nvda))
daily_res=daily_probabilities(state,k=k_daily)
intra_res=intraday_probability(intra,k=k_intra) if not intra.empty else None

gamma=None; gex_by=None; gamma_error=None
if token:
    try:
        chain=fetch_live_chain(CFG.ticker,token=token)
        gamma,gex_by=compute_gamma_metrics(chain,spot=spot,max_dte=max_dte)
    except Exception as e:
        gamma_error=str(e)

c1,c2,c3,c4=st.columns(4)
c1.metric("IREN",f"${spot:.2f}")
if intra_res:
    c2.metric("P(cerrar arriba hoy)",pct(intra_res.probabilities.get("close_up")))
    c3.metric("Muestra intradía",str(intra_res.n))
    c4.metric("Confianza intradía",intra_res.confidence)
else:
    c2.metric("P(cerrar arriba hoy)","—"); c3.metric("Muestra intradía","—"); c4.metric("Confianza intradía","—")

st.subheader("En castellano")
st.info(plain_summary(intra_res,daily_res,gamma))

def balance_label(r):
    if r is None: return "SIN DATOS"
    up=r.probabilities.get("touch_+5%",0)
    dn=r.probabilities.get("touch_-5%",0)
    if up-dn>=0.18: return "🟢 Sesgo favorable"
    if dn-up>=0.18: return "🔴 Riesgo de caída elevado"
    return "🟡 Sin ventaja clara"

st.subheader("Radar rápido")
cols=st.columns(4)
cols[0].markdown(f"**HOY**  \n{('🟢' if intra_res and intra_res.probabilities.get('close_up',.5)>.58 else '🔴' if intra_res and intra_res.probabilities.get('close_up',.5)<.42 else '🟡')} {pct(intra_res.probabilities.get('close_up')) if intra_res else '—'} cerrar arriba")
for i,h in enumerate(["1D","3D","5D"],start=1):
    r=next((x for x in daily_res if x.horizon==h),None)
    cols[i].markdown(f"**{h}**  \n{balance_label(r)}  \nConfianza: {r.confidence if r else '—'}")

st.subheader("Probabilidades de recorrido")
rows=[]
if intra_res:
    rows.append({"Horizonte":"Resto de hoy","Muestra":intra_res.n,"Confianza":intra_res.confidence,**{k:pct(v) for k,v in intra_res.probabilities.items()},"Retorno medio":pct(intra_res.expected_return),"P10":pct(intra_res.q10_return),"P90":pct(intra_res.q90_return)})
for r in daily_res:
    rows.append({"Horizonte":r.horizon,"Muestra":r.n,"Confianza":r.confidence,**{k:pct(v) for k,v in r.probabilities.items()},"Retorno medio":pct(r.expected_return),"P10":pct(r.q10_return),"P90":pct(r.q90_return)})
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

st.subheader(f"Probabilidad empírica de tocar ${target:.2f}")
tcols=st.columns(3)
for col,h in zip(tcols,[1,3,5]):
    p,n=target_touch_probability(state,target,h,k=k_daily)
    col.metric(f"En ≤ {h} sesión{'es' if h>1 else ''}",pct(p),f"n={n}")

st.subheader("Precio y medias")
plot=iren.tail(260).copy()
plot["EMA21"]=plot["Close"].ewm(span=21,adjust=False).mean()
plot["EMA50"]=plot["Close"].ewm(span=50,adjust=False).mean()
fig=go.Figure()
fig.add_trace(go.Candlestick(x=plot.index,open=plot.Open,high=plot.High,low=plot.Low,close=plot.Close,name="IREN"))
fig.add_trace(go.Scatter(x=plot.index,y=plot.EMA21,name="EMA21"))
fig.add_trace(go.Scatter(x=plot.index,y=plot.EMA50,name="EMA50"))
if gamma:
    for val,name in [(gamma.call_wall,"Call Wall"),(gamma.put_wall,"Put Wall"),(gamma.gamma_flip_proxy,"Gamma Flip proxy")]:
        if val:
            fig.add_hline(y=val,line_dash="dot",annotation_text=name)
fig.update_layout(height=520,xaxis_rangeslider_visible=False)
st.plotly_chart(fig,use_container_width=True)

st.subheader("Gamma / opciones")
if gamma:
    g1,g2,g3,g4=st.columns(4)
    g1.metric("Net GEX proxy",f"{gamma.net_gex:,.0f}")
    g2.metric("Call Wall",f"${gamma.call_wall:.2f}" if gamma.call_wall else "—")
    g3.metric("Put Wall",f"${gamma.put_wall:.2f}" if gamma.put_wall else "—")
    g4.metric("Gamma Flip proxy",f"${gamma.gamma_flip_proxy:.2f}" if gamma.gamma_flip_proxy else "—")
    st.caption("GEX/Flip son estimaciones bajo una convención de posicionamiento dealer; el OI no revela quién está largo/corto. Úsalos como mapa, no como verdad física.")
    if gex_by is not None and not gex_by.empty:
        near=gex_by[(gex_by.strike>=spot*.75)&(gex_by.strike<=spot*1.25)]
        gf=go.Figure()
        gf.add_bar(x=near.strike,y=near.call_gex,name="Call GEX")
        gf.add_bar(x=near.strike,y=near.put_gex,name="Put GEX")
        gf.update_layout(barmode="relative",height=420)
        st.plotly_chart(gf,use_container_width=True)
elif gamma_error:
    st.warning(f"ORATS no respondió: {gamma_error}")
else:
    st.warning("Gamma desactivado. Añade ORATS_TOKEN para activar cadena live, GEX, walls y Gamma Flip proxy.")

try:
    payload={
        "daily":[{"horizon":r.horizon,"n":r.n,"probabilities":r.probabilities,"expected_return":r.expected_return,"confidence":r.confidence} for r in daily_res],
        "intraday":({"n":intra_res.n,"probabilities":intra_res.probabilities,"expected_return":intra_res.expected_return,"confidence":intra_res.confidence} if intra_res else None),
        "gamma":gamma.__dict__ if gamma else None,
    }
    save_snapshot("data/iren_engine.db",CFG.ticker,spot,payload)
except Exception:
    pass

with st.expander("Qué significan estas probabilidades"):
    st.markdown("""
- **No son una predicción determinista.** El motor busca estados históricos de IREN parecidos al actual y mide qué ocurrió después.
- **Tocar +5%** significa que el máximo posterior alcanzó al menos +5% respecto al precio de referencia, aunque luego cerrase abajo.
- **P10 / P90** muestran un intervalo empírico de retornos de cierre de los vecinos históricos.
- **Muestra** importa: 70% con 20 casos no vale lo mismo que 62% con 140 casos.
- La V1 diaria usa precio/técnicos de IREN + BTC + QQQ + NVDA. La parte Gamma actual se muestra aparte hasta disponer de histórico suficiente de opciones para entrenarla sin engañarnos.
""")
