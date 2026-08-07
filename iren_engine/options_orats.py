from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
import os
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

BASE = "https://api.orats.io/datav2"

@dataclass
class GammaMetrics:
    spot: float
    net_gex: float
    call_wall: float | None
    put_wall: float | None
    gamma_flip_proxy: float | None
    call_put_volume_ratio: float | None
    chain_rows: int
    max_dte: int


def _token(explicit: str | None = None) -> str | None:
    return explicit or os.getenv("ORATS_TOKEN")


def fetch_live_chain(ticker: str, token: str | None = None, derived: bool = False) -> pd.DataFrame:
    tok=_token(token)
    if not tok:
        raise ValueError("Falta ORATS_TOKEN")
    path="live/derived/strikes" if derived else "live/strikes"
    r=requests.get(f"{BASE}/{path}", params={"token":tok,"ticker":ticker}, timeout=20)
    r.raise_for_status()
    payload=r.json()
    return pd.DataFrame(payload.get("data", []))


def fetch_live_one_minute_chain(ticker: str, token: str | None = None, derived: bool = False) -> pd.DataFrame:
    tok=_token(token)
    if not tok:
        raise ValueError("Falta ORATS_TOKEN")
    path="live/derived/one-minute/strikes/chain" if derived else "live/one-minute/strikes/chain"
    r=requests.get(f"{BASE}/{path}", params={"token":tok,"ticker":ticker}, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def fetch_hist_eod(ticker: str, trade_date: str, token: str | None = None) -> pd.DataFrame:
    tok=_token(token)
    if not tok:
        raise ValueError("Falta ORATS_TOKEN")
    r=requests.get(f"{BASE}/hist/strikes", params={"token":tok,"ticker":ticker,"tradeDate":trade_date}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("data", []))


def _bs_gamma(S: float, K: np.ndarray, T: np.ndarray, sigma: np.ndarray, r: float=0.04) -> np.ndarray:
    S=max(float(S),1e-6)
    K=np.maximum(K.astype(float),1e-6)
    T=np.maximum(T.astype(float),1/(365*24*2))
    sigma=np.clip(sigma.astype(float),0.03,5.0)
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.pdf(d1)/(S*sigma*np.sqrt(T))


def compute_gamma_metrics(chain: pd.DataFrame, spot: float | None = None, max_dte: int = 45) -> tuple[GammaMetrics, pd.DataFrame]:
    if chain.empty:
        raise ValueError("Cadena de opciones vacía")
    c=chain.copy()
    numeric=["strike","dte","gamma","callOpenInterest","putOpenInterest","callVolume","putVolume","smvVol","callMidIv","putMidIv","stockPrice","spotPrice"]
    for col in numeric:
        if col in c:
            c[col]=pd.to_numeric(c[col], errors="coerce")
    if spot is None:
        for col in ("spotPrice","stockPrice"):
            if col in c and c[col].notna().any():
                spot=float(c[col].dropna().median()); break
    if spot is None:
        raise ValueError("No se pudo determinar spot")
    if "dte" in c:
        c=c[(c["dte"]>=0)&(c["dte"]<=max_dte)]
    c=c.dropna(subset=["strike"])
    c["callOpenInterest"]=c.get("callOpenInterest",0).fillna(0)
    c["putOpenInterest"]=c.get("putOpenInterest",0).fillna(0)
    if "gamma" not in c or c["gamma"].isna().all():
        iv=c.get("smvVol", pd.Series(0.6,index=c.index)).fillna(0.6)
        T=c.get("dte", pd.Series(7,index=c.index)).fillna(7)/365.0
        c["gamma"]=_bs_gamma(spot,c["strike"].to_numpy(),T.to_numpy(),iv.to_numpy())
    scale=100.0*spot*spot*0.01
    c["call_gex"]=c["gamma"]*c["callOpenInterest"]*scale
    c["put_gex"]=-c["gamma"]*c["putOpenInterest"]*scale
    c["net_gex_row"]=c["call_gex"]+c["put_gex"]
    by=c.groupby("strike",as_index=False)[["call_gex","put_gex","net_gex_row"]].sum().sort_values("strike")
    call_wall=float(by.loc[by["call_gex"].idxmax(),"strike"]) if len(by) and by["call_gex"].max()>0 else None
    put_wall=float(by.loc[by["put_gex"].idxmin(),"strike"]) if len(by) and by["put_gex"].min()<0 else None

    scan=np.linspace(spot*0.70, spot*1.30, 121)
    K=c["strike"].to_numpy(float)
    T=np.maximum(c.get("dte",pd.Series(7,index=c.index)).fillna(7).to_numpy(float)/365.0,1/(365*24*2))
    iv=c.get("smvVol",pd.Series(np.nan,index=c.index)).fillna(c.get("callMidIv",0.6)).fillna(0.6).to_numpy(float)
    coi=c["callOpenInterest"].to_numpy(float); poi=c["putOpenInterest"].to_numpy(float)
    vals=[]
    for s in scan:
        g=_bs_gamma(s,K,T,iv)
        vals.append(float(np.sum(g*(coi-poi)*100*s*s*0.01)))
    flip=None
    for a,b,sa,sb in zip(vals[:-1],vals[1:],scan[:-1],scan[1:]):
        if a==0 or a*b<0:
            flip=float(sa + (0-a)*(sb-sa)/(b-a)) if b!=a else float(sa)
            break
    cv=float(c.get("callVolume",pd.Series(0,index=c.index)).fillna(0).sum())
    pv=float(c.get("putVolume",pd.Series(0,index=c.index)).fillna(0).sum())
    ratio=cv/pv if pv>0 else None
    metrics=GammaMetrics(
        spot=float(spot),net_gex=float(c["net_gex_row"].sum()),call_wall=call_wall,put_wall=put_wall,
        gamma_flip_proxy=flip,call_put_volume_ratio=ratio,chain_rows=len(c),max_dte=max_dte
    )
    return metrics, by
