from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


@dataclass
class ProbabilityResult:
    horizon: str
    n: int
    probabilities: Dict[str, float]
    expected_return: float | None = None
    median_return: float | None = None
    q10_return: float | None = None
    q90_return: float | None = None
    confidence: str = "BAJA"


def _confidence(n: int, mean_distance: float | None = None) -> str:
    if n >= 100 and (mean_distance is None or mean_distance < 3.0):
        return "ALTA"
    if n >= 50:
        return "MEDIA"
    return "BAJA"


def _nearest_rows(frame: pd.DataFrame, feature_cols: list[str], current_idx, k: int):
    usable = frame.dropna(subset=feature_cols).copy()
    if current_idx not in usable.index:
        raise ValueError("El estado actual no tiene todas las variables necesarias.")
    cur = usable.loc[[current_idx], feature_cols]
    hist = usable.loc[usable.index < current_idx, feature_cols]
    if len(hist) == 0:
        return [], None
    scaler = StandardScaler().fit(hist)
    X = scaler.transform(hist)
    q = scaler.transform(cur)
    k2 = min(k, len(hist))
    nn = NearestNeighbors(n_neighbors=k2).fit(X)
    dist, ind = nn.kneighbors(q)
    ids = hist.index[ind[0]]
    return ids, float(np.mean(dist[0]))


def _daily_date_index(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z.index = pd.to_datetime([getattr(i, "date", lambda: i)() for i in z.index])
    z = z[~z.index.duplicated(keep="last")].sort_index()
    return z


def build_daily_state(iren: pd.DataFrame, btc: pd.DataFrame, qqq: pd.DataFrame, nvda: pd.DataFrame | None = None) -> pd.DataFrame:
    from .indicators import add_daily_features
    iren = _daily_date_index(iren)
    btc = _daily_date_index(btc)
    qqq = _daily_date_index(qqq)
    nvda = _daily_date_index(nvda) if nvda is not None and not nvda.empty else nvda
    base = add_daily_features(iren)
    keep = ["ret1", "ret5", "ret20", "dist_ema8", "dist_ema21", "dist_ema50", "rsi14", "atrp", "vol_z", "range"]
    out = base[keep + ["Open", "High", "Low", "Close", "Volume"]].copy()
    for name, df in [("btc", btc), ("qqq", qqq), ("nvda", nvda)]:
        if df is None or df.empty:
            continue
        f = add_daily_features(df)
        z = f[["ret1", "ret5", "ret20", "dist_ema21", "rsi14"]].add_prefix(name + "_")
        out = out.join(z, how="left")
    return out


def daily_probabilities(state: pd.DataFrame, horizons=(1, 3, 5), k: int = 120, thresholds=(0.03, 0.05, 0.10)) -> list[ProbabilityResult]:
    feature_cols = [c for c in state.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    current_idx = state.dropna(subset=feature_cols).index[-1]
    ids, md = _nearest_rows(state, feature_cols, current_idx, k)
    results = []
    loc_map = {idx: i for i, idx in enumerate(state.index)}
    for h in horizons:
        rows = []
        for idx in ids:
            i = loc_map[idx]
            if i + h >= len(state):
                continue
            entry = float(state.iloc[i]["Close"])
            fut = state.iloc[i+1:i+h+1]
            if fut.empty or not np.isfinite(entry):
                continue
            rows.append({
                "ret": float(fut.iloc[-1]["Close"] / entry - 1),
                "up": float(fut["High"].max() / entry - 1),
                "down": float(fut["Low"].min() / entry - 1),
            })
        r = pd.DataFrame(rows)
        probs: Dict[str, float] = {}
        if not r.empty:
            for t in thresholds:
                probs[f"touch_+{int(t*100)}%"] = float((r["up"] >= t).mean())
                probs[f"touch_-{int(t*100)}%"] = float((r["down"] <= -t).mean())
            probs["close_up"] = float((r["ret"] > 0).mean())
            results.append(ProbabilityResult(
                horizon=f"{h}D", n=len(r), probabilities=probs,
                expected_return=float(r["ret"].mean()), median_return=float(r["ret"].median()),
                q10_return=float(r["ret"].quantile(.10)), q90_return=float(r["ret"].quantile(.90)),
                confidence=_confidence(len(r), md)
            ))
    return results


def target_touch_probability(state: pd.DataFrame, target_price: float, horizon: int, k: int = 120) -> tuple[float | None, int]:
    feature_cols = [c for c in state.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    current_idx = state.dropna(subset=feature_cols).index[-1]
    current = float(state.loc[current_idx, "Close"])
    rel = target_price / current - 1
    ids, _ = _nearest_rows(state, feature_cols, current_idx, k)
    loc_map = {idx: i for i, idx in enumerate(state.index)}
    hits=[]
    for idx in ids:
        i=loc_map[idx]
        if i+horizon >= len(state):
            continue
        entry=float(state.iloc[i]["Close"])
        fut=state.iloc[i+1:i+horizon+1]
        if rel >= 0:
            hits.append(float(fut["High"].max()/entry-1 >= rel))
        else:
            hits.append(float(fut["Low"].min()/entry-1 <= rel))
    return (float(np.mean(hits)) if hits else None, len(hits))


def intraday_probability(intra: pd.DataFrame, k: int = 160, thresholds=(0.01,0.02,0.03)) -> ProbabilityResult | None:
    from .indicators import add_intraday_features
    x = add_intraday_features(intra)
    if x.empty:
        return None
    feat = ["ret_5m","ret_15m","ret_60m","dist_ema9","dist_ema21","rsi14","dist_vwap","vol_z_20","session_frac"]
    usable = x.dropna(subset=feat).copy()
    if len(usable) < 100:
        return None
    current_idx = usable.index[-1]
    cur_frac = float(usable.loc[current_idx, "session_frac"])
    if usable.index.tz is None:
        usable_ny = usable.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        usable_ny = usable.index.tz_convert("America/New_York")
    current_day = usable_ny[-1].date()
    hist_mask = pd.Series([d.date() < current_day for d in usable_ny], index=usable.index)
    hist = usable.loc[hist_mask].copy()
    hist = hist[(hist["session_frac"] - cur_frac).abs() <= 0.10]
    if len(hist) < 30:
        return None
    scaler=StandardScaler().fit(hist[feat])
    X=scaler.transform(hist[feat]); q=scaler.transform(usable.loc[[current_idx],feat])
    kk=min(k,len(hist)); nn=NearestNeighbors(n_neighbors=kk).fit(X)
    dist,ind=nn.kneighbors(q)
    ids=hist.index[ind[0]]

    y=x.copy()
    if y.index.tz is None:
        idx_ny=y.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        idx_ny=y.index.tz_convert("America/New_York")
    day=pd.Series(idx_ny.date,index=y.index)
    rows=[]
    for idx in ids:
        d=day.loc[idx]
        same=y.loc[(day==d) & (y.index>=idx)]
        if len(same)<2:
            continue
        entry=float(y.loc[idx,"Close"])
        rows.append({
            "ret": float(same.iloc[-1]["Close"]/entry-1),
            "up": float(same["High"].max()/entry-1),
            "down": float(same["Low"].min()/entry-1),
        })
    r=pd.DataFrame(rows)
    if r.empty:
        return None
    probs={}
    for t in thresholds:
        probs[f"touch_+{int(t*100)}%"] = float((r["up"]>=t).mean())
        probs[f"touch_-{int(t*100)}%"] = float((r["down"]<=-t).mean())
    probs["close_up"] = float((r["ret"]>0).mean())
    return ProbabilityResult(
        horizon="HOY", n=len(r), probabilities=probs,
        expected_return=float(r["ret"].mean()), median_return=float(r["ret"].median()),
        q10_return=float(r["ret"].quantile(.10)), q90_return=float(r["ret"].quantile(.90)),
        confidence=_confidence(len(r), float(np.mean(dist[0])))
    )
