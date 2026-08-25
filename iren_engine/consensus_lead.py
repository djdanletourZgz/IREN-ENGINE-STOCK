from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConsensusLeadConfig:
    lead_window_bars: int = 6
    beta_window_bars: int = 390
    beta_min_obs: int = 156
    vol_window_bars: int = 390
    factor_z_threshold: float = 1.0
    gap_z_threshold: float = 0.75
    breadth_threshold: float = 0.70
    cooldown_bars: int = 24


def _to_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    z = df.copy().sort_index()
    idx = pd.DatetimeIndex(z.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")
    z.index = idx
    return z[~z.index.duplicated(keep="last")]


def _regular_session(df: pd.DataFrame) -> pd.DataFrame:
    z = _to_ny_index(df)
    if z.empty:
        return z
    mins = z.index.hour * 60 + z.index.minute
    return z.loc[(mins >= 570) & (mins <= 960)].copy()


def align_intraday(market: Dict[str, pd.DataFrame], ticker: str, peers: Iterable[str]) -> pd.DataFrame:
    names = [ticker, *list(peers)]
    closes = {}
    for name in names:
        df = _regular_session(market.get(name, pd.DataFrame()))
        if df.empty or "Close" not in df:
            continue
        closes[name] = df["Close"].astype(float).rename(name)
    if ticker not in closes:
        return pd.DataFrame()
    return pd.concat(closes.values(), axis=1, join="inner").sort_index().dropna()


def _session_key(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.date, index=index)


def _session_pct_change(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    session = _session_key(close.index)
    return close.groupby(session).pct_change(periods)


def _future_session_return(series: pd.Series, bars: int) -> pd.Series:
    session = _session_key(series.index)
    shifted = series.groupby(session).shift(-bars)
    return shifted / series - 1.0


def _to_close_return(series: pd.Series) -> pd.Series:
    session = _session_key(series.index)
    close_px = series.groupby(session).transform("last")
    return close_px / series - 1.0


def _rolling_beta_past_only(y: pd.Series, x: pd.Series, window: int, min_obs: int) -> pd.Series:
    cov = y.rolling(window, min_periods=min_obs).cov(x)
    var = x.rolling(window, min_periods=min_obs).var()
    return (cov / var.replace(0, np.nan)).clip(-5.0, 8.0).shift(1)


def build_consensus_frame(close: pd.DataFrame, ticker: str = "IREN", peers: Iterable[str] | None = None, cfg: ConsensusLeadConfig = ConsensusLeadConfig()) -> pd.DataFrame:
    if close is None or close.empty or ticker not in close:
        return pd.DataFrame()
    peer_cols = [p for p in (list(peers) if peers is not None else list(close.columns)) if p != ticker and p in close]
    if len(peer_cols) < 3:
        return pd.DataFrame()

    out = pd.DataFrame(index=close.index)
    ret5 = _session_pct_change(close[[ticker, *peer_cols]], 1)
    retw = _session_pct_change(close[[ticker, *peer_cols]], cfg.lead_window_bars)
    factor5 = ret5[peer_cols].median(axis=1, skipna=True)
    factorw = (1.0 + factor5.fillna(0.0)).groupby(_session_key(close.index)).rolling(cfg.lead_window_bars, min_periods=cfg.lead_window_bars).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1.0
    iren5 = ret5[ticker]
    beta = _rolling_beta_past_only(iren5, factor5, cfg.beta_window_bars, cfg.beta_min_obs)
    expected = beta * factorw
    actual = retw[ticker]
    gap = expected - actual
    factor_sigma5 = factor5.rolling(cfg.vol_window_bars, min_periods=cfg.beta_min_obs).std().shift(1)
    iren_sigma5 = iren5.rolling(cfg.vol_window_bars, min_periods=cfg.beta_min_obs).std().shift(1)
    scale = np.sqrt(float(cfg.lead_window_bars))
    factor_z = factorw / (factor_sigma5 * scale).replace(0, np.nan)
    gap_z = gap / (iren_sigma5 * scale).replace(0, np.nan)
    iren_z = actual / (iren_sigma5 * scale).replace(0, np.nan)
    peer_window = retw[peer_cols]
    factor_sign = np.sign(factorw).replace(0, np.nan)
    breadth = (peer_window.mul(factor_sign, axis=0) > 0).mean(axis=1, skipna=True)
    directional_gap_z = gap_z * factor_sign

    out["factor_ret_window"] = factorw
    out["factor_z"] = factor_z
    out["factor_sign"] = factor_sign
    out["breadth"] = breadth
    out["median_peer_ret_window"] = peer_window.median(axis=1, skipna=True)
    out["iren_ret_window"] = actual
    out["iren_z"] = iren_z
    out["beta"] = beta
    out["expected_iren_ret"] = expected
    out["gap"] = gap
    out["gap_z"] = gap_z
    out["directional_gap_z"] = directional_gap_z

    for hbars, label in [(3, "15m"), (6, "30m"), (12, "60m"), (24, "120m")]:
        out[f"future_{label}"] = _future_session_return(close[ticker], hbars)
    out["future_close"] = _to_close_return(close[ticker])
    out["session"] = pd.Series(close.index.date, index=close.index).astype(str)
    mins = close.index.hour * 60 + close.index.minute
    out["minutes_from_open"] = np.asarray(mins - 570, dtype=float)
    return out


def _select_episode_rows(mask: pd.Series, cooldown_bars: int) -> pd.Index:
    idx = mask.index
    positions = np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))
    if len(positions) == 0:
        return pd.Index([])
    chosen, last, last_day = [], -10**9, None
    for pos in positions:
        day = idx[pos].date()
        if last_day != day or pos - last >= int(cooldown_bars):
            chosen.append(pos)
            last = pos
            last_day = day
    return idx[chosen]


def _finalize_events(e: pd.DataFrame, event_flag: bool) -> pd.DataFrame:
    if e.empty:
        return e
    e = e.copy()
    e["direction"] = np.where(e["factor_sign"] > 0, "ALCISTA", "BAJISTA")
    e["event"] = event_flag
    for h in ["15m", "30m", "60m", "120m", "close"]:
        fcol = f"future_{h}"
        e[f"signed_{fcol}"] = e[fcol] * e["factor_sign"]
        e[f"follow_{fcol}"] = (e[f"signed_{fcol}"] > 0).astype(float)
    return e.reset_index(names="timestamp")


def detect_consensus_events(frame: pd.DataFrame, cfg: ConsensusLeadConfig = ConsensusLeadConfig()) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    mask = (frame["factor_z"].abs() >= cfg.factor_z_threshold) & (frame["breadth"] >= cfg.breadth_threshold) & (frame["directional_gap_z"] >= cfg.gap_z_threshold)
    ids = _select_episode_rows(mask, cfg.cooldown_bars)
    return _finalize_events(frame.loc[ids].copy(), True) if len(ids) else pd.DataFrame()


def detect_consensus_controls(frame: pd.DataFrame, cfg: ConsensusLeadConfig = ConsensusLeadConfig()) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    mask = (frame["factor_z"].abs() >= cfg.factor_z_threshold) & (frame["breadth"] >= cfg.breadth_threshold) & (frame["directional_gap_z"] < cfg.gap_z_threshold * 0.35)
    ids = _select_episode_rows(mask, cfg.cooldown_bars)
    return _finalize_events(frame.loc[ids].copy(), False) if len(ids) else pd.DataFrame()


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def horizon_summary(events: pd.DataFrame, label: str = "CONSENSUS_GAP") -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    rows = []
    for h in ["15m", "30m", "60m", "120m", "close"]:
        fcol = f"future_{h}"
        scol = f"signed_{fcol}"
        z = events.dropna(subset=[fcol, scol]).copy()
        if z.empty:
            continue
        follows = (z[scol] > 0).astype(int)
        lo, hi = _wilson_interval(int(follows.sum()), int(len(z)))
        rows.append({"Grupo": label, "Horizonte": h.upper(), "N episodios": int(len(z)), "Follow rate": float(follows.mean()), "Wilson low": lo, "Wilson high": hi, "Retorno firmado medio": float(z[scol].mean()), "Retorno firmado mediano": float(z[scol].median()), "Gap z medio": float(z["directional_gap_z"].mean()), "Breadth medio": float(z["breadth"].mean())})
    return pd.DataFrame(rows)


def direction_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    parts = [horizon_summary(g, label=str(direction)) for direction, g in events.groupby("direction")]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stability_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    e = events.sort_values("timestamp").copy()
    days = sorted(pd.to_datetime(e["timestamp"]).dt.date.unique())
    if len(days) < 8:
        return pd.DataFrame()
    split = days[len(days) // 2]
    e["period"] = np.where(pd.to_datetime(e["timestamp"]).dt.date < split, "PRIMERA MITAD", "SEGUNDA MITAD")
    parts = [horizon_summary(g, label=str(period)) for period, g in e.groupby("period")]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def gap_bins(events: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    if events is None or events.empty or len(events) < bins * 5:
        return pd.DataFrame()
    e = events.copy()
    try:
        e["gap_bucket"] = pd.qcut(e["directional_gap_z"], q=bins, labels=[f"Q{i+1}" for i in range(bins)], duplicates="drop")
    except Exception:
        return pd.DataFrame()
    rows = []
    for bucket, g in e.groupby("gap_bucket", observed=True):
        for h in ["15m", "30m", "60m", "120m", "close"]:
            scol = f"signed_future_{h}"
            z = g.dropna(subset=[scol])
            if not z.empty:
                rows.append({"Gap bucket": str(bucket), "Horizonte": h.upper(), "N": int(len(z)), "Gap z medio": float(z["directional_gap_z"].mean()), "Follow rate": float((z[scol] > 0).mean()), "Retorno firmado medio": float(z[scol].mean())})
    return pd.DataFrame(rows)


def compare_with_controls(events: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    a, b = horizon_summary(events, "CONSENSUS GAP"), horizon_summary(controls, "CONSENSUS SIN GAP")
    if a.empty and b.empty:
        return pd.DataFrame()
    raw = pd.concat([a, b], ignore_index=True)
    rows = []
    for h in raw["Horizonte"].unique():
        ga = raw[(raw["Horizonte"] == h) & (raw["Grupo"] == "CONSENSUS GAP")]
        gb = raw[(raw["Horizonte"] == h) & (raw["Grupo"] == "CONSENSUS SIN GAP")]
        def one(df, col): return float(df.iloc[0][col]) if not df.empty else np.nan
        rows.append({"Horizonte": h, "N gap": one(ga, "N episodios"), "N control": one(gb, "N episodios"), "Follow gap": one(ga, "Follow rate"), "Follow control": one(gb, "Follow rate"), "Delta follow": one(ga, "Follow rate") - one(gb, "Follow rate"), "Ret firmado gap": one(ga, "Retorno firmado medio"), "Ret firmado control": one(gb, "Retorno firmado medio"), "Delta ret firmado": one(ga, "Retorno firmado medio") - one(gb, "Retorno firmado medio")})
    return pd.DataFrame(rows)


def t3_market_audit(daily_market: Dict[str, pd.DataFrame], ticker: str, peers: Iterable[str], test_window: int = 350) -> pd.DataFrame:
    closes = {}
    for name in [ticker, *list(peers)]:
        df = daily_market.get(name, pd.DataFrame())
        if df is None or df.empty or "Close" not in df:
            continue
        s = df["Close"].astype(float).copy()
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        s.index = pd.to_datetime(idx.date)
        closes[name] = s.rename(name)
    if ticker not in closes or len(closes) < 4:
        return pd.DataFrame()
    px = pd.concat(closes.values(), axis=1, join="inner").dropna().sort_index()
    if test_window and len(px) > test_window:
        px = px.iloc[-test_window:]
    ret = px.pct_change().dropna()
    if len(ret) < 30:
        return pd.DataFrame()
    peer_cols = [c for c in ret.columns if c != ticker]
    rows = []
    for i, chunk in enumerate(np.array_split(ret.index.to_numpy(), 3), start=1):
        g = ret.loc[pd.to_datetime(chunk)].copy()
        if g.empty:
            continue
        corr = g[peer_cols].corr().to_numpy(dtype=float)
        upper = corr[np.triu_indices_from(corr, k=1)] if len(peer_cols) > 1 else np.array([])
        abs_move = g[ticker].abs().dropna()
        rows.append({"Periodo": f"T{i}", "Desde": pd.Timestamp(g.index.min()).date(), "Hasta": pd.Timestamp(g.index.max()).date(), "N días": int(len(g)), "Vol IREN diaria": float(g[ticker].std()), "Abs move IREN medio": float(g[ticker].abs().mean()), "Dispersion AI media": float(g[peer_cols].std(axis=1).mean()), "Correlación interna AI": float(np.nanmean(upper)) if len(upper) else np.nan, "Top3 share abs move IREN": float(abs_move.nlargest(3).sum() / abs_move.sum()) if abs_move.sum() > 0 else np.nan, "Peor día IREN": float(g[ticker].min()), "Mejor día IREN": float(g[ticker].max())})
    return pd.DataFrame(rows)
