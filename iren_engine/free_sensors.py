from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import requests

from .consensus_lead import (
    ConsensusLeadConfig,
    align_intraday,
    build_consensus_frame,
    detect_consensus_events,
)


@dataclass(frozen=True)
class FreeSensorsConfig:
    # Sensor 1: señal descubierta en Consensus Lead V1, ahora pre-registrada
    # como persistencia de la divergencia (dirección = -factor_sign).
    divergence_lead_bars_5m: int = 6
    divergence_factor_z: float = 1.0
    divergence_gap_z: float = 0.75
    divergence_breadth: float = 0.70
    divergence_cooldown_bars_5m: int = 24

    # Sensor 2: proxy gratuito de flujo/participación usando OHLCV.
    flow_window_bars: int = 6
    flow_score_threshold: float = 0.35
    rvol_threshold: float = 1.20
    flow_cooldown_bars: int = 24

    # Validación 1h de más historia.
    hourly_beta_window: int = 120
    hourly_beta_min_obs: int = 40
    hourly_cooldown_bars: int = 2

    # Sensor 3: FINRA Consolidated NMS daily short-sale volume.
    finra_calendar_days: int = 90
    finra_z_window: int = 20


FREE_CFG = FreeSensorsConfig()


def _to_ny(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    z = df.copy().sort_index()
    idx = pd.DatetimeIndex(z.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")
    z.index = idx
    return z[~z.index.duplicated(keep="last")]


def regular_session_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    z = _to_ny(df)
    if z.empty:
        return z
    mins = z.index.hour * 60 + z.index.minute
    return z.loc[(mins >= 570) & (mins <= 960)].copy()


def _session_key(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.date, index=index)


def _future_same_session(series: pd.Series, bars: int) -> pd.Series:
    session = _session_key(series.index)
    future = series.groupby(session).shift(-bars)
    return future / series - 1.0


def _to_session_close(series: pd.Series) -> pd.Series:
    session = _session_key(series.index)
    close_px = series.groupby(session).transform("last")
    return close_px / series - 1.0


def _select_episodes(mask: pd.Series, cooldown_bars: int) -> pd.Index:
    positions = np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))
    if not len(positions):
        return pd.Index([])
    idx = mask.index
    chosen: list[int] = []
    last_pos = -10**9
    last_day = None
    for pos in positions:
        day = idx[pos].date()
        if day != last_day or pos - last_pos >= int(cooldown_bars):
            chosen.append(pos)
            last_pos = int(pos)
            last_day = day
    return idx[chosen]


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def build_divergence_events_5m(
    intraday_market: Dict[str, pd.DataFrame],
    ticker: str,
    peers: Iterable[str],
    cfg: FreeSensorsConfig = FREE_CFG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = align_intraday(intraday_market, ticker, peers)
    ccfg = ConsensusLeadConfig(
        lead_window_bars=cfg.divergence_lead_bars_5m,
        beta_window_bars=390,
        beta_min_obs=156,
        vol_window_bars=390,
        factor_z_threshold=cfg.divergence_factor_z,
        gap_z_threshold=cfg.divergence_gap_z,
        breadth_threshold=cfg.divergence_breadth,
        cooldown_bars=cfg.divergence_cooldown_bars_5m,
    )
    frame = build_consensus_frame(close, ticker, peers, ccfg)
    events = detect_consensus_events(frame, ccfg)
    if events.empty:
        return close, events
    events = events.copy()
    events["signal_sign"] = -np.sign(events["factor_sign"].astype(float))
    events["signal_label"] = np.where(events["signal_sign"] > 0, "IREN FUERTE / ALCISTA", "IREN DEBIL / BAJISTA")
    # La hipótesis ahora es persistencia del residual, por tanto el retorno firmado
    # es el opuesto al signed_future_* del antiguo test de catch-up.
    for h in ["15m", "30m", "60m", "120m", "close"]:
        raw = events.get(f"future_{h}")
        if raw is not None:
            events[f"div_signed_{h}"] = raw.astype(float) * events["signal_sign"]
    if not close.empty and ticker in close:
        px = close[ticker].rename("event_price")
        events = events.merge(px, left_on="timestamp", right_index=True, how="left")
    return close, events


def add_cross_day_returns(events: pd.DataFrame, daily_iren: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty or daily_iren is None or daily_iren.empty:
        return events.copy() if events is not None else pd.DataFrame()
    e = events.copy()
    d = daily_iren.copy().sort_index()
    d_dates = pd.Index(pd.DatetimeIndex(d.index).date)
    dclose = d["Close"].astype(float).to_numpy()
    date_to_pos = {dt: i for i, dt in enumerate(d_dates)}

    out_1, out_2, out_3 = [], [], []
    for _, row in e.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        day = ts.date()
        pos = date_to_pos.get(day)
        entry = float(row.get("event_price", np.nan))
        sign = float(row.get("signal_sign", np.nan))
        vals = []
        for step in (1, 2, 3):
            if pos is None or not np.isfinite(entry) or entry <= 0 or pos + step >= len(dclose):
                vals.append(np.nan)
            else:
                vals.append((float(dclose[pos + step]) / entry - 1.0) * sign)
        out_1.append(vals[0]); out_2.append(vals[1]); out_3.append(vals[2])
    e["div_signed_1D"] = out_1
    e["div_signed_2D"] = out_2
    e["div_signed_3D"] = out_3
    return e


def signed_horizon_summary(
    data: pd.DataFrame,
    prefix: str,
    horizons: Iterable[str],
    group: str = "DIVERGENCIA",
) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    rows = []
    for h in horizons:
        col = f"{prefix}{h}"
        if col not in data:
            continue
        z = data[[col]].dropna()
        if z.empty:
            continue
        vals = z[col].astype(float)
        hits = (vals > 0).astype(int)
        lo, hi = _wilson_interval(int(hits.sum()), int(len(hits)))
        rows.append({
            "Grupo": group,
            "Horizonte": h.upper(),
            "N": int(len(vals)),
            "Acierto direccion": float(hits.mean()),
            "Wilson low": lo,
            "Wilson high": hi,
            "Ret firmado medio": float(vals.mean()),
            "Ret firmado mediano": float(vals.median()),
        })
    return pd.DataFrame(rows)


def divergence_life_summary(events: pd.DataFrame) -> pd.DataFrame:
    return signed_horizon_summary(
        events,
        "div_signed_",
        ["30m", "60m", "120m", "close", "1D", "2D", "3D"],
        group="DIVERGENCIA",
    )


def divergence_stability(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty or len(events) < 12:
        return pd.DataFrame()
    e = events.sort_values("timestamp").copy()
    split = len(e) // 2
    e["period"] = ["PRIMERA MITAD"] * split + ["SEGUNDA MITAD"] * (len(e) - split)
    parts = []
    for period, g in e.groupby("period", sort=False):
        s = signed_horizon_summary(g, "div_signed_", ["30m", "60m", "120m", "close", "1D", "2D", "3D"], group=period)
        if not s.empty:
            parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_price_volume_flow(iren_intraday: pd.DataFrame, cfg: FreeSensorsConfig = FREE_CFG) -> pd.DataFrame:
    z = regular_session_ohlcv(iren_intraday)
    if z.empty or not {"Close", "Volume"}.issubset(z.columns):
        return pd.DataFrame()
    out = z.copy()
    session = _session_key(out.index)
    out["ret5"] = out.groupby(session)["Close"].pct_change()
    out["ret30"] = out.groupby(session)["Close"].pct_change(cfg.flow_window_bars)

    if {"High", "Low"}.issubset(out.columns):
        typical = (out["High"].astype(float) + out["Low"].astype(float) + out["Close"].astype(float)) / 3.0
    else:
        typical = out["Close"].astype(float)
    pv = typical * out["Volume"].astype(float)
    cum_pv = pv.groupby(session).cumsum()
    cum_vol = out["Volume"].astype(float).groupby(session).cumsum().replace(0, np.nan)
    out["vwap"] = cum_pv / cum_vol
    out["vwap_dev"] = out["Close"].astype(float) / out["vwap"] - 1.0

    minute = out.index.hour * 60 + out.index.minute
    vol = out["Volume"].astype(float)
    baseline = vol.groupby(minute).transform(lambda s: s.shift(1).rolling(20, min_periods=8).median())
    out["rvol"] = vol / baseline.replace(0, np.nan)

    signed_vol = np.sign(out["ret5"].fillna(0.0)) * vol
    signed_sum = signed_vol.groupby(session).rolling(cfg.flow_window_bars, min_periods=cfg.flow_window_bars).sum().reset_index(level=0, drop=True)
    total_sum = vol.groupby(session).rolling(cfg.flow_window_bars, min_periods=cfg.flow_window_bars).sum().reset_index(level=0, drop=True)
    out["signed_volume_ratio"] = signed_sum / total_sum.replace(0, np.nan)

    sigma5 = out["ret5"].rolling(390, min_periods=156).std().shift(1)
    bars_from_open = out.groupby(session).cumcount().astype(float) + 1.0
    out["vwap_z"] = out["vwap_dev"] / (sigma5 * np.sqrt(bars_from_open)).replace(0, np.nan)
    sv = out["signed_volume_ratio"].clip(-1.0, 1.0)
    vz = (out["vwap_z"] / 2.0).clip(-1.0, 1.0)
    out["flow_score"] = 0.55 * sv + 0.45 * vz
    out["flow_sign"] = np.sign(out["flow_score"])

    for bars, label in [(6, "30m"), (12, "60m"), (24, "120m")]:
        out[f"future_{label}"] = _future_same_session(out["Close"].astype(float), bars)
    out["future_close"] = _to_session_close(out["Close"].astype(float))
    return out


def detect_flow_events(flow: pd.DataFrame, cfg: FreeSensorsConfig = FREE_CFG) -> pd.DataFrame:
    if flow is None or flow.empty:
        return pd.DataFrame()
    mask = (
        (flow["flow_score"].abs() >= cfg.flow_score_threshold)
        & (flow["rvol"] >= cfg.rvol_threshold)
        & flow["flow_sign"].ne(0)
    )
    ids = _select_episodes(mask, cfg.flow_cooldown_bars)
    if not len(ids):
        return pd.DataFrame()
    e = flow.loc[ids].copy().reset_index(names="timestamp")
    for h in ["30m", "60m", "120m", "close"]:
        e[f"flow_signed_{h}"] = e[f"future_{h}"] * e["flow_sign"]
    return e


def flow_summary(events: pd.DataFrame) -> pd.DataFrame:
    return signed_horizon_summary(events, "flow_signed_", ["30m", "60m", "120m", "close"], group="FLOW")


def combine_divergence_and_flow(div_events: pd.DataFrame, flow: pd.DataFrame, cfg: FreeSensorsConfig = FREE_CFG) -> tuple[pd.DataFrame, pd.DataFrame]:
    if div_events is None or div_events.empty or flow is None or flow.empty:
        return pd.DataFrame(), pd.DataFrame()
    cols = ["flow_score", "flow_sign", "rvol"]
    e = div_events.merge(flow[cols], left_on="timestamp", right_index=True, how="left", suffixes=("", "_flow"))
    e["flow_confirms"] = (
        (np.sign(e["flow_score"]) == np.sign(e["signal_sign"]))
        & (e["flow_score"].abs() >= cfg.flow_score_threshold)
        & (e["rvol"] >= cfg.rvol_threshold)
    )
    e["flow_opposes"] = (
        (np.sign(e["flow_score"]) == -np.sign(e["signal_sign"]))
        & (e["flow_score"].abs() >= cfg.flow_score_threshold)
        & (e["rvol"] >= cfg.rvol_threshold)
    )
    parts = []
    for label, g in [
        ("DIVERGENCIA SOLA", e),
        ("DIVERGENCIA + FLOW CONFIRMA", e[e["flow_confirms"]]),
        ("DIVERGENCIA + FLOW SE OPONE", e[e["flow_opposes"]]),
    ]:
        s = signed_horizon_summary(g, "div_signed_", ["30m", "60m", "120m", "close", "1D", "2D", "3D"], group=label)
        if not s.empty:
            parts.append(s)
    return e, (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())


def _hourly_regular_close(market: Dict[str, pd.DataFrame], ticker: str, peers: Iterable[str]) -> pd.DataFrame:
    closes = {}
    for name in [ticker, *list(peers)]:
        df = regular_session_ohlcv(market.get(name, pd.DataFrame()))
        if not df.empty and "Close" in df:
            closes[name] = df["Close"].astype(float).rename(name)
    if ticker not in closes:
        return pd.DataFrame()
    return pd.concat(closes.values(), axis=1, join="inner").sort_index().dropna()


def hourly_divergence_validation(
    hourly_market: Dict[str, pd.DataFrame],
    ticker: str,
    peers: Iterable[str],
    cfg: FreeSensorsConfig = FREE_CFG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = _hourly_regular_close(hourly_market, ticker, peers)
    if close.empty or close.shape[1] < 4:
        return pd.DataFrame(), pd.DataFrame()
    peer_cols = [c for c in close.columns if c != ticker]
    session = _session_key(close.index)
    rets = close.groupby(session).pct_change()
    factor = rets[peer_cols].median(axis=1, skipna=True)
    ir = rets[ticker]
    cov = ir.rolling(cfg.hourly_beta_window, min_periods=cfg.hourly_beta_min_obs).cov(factor)
    var = factor.rolling(cfg.hourly_beta_window, min_periods=cfg.hourly_beta_min_obs).var()
    beta = (cov / var.replace(0, np.nan)).clip(-5.0, 8.0).shift(1)
    expected = beta * factor
    gap = expected - ir
    fsig = factor.rolling(cfg.hourly_beta_window, min_periods=cfg.hourly_beta_min_obs).std().shift(1)
    isig = ir.rolling(cfg.hourly_beta_window, min_periods=cfg.hourly_beta_min_obs).std().shift(1)
    factor_z = factor / fsig.replace(0, np.nan)
    gap_z = gap / isig.replace(0, np.nan)
    sign = np.sign(factor).replace(0, np.nan)
    breadth = (rets[peer_cols].mul(sign, axis=0) > 0).mean(axis=1, skipna=True)
    directional_gap = gap_z * sign
    frame = pd.DataFrame({
        "factor": factor,
        "factor_z": factor_z,
        "beta": beta,
        "gap_z": gap_z,
        "directional_gap_z": directional_gap,
        "breadth": breadth,
        "factor_sign": sign,
        "signal_sign": -sign,
    })
    frame["future_1h"] = _future_same_session(close[ticker], 1)
    frame["future_2h"] = _future_same_session(close[ticker], 2)
    frame["future_close"] = _to_session_close(close[ticker])
    mask = (
        frame["factor_z"].abs().ge(cfg.divergence_factor_z)
        & frame["breadth"].ge(cfg.divergence_breadth)
        & frame["directional_gap_z"].ge(cfg.divergence_gap_z)
    )
    ids = _select_episodes(mask, cfg.hourly_cooldown_bars)
    if not len(ids):
        return frame, pd.DataFrame()
    e = frame.loc[ids].copy().reset_index(names="timestamp")
    for h in ["1h", "2h", "close"]:
        e[f"hour_signed_{h}"] = e[f"future_{h}"] * e["signal_sign"]
    summary = signed_horizon_summary(e, "hour_signed_", ["1h", "2h", "close"], group="VALIDACION 1H")
    # Estabilidad temporal en dos mitades de los episodios.
    if len(e) >= 12:
        split = len(e) // 2
        e["period"] = ["1H PRIMERA MITAD"] * split + ["1H SEGUNDA MITAD"] * (len(e) - split)
        parts = [summary]
        for period, g in e.groupby("period", sort=False):
            s = signed_horizon_summary(g, "hour_signed_", ["1h", "2h", "close"], group=period)
            if not s.empty:
                parts.append(s)
        summary = pd.concat(parts, ignore_index=True)
    return e, summary


def _finra_one_day(date: pd.Timestamp, symbol: str, timeout: float = 7.0) -> dict | None:
    ds = pd.Timestamp(date).strftime("%Y%m%d")
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ds}.txt"
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "IREN-Engine/1.0 research"})
        if r.status_code != 200 or "Date|Symbol|" not in r.text[:100]:
            return None
        target = f"|{symbol}|"
        for line in r.text.splitlines()[1:]:
            if target in line:
                parts = line.split("|")
                if len(parts) < 6:
                    continue
                return {
                    "date": pd.to_datetime(parts[0], format="%Y%m%d"),
                    "symbol": parts[1],
                    "short_volume": float(parts[2]),
                    "short_exempt_volume": float(parts[3]),
                    "total_volume": float(parts[4]),
                    "market": parts[5],
                }
    except Exception:
        return None
    return None


def fetch_finra_short_volume(symbol: str = "IREN", calendar_days: int = 90, workers: int = 8) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    start = end - pd.Timedelta(days=int(calendar_days))
    candidates = pd.bdate_range(start, end)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(_finra_one_day, d, symbol): d for d in candidates}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception:
                row = None
            if row:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    out["short_ratio"] = out["short_volume"] / out["total_volume"].replace(0, np.nan)
    return out.reset_index(drop=True)


def finra_features(finra: pd.DataFrame, daily_iren: pd.DataFrame, z_window: int = 20) -> pd.DataFrame:
    if finra is None or finra.empty or daily_iren is None or daily_iren.empty:
        return pd.DataFrame()
    f = finra.copy().sort_values("date")
    # Fuerza una resolución homogénea para evitar diferencias pandas s/us/ns.
    f["date"] = pd.to_datetime(f["date"], errors="coerce").astype("datetime64[ns]")
    ratio = f["short_ratio"].astype(float)
    mean = ratio.shift(1).rolling(z_window, min_periods=max(8, z_window // 2)).mean()
    std = ratio.shift(1).rolling(z_window, min_periods=max(8, z_window // 2)).std()
    f["short_ratio_z"] = (ratio - mean) / std.replace(0, np.nan)
    f["short_ratio_delta"] = ratio.diff()

    d = daily_iren[["Close"]].copy().sort_index()
    d["date"] = pd.to_datetime(pd.DatetimeIndex(d.index).date).astype("datetime64[ns]")
    d = d.drop_duplicates("date", keep="last").reset_index(drop=True)
    for h in (1, 2, 3):
        d[f"future_{h}D"] = d["Close"].shift(-h) / d["Close"] - 1.0
    x = f.merge(d[["date", "Close", "future_1D", "future_2D", "future_3D"]], on="date", how="left")
    return x


def finra_short_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame()
    rows = []
    for h in (1, 2, 3):
        col = f"future_{h}D"
        z = features[["short_ratio_z", col]].dropna().copy()
        if len(z) < 12:
            continue
        corr = float(z["short_ratio_z"].corr(z[col], method="spearman"))
        try:
            z["bucket"] = pd.qcut(z["short_ratio_z"], 3, labels=["SHORT BAJO", "SHORT NORMAL", "SHORT ALTO"], duplicates="drop")
        except Exception:
            continue
        for bucket, g in z.groupby("bucket", observed=True):
            rows.append({
                "Horizonte": f"{h}D",
                "Bucket": str(bucket),
                "N": int(len(g)),
                "Short z medio": float(g["short_ratio_z"].mean()),
                "P(sube)": float((g[col] > 0).mean()),
                "Ret medio": float(g[col].mean()),
                "Spearman global": corr,
            })
    return pd.DataFrame(rows)


def attach_prior_finra_to_divergence(events: pd.DataFrame, finra_features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events is None or events.empty or finra_features_df is None or finra_features_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    e = events.copy()

    # `merge_asof` exige que ambas claves tengan EXACTAMENTE el mismo dtype.
    # Pandas puede crear datetime64[s], [us] o [ns] según el origen; normalizamos
    # explícitamente a nanosegundos y a fecha local de NY antes de unir.
    ts = pd.to_datetime(e["timestamp"], errors="coerce", utc=True)
    event_date = ts.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    e["event_date"] = event_date.astype("datetime64[ns]")

    f = finra_features_df[["date", "short_ratio", "short_ratio_z", "short_ratio_delta"]].copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce").astype("datetime64[ns]")
    f = f.dropna(subset=["date"]).sort_values("date")

    # Para un evento intradía del día D sólo usamos FINRA de D-1 o anterior,
    # porque el volumen short de D se publica después del cierre.
    e = e.dropna(subset=["event_date"]).sort_values("event_date")
    merged = pd.merge_asof(
        e,
        f,
        left_on="event_date",
        right_on="date",
        direction="backward",
        allow_exact_matches=False,
    )
    merged["finra_context"] = np.select(
        [merged["short_ratio_z"] >= 1.0, merged["short_ratio_z"] <= -1.0],
        ["SHORT VOL ALTO", "SHORT VOL BAJO"],
        default="SHORT VOL NORMAL",
    )
    parts = []
    for context, g in merged.groupby("finra_context"):
        s = signed_horizon_summary(g, "div_signed_", ["30m", "60m", "120m", "close", "1D", "2D", "3D"], group=context)
        if not s.empty:
            parts.append(s)
    return merged, (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())