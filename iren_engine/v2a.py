from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable
import math

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .indicators import add_daily_features
from .probability import ProbabilityResult


AI_BASKET_DEFAULT = ("SOXX", "NVDA", "MU", "MRVL", "AVGO", "AMD", "VRT", "CRWV", "NBIS")
CONTEXT_TICKERS_DEFAULT = ("QQQ", "^VIX", "BTC-USD")


def _date_index(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z.index = pd.to_datetime([getattr(i, "date", lambda: i)() for i in z.index])
    z = z[~z.index.duplicated(keep="last")].sort_index()
    return z


def _close(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or df.empty or "Close" not in df:
        return None
    return _date_index(df)["Close"].astype(float)


def _rolling_factor_return(returns: pd.DataFrame, iren_ret: pd.Series, corr_window: int = 60) -> tuple[pd.Series, pd.Series]:
    """Factor IA dinámico ponderado por correlación reciente con IREN.

    Todos los pesos se calculan con información disponible hasta cada fecha.
    Correlaciones negativas no reciben peso; si no hay historial suficiente,
    se usa media equiponderada entre miembros disponibles.
    """
    if returns.empty:
        idx = iren_ret.index
        return pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)

    rets = returns.reindex(iren_ret.index)
    corr_cols = {}
    for c in rets.columns:
        corr_cols[c] = iren_ret.rolling(corr_window, min_periods=max(20, corr_window // 3)).corr(rets[c]).clip(lower=0)
    corr = pd.DataFrame(corr_cols, index=rets.index)
    weights = corr.where(rets.notna()).fillna(0.0)
    denom = weights.sum(axis=1)
    weighted = (rets.fillna(0.0) * weights).sum(axis=1) / denom.replace(0, np.nan)

    equal = rets.mean(axis=1, skipna=True)
    factor = weighted.fillna(equal)
    corr_mean = corr.where(rets.notna()).mean(axis=1, skipna=True)
    return factor, corr_mean


def build_v2a_state(
    iren: pd.DataFrame,
    market: Dict[str, pd.DataFrame],
    ai_tickers: Iterable[str] = AI_BASKET_DEFAULT,
) -> pd.DataFrame:
    """Estado V2-A: IREN + universo IA + fuerza relativa + régimen de riesgo.

    No contiene Gamma ni noticias. Está diseñado para backtest walk-forward sin
    usar datos futuros.
    """
    iren = _date_index(iren)
    base = add_daily_features(iren)
    keep = [
        "ret1", "ret3", "ret5", "ret20",
        "dist_ema8", "dist_ema21", "dist_ema50",
        "rsi14", "atrp", "vol_z", "range",
    ]
    out = base[keep + ["Open", "High", "Low", "Close", "Volume"]].copy()

    peer_ret = {}
    for ticker in ai_tickers:
        s = _close(market.get(ticker))
        if s is not None:
            peer_ret[ticker] = s.pct_change().reindex(out.index)
    peer_returns = pd.DataFrame(peer_ret, index=out.index)

    factor_ret1, corr_mean = _rolling_factor_return(peer_returns, out["ret1"], corr_window=60)
    factor_index = (1.0 + factor_ret1.fillna(0.0)).cumprod()

    out["ai_ret1"] = factor_ret1
    out["ai_ret3"] = factor_index.pct_change(3)
    out["ai_ret5"] = factor_index.pct_change(5)
    out["ai_ret20"] = factor_index.pct_change(20)
    out["ai_dist_ema21"] = factor_index / factor_index.ewm(span=21, adjust=False).mean() - 1
    out["ai_vol20"] = factor_ret1.rolling(20, min_periods=10).std()
    out["ai_corr60"] = corr_mean

    if not peer_returns.empty:
        positive = peer_returns.gt(0).where(peer_returns.notna())
        out["ai_breadth1"] = positive.mean(axis=1, skipna=True)
        out["ai_dispersion1"] = peer_returns.std(axis=1, skipna=True)
        out["ai_members"] = peer_returns.notna().sum(axis=1).astype(float)
    else:
        out["ai_breadth1"] = np.nan
        out["ai_dispersion1"] = np.nan
        out["ai_members"] = 0.0

    out["rel_ai_1"] = out["ret1"] - out["ai_ret1"]
    out["rel_ai_3"] = out["ret3"] - out["ai_ret3"]
    out["rel_ai_5"] = out["ret5"] - out["ai_ret5"]
    out["rel_ai_20"] = out["ret20"] - out["ai_ret20"]
    out["rel_ai_accel"] = out["rel_ai_5"] - out["rel_ai_20"] / 4.0

    qqq = _close(market.get("QQQ"))
    if qqq is not None:
        q = qqq.reindex(out.index)
        out["qqq_ret1"] = q.pct_change(1)
        out["qqq_ret5"] = q.pct_change(5)
    vix = _close(market.get("^VIX"))
    if vix is not None:
        v = vix.reindex(out.index)
        out["vix_ret1"] = v.pct_change(1)
        out["vix_level_z"] = (v - v.rolling(60, min_periods=20).mean()) / v.rolling(60, min_periods=20).std()
    btc = _close(market.get("BTC-USD"))
    if btc is not None:
        b = btc.reindex(out.index)
        out["btc_ret1"] = b.pct_change(1)
        out["btc_ret5"] = b.pct_change(5)

    return out


def v2a_feature_columns(state: pd.DataFrame) -> list[str]:
    preferred = [
        "ret1", "ret3", "ret5", "ret20", "dist_ema8", "dist_ema21", "dist_ema50",
        "rsi14", "atrp", "vol_z", "range",
        "ai_ret1", "ai_ret3", "ai_ret5", "ai_ret20", "ai_dist_ema21", "ai_vol20",
        "ai_breadth1", "ai_dispersion1", "ai_corr60",
        "rel_ai_1", "rel_ai_3", "rel_ai_5", "rel_ai_20", "rel_ai_accel",
        "qqq_ret1", "qqq_ret5", "vix_ret1", "vix_level_z", "btc_ret1", "btc_ret5",
    ]
    return [c for c in preferred if c in state.columns]


def _weighted_neighbors(
    frame: pd.DataFrame,
    feature_cols: list[str],
    current_idx,
    k: int,
    lookback_rows: int = 756,
    recency_half_life: int = 252,
):
    usable = frame.dropna(subset=feature_cols).copy()
    if current_idx not in usable.index:
        raise ValueError("El estado actual no tiene todas las variables V2-A necesarias.")

    hist = usable.loc[usable.index < current_idx].copy()
    if lookback_rows and len(hist) > lookback_rows:
        hist = hist.iloc[-lookback_rows:]
    if hist.empty:
        return pd.Index([]), np.array([]), np.array([]), np.nan, np.nan

    cur = usable.loc[[current_idx], feature_cols]
    scaler = StandardScaler().fit(hist[feature_cols])
    X = scaler.transform(hist[feature_cols])
    q = scaler.transform(cur)
    kk = min(int(k), len(hist))
    nn = NearestNeighbors(n_neighbors=kk).fit(X)
    dist, ind = nn.kneighbors(q)
    ids = hist.index[ind[0]]
    d = dist[0].astype(float)

    norm_d = d / max(math.sqrt(len(feature_cols)), 1.0)
    sim_w = 1.0 / np.square(1.0 + norm_d)

    positions = {idx: i for i, idx in enumerate(usable.index)}
    cur_pos = positions[current_idx]
    ages = np.array([max(cur_pos - positions[idx], 1) for idx in ids], dtype=float)
    if recency_half_life > 0:
        recency_w = np.power(0.5, ages / float(recency_half_life))
    else:
        recency_w = np.ones_like(ages)
    weights = sim_w * recency_w
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    ess = float(1.0 / np.sum(np.square(weights)))
    mean_norm_distance = float(np.average(norm_d, weights=weights))
    return ids, weights, ages, ess, mean_norm_distance


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = weights.sum()
    if total <= 0:
        return float(np.quantile(values, q))
    cum = np.cumsum(weights) / total
    return float(np.interp(q, cum, values))


def _quality_score(ess: float, mean_norm_distance: float, p_up: float) -> float:
    """Claridad estadística 0-100; NO equivale todavía a accuracy histórica."""
    n_score = np.clip(ess / 70.0, 0.0, 1.0)
    sim_score = float(np.exp(-max(mean_norm_distance, 0.0))) if np.isfinite(mean_norm_distance) else 0.0
    edge_score = np.clip(abs(float(p_up) - 0.5) / 0.18, 0.0, 1.0)
    return float(np.clip(100.0 * (0.45 * n_score + 0.35 * sim_score + 0.20 * edge_score), 0.0, 95.0))


def _confidence_label(quality: float) -> str:
    if quality >= 70:
        return "ALTA"
    if quality >= 55:
        return "MEDIA"
    return "BAJA"


def v2a_probabilities(
    state: pd.DataFrame,
    horizons=(1, 3, 5),
    k: int = 120,
    thresholds=(0.03, 0.05, 0.10),
    lookback_rows: int = 756,
    recency_half_life: int = 252,
) -> list[ProbabilityResult]:
    features = v2a_feature_columns(state)
    if not features:
        return []
    usable = state.dropna(subset=features)
    if usable.empty:
        return []
    current_idx = usable.index[-1]
    ids, weights, _ages, ess, md = _weighted_neighbors(
        state, features, current_idx, k, lookback_rows=lookback_rows, recency_half_life=recency_half_life
    )
    if len(ids) == 0:
        return []

    loc = {idx: i for i, idx in enumerate(state.index)}
    results: list[ProbabilityResult] = []
    for h in horizons:
        rows = []
        ws = []
        for idx, w in zip(ids, weights):
            i = loc[idx]
            if i + h >= len(state):
                continue
            entry = float(state.iloc[i]["Close"])
            fut = state.iloc[i + 1 : i + h + 1]
            if len(fut) < h or not np.isfinite(entry):
                continue
            vals = fut[["High", "Low", "Close"]].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                continue
            rows.append([
                float(fut.iloc[-1]["Close"] / entry - 1),
                float(fut["High"].max() / entry - 1),
                float(fut["Low"].min() / entry - 1),
            ])
            ws.append(float(w))
        if not rows:
            continue
        arr = np.asarray(rows, dtype=float)
        ww = np.asarray(ws, dtype=float)
        ww = ww / ww.sum()
        ret, up, down = arr[:, 0], arr[:, 1], arr[:, 2]
        probs: Dict[str, float] = {"close_up": float(np.sum(ww * (ret > 0)))}
        for t in thresholds:
            key = int(t * 100)
            probs[f"touch_+{key}%"] = float(np.sum(ww * (up >= t)))
            probs[f"touch_-{key}%"] = float(np.sum(ww * (down <= -t)))

        qscore = _quality_score(ess, md, probs["close_up"])
        r = ProbabilityResult(
            horizon=f"{h}D",
            n=len(ret),
            probabilities=probs,
            expected_return=float(np.sum(ww * ret)),
            median_return=_weighted_quantile(ret, ww, 0.50),
            q10_return=_weighted_quantile(ret, ww, 0.10),
            q90_return=_weighted_quantile(ret, ww, 0.90),
            confidence=_confidence_label(qscore),
        )
        r.quality_score = qscore
        r.effective_n = ess
        r.mean_distance = md
        results.append(r)
    return results


def v2a_target_touch_probability(
    state: pd.DataFrame,
    target_price: float,
    horizon: int,
    k: int = 120,
    lookback_rows: int = 756,
    recency_half_life: int = 252,
) -> tuple[float | None, int]:
    features = v2a_feature_columns(state)
    usable = state.dropna(subset=features)
    if usable.empty:
        return None, 0
    current_idx = usable.index[-1]
    current = float(state.loc[current_idx, "Close"])
    rel = float(target_price / current - 1.0)
    ids, weights, _ages, _ess, _md = _weighted_neighbors(
        state, features, current_idx, k, lookback_rows=lookback_rows, recency_half_life=recency_half_life
    )
    loc = {idx: i for i, idx in enumerate(state.index)}
    hits = []
    ws = []
    for idx, w in zip(ids, weights):
        i = loc[idx]
        if i + horizon >= len(state):
            continue
        entry = float(state.iloc[i]["Close"])
        fut = state.iloc[i + 1 : i + horizon + 1]
        if len(fut) < horizon:
            continue
        if rel >= 0:
            hit = float(fut["High"].max() / entry - 1 >= rel)
        else:
            hit = float(fut["Low"].min() / entry - 1 <= rel)
        hits.append(hit)
        ws.append(float(w))
    if not hits:
        return None, 0
    ww = np.asarray(ws, dtype=float)
    ww = ww / ww.sum()
    return float(np.sum(ww * np.asarray(hits, dtype=float))), len(hits)


def current_context(state: pd.DataFrame) -> dict:
    features = v2a_feature_columns(state)
    usable = state.dropna(subset=[c for c in features if c not in ("btc_ret1", "btc_ret5")])
    if usable.empty:
        return {}
    row = usable.iloc[-1]
    ai5 = float(row.get("ai_ret5", np.nan))
    breadth = float(row.get("ai_breadth1", np.nan))
    rel5 = float(row.get("rel_ai_5", np.nan))
    volz = float(row.get("vol_z", np.nan))

    if np.isfinite(ai5) and np.isfinite(breadth) and ai5 >= 0.02 and breadth >= 0.55:
        ai_label = "🟢 IA RISK-ON"
    elif np.isfinite(ai5) and np.isfinite(breadth) and ai5 <= -0.02 and breadth <= 0.45:
        ai_label = "🔴 IA RISK-OFF"
    else:
        ai_label = "🟡 IA MIXTA"

    if np.isfinite(rel5) and rel5 >= 0.03:
        rel_label = "🟢 IREN lidera al universo IA"
    elif np.isfinite(rel5) and rel5 <= -0.03:
        rel_label = "🔴 IREN se rezaga frente al universo IA"
    else:
        rel_label = "🟡 IREN acompaña al universo IA"

    if np.isfinite(volz) and volz >= 1.5:
        volume_label = "Volumen anormalmente alto"
    elif np.isfinite(volz) and volz <= -1.0:
        volume_label = "Volumen débil"
    else:
        volume_label = "Volumen normal"

    return {
        "ai_label": ai_label,
        "relative_label": rel_label,
        "volume_label": volume_label,
        "ai_ret5": ai5,
        "ai_breadth1": breadth,
        "rel_ai_5": rel5,
        "ire_ret5": float(row.get("ret5", np.nan)),
    }
