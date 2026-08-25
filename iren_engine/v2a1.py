from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler

from .probability import ProbabilityResult
from .v2a import build_v2a_state


@dataclass
class FitDiagnostics:
    n_train: int
    base_rate: float
    ood_score: float
    ood: bool
    top_drivers: list[str]


def _exp_weights(n: int, half_life: float) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    age = np.arange(n - 1, -1, -1, dtype=float)
    if half_life <= 0:
        w = np.ones(n, dtype=float)
    else:
        w = np.power(0.5, age / float(half_life))
    return w / w.sum()


def _causal_beta(
    iren_ret: pd.Series,
    factor_ret: pd.Series,
    window: int = 120,
    half_life: float = 18.0,
    min_obs: int = 40,
) -> tuple[pd.Series, pd.Series]:
    """Beta e intercepto robustos. Cada fila t usa sólo observaciones < t."""
    idx = iren_ret.index
    beta = pd.Series(np.nan, index=idx, dtype=float)
    alpha = pd.Series(np.nan, index=idx, dtype=float)
    y_all = iren_ret.to_numpy(dtype=float)
    x_all = factor_ret.to_numpy(dtype=float)

    for i in range(len(idx)):
        start = max(0, i - int(window))
        x = x_all[start:i]
        y = y_all[start:i]
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < int(min_obs):
            continue
        x = x[valid]
        y = y[valid]
        w = _exp_weights(len(x), half_life)
        try:
            model = HuberRegressor(
                epsilon=1.35,
                alpha=0.0001,
                fit_intercept=True,
                max_iter=200,
            )
            model.fit(x.reshape(-1, 1), y, sample_weight=w)
            beta.iloc[i] = float(model.coef_[0])
            alpha.iloc[i] = float(model.intercept_)
        except Exception:
            xm = float(np.sum(w * x))
            ym = float(np.sum(w * y))
            varx = float(np.sum(w * np.square(x - xm)))
            if varx <= 1e-12:
                continue
            cov = float(np.sum(w * (x - xm) * (y - ym)))
            b = cov / varx
            beta.iloc[i] = b
            alpha.iloc[i] = ym - b * xm
    return beta, alpha


def _transition_label(row: pd.Series) -> str:
    ai5 = float(row.get("ai_ret5", np.nan))
    accel = float(row.get("ai_accel3", np.nan))
    breadth_d = float(row.get("breadth_delta3", np.nan))
    ai1 = float(row.get("ai_ret1", np.nan))
    breadth = float(row.get("ai_breadth1", np.nan))
    if not np.isfinite(ai5) or not np.isfinite(accel):
        return "SIN DATOS"
    if ai5 < 0:
        if ai1 > 0 and np.isfinite(breadth) and breadth >= 0.55:
            return "RISK-OFF → PRIMER REBOTE"
        if accel > 0 and (not np.isfinite(breadth_d) or breadth_d >= 0):
            return "RISK-OFF → ESTABILIZANDO"
        if accel < 0:
            return "RISK-OFF → ACELERANDO"
        return "RISK-OFF → MIXTO"
    if ai5 > 0:
        if accel > 0:
            return "RISK-ON → ACELERANDO"
        if accel < 0 and (not np.isfinite(breadth_d) or breadth_d <= 0):
            return "RISK-ON → AGOTÁNDOSE"
        return "RISK-ON → MIXTO"
    return "MIXTO"


def build_v2a1_state(
    iren: pd.DataFrame,
    market: dict[str, pd.DataFrame],
    ai_tickers: Iterable[str],
    beta_window: int = 120,
    beta_half_life: float = 18.0,
    beta_min_obs: int = 40,
) -> pd.DataFrame:
    """V2-A.1 = V2-A + residual IREN/AI + variables de transición causales."""
    out = build_v2a_state(iren, market, ai_tickers=ai_tickers).copy()

    beta, alpha = _causal_beta(
        out["ret1"],
        out["ai_ret1"],
        window=beta_window,
        half_life=beta_half_life,
        min_obs=beta_min_obs,
    )
    out["ai_beta"] = beta
    out["ai_alpha"] = alpha
    out["iren_expected_ret1"] = out["ai_alpha"] + out["ai_beta"] * out["ai_ret1"]
    out["resid_1"] = out["ret1"] - out["iren_expected_ret1"]
    out["resid_3"] = out["resid_1"].rolling(3, min_periods=2).sum()
    out["resid_5"] = out["resid_1"].rolling(5, min_periods=3).sum()
    out["resid_delta"] = out["resid_1"] - out["resid_1"].shift(1)

    rmean = out["resid_1"].shift(1).rolling(60, min_periods=30).mean()
    rstd = out["resid_1"].shift(1).rolling(60, min_periods=30).std().replace(0, np.nan)
    out["resid_z60"] = (out["resid_1"] - rmean) / rstd
    out["resid_disp_adj"] = out["resid_1"] / out["ai_dispersion1"].replace(0, np.nan)

    out["ai_accel3"] = out["ai_ret3"] - out["ai_ret3"].shift(3)
    breadth3 = out["ai_breadth1"].rolling(3, min_periods=2).mean()
    out["breadth_delta3"] = breadth3 - breadth3.shift(3)
    out["dispersion_delta3"] = out["ai_dispersion1"] - out["ai_dispersion1"].shift(3)

    neg_ai5 = (-out["ai_ret5"]).clip(lower=0)
    pos_ai5 = out["ai_ret5"].clip(lower=0)
    pos_accel = out["ai_accel3"].clip(lower=0)
    neg_accel = (-out["ai_accel3"]).clip(lower=0)
    pos_breadth_d = out["breadth_delta3"].clip(lower=0)
    neg_breadth_d = (-out["breadth_delta3"]).clip(lower=0)

    out["riskoff_stabilize_score"] = neg_ai5 * pos_accel * (1.0 + pos_breadth_d)
    out["riskoff_accel_score"] = neg_ai5 * neg_accel
    out["riskon_accel_score"] = pos_ai5 * pos_accel * (1.0 + pos_breadth_d)
    out["riskon_fade_score"] = pos_ai5 * neg_accel * (1.0 + neg_breadth_d)
    out["transition_label"] = out.apply(_transition_label, axis=1)
    return out


V2A1_FEATURES = [
    "ret1", "ret5", "dist_ema21", "atrp", "vol_z",
    "ai_ret1", "ai_ret5", "ai_breadth1", "ai_dispersion1", "vix_ret1",
    "resid_1", "resid_3", "resid_z60", "resid_delta",
    "ai_accel3", "breadth_delta3", "dispersion_delta3",
    "riskoff_stabilize_score", "riskoff_accel_score",
    "riskon_accel_score", "riskon_fade_score",
]


def v2a1_feature_columns(state: pd.DataFrame) -> list[str]:
    return [c for c in V2A1_FEATURES if c in state.columns]


def _fit_one(
    state: pd.DataFrame,
    pos: int,
    horizon: int,
    features: list[str],
    model_start: str = "2024-11-01",
    lookback_rows: int = 504,
    min_train: int = 80,
    c_value: float = 0.25,
) -> tuple[float | None, FitDiagnostics | None, dict[str, float]]:
    if pos < horizon or pos >= len(state):
        return None, None, {}
    cur = state.iloc[pos]
    if cur[features].isna().any():
        return None, None, {}

    start_ts = pd.Timestamp(model_start)
    start_pos = int(state.index.searchsorted(start_ts, side="left"))
    last_train_pos = pos - horizon
    first_train_pos = max(start_pos, last_train_pos - int(lookback_rows) + 1)
    if last_train_pos < first_train_pos:
        return None, None, {}

    close = state["Close"].to_numpy(dtype=float)
    train_idx = np.arange(first_train_pos, last_train_pos + 1, dtype=int)
    feat = state.iloc[train_idx][features]
    future_close = close[train_idx + int(horizon)]
    entry_close = close[train_idx]
    valid = (
        feat.notna().all(axis=1).to_numpy()
        & np.isfinite(future_close)
        & np.isfinite(entry_close)
        & (entry_close > 0)
    )
    train_idx = train_idx[valid]
    if len(train_idx) < int(min_train):
        return None, None, {}

    X = state.iloc[train_idx][features].to_numpy(dtype=float)
    y = (close[train_idx + int(horizon)] > close[train_idx]).astype(int)
    if len(np.unique(y)) < 2:
        return None, None, {}

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    q = scaler.transform(cur[features].to_numpy(dtype=float).reshape(1, -1))

    model = LogisticRegression(
        penalty="l2",
        C=float(c_value),
        solver="liblinear",
        max_iter=500,
    )
    model.fit(Xs, y)
    p_up = float(model.predict_proba(q)[0, 1])

    z = q[0]
    ood_score = float(np.sqrt(np.mean(np.square(z))))
    ood = bool(np.max(np.abs(z)) >= 4.0 or ood_score >= 2.5)
    contribution = model.coef_[0] * z
    order = np.argsort(np.abs(contribution))[::-1][:4]
    drivers = [
        f"{features[j]} {'↑' if contribution[j] >= 0 else '↓'}"
        for j in order
    ]
    diag = FitDiagnostics(
        n_train=int(len(y)),
        base_rate=float(np.mean(y)),
        ood_score=ood_score,
        ood=ood,
        top_drivers=drivers,
    )
    coefs = {f"coef_{f}": float(v) for f, v in zip(features, model.coef_[0])}
    return p_up, diag, coefs


def v2a1_probabilities(
    state: pd.DataFrame,
    horizons=(1, 3, 5),
    model_start: str = "2024-11-01",
    lookback_rows: int = 504,
    min_train: int = 80,
    c_value: float = 0.25,
) -> list[ProbabilityResult]:
    features = v2a1_feature_columns(state)
    if not features or state.empty:
        return []
    pos = len(state) - 1
    results: list[ProbabilityResult] = []
    for h in horizons:
        p, diag, _ = _fit_one(
            state,
            pos,
            int(h),
            features,
            model_start=model_start,
            lookback_rows=lookback_rows,
            min_train=min_train,
            c_value=c_value,
        )
        if p is None or diag is None:
            continue
        r = ProbabilityResult(
            horizon=f"{int(h)}D",
            n=diag.n_train,
            probabilities={"close_up": p},
            confidence="NO VALIDADA",
        )
        r.base_rate = diag.base_rate
        r.ood_score = diag.ood_score
        r.ood = diag.ood
        r.top_drivers = diag.top_drivers
        results.append(r)
    return results


def current_v2a1_context(state: pd.DataFrame) -> dict:
    if state.empty:
        return {}
    row = state.iloc[-1]
    resid = float(row.get("resid_1", np.nan))
    rz = float(row.get("resid_z60", np.nan))
    beta = float(row.get("ai_beta", np.nan))
    if np.isfinite(rz):
        if rz >= 1.0:
            residual_label = "🟢 IREN mucho más fuerte que su factor IA"
        elif rz <= -1.0:
            residual_label = "🔴 IREN mucho más débil que su factor IA"
        else:
            residual_label = "🟡 Residual IREN/IA normal"
    else:
        residual_label = "Residual sin muestra"
    ai_ret5 = row.get("ai_ret5", np.nan)
    ai_breadth1 = row.get("ai_breadth1", np.nan)
    return {
        "transition": str(row.get("transition_label", "—")),
        "ai_beta": beta if np.isfinite(beta) else None,
        "resid_1": resid if np.isfinite(resid) else None,
        "resid_z60": rz if np.isfinite(rz) else None,
        "residual_label": residual_label,
        "ai_ret5": float(ai_ret5) if pd.notna(ai_ret5) and np.isfinite(ai_ret5) else None,
        "ai_breadth1": float(ai_breadth1) if pd.notna(ai_breadth1) and np.isfinite(ai_breadth1) else None,
    }


def walk_forward_v2a1(
    state: pd.DataFrame,
    test_window: int = 350,
    horizons=(1, 3, 5),
    model_start: str = "2024-11-01",
    lookback_rows: int = 504,
    min_train: int = 80,
    c_value: float = 0.25,
    model_name: str = "V2-A.1",
) -> pd.DataFrame:
    """Backtest rápido del motor direccional V2-A.1; targets siempre resueltos antes de entrenar."""
    if state is None or state.empty:
        return pd.DataFrame()
    state = state.sort_index().copy()
    features = v2a1_feature_columns(state)
    if not features:
        return pd.DataFrame()
    max_h = int(max(horizons))

    positions = list(range(0, len(state) - max_h))
    if test_window and len(positions) > int(test_window):
        positions = positions[-int(test_window):]

    close = state["Close"].to_numpy(dtype=float)
    records = []
    for pos in positions:
        entry = close[pos]
        if not np.isfinite(entry) or entry <= 0:
            continue
        for h in horizons:
            p, diag, coefs = _fit_one(
                state,
                pos,
                int(h),
                features,
                model_start=model_start,
                lookback_rows=lookback_rows,
                min_train=min_train,
                c_value=c_value,
            )
            if p is None or diag is None or pos + int(h) >= len(state):
                continue
            actual_ret = float(close[pos + int(h)] / entry - 1.0)
            rec = {
                "model": model_name,
                "date": pd.Timestamp(state.index[pos]),
                "pos": int(pos),
                "resolved_pos": int(pos + int(h)),
                "horizon": int(h),
                "entry": float(entry),
                "n_train": int(diag.n_train),
                "p_close_up": float(p),
                "base_p_close_up": float(diag.base_rate),
                "actual_close_up": float(actual_ret > 0),
                "actual_return": actual_ret,
                "ood_score": float(diag.ood_score),
                "ood": bool(diag.ood),
                "transition": str(state.iloc[pos].get("transition_label", "—")),
            }
            for c in [
                "ai_beta", "resid_1", "resid_3", "resid_5", "resid_z60", "resid_delta",
                "ai_ret1", "ai_ret5", "ai_breadth1", "ai_dispersion1", "ai_accel3",
                "breadth_delta3", "dispersion_delta3", "atrp", "vol_z", "vix_ret1",
            ]:
                if c in state.columns:
                    v = state.iloc[pos].get(c, np.nan)
                    rec[c] = float(v) if pd.notna(v) and np.isfinite(v) else np.nan
            rec.update(coefs)
            records.append(rec)
    return pd.DataFrame(records)


def direction_metrics(bt: pd.DataFrame, upper: float = 0.60, lower: float = 0.40) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()
    rows = []
    for h, g in bt.groupby("horizon"):
        z = g.dropna(subset=["p_close_up", "base_p_close_up", "actual_close_up", "actual_return"]).copy()
        if z.empty:
            continue
        p = z["p_close_up"].astype(float).clip(0, 1)
        b = z["base_p_close_up"].astype(float).clip(0, 1)
        y = z["actual_close_up"].astype(float)
        brier = float(np.mean(np.square(p - y)))
        brier_base = float(np.mean(np.square(b - y)))
        skill = np.nan if brier_base <= 0 else float(1.0 - brier / brier_base)
        sig = z[(z["p_close_up"] >= upper) | (z["p_close_up"] <= lower)].copy()
        if not sig.empty:
            pred_up = sig["p_close_up"] >= upper
            hit = pred_up.astype(float) == sig["actual_close_up"].astype(float)
            signed_ret = np.where(pred_up, sig["actual_return"], -sig["actual_return"])
            hit_rate = float(hit.mean())
            coverage = float(len(sig) / len(z))
            signed = float(np.mean(signed_ret))
        else:
            hit_rate = coverage = signed = np.nan
        rows.append({
            "Horizonte": f"{int(h)}D",
            "N": int(len(z)),
            "Brier": brier,
            "Brier benchmark": brier_base,
            "Brier skill": skill,
            "Señales fuertes": int(len(sig)),
            "Cobertura": coverage,
            "Acierto señales": hit_rate,
            "Retorno firmado medio": signed,
            "OOD": float(z["ood"].mean()) if "ood" in z else np.nan,
        })
    return pd.DataFrame(rows)


def performance_by_period_v2a1(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()
    dates = sorted(pd.to_datetime(bt["date"].dropna().unique()))
    if len(dates) < 9:
        return pd.DataFrame()
    chunks = np.array_split(np.asarray(dates), 3)
    rows = []
    for j, chunk in enumerate(chunks, start=1):
        z = bt[pd.to_datetime(bt["date"]).isin(pd.to_datetime(chunk))]
        m = direction_metrics(z)
        for _, r in m.iterrows():
            d = r.to_dict()
            d.update({"Periodo": f"T{j}", "Desde": pd.Timestamp(chunk[0]).date(), "Hasta": pd.Timestamp(chunk[-1]).date()})
            rows.append(d)
    return pd.DataFrame(rows)


def performance_by_volatility_v2a1(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty or "atrp" not in bt.columns:
        return pd.DataFrame()
    z = bt.copy()
    try:
        z["vol_regime"] = pd.qcut(z["atrp"], 3, labels=["VOL BAJA", "VOL MEDIA", "VOL ALTA"], duplicates="drop")
    except Exception:
        return pd.DataFrame()
    rows = []
    for regime, g in z.groupby("vol_regime", observed=False):
        m = direction_metrics(g)
        for _, r in m.iterrows():
            d = r.to_dict()
            d["Régimen volatilidad"] = str(regime)
            rows.append(d)
    return pd.DataFrame(rows)


def performance_by_transition_v2a1(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty or "transition" not in bt.columns:
        return pd.DataFrame()
    rows = []
    for transition, g in bt.groupby("transition"):
        if len(g) < 12:
            continue
        m = direction_metrics(g)
        for _, r in m.iterrows():
            d = r.to_dict()
            d["Transición"] = str(transition)
            rows.append(d)
    return pd.DataFrame(rows)


def calibration_bins_v2a1(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()
    labels = ["<40%", "40–50%", "50–60%", "≥60%"]
    rows = []
    for h, g in bt.groupby("horizon"):
        z = g.dropna(subset=["p_close_up", "actual_close_up"]).copy()
        if z.empty:
            continue
        z["bucket"] = pd.cut(
            z["p_close_up"],
            bins=[0.0, 0.4, 0.5, 0.6, 1.000001],
            labels=labels,
            include_lowest=True,
            right=False,
        )
        for bucket, b in z.groupby("bucket", observed=False):
            if b.empty:
                continue
            rows.append({
                "Horizonte": f"{int(h)}D",
                "Rango probabilidad": str(bucket),
                "N": int(len(b)),
                "Prob. media": float(b["p_close_up"].mean()),
                "Ocurrió": float(b["actual_close_up"].mean()),
                "Diferencia": float(b["actual_close_up"].mean() - b["p_close_up"].mean()),
            })
    return pd.DataFrame(rows)


def residual_deciles_v2a1(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty or "resid_z60" not in bt.columns:
        return pd.DataFrame()
    rows = []
    for h, g in bt.groupby("horizon"):
        z = g.dropna(subset=["resid_z60", "actual_return"]).copy()
        if len(z) < 40:
            continue
        try:
            z["decile"] = pd.qcut(z["resid_z60"], 10, labels=False, duplicates="drop") + 1
        except Exception:
            continue
        for d, b in z.groupby("decile"):
            rows.append({
                "Horizonte": f"{int(h)}D",
                "Decil residual": int(d),
                "N": int(len(b)),
                "Residual z medio": float(b["resid_z60"].mean()),
                "Retorno forward medio": float(b["actual_return"].mean()),
                "P cerrar arriba real": float(b["actual_close_up"].mean()),
            })
    return pd.DataFrame(rows)


def _wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def attach_empirical_reliability_v2a1(
    bt: pd.DataFrame,
    min_n: int = 20,
    max_history: int = 140,
    probability_radius: float = 0.10,
) -> pd.DataFrame:
    """Fiabilidad ex-ante: cada fila usa sólo predicciones OOS anteriores ya resueltas."""
    if bt is None or bt.empty:
        return pd.DataFrame()
    out = bt.copy().sort_values(["horizon", "pos"]).reset_index(drop=True)
    out["pred_up"] = out["p_close_up"] >= 0.5
    out["hit"] = out["pred_up"].astype(float) == out["actual_close_up"].astype(float)
    out["reliability"] = np.nan
    out["reliability_n"] = 0
    out["reliability_low95"] = np.nan
    out["reliability_high95"] = np.nan

    for h, idxs in out.groupby("horizon").groups.items():
        ordered = list(idxs)
        hist_all = out.loc[ordered]
        for idx in ordered:
            row = out.loc[idx]
            pos = int(row["pos"])
            p = float(row["p_close_up"])
            side = bool(row["pred_up"])
            hist = hist_all[
                (hist_all["resolved_pos"] <= pos)
                & (hist_all.index != idx)
                & (hist_all["pred_up"] == side)
                & ((hist_all["p_close_up"] - p).abs() <= float(probability_radius))
            ].sort_values("pos")
            if len(hist) > int(max_history):
                hist = hist.iloc[-int(max_history):]
            if len(hist) < int(min_n):
                continue
            hits = int(hist["hit"].sum())
            n = int(len(hist))
            lo, hi = _wilson_interval(hits, n)
            out.at[idx, "reliability"] = float(hits / n)
            out.at[idx, "reliability_n"] = n
            out.at[idx, "reliability_low95"] = lo
            out.at[idx, "reliability_high95"] = hi
    return out


def reliability_summary_v2a1(bt_with_rel: pd.DataFrame) -> pd.DataFrame:
    if bt_with_rel is None or bt_with_rel.empty or "reliability" not in bt_with_rel.columns:
        return pd.DataFrame()
    z = bt_with_rel.dropna(subset=["reliability", "hit"]).copy()
    if z.empty:
        return pd.DataFrame()
    z["reliability_bucket"] = pd.cut(
        z["reliability"],
        bins=[0.0, 0.50, 0.60, 0.70, 1.000001],
        labels=["<50%", "50–60%", "60–70%", "≥70%"],
        include_lowest=True,
        right=False,
    )
    rows = []
    for (h, bucket), g in z.groupby(["horizon", "reliability_bucket"], observed=False):
        if g.empty:
            continue
        rows.append({
            "Horizonte": f"{int(h)}D",
            "Fiabilidad declarada": str(bucket),
            "N": int(len(g)),
            "Fiabilidad media": float(g["reliability"].mean()),
            "Acierto real": float(g["hit"].mean()),
            "IC95 inferior medio": float(g["reliability_low95"].mean()),
        })
    return pd.DataFrame(rows)
