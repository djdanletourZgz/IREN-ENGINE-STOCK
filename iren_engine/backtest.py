from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .probability import daily_probabilities


def _build_base_rate_cache(
    state: pd.DataFrame,
    horizons=(1, 3, 5),
    thresholds=(0.03, 0.05, 0.10),
) -> dict[int, dict[str, np.ndarray]]:
    """Precalcula benchmarks rolling sin leakage en O(n*h).

    Para una predicción emitida en ``pos`` y horizonte ``h`` sólo cuenta
    entradas ``i <= pos-h``: esos resultados ya habían terminado en esa fecha.
    Sustituye el antiguo bucle que reconstruía TODO el histórico para cada
    predicción, horizonte y modelo.
    """
    n = len(state)
    if n == 0:
        return {}

    close = state["Close"].to_numpy(dtype=float)
    high = state["High"].to_numpy(dtype=float)
    low = state["Low"].to_numpy(dtype=float)
    cache: dict[int, dict[str, np.ndarray]] = {}

    for h in horizons:
        events: dict[str, np.ndarray] = {
            "close_up": np.full(n, np.nan, dtype=float),
        }
        for t in thresholds:
            key = int(t * 100)
            events[f"touch_up_{key}"] = np.full(n, np.nan, dtype=float)
            events[f"touch_down_{key}"] = np.full(n, np.nan, dtype=float)

        valid_entry = np.zeros(n, dtype=bool)
        last_entry = n - h
        for i in range(max(last_entry, 0)):
            entry = close[i]
            fut_close = close[i + h]
            fut_high = high[i + 1 : i + h + 1]
            fut_low = low[i + 1 : i + h + 1]
            if (
                not np.isfinite(entry)
                or not np.isfinite(fut_close)
                or len(fut_high) < h
                or len(fut_low) < h
                or not np.isfinite(fut_high).all()
                or not np.isfinite(fut_low).all()
            ):
                continue
            valid_entry[i] = True
            ret = fut_close / entry - 1.0
            up = float(np.max(fut_high) / entry - 1.0)
            down = float(np.min(fut_low) / entry - 1.0)
            events["close_up"][i] = float(ret > 0)
            for t in thresholds:
                key = int(t * 100)
                events[f"touch_up_{key}"][i] = float(up >= t)
                events[f"touch_down_{key}"][i] = float(down <= -t)

        valid_cum = np.cumsum(valid_entry.astype(int))
        out: dict[str, np.ndarray] = {
            "n": np.zeros(n, dtype=float),
        }
        cumsums = {
            name: np.cumsum(np.nan_to_num(values, nan=0.0))
            for name, values in events.items()
        }
        for name in events:
            out[name] = np.full(n, np.nan, dtype=float)

        for pos in range(n):
            cutoff = pos - h
            if cutoff < 0:
                continue
            count = int(valid_cum[cutoff])
            if count <= 0:
                continue
            out["n"][pos] = float(count)
            for name, cs in cumsums.items():
                out[name][pos] = float(cs[cutoff] / count)
        cache[int(h)] = out

    return cache


def _cached_base_rates(
    cache: dict[int, dict[str, np.ndarray]],
    pos: int,
    horizon: int,
) -> dict[str, float]:
    hcache = cache.get(int(horizon), {})
    if not hcache:
        return {}
    n_arr = hcache.get("n")
    if n_arr is None or pos >= len(n_arr) or n_arr[pos] <= 0:
        return {}
    out = {"n": int(n_arr[pos])}
    for key, arr in hcache.items():
        if key == "n" or pos >= len(arr):
            continue
        v = arr[pos]
        if np.isfinite(v):
            out[key] = float(v)
    return out


def walk_forward_backtest(
    state: pd.DataFrame,
    k: int = 120,
    min_train: int = 220,
    test_window: int = 350,
    horizons=(1, 3, 5),
    thresholds=(0.03, 0.05, 0.10),
    predictor: Callable | None = None,
    model_name: str = "V1",
    predictor_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Predicción histórica sin mirar futuro, compatible con V1 y V2-A."""
    if state is None or state.empty:
        return pd.DataFrame()

    predictor = predictor or daily_probabilities
    predictor_kwargs = predictor_kwargs or {}
    state = state.sort_index().copy()
    max_h = max(horizons)

    positions = [
        pos
        for pos in range(min_train, len(state) - max_h)
        if np.isfinite(state.iloc[pos]["Close"])
    ]
    if test_window and len(positions) > test_window:
        positions = positions[-test_window:]

    # Coste O(n) una sola vez por modelo, en vez de reconstruir el benchmark
    # miles de veces dentro del walk-forward.
    base_cache = _build_base_rate_cache(state, horizons=horizons, thresholds=thresholds)

    records = []
    for pos in positions:
        # Una vista/copia pequeña sigue garantizando que el predictor no ve futuro.
        truncated = state.iloc[: pos + 1]
        try:
            pred = predictor(
                truncated,
                horizons=horizons,
                k=k,
                thresholds=thresholds,
                **predictor_kwargs,
            )
        except Exception:
            continue
        if not pred:
            continue

        pred_by_h = {int(r.horizon.replace("D", "")): r for r in pred}
        entry = float(state.iloc[pos]["Close"])
        test_date = state.index[pos]

        for h in horizons:
            r = pred_by_h.get(h)
            if r is None:
                continue

            fut = state.iloc[pos + 1 : pos + h + 1]
            if len(fut) < h:
                continue

            actual_ret = float(fut.iloc[-1]["Close"] / entry - 1)
            actual_up = float(fut["High"].max() / entry - 1)
            actual_down = float(fut["Low"].min() / entry - 1)
            base = _cached_base_rates(base_cache, pos, h)

            rec = {
                "model": model_name,
                "date": pd.Timestamp(test_date),
                "horizon": h,
                "entry": entry,
                "n_neighbors": r.n,
                "confidence": r.confidence,
                "quality_score": getattr(r, "quality_score", np.nan),
                "effective_n": getattr(r, "effective_n", np.nan),
                "mean_distance": getattr(r, "mean_distance", np.nan),
                "p_close_up": r.probabilities.get("close_up", np.nan),
                "base_p_close_up": base.get("close_up", np.nan),
                "base_n": base.get("n", np.nan),
                "actual_close_up": float(actual_ret > 0),
                "expected_return": r.expected_return,
                "actual_return": actual_ret,
                "q10": r.q10_return,
                "q90": r.q90_return,
                "inside_p10_p90": float(
                    r.q10_return is not None
                    and r.q90_return is not None
                    and r.q10_return <= actual_ret <= r.q90_return
                ),
            }

            for c in [
                "ret1", "ret5", "ai_ret1", "ai_ret5", "ai_breadth1",
                "ai_corr60", "rel_ai_1", "rel_ai_5", "qqq_ret5",
                "vix_ret1", "vix_level_z", "btc_ret5", "vol_z", "atrp",
            ]:
                if c in state.columns:
                    v = state.iloc[pos].get(c, np.nan)
                    rec[c] = float(v) if pd.notna(v) else np.nan

            for t in thresholds:
                key = int(t * 100)
                rec[f"p_touch_up_{key}"] = r.probabilities.get(f"touch_+{key}%", np.nan)
                rec[f"p_touch_down_{key}"] = r.probabilities.get(f"touch_-{key}%", np.nan)
                rec[f"base_p_touch_up_{key}"] = base.get(f"touch_up_{key}", np.nan)
                rec[f"base_p_touch_down_{key}"] = base.get(f"touch_down_{key}", np.nan)
                rec[f"actual_touch_up_{key}"] = float(actual_up >= t)
                rec[f"actual_touch_down_{key}"] = float(actual_down <= -t)

            records.append(rec)

    return pd.DataFrame(records)


def calibration_summary(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()

    events = [("Cerrar arriba", "p_close_up", "base_p_close_up", "actual_close_up")]
    for key in (3, 5, 10):
        events.extend(
            [
                (f"Tocar +{key}%", f"p_touch_up_{key}", f"base_p_touch_up_{key}", f"actual_touch_up_{key}"),
                (f"Tocar -{key}%", f"p_touch_down_{key}", f"base_p_touch_down_{key}", f"actual_touch_down_{key}"),
            ]
        )

    rows = []
    group_cols = ["model", "horizon"] if "model" in bt.columns else ["horizon"]
    for keys, g in bt.groupby(group_cols):
        if isinstance(keys, tuple):
            model, h = keys
        else:
            model, h = "V1", keys
        for label, pcol, bcol, ycol in events:
            required = [pcol, ycol]
            if bcol in g.columns:
                required.append(bcol)
            z = g[required].dropna().copy()
            if z.empty:
                continue

            p = z[pcol].astype(float).clip(0, 1)
            y = z[ycol].astype(float)
            observed = float(y.mean())
            brier = float(np.mean((p - y) ** 2))
            if bcol in z.columns:
                bp = z[bcol].astype(float).clip(0, 1)
                base_brier = float(np.mean((bp - y) ** 2))
                base_prob = float(bp.mean())
            else:
                base_brier = np.nan
                base_prob = np.nan
            skill = np.nan if not np.isfinite(base_brier) or base_brier <= 0 else float(1 - brier / base_brier)
            rows.append(
                {
                    "Modelo": model,
                    "Horizonte": f"{h}D",
                    "Evento": label,
                    "N": len(z),
                    "Prob. media": float(p.mean()),
                    "Ocurrió": observed,
                    "Error calibración": float(abs(p.mean() - observed)),
                    "Brier": brier,
                    "Benchmark prob. media": base_prob,
                    "Brier benchmark rolling": base_brier,
                    "Brier skill": skill,
                }
            )
    return pd.DataFrame(rows)


def directional_summary(bt: pd.DataFrame, upper: float = 0.60, lower: float = 0.40) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["model", "horizon"] if "model" in bt.columns else ["horizon"]
    for keys, g in bt.groupby(group_cols):
        if isinstance(keys, tuple):
            model, h = keys
        else:
            model, h = "V1", keys
        cols = ["p_close_up", "actual_close_up", "actual_return"]
        if "quality_score" in g.columns:
            cols.append("quality_score")
        z = g[cols].dropna(subset=["p_close_up", "actual_close_up", "actual_return"]).copy()
        sig = z[(z.p_close_up >= upper) | (z.p_close_up <= lower)].copy()
        if sig.empty:
            rows.append({"Modelo": model, "Horizonte": f"{h}D", "Señales": 0, "Cobertura": 0.0, "Acierto": np.nan, "Retorno medio real": np.nan})
            continue
        sig["pred_up"] = sig.p_close_up >= upper
        sig["hit"] = sig.pred_up.astype(float) == sig.actual_close_up.astype(float)
        signed_ret = np.where(sig.pred_up, sig.actual_return, -sig.actual_return)
        rows.append(
            {
                "Modelo": model,
                "Horizonte": f"{h}D",
                "Señales": len(sig),
                "Cobertura": float(len(sig) / len(z)),
                "Acierto": float(sig.hit.mean()),
                "Retorno medio real": float(np.mean(signed_ret)),
                "Calidad media": float(sig["quality_score"].mean()) if "quality_score" in sig and sig["quality_score"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_bins(bt: pd.DataFrame, bins=(0.0, 0.4, 0.5, 0.6, 1.000001)) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()

    labels = ["<40%", "40–50%", "50–60%", "≥60%"]
    rows = []
    group_cols = ["model", "horizon"] if "model" in bt.columns else ["horizon"]
    for keys, g in bt.groupby(group_cols):
        if isinstance(keys, tuple):
            model, h = keys
        else:
            model, h = "V1", keys
        z = g[["p_close_up", "actual_close_up"]].dropna().copy()
        if z.empty:
            continue
        z["bucket"] = pd.cut(z["p_close_up"], bins=list(bins), labels=labels, include_lowest=True, right=False)
        for bucket, b in z.groupby("bucket", observed=False):
            if b.empty:
                continue
            rows.append(
                {
                    "Modelo": model,
                    "Horizonte": f"{h}D",
                    "Rango probabilidad": str(bucket),
                    "N": len(b),
                    "Prob. media": float(b["p_close_up"].mean()),
                    "Ocurrió": float(b["actual_close_up"].mean()),
                    "Diferencia": float(b["actual_close_up"].mean() - b["p_close_up"].mean()),
                }
            )
    return pd.DataFrame(rows)


def performance_by_period(bt: pd.DataFrame) -> pd.DataFrame:
    """Diagnóstico de estabilidad: divide cronológicamente cada modelo en 3 tramos."""
    if bt is None or bt.empty:
        return pd.DataFrame()
    rows = []
    models = bt["model"].dropna().unique() if "model" in bt.columns else ["V1"]
    for model in models:
        gm = bt[bt["model"] == model] if "model" in bt.columns else bt
        dates = sorted(pd.to_datetime(gm["date"].dropna().unique()))
        if len(dates) < 9:
            continue
        chunks = np.array_split(np.asarray(dates), 3)
        for j, chunk in enumerate(chunks, start=1):
            if len(chunk) == 0:
                continue
            z = gm[pd.to_datetime(gm["date"]).isin(pd.to_datetime(chunk))]
            for h, gh in z.groupby("horizon"):
                d = directional_summary(gh.assign(model=str(model)))
                c = calibration_summary(gh.assign(model=str(model)))
                core = c[c["Evento"] == "Cerrar arriba"]
                rows.append(
                    {
                        "Modelo": str(model),
                        "Periodo": f"T{j}",
                        "Desde": pd.to_datetime(chunk[0]).date(),
                        "Hasta": pd.to_datetime(chunk[-1]).date(),
                        "Horizonte": f"{h}D",
                        "N": int(len(gh)),
                        "Brier skill": float(core["Brier skill"].mean()) if not core.empty else np.nan,
                        "Acierto señales": float(d["Acierto"].mean()) if not d.empty else np.nan,
                        "Cobertura": float(d["Cobertura"].mean()) if not d.empty else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def performance_by_ai_regime(bt: pd.DataFrame) -> pd.DataFrame:
    """Diagnóstico V2-A por régimen IA definido ex ante con ai_ret5/breadth."""
    if bt is None or bt.empty or "ai_ret5" not in bt.columns:
        return pd.DataFrame()
    z = bt.copy()
    breadth = z.get("ai_breadth1", pd.Series(np.nan, index=z.index))
    risk_on = (z["ai_ret5"] >= 0.02) & (breadth >= 0.55)
    risk_off = (z["ai_ret5"] <= -0.02) & (breadth <= 0.45)
    z["ai_regime"] = np.select([risk_on, risk_off], ["RISK-ON", "RISK-OFF"], default="MIXTO")
    rows = []
    for (model, regime, h), g in z.groupby(["model", "ai_regime", "horizon"]):
        if len(g) < 10:
            continue
        c = calibration_summary(g)
        d = directional_summary(g)
        core = c[c["Evento"] == "Cerrar arriba"]
        rows.append(
            {
                "Modelo": model,
                "Régimen IA": regime,
                "Horizonte": f"{h}D",
                "N": len(g),
                "Brier skill": float(core["Brier skill"].mean()) if not core.empty else np.nan,
                "Acierto señales": float(d["Acierto"].mean()) if not d.empty else np.nan,
                "Cobertura": float(d["Cobertura"].mean()) if not d.empty else np.nan,
                "Retorno medio IREN": float(g["actual_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def model_comparison(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()
    cal = calibration_summary(bt)
    d = directional_summary(bt)
    rows = []
    models = sorted(bt["model"].dropna().unique()) if "model" in bt.columns else ["V1"]
    for model in models:
        cm = cal[(cal["Modelo"] == model) & (cal["Evento"] == "Cerrar arriba")]
        dm = d[d["Modelo"] == model]
        rows.append(
            {
                "Modelo": model,
                "Brier skill dirección": float(cm["Brier skill"].mean()) if not cm.empty else np.nan,
                "Acierto señales fuertes": float(dm["Acierto"].mean()) if not dm.empty else np.nan,
                "Cobertura señales fuertes": float(dm["Cobertura"].mean()) if not dm.empty else np.nan,
                "Retorno firmado medio": float(dm["Retorno medio real"].mean()) if not dm.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def human_verdict(cal: pd.DataFrame, direction: pd.DataFrame, model: str | None = None) -> str:
    if cal is None or cal.empty:
        return "No hay muestra suficiente para juzgar el modelo."
    c = cal.copy()
    d = direction.copy() if direction is not None else pd.DataFrame()
    if model is not None and "Modelo" in c.columns:
        c = c[c["Modelo"] == model]
        if not d.empty and "Modelo" in d.columns:
            d = d[d["Modelo"] == model]
    core = c[c["Evento"] == "Cerrar arriba"].copy()
    skills = core["Brier skill"].replace([np.inf, -np.inf], np.nan).dropna()
    mean_skill = float(skills.mean()) if not skills.empty else np.nan
    hits = d["Acierto"].replace([np.inf, -np.inf], np.nan).dropna() if not d.empty else pd.Series(dtype=float)
    mean_hit = float(hits.mean()) if not hits.empty else np.nan

    if np.isfinite(mean_skill) and mean_skill > 0.05 and np.isfinite(mean_hit) and mean_hit >= 0.55:
        return "🟢 PROMETEDOR: fuera de muestra mejora al benchmark rolling y las señales fuertes muestran ventaja. Hay que confirmar estabilidad por periodos."
    if np.isfinite(mean_skill) and mean_skill > 0 and np.isfinite(mean_hit) and mean_hit >= 0.50:
        return "🟡 EDGE DÉBIL: añade algo de información, pero todavía no suficiente para confiar dinero importante."
    return "🔴 NO DEMOSTRADO: no mejora de forma consistente al benchmark o falla cuando se moja."
