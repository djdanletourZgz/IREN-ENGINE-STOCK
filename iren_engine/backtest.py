from __future__ import annotations

import numpy as np
import pandas as pd

from .probability import daily_probabilities


def walk_forward_backtest(
    state: pd.DataFrame,
    k: int = 120,
    min_train: int = 220,
    test_window: int = 350,
    horizons=(1, 3, 5),
    thresholds=(0.03, 0.05, 0.10),
) -> pd.DataFrame:
    """Reproduce la predicción histórica sin mirar datos futuros.

    En cada fecha de test se trunca el dataframe exactamente en esa fecha,
    se ejecuta el mismo motor de probabilidades de la app y, sólo después,
    se compara contra lo que ocurrió en el dataframe completo.
    """
    if state is None or state.empty:
        return pd.DataFrame()

    state = state.sort_index().copy()
    feature_cols = [c for c in state.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    max_h = max(horizons)

    positions = []
    for pos in range(min_train, len(state) - max_h):
        row = state.iloc[pos]
        if row[feature_cols].notna().all() and np.isfinite(row["Close"]):
            positions.append(pos)

    if test_window and len(positions) > test_window:
        positions = positions[-test_window:]

    records = []
    for pos in positions:
        truncated = state.iloc[: pos + 1].copy()
        try:
            pred = daily_probabilities(truncated, horizons=horizons, k=k, thresholds=thresholds)
        except Exception:
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

            rec = {
                "date": pd.Timestamp(test_date),
                "horizon": h,
                "entry": entry,
                "n_neighbors": r.n,
                "confidence": r.confidence,
                "p_close_up": r.probabilities.get("close_up", np.nan),
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

            for t in thresholds:
                key = int(t * 100)
                rec[f"p_touch_up_{key}"] = r.probabilities.get(f"touch_+{key}%", np.nan)
                rec[f"p_touch_down_{key}"] = r.probabilities.get(f"touch_-{key}%", np.nan)
                rec[f"actual_touch_up_{key}"] = float(actual_up >= t)
                rec[f"actual_touch_down_{key}"] = float(actual_down <= -t)

            records.append(rec)

    return pd.DataFrame(records)


def calibration_summary(bt: pd.DataFrame) -> pd.DataFrame:
    """Métricas de calibración y Brier frente al base-rate del propio periodo."""
    if bt is None or bt.empty:
        return pd.DataFrame()

    events = [("Cerrar arriba", "p_close_up", "actual_close_up")]
    for key in (3, 5, 10):
        events.extend([
            (f"Tocar +{key}%", f"p_touch_up_{key}", f"actual_touch_up_{key}"),
            (f"Tocar -{key}%", f"p_touch_down_{key}", f"actual_touch_down_{key}"),
        ])

    rows = []
    for h, g in bt.groupby("horizon"):
        for label, pcol, ycol in events:
            z = g[[pcol, ycol]].dropna()
            if z.empty:
                continue
            p = z[pcol].astype(float).clip(0, 1)
            y = z[ycol].astype(float)
            observed = float(y.mean())
            brier = float(np.mean((p - y) ** 2))
            base_brier = float(np.mean((observed - y) ** 2))
            skill = np.nan if base_brier <= 0 else float(1 - brier / base_brier)
            rows.append({
                "Horizonte": f"{h}D",
                "Evento": label,
                "N": len(z),
                "Prob. media": float(p.mean()),
                "Ocurrió": observed,
                "Error calibración": float(abs(p.mean() - observed)),
                "Brier": brier,
                "Brier base-rate": base_brier,
                "Brier skill": skill,
            })
    return pd.DataFrame(rows)


def directional_summary(bt: pd.DataFrame, upper: float = 0.60, lower: float = 0.40) -> pd.DataFrame:
    """Evalúa sólo señales direccionales donde el modelo se moja (>60% o <40%)."""
    if bt is None or bt.empty:
        return pd.DataFrame()

    rows = []
    for h, g in bt.groupby("horizon"):
        z = g[["p_close_up", "actual_close_up", "actual_return"]].dropna().copy()
        sig = z[(z.p_close_up >= upper) | (z.p_close_up <= lower)].copy()
        if sig.empty:
            rows.append({"Horizonte": f"{h}D", "Señales": 0, "Cobertura": 0.0, "Acierto": np.nan, "Retorno medio real": np.nan})
            continue
        sig["pred_up"] = sig.p_close_up >= upper
        sig["hit"] = sig.pred_up.astype(float) == sig.actual_close_up.astype(float)
        signed_ret = np.where(sig.pred_up, sig.actual_return, -sig.actual_return)
        rows.append({
            "Horizonte": f"{h}D",
            "Señales": len(sig),
            "Cobertura": float(len(sig) / len(z)),
            "Acierto": float(sig.hit.mean()),
            "Retorno medio real": float(np.mean(signed_ret)),
        })
    return pd.DataFrame(rows)


def human_verdict(cal: pd.DataFrame, direction: pd.DataFrame) -> str:
    if cal is None or cal.empty:
        return "No hay muestra suficiente para juzgar el modelo."

    core = cal[cal["Evento"] == "Cerrar arriba"].copy()
    skills = core["Brier skill"].replace([np.inf, -np.inf], np.nan).dropna()
    mean_skill = float(skills.mean()) if not skills.empty else np.nan

    hits = direction["Acierto"].replace([np.inf, -np.inf], np.nan).dropna() if direction is not None and not direction.empty else pd.Series(dtype=float)
    mean_hit = float(hits.mean()) if not hits.empty else np.nan

    if np.isfinite(mean_skill) and mean_skill > 0.05 and np.isfinite(mean_hit) and mean_hit >= 0.55:
        return "🟢 PROMETEDOR: fuera de muestra el motor mejora al base-rate y las señales fuertes muestran ventaja. Aún hay que confirmar estabilidad por periodos y añadir costes."
    if np.isfinite(mean_skill) and mean_skill > 0:
        return "🟡 EDGE DÉBIL: hay algo de mejora estadística, pero todavía no suficiente para confiar dinero importante. Necesita más muestra y validación por regímenes."
    return "🔴 NO DEMOSTRADO: con esta muestra el motor no mejora de forma consistente al benchmark. Hay que modificarlo antes de fiarnos de sus probabilidades."
