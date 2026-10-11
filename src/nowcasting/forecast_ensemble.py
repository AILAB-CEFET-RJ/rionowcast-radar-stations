"""Causal post-processing ensembles over aligned forecast records."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from nowcasting.forecast_evaluation import align_records, validate_records


def _continuous_score(prediction: np.ndarray, observed: np.ndarray, metric: str) -> float:
    error = prediction - observed
    if metric == "mae":
        return float(np.mean(np.abs(error)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(error ** 2)))
    raise ValueError(f"Métrica de seleção desconhecida: {metric}.")


def select_horizon_weights(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    weights: Iterable[float],
    *,
    metric: str = "mae",
) -> tuple[dict[int, float], dict[int, list[dict[str, float]]]]:
    """Select one candidate weight per horizon using validation records only."""
    aligned = align_records({"baseline": baseline, "candidate": candidate})
    baseline, candidate = aligned["baseline"], aligned["candidate"]
    candidate_weights = sorted(set(float(weight) for weight in weights))
    if not candidate_weights or any(weight < 0 or weight > 1 for weight in candidate_weights):
        raise ValueError("Os pesos devem estar no intervalo fechado [0, 1].")
    selected: dict[int, float] = {}
    reports: dict[int, list[dict[str, float]]] = {}
    for horizon in sorted(baseline["horizon"].unique()):
        base = baseline.loc[(baseline["horizon"] == horizon) & baseline["is_observed"]]
        current = candidate.loc[(candidate["horizon"] == horizon) & candidate["is_observed"]]
        if not len(base):
            raise ValueError(f"Horizonte {horizon}: não há observações para selecionar peso.")
        observed = base["observed_mm_15min"].to_numpy(float)
        base_prediction = base["predicted_mm_15min"].to_numpy(float)
        candidate_prediction = current["predicted_mm_15min"].to_numpy(float)
        horizon_reports = []
        for weight in candidate_weights:
            prediction = (1.0 - weight) * base_prediction + weight * candidate_prediction
            horizon_reports.append({"candidate_weight": weight, metric: _continuous_score(prediction, observed, metric)})
        best = min(horizon_reports, key=lambda item: (item[metric], item["candidate_weight"]))
        selected[int(horizon)] = float(best["candidate_weight"])
        reports[int(horizon)] = horizon_reports
    return selected, reports


def blend_horizons(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    weights: dict[int, float],
    *,
    experiment_id: str,
) -> pd.DataFrame:
    """Blend predictions in physical precipitation units using fixed horizon weights."""
    aligned = align_records({"baseline": baseline, "candidate": candidate})
    base, current = aligned["baseline"], aligned["candidate"]
    horizons = {int(value) for value in base["horizon"].unique()}
    if horizons != set(weights):
        raise ValueError("Os pesos devem cobrir exatamente todos os horizontes dos registros.")
    result = base.copy()
    applied = result["horizon"].map(weights).to_numpy(float)
    prediction = ((1.0 - applied) * base["predicted_mm_15min"].to_numpy(float)
                  + applied * current["predicted_mm_15min"].to_numpy(float))
    result["predicted_mm_15min"] = np.maximum(prediction, 0.0)
    result = result.drop(columns="experiment_id", errors="ignore")
    result.insert(0, "experiment_id", experiment_id)
    return validate_records(result, experiment_id).assign(experiment_id=experiment_id)
