"""Canonical station-forecast records and deterministic evaluation metrics."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ("year", "target_timestamp", "horizon", "station_id")
REQUIRED_COLUMNS = (*KEY_COLUMNS, "predicted_mm_15min", "observed_mm_15min", "is_observed")
INTENSITY_BINS = (("weak", 0.0, 1.25), ("moderate", 1.25, 6.25),
                  ("strong", 6.25, 12.5), ("extreme", 12.5, float("inf")))


def validate_records(records: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(records.columns)
    if missing:
        raise ValueError(f"{name}: colunas ausentes: {sorted(missing)}")
    result = records.loc[:, REQUIRED_COLUMNS].copy()
    result["target_timestamp"] = pd.to_datetime(result["target_timestamp"], utc=True, errors="raise")
    result["is_observed"] = result["is_observed"].astype(bool)
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{name}: há chaves de previsão duplicadas.")
    if (result["predicted_mm_15min"] < 0).any() or (result["observed_mm_15min"] < 0).any():
        raise ValueError(f"{name}: precipitação negativa não é válida.")
    return result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def align_records(named_records: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Require equal keys, masks and observed values across all experiments."""
    if len(named_records) < 2:
        raise ValueError("Informe ao menos dois experimentos para avaliação comparativa.")
    aligned = {name: validate_records(records, name) for name, records in named_records.items()}
    names = list(aligned)
    reference = aligned[names[0]]
    for name in names[1:]:
        current = aligned[name]
        if not reference.loc[:, KEY_COLUMNS].equals(current.loc[:, KEY_COLUMNS]):
            raise ValueError(f"{name}: chaves diferentes do experimento de referência {names[0]}.")
        if not reference["is_observed"].equals(current["is_observed"]):
            raise ValueError(f"{name}: máscara de observação diferente da referência.")
        if not np.allclose(reference["observed_mm_15min"], current["observed_mm_15min"], equal_nan=True):
            raise ValueError(f"{name}: alvo observado diferente da referência.")
    return aligned


def _continuous(prediction: np.ndarray, observed: np.ndarray) -> dict:
    if not len(observed):
        return {"n": 0, "rmse": None, "mae": None, "bias": None}
    error = prediction - observed
    return {"n": int(len(error)), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))}


def _categorical(prediction: np.ndarray, observed: np.ndarray, threshold: float) -> dict:
    return categorical_with_decision_threshold(prediction, observed, threshold, threshold)


def categorical_with_decision_threshold(
    prediction: np.ndarray, observed: np.ndarray, observed_threshold: float, decision_threshold: float,
) -> dict:
    forecast_event, observed_event = prediction >= decision_threshold, observed >= observed_threshold
    hits = int(np.sum(forecast_event & observed_event))
    misses = int(np.sum(~forecast_event & observed_event))
    false_alarms = int(np.sum(forecast_event & ~observed_event))
    correct_negatives = int(np.sum(~forecast_event & ~observed_event))
    total = hits + misses + false_alarms + correct_negatives
    pod = hits / (hits + misses) if hits + misses else None
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else None
    success_ratio = hits / (hits + false_alarms) if hits + false_alarms else None
    csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else None
    frequency_bias = ((hits + false_alarms) / (hits + misses)) if hits + misses else None
    random_hits = ((hits + misses) * (hits + false_alarms) / total) if total else 0.0
    ets_denominator = hits + misses + false_alarms - random_hits
    ets = (hits - random_hits) / ets_denominator if ets_denominator else None
    hss_denominator = ((hits + misses) * (misses + correct_negatives)
                       + (hits + false_alarms) * (false_alarms + correct_negatives))
    hss = (2 * (hits * correct_negatives - misses * false_alarms) / hss_denominator
           if hss_denominator else None)
    return {"n": total, "hits": hits, "misses": misses, "false_alarms": false_alarms,
            "correct_negatives": correct_negatives, "pod": pod, "far": far,
            "success_ratio": success_ratio, "csi": csi, "frequency_bias": frequency_bias,
            "ets": ets, "hss": hss}


def event_metrics(records: pd.DataFrame, threshold: float, *, gap_minutes: int = 15) -> dict:
    """Evaluate contiguous station events using the available forecast schedule.

    An event is a contiguous observed exceedance at one station. Detection is
    any thresholded forecast on an observed event timestep; lead is the largest
    available horizon among detections. This is intentionally station-based and
    does not claim a spatially continuous precipitation-event truth.
    """
    valid = records.loc[records["is_observed"]].copy()
    valid["target_timestamp"] = pd.to_datetime(valid["target_timestamp"], utc=True, errors="raise")
    event_count = detected = false_alert_records = 0
    lead_minutes: list[float] = []
    for _, group in valid.groupby("station_id", sort=False):
        group = group.sort_values("target_timestamp").reset_index(drop=True)
        observed_event = group["observed_mm_15min"].to_numpy() >= threshold
        forecast_event = group["predicted_mm_15min"].to_numpy() >= threshold
        timestamps = group["target_timestamp"].to_numpy()
        event_indices: list[list[int]] = []
        current: list[int] = []
        for index, is_event in enumerate(observed_event):
            contiguous = index and (timestamps[index] - timestamps[index - 1]) <= np.timedelta64(gap_minutes, "m")
            if is_event and (not current or contiguous):
                current.append(index)
            elif is_event:
                if current:
                    event_indices.append(current)
                current = [index]
            elif current:
                event_indices.append(current)
                current = []
        if current:
            event_indices.append(current)
        event_mask = np.zeros(len(group), dtype=bool)
        for indices in event_indices:
            event_count += 1
            event_mask[indices] = True
            available = np.asarray(indices)[forecast_event[indices]]
            if len(available):
                detected += 1
                lead_minutes.append(float(group.loc[available, "horizon"].max() * 15))
        false_alert_records += int(np.sum(forecast_event & ~event_mask))
    return {"threshold_mm_15min": threshold, "event_gap_minutes": gap_minutes,
            "events": event_count, "detected_events": detected,
            "event_detection_fraction": detected / event_count if event_count else None,
            "median_detected_lead_minutes": float(np.median(lead_minutes)) if lead_minutes else None,
            "max_detected_lead_minutes": float(np.max(lead_minutes)) if lead_minutes else None,
            "false_alert_records": false_alert_records}


def select_decision_threshold(
    records: pd.DataFrame, observed_threshold: float, candidates: Iterable[float],
) -> tuple[float, dict]:
    valid = records.loc[records["is_observed"]]
    prediction = valid["predicted_mm_15min"].to_numpy(float)
    observed = valid["observed_mm_15min"].to_numpy(float)
    scored = []
    for candidate in candidates:
        current = categorical_with_decision_threshold(prediction, observed, observed_threshold, candidate)
        scored.append((candidate, current))
    usable = [(candidate, current) for candidate, current in scored if current["csi"] is not None]
    if not usable:
        raise ValueError("Não há eventos suficientes na validação para selecionar limiar de decisão.")
    # Ties prefer lower FAR and then the decision closest to the physical event threshold.
    return max(usable, key=lambda item: (item[1]["csi"], -(item[1]["far"] or 0.0), -abs(item[0] - observed_threshold)))


def evaluate_records(records: pd.DataFrame, thresholds: Iterable[float]) -> dict:
    valid = records.loc[records["is_observed"]]
    prediction = valid["predicted_mm_15min"].to_numpy(dtype=float)
    observed = valid["observed_mm_15min"].to_numpy(dtype=float)
    result = {"global": _continuous(prediction, observed), "horizons": {}, "intensity": {}, "thresholds": {}}
    for horizon, group in valid.groupby("horizon", sort=True):
        result["horizons"][str(int(horizon))] = _continuous(
            group["predicted_mm_15min"].to_numpy(float), group["observed_mm_15min"].to_numpy(float))
    for name, low, high in INTENSITY_BINS:
        selection = (observed >= low) & (observed < high)
        result["intensity"][name] = _continuous(prediction[selection], observed[selection])
    for threshold in thresholds:
        threshold_key = f"{threshold:g}"
        result["thresholds"][threshold_key] = {"global": _categorical(prediction, observed, threshold), "horizons": {}}
        for horizon, group in valid.groupby("horizon", sort=True):
            result["thresholds"][threshold_key]["horizons"][str(int(horizon))] = _categorical(
                group["predicted_mm_15min"].to_numpy(float), group["observed_mm_15min"].to_numpy(float), threshold)
    return result


def daily_aggregates(records: pd.DataFrame, thresholds: Iterable[float], horizon: int | None = None) -> pd.DataFrame:
    valid = records.loc[records["is_observed"]].copy()
    valid["target_timestamp"] = pd.to_datetime(valid["target_timestamp"], utc=True, errors="raise")
    if horizon is not None:
        valid = valid.loc[valid["horizon"] == horizon]
    valid["local_day"] = valid["target_timestamp"].dt.tz_convert("America/Sao_Paulo").dt.date.astype(str)
    valid["error"] = valid["predicted_mm_15min"] - valid["observed_mm_15min"]
    output = valid.groupby("local_day", sort=True).agg(
        se=("error", lambda values: float(np.square(values).sum())),
        ae=("error", lambda values: float(np.abs(values).sum())),
        bias=("error", "sum"), n=("error", "size"),
    )
    for threshold in thresholds:
        forecast = valid["predicted_mm_15min"] >= threshold
        observed = valid["observed_mm_15min"] >= threshold
        prefix = f"t{threshold:g}_"
        output[prefix + "hits"] = (forecast & observed).groupby(valid["local_day"]).sum()
        output[prefix + "misses"] = ((~forecast) & observed).groupby(valid["local_day"]).sum()
        output[prefix + "false_alarms"] = (forecast & (~observed)).groupby(valid["local_day"]).sum()
        output[prefix + "correct_negatives"] = ((~forecast) & (~observed)).groupby(valid["local_day"]).sum()
    return output.fillna(0.0)


def _continuous_from_sums(se: float, ae: float, bias: float, n: float) -> tuple[float, float, float]:
    return (float(np.sqrt(se / n)), float(ae / n), float(bias / n)) if n else (np.nan, np.nan, np.nan)


def _csi_from_sums(hits: float, misses: float, false_alarms: float) -> float:
    denominator = hits + misses + false_alarms
    return float(hits / denominator) if denominator else np.nan


def paired_daily_bootstrap(
    baseline: pd.DataFrame, candidate: pd.DataFrame, thresholds: Iterable[float], *, replicates: int, seed: int,
) -> dict:
    """Bootstrap paired by local day from already aligned forecast records."""
    if replicates <= 0:
        return {}
    base_daily, candidate_daily = daily_aggregates(baseline, thresholds), daily_aggregates(candidate, thresholds)
    if not base_daily.index.equals(candidate_daily.index):
        raise ValueError("Dias de bootstrap diferem entre previsão de referência e candidata.")
    rng, day_count = np.random.default_rng(seed), len(base_daily)
    if not day_count:
        raise ValueError("Não há dias observados para bootstrap.")
    outcomes = {"mae_difference": [], "rmse_difference": [], "skill_mae": [], "skill_rmse": []}
    for threshold in thresholds:
        outcomes[f"csi_difference_threshold_{threshold:g}"] = []
    for _ in range(replicates):
        draw = rng.integers(0, day_count, size=day_count)
        base = base_daily.iloc[draw].sum(axis=0)
        current = candidate_daily.iloc[draw].sum(axis=0)
        base_rmse, base_mae, _ = _continuous_from_sums(base.se, base.ae, base.bias, base.n)
        current_rmse, current_mae, _ = _continuous_from_sums(current.se, current.ae, current.bias, current.n)
        outcomes["mae_difference"].append(current_mae - base_mae)
        outcomes["rmse_difference"].append(current_rmse - base_rmse)
        outcomes["skill_mae"].append(1.0 - current_mae / base_mae if base_mae else np.nan)
        outcomes["skill_rmse"].append(1.0 - current_rmse / base_rmse if base_rmse else np.nan)
        for threshold in thresholds:
            prefix = f"t{threshold:g}_"
            base_csi = _csi_from_sums(base[prefix + "hits"], base[prefix + "misses"], base[prefix + "false_alarms"])
            current_csi = _csi_from_sums(current[prefix + "hits"], current[prefix + "misses"], current[prefix + "false_alarms"])
            outcomes[f"csi_difference_threshold_{threshold:g}"].append(current_csi - base_csi)
    result = {}
    for name, values in outcomes.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        result[name] = ({"estimate": float(np.mean(finite)),
                         "ci95": [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]}
                        if len(finite) else {"estimate": None, "ci95": [None, None]})
    return result
