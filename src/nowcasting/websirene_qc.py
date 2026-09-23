"""Quality-control utilities for WebSirene precipitation observations."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable
import json
import math

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "version": "v1",
    "temporal_resolution_minutes": 15,
    "thresholds": {
        "expected_interval_minutes": 5,
        "gap_multiplier": 6,
        "constant_run_length": 12,
        "constant_value_min_mm": 0.1,
        "suspect_m15_mm": 50.0,
        "reject_m15_mm": 175.0,
    },
    "station_acceptance": {
        "min_accepted_observations": 1000,
        "min_years_with_accepted_data": 3,
        "max_rejected_fraction": 0.01,
        "max_suspect_fraction": 0.01,
    },
}

REJECTED_FLAGS = {
    "invalid_timestamp",
    "invalid_m15",
    "negative_m15",
    "m15_above_reject_threshold",
}
SUSPECT_FLAGS = {
    "duplicate_timestamp_conflict",
    "constant_positive_run",
    "m15_above_suspect_threshold",
}


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge a user configuration into the audited defaults."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None) -> dict:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    with path.open(encoding="utf-8") as handle:
        return merge_config(DEFAULT_CONFIG, json.load(handle))


def _append_flags(flags: list[list[str]], indexes: Iterable[int], flag: str) -> None:
    for index in indexes:
        if flag not in flags[index]:
            flags[index].append(flag)


def _mark_constant_runs(
    timestamps: pd.Series,
    values: pd.Series,
    flags: list[list[str]],
    config: dict,
) -> None:
    threshold = config["thresholds"]
    expected_delta = pd.Timedelta(minutes=threshold["expected_interval_minutes"])
    minimum_value = threshold["constant_value_min_mm"]
    minimum_run = threshold["constant_run_length"]
    valid_indexes = [
        index
        for index in timestamps.index
        if pd.notna(timestamps.loc[index])
        and pd.notna(values.loc[index])
        and values.loc[index] >= minimum_value
    ]
    run: list[int] = []
    previous_timestamp = None
    previous_value = None
    for index in valid_indexes:
        timestamp = timestamps.loc[index]
        value = values.loc[index]
        contiguous = previous_timestamp is not None and timestamp - previous_timestamp <= expected_delta
        same_value = previous_value is not None and np.isclose(value, previous_value, rtol=0, atol=1e-6)
        if contiguous and same_value:
            run.append(index)
        else:
            if len(run) >= minimum_run:
                _append_flags(flags, run, "constant_positive_run")
            run = [index]
        previous_timestamp = timestamp
        previous_value = value
    if len(run) >= minimum_run:
        _append_flags(flags, run, "constant_positive_run")


def audit_observations(frame: pd.DataFrame, station_id: int, year: int, config: dict) -> pd.DataFrame:
    """Return observations with deterministic QC flags and an aggregate status."""
    required = {"observation_datetime", "m15"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"WebSirene: colunas ausentes: {sorted(missing)}")

    audited = frame.copy().reset_index(drop=True)
    audited["station_id"] = station_id
    audited["year"] = year
    audited["observation_datetime"] = pd.to_datetime(
        audited["observation_datetime"], utc=True, errors="coerce"
    )
    audited["m15"] = pd.to_numeric(audited["m15"], errors="coerce")
    audited["timestamp_15min"] = audited["observation_datetime"].dt.floor(
        f"{config['temporal_resolution_minutes']}min"
    )
    flags: list[list[str]] = [[] for _ in range(len(audited))]

    _append_flags(flags, audited.index[audited["observation_datetime"].isna()], "invalid_timestamp")
    invalid_values = audited["m15"].isna() | ~np.isfinite(audited["m15"].fillna(0))
    _append_flags(flags, audited.index[invalid_values], "invalid_m15")
    _append_flags(flags, audited.index[audited["m15"].lt(0)], "negative_m15")

    thresholds = config["thresholds"]
    _append_flags(
        flags,
        audited.index[audited["m15"].gt(thresholds["suspect_m15_mm"])],
        "m15_above_suspect_threshold",
    )
    _append_flags(
        flags,
        audited.index[audited["m15"].gt(thresholds["reject_m15_mm"])],
        "m15_above_reject_threshold",
    )

    valid_time = audited["observation_datetime"].notna()
    for _, group in audited.loc[valid_time].groupby("observation_datetime", sort=False):
        if len(group) < 2:
            continue
        flag = "duplicate_timestamp" if group["m15"].nunique(dropna=False) == 1 else "duplicate_timestamp_conflict"
        _append_flags(flags, group.index, flag)

    ordered = audited.loc[valid_time].sort_values("observation_datetime")
    expected_gap = pd.Timedelta(
        minutes=thresholds["expected_interval_minutes"] * thresholds["gap_multiplier"]
    )
    gaps = ordered["observation_datetime"].diff().gt(expected_gap)
    _append_flags(flags, ordered.index[gaps.fillna(False)], "gap_before")
    _mark_constant_runs(
        ordered["observation_datetime"], ordered["m15"], flags, config
    )

    statuses = []
    for observation_flags in flags:
        flag_set = set(observation_flags)
        if flag_set & REJECTED_FLAGS:
            statuses.append("rejected")
        elif flag_set & SUSPECT_FLAGS:
            statuses.append("suspect")
        else:
            statuses.append("accepted")
    audited["qc_flags"] = [";".join(items) for items in flags]
    audited["qc_status"] = statuses
    audited["qc_version"] = config["version"]
    return audited


def station_summary(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aggregate audited observations and derive a preliminary station whitelist."""
    rows = []
    acceptance = config["station_acceptance"]
    for station_id, group in frame.groupby("station_id", sort=True):
        counts = group["qc_status"].value_counts()
        total = len(group)
        accepted = int(counts.get("accepted", 0))
        rejected = int(counts.get("rejected", 0))
        suspect = int(counts.get("suspect", 0))
        accepted_years = int(group.loc[group["qc_status"] == "accepted", "year"].nunique())
        reasons = []
        if accepted < acceptance["min_accepted_observations"]:
            reasons.append("few_accepted_observations")
        if accepted_years < acceptance["min_years_with_accepted_data"]:
            reasons.append("few_years_with_accepted_data")
        if total and rejected / total > acceptance["max_rejected_fraction"]:
            reasons.append("rejected_fraction_above_limit")
        if total and suspect / total > acceptance["max_suspect_fraction"]:
            reasons.append("suspect_fraction_above_limit")
        flag_counter = Counter(
            flag
            for flags in group["qc_flags"].fillna("")
            for flag in flags.split(";")
            if flag
        )
        names = group["nome"].dropna().astype(str) if "nome" in group else pd.Series(dtype=str)
        rows.append(
            {
                "station_id": station_id,
                "station_name": names.iloc[0] if not names.empty else None,
                "observations": total,
                "accepted_observations": accepted,
                "suspect_observations": suspect,
                "rejected_observations": rejected,
                "accepted_fraction": accepted / total if total else 0.0,
                "suspect_fraction": suspect / total if total else 0.0,
                "rejected_fraction": rejected / total if total else 0.0,
                "years_with_accepted_data": accepted_years,
                "first_timestamp": group["observation_datetime"].min(),
                "last_timestamp": group["observation_datetime"].max(),
                "max_m15": group["m15"].max(),
                "flag_counts": json.dumps(dict(sorted(flag_counter.items())), ensure_ascii=False),
                "station_status": "approved" if not reasons else "conditional",
                "station_status_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def write_qc_report(path: Path, summary: pd.DataFrame, config: dict) -> None:
    total = int(summary["observations"].sum()) if not summary.empty else 0
    accepted = int(summary["accepted_observations"].sum()) if not summary.empty else 0
    suspect = int(summary["suspect_observations"].sum()) if not summary.empty else 0
    rejected = int(summary["rejected_observations"].sum()) if not summary.empty else 0
    approved = int((summary["station_status"] == "approved").sum()) if not summary.empty else 0
    lines = [
        "# Auditoria WebSirene",
        "",
        f"Versao de QC: `{config['version']}`.",
        "",
        "| Medida | Valor |",
        "|---|---:|",
        f"| Estacoes auditadas | {len(summary)} |",
        f"| Estacoes aprovadas preliminarmente | {approved} |",
        f"| Observacoes | {total} |",
        f"| Aceitas | {accepted} |",
        f"| Suspeitas | {suspect} |",
        f"| Rejeitadas | {rejected} |",
        "",
        "A whitelist e preliminar: eventos marcados como suspeitos exigem revisao "
        "contra outras fontes antes de serem usados como supervisao.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def haversine_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    radius_km = 6371.0088
    lat_a, lon_a, lat_b, lon_b = map(math.radians, (latitude_a, longitude_a, latitude_b, longitude_b))
    return 2 * radius_km * math.asin(
        math.sqrt(
            math.sin((lat_b - lat_a) / 2) ** 2
            + math.cos(lat_a) * math.cos(lat_b) * math.sin((lon_b - lon_a) / 2) ** 2
        )
    )


def nearest_reference_stations(
    websirene_mapping: pd.DataFrame,
    reference_mapping: pd.DataFrame,
    max_distance_km: float,
) -> pd.DataFrame:
    """Match each WebSirene station to its nearest reference station in range."""
    required = {"station_id", "latitude", "longitude"}
    for name, mapping in (("WebSirene", websirene_mapping), ("referencia", reference_mapping)):
        missing = required - set(mapping.columns)
        if missing:
            raise ValueError(f"Mapeamento {name}: colunas ausentes: {sorted(missing)}")

    rows = []
    reference = reference_mapping.dropna(subset=["station_id", "latitude", "longitude"])
    for web in websirene_mapping.dropna(subset=["station_id", "latitude", "longitude"]).itertuples():
        distances = reference.apply(
            lambda item: haversine_km(web.latitude, web.longitude, item.latitude, item.longitude),
            axis=1,
        )
        if distances.empty:
            continue
        nearest_index = distances.idxmin()
        distance = float(distances.loc[nearest_index])
        if distance <= max_distance_km:
            reference_row = reference.loc[nearest_index]
            rows.append(
                {
                    "websirene_station_id": int(web.station_id),
                    "alertario_station_id": reference_row["station_id"],
                    "distance_km": distance,
                }
            )
    return pd.DataFrame(rows)


def paired_metrics(pairs: pd.DataFrame) -> dict[str, float | int | None]:
    """Compute transparent agreement metrics for timestamp-aligned station readings."""
    pairs = pairs.dropna(subset=["websirene_m15", "reference_m15"])
    if pairs.empty:
        return {"n": 0, "mae": None, "bias": None, "pearson_r": None}
    error = pairs["websirene_m15"] - pairs["reference_m15"]
    correlation = None
    if pairs["websirene_m15"].nunique() > 1 and pairs["reference_m15"].nunique() > 1:
        correlation = pairs["websirene_m15"].corr(pairs["reference_m15"])
    return {
        "n": int(len(pairs)),
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
        "pearson_r": float(correlation) if correlation is not None and pd.notna(correlation) else None,
    }
