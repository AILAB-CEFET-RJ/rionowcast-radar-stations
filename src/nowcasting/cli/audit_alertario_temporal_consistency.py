#!/usr/bin/env python3
"""Audit temporal consistency among AlertaRio rolling accumulations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ACCUMULATION_WINDOWS = {"m10": 2, "m15": 3, "h01": 12}
FLAG_COLUMNS = (
    "duplicate_exact_timestamp",
    "conflicting_duplicate_timestamp",
    "high_m05",
    "isolated_m05_spike",
    "high_m15",
    "isolated_m15_spike",
    "accumulation_hierarchy_violation",
    "m10_inconsistent",
    "m15_inconsistent",
    "h01_inconsistent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audita, sem excluir dados, a coerencia temporal entre m05, m10, "
            "m15 e h01 nas series originais do AlertaRio."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--sentinel", type=float, default=-99.99)
    parser.add_argument("--cadence-minutes", type=int, default=5)
    parser.add_argument("--consistency-tolerance-mm", type=float, default=0.11)
    parser.add_argument("--high-m05-mm", type=float, default=50.0)
    parser.add_argument("--high-m15-mm", type=float, default=50.0)
    parser.add_argument("--quiet-neighbor-mm", type=float, default=1.0)
    parser.add_argument("--max-example-rows", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _valid_accumulation(series: pd.Series, sentinel: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & np.isfinite(numeric) & numeric.ge(0) & numeric.ne(sentinel)


def _collapse_exact_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["station_id", "timestamp"]
    values = ["m05", "m10", "m15", "h01"]
    grouped = data.groupby(keys, sort=True, dropna=False)
    duplicate_count = grouped.size().rename("duplicate_count")
    conflicts = grouped[values].nunique(dropna=False).max(axis=1).gt(1).rename(
        "conflicting_duplicate_timestamp"
    )
    collapsed = grouped.agg(
        station_name=("station_name", "first"),
        m05=("m05", "max"),
        m10=("m10", "max"),
        m15=("m15", "max"),
        h01=("h01", "max"),
    ).join((duplicate_count, conflicts)).reset_index()
    collapsed["duplicate_exact_timestamp"] = collapsed["duplicate_count"].gt(1)
    return collapsed


def _audit_station(
    station: pd.DataFrame,
    *,
    sentinel: float,
    cadence_minutes: int,
    consistency_tolerance_mm: float,
    high_m05_mm: float,
    high_m15_mm: float,
    quiet_neighbor_mm: float,
) -> pd.DataFrame:
    result = station.sort_values("timestamp").reset_index(drop=True).copy()
    cadence = pd.Timedelta(minutes=cadence_minutes)
    safe = ~result["conflicting_duplicate_timestamp"]

    valid = {
        column: _valid_accumulation(result[column], sentinel)
        for column in ("m05", "m10", "m15", "h01")
    }
    previous_timestamp = result["timestamp"].shift(1)
    next_timestamp = result["timestamp"].shift(-1)
    previous_is_adjacent = result["timestamp"].sub(previous_timestamp).eq(cadence)
    next_is_adjacent = next_timestamp.sub(result["timestamp"]).eq(cadence)

    result["previous_m05"] = result["m05"].shift(1)
    result["next_m05"] = result["m05"].shift(-1)
    result["previous_m15"] = result["m15"].shift(1)
    result["next_m15"] = result["m15"].shift(-1)
    result["high_m05"] = valid["m05"] & result["m05"].ge(high_m05_mm)
    result["high_m15"] = valid["m15"] & result["m15"].ge(high_m15_mm)

    adjacent_safe = safe & safe.shift(1, fill_value=False) & safe.shift(-1, fill_value=False)
    adjacent_m05_valid = (
        valid["m05"]
        & valid["m05"].shift(1, fill_value=False)
        & valid["m05"].shift(-1, fill_value=False)
    )
    adjacent_m15_valid = (
        valid["m15"]
        & valid["m15"].shift(1, fill_value=False)
        & valid["m15"].shift(-1, fill_value=False)
    )
    result["isolated_m05_spike"] = (
        result["high_m05"]
        & adjacent_safe
        & adjacent_m05_valid
        & previous_is_adjacent
        & next_is_adjacent
        & result["previous_m05"].le(quiet_neighbor_mm)
        & result["next_m05"].le(quiet_neighbor_mm)
    )
    result["isolated_m15_spike"] = (
        result["high_m15"]
        & adjacent_safe
        & adjacent_m15_valid
        & previous_is_adjacent
        & next_is_adjacent
        & result["previous_m15"].le(quiet_neighbor_mm)
        & result["next_m15"].le(quiet_neighbor_mm)
    )

    hierarchy_comparable = safe.copy()
    for column in ("m05", "m10", "m15", "h01"):
        hierarchy_comparable &= valid[column]
    result["accumulation_hierarchy_comparable"] = hierarchy_comparable
    result["accumulation_hierarchy_violation"] = hierarchy_comparable & (
        result["m10"].add(consistency_tolerance_mm).lt(result["m05"])
        | result["m15"].add(consistency_tolerance_mm).lt(result["m10"])
        | result["h01"].add(consistency_tolerance_mm).lt(result["m15"])
    )

    for column, window in ACCUMULATION_WINDOWS.items():
        shifted_values = [result["m05"].shift(step) for step in range(window)]
        expected = pd.concat(shifted_values, axis=1).sum(axis=1, min_count=window)
        comparable = safe & valid[column]
        for step in range(window):
            comparable &= safe.shift(step, fill_value=False)
            comparable &= valid["m05"].shift(step, fill_value=False)
        for step in range(window - 1):
            newer = result["timestamp"].shift(step)
            older = result["timestamp"].shift(step + 1)
            comparable &= newer.sub(older).eq(cadence)
        error = result[column].sub(expected)
        result[f"expected_{column}_from_m05"] = expected.where(comparable)
        result[f"{column}_consistency_error_mm"] = error.where(comparable)
        result[f"{column}_comparable"] = comparable
        result[f"{column}_inconsistent"] = (
            comparable & error.abs().gt(consistency_tolerance_mm)
        )

    result["any_temporal_flag"] = result[list(FLAG_COLUMNS)].any(axis=1)
    error_columns = [f"{column}_consistency_error_mm" for column in ACCUMULATION_WINDOWS]
    result["max_consistency_abs_error_mm"] = result[error_columns].abs().max(axis=1)
    return result


def audit_temporal_consistency(
    frame: pd.DataFrame,
    *,
    sentinel: float = -99.99,
    cadence_minutes: int = 5,
    consistency_tolerance_mm: float = 0.11,
    high_m05_mm: float = 50.0,
    high_m15_mm: float = 50.0,
    quiet_neighbor_mm: float = 1.0,
) -> pd.DataFrame:
    """Return one diagnostic row per station/timestamp without filtering observations."""
    required = {"estacao_id", "dia_utc", "m05", "m10", "m15", "h01"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes deve ser positivo.")
    if consistency_tolerance_mm < 0 or min(high_m05_mm, high_m15_mm, quiet_neighbor_mm) < 0:
        raise ValueError("Limiares da auditoria não podem ser negativos.")

    data = frame.rename(columns={"estacao_id": "station_id", "dia_utc": "timestamp"}).copy()
    data["station_name"] = data.get("estacao", "")
    data["station_id"] = pd.to_numeric(data["station_id"], errors="coerce")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for column in ("m05", "m10", "m15", "h01"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["station_id", "timestamp"]).copy()
    data["station_id"] = data["station_id"].astype(int)
    collapsed = _collapse_exact_timestamps(data)
    audited = [
        _audit_station(
            station,
            sentinel=sentinel,
            cadence_minutes=cadence_minutes,
            consistency_tolerance_mm=consistency_tolerance_mm,
            high_m05_mm=high_m05_mm,
            high_m15_mm=high_m15_mm,
            quiet_neighbor_mm=quiet_neighbor_mm,
        )
        for _, station in collapsed.groupby("station_id", sort=True)
    ]
    return pd.concat(audited, ignore_index=True) if audited else collapsed


def summarize_station_year(audited: pd.DataFrame) -> pd.DataFrame:
    data = audited.copy()
    data["year"] = data["timestamp"].dt.year
    aggregations: dict[str, tuple[str, str]] = {
        "rows": ("timestamp", "size"),
        "first_timestamp": ("timestamp", "min"),
        "last_timestamp": ("timestamp", "max"),
        "max_m05": ("m05", "max"),
        "max_m15": ("m15", "max"),
        "any_temporal_flag": ("any_temporal_flag", "sum"),
    }
    for flag in FLAG_COLUMNS:
        aggregations[flag] = (flag, "sum")
    for column in ACCUMULATION_WINDOWS:
        aggregations[f"{column}_comparable"] = (f"{column}_comparable", "sum")
    return data.groupby(["station_id", "station_name", "year"], as_index=False).agg(**aggregations)


def summarize_flagged_episodes(
    audited: pd.DataFrame, *, cadence_minutes: int = 5,
) -> pd.DataFrame:
    """Group consecutive flagged timestamps from the same station into episodes."""
    output_columns = [
        "station_id",
        "station_name",
        "episode_number",
        "start_timestamp",
        "end_timestamp",
        "span_minutes",
        "flagged_timestamps",
        "max_m05",
        "max_m15",
        "max_consistency_abs_error_mm",
        *[f"{flag}_count" for flag in FLAG_COLUMNS],
    ]
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes deve ser positivo.")

    flagged = audited.loc[audited["any_temporal_flag"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=output_columns)

    flagged = flagged.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    cadence = pd.Timedelta(minutes=cadence_minutes)
    time_difference = flagged.groupby("station_id")["timestamp"].diff()
    episode_start = time_difference.isna() | time_difference.gt(cadence)
    flagged["episode_number"] = (
        episode_start.groupby(flagged["station_id"]).cumsum().astype(int)
    )

    aggregations: dict[str, tuple[str, str]] = {
        "start_timestamp": ("timestamp", "min"),
        "end_timestamp": ("timestamp", "max"),
        "flagged_timestamps": ("timestamp", "size"),
        "max_m05": ("m05", "max"),
        "max_m15": ("m15", "max"),
        "max_consistency_abs_error_mm": ("max_consistency_abs_error_mm", "max"),
    }
    for flag in FLAG_COLUMNS:
        aggregations[f"{flag}_count"] = (flag, "sum")
    episodes = flagged.groupby(
        ["station_id", "station_name", "episode_number"], as_index=False,
    ).agg(**aggregations)
    episodes["span_minutes"] = (
        episodes["end_timestamp"].sub(episodes["start_timestamp"]).dt.total_seconds() / 60
    ).astype(int)
    return episodes[output_columns]


def load_year_range(
    files: list[Path], year_start: int, year_end: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    start = pd.Timestamp(year=year_start, month=1, day=1, tz="UTC")
    stop = pd.Timestamp(year=year_end + 1, month=1, day=1, tz="UTC")
    selected_frames = []
    rows_scanned = 0
    invalid_timestamps = 0
    columns = ["estacao_id", "estacao", "dia_utc", "m05", "m10", "m15", "h01"]
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=columns)
        rows_scanned += len(frame)
        frame["dia_utc"] = pd.to_datetime(frame["dia_utc"], utc=True, errors="coerce")
        invalid_timestamps += int(frame["dia_utc"].isna().sum())
        selected = frame.loc[frame["dia_utc"].ge(start) & frame["dia_utc"].lt(stop)]
        if not selected.empty:
            selected_frames.append(selected)
        if index == 1 or index % 25 == 0 or index == len(files):
            print(f"files {index}/{len(files)}", flush=True)
    if not selected_frames:
        raise ValueError(f"Nenhuma observacao encontrada entre {year_start} e {year_end}.")
    return pd.concat(selected_frames, ignore_index=True), {
        "files": len(files),
        "rows_scanned": rows_scanned,
        "invalid_timestamps": invalid_timestamps,
        "rows_in_scope": sum(len(frame) for frame in selected_frames),
    }


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("--year-end deve ser maior ou igual a --year-start.")
    if args.max_example_rows < 0:
        raise ValueError("--max-example-rows não pode ser negativo.")
    files = sorted(args.input_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {args.input_root}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} não está vazio; use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source, source_stats = load_year_range(files, args.year_start, args.year_end)
    audited = audit_temporal_consistency(
        source,
        sentinel=args.sentinel,
        cadence_minutes=args.cadence_minutes,
        consistency_tolerance_mm=args.consistency_tolerance_mm,
        high_m05_mm=args.high_m05_mm,
        high_m15_mm=args.high_m15_mm,
        quiet_neighbor_mm=args.quiet_neighbor_mm,
    )
    station_year = summarize_station_year(audited)
    flagged = audited.loc[audited["any_temporal_flag"]].copy()
    episodes = summarize_flagged_episodes(
        audited, cadence_minutes=args.cadence_minutes,
    )
    flagged.to_parquet(args.output_dir / "flagged_observations.parquet", index=False)
    station_year.to_csv(args.output_dir / "station_year_summary.csv", index=False)
    episodes.to_csv(args.output_dir / "flagged_episodes.csv", index=False)

    examples = flagged.sort_values(
        ["max_consistency_abs_error_mm", "m05", "m15"],
        ascending=[False, False, False], na_position="last",
    ).head(args.max_example_rows)
    examples.to_csv(args.output_dir / "flagged_examples.csv", index=False)

    totals = {flag: int(audited[flag].sum()) for flag in FLAG_COLUMNS}
    totals.update({
        f"{column}_comparable": int(audited[f"{column}_comparable"].sum())
        for column in ACCUMULATION_WINDOWS
    })
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "input_root": str(args.input_root),
        "year_start": args.year_start,
        "year_end": args.year_end,
        "source": source_stats,
        "stations": int(audited["station_id"].nunique()),
        "audited_station_timestamps": int(len(audited)),
        "flagged_station_timestamps": int(len(flagged)),
        "flagged_episodes": int(len(episodes)),
        "parameters": {
            "sentinel": args.sentinel,
            "cadence_minutes": args.cadence_minutes,
            "consistency_tolerance_mm": args.consistency_tolerance_mm,
            "high_m05_mm": args.high_m05_mm,
            "high_m15_mm": args.high_m15_mm,
            "quiet_neighbor_mm": args.quiet_neighbor_mm,
        },
        "assumptions_under_audit": {
            "m10": "sum of current and previous m05 at five-minute cadence",
            "m15": "sum of current and two previous m05 values at five-minute cadence",
            "h01": "sum of current and eleven previous m05 values at five-minute cadence",
        },
        "totals": totals,
        "outputs": {
            "flagged_observations": "flagged_observations.parquet",
            "flagged_examples": "flagged_examples.csv",
            "flagged_episodes": "flagged_episodes.csv",
            "station_year_summary": "station_year_summary.csv",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Auditoria temporal concluida | linhas={len(audited)} | flags={len(flagged)} | "
        f"saida={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
