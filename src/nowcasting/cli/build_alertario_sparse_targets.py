#!/usr/bin/env python3
"""Build provenance-preserving sparse AlertaRio targets from raw Parquets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.station_geometry import direct_resampled_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera targets AlertaRio esparsos com station_id a partir dos Parquets originais."
    )
    parser.add_argument("--alertario-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True,
                        help="Raiz que contem radar_timestamps.npy por ano.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Raiz de dataset nova, ja contendo os diretorios year=<ano>.")
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height-orig", type=int, default=656)
    parser.add_argument("--width-orig", type=int, default=654)
    parser.add_argument(
        "--observation-to-radar-offset-minutes", type=int, default=0,
        help="Offset para converter o timestamp da observação ao timestamp inicial da janela de radar.",
    )
    parser.add_argument(
        "--max-m15-mm", type=float,
        help=(
            "Limite superior auditável para m15. Se qualquer leitura ultrapassar "
            "o limite, a observação da estação na janela de 15 minutos é excluída "
            "como ausente; o valor não é truncado nem substituído por zero."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, np.generic):
        value = value.item()
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def load_mapping(args: argparse.Namespace) -> pd.DataFrame:
    mapping = pd.read_csv(args.mapping)
    required = {"station_id", "pixel_i", "pixel_j"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"{args.mapping}: colunas ausentes: {sorted(missing)}")
    mapping = mapping.dropna(subset=required).copy()
    mapping["station_id"] = mapping["station_id"].astype(int)
    if mapping["station_id"].duplicated().any():
        raise ValueError("O mapeamento possui station_id duplicado.")
    rows, columns = direct_resampled_pixels(
        mapping["pixel_i"].to_numpy(float), mapping["pixel_j"].to_numpy(float),
        height_orig=args.height_orig, width_orig=args.width_orig, height=args.height, width=args.width,
    )
    mapping["row"] = rows
    mapping["column"] = columns
    if mapping.duplicated(["row", "column"]).any():
        raise ValueError("Duas estações ocupam o mesmo pixel na resolução de destino.")
    return mapping[["station_id", "row", "column"]]


def load_observations_in_interval(
    parquet_files: list[Path], start: pd.Timestamp, stop: pd.Timestamp, mapping: pd.DataFrame,
    max_m15_mm: float | None = None,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    frames = []
    above_max_groups = []
    stats = {"rows_scanned": 0, "invalid_datetime": 0, "missing_m15": 0, "sentinel_m15": 0,
             "negative_m15": 0, "above_max_m15": 0,
             "station_windows_flagged_above_max_m15": 0,
             "station_windows_with_valid_values_discarded_above_max_m15": 0,
             "station_windows_excluded_above_max_m15": 0, "unmapped_station": 0}
    for path in parquet_files:
        frame = pd.read_parquet(path, columns=["estacao_id", "dia_utc", "m15"])
        frame = frame.rename(columns={"estacao_id": "station_id", "dia_utc": "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        invalid_datetime = frame["timestamp"].isna()
        stats["invalid_datetime"] += int(invalid_datetime.sum())
        frame = frame.loc[~invalid_datetime & frame["timestamp"].ge(start) & frame["timestamp"].lt(stop)].copy()
        if frame.empty:
            continue
        stats["rows_scanned"] += len(frame)
        frame["m15"] = pd.to_numeric(frame["m15"], errors="coerce")
        stats["missing_m15"] += int(frame["m15"].isna().sum())
        stats["sentinel_m15"] += int((frame["m15"] == -99.99).sum())
        stats["negative_m15"] += int((frame["m15"] < 0).sum())
        frame = frame.loc[frame["m15"].notna() & frame["m15"].ge(0)].copy()
        frame["timestamp"] = frame["timestamp"].dt.floor("15min")
        if max_m15_mm is not None:
            above_max = frame["m15"].gt(max_m15_mm)
            stats["above_max_m15"] += int(above_max.sum())
            if above_max.any():
                above_max_groups.append(frame.loc[above_max, ["station_id", "timestamp"]])
                frame = frame.loc[~above_max].copy()
        if not frame.empty:
            frames.append(frame)
    if above_max_groups:
        flagged_groups = pd.concat(above_max_groups, ignore_index=True).drop_duplicates()
        stats["station_windows_flagged_above_max_m15"] = len(flagged_groups)
    else:
        flagged_groups = pd.DataFrame(columns=["station_id", "timestamp"])
    if not frames:
        empty = pd.DataFrame(columns=["station_id", "timestamp", "m15", "row", "column"])
        return empty, stats, flagged_groups
    observations = pd.concat(frames, ignore_index=True)
    observations["station_id"] = observations["station_id"].astype(int)
    observations = observations.groupby(["station_id", "timestamp"], as_index=False)["m15"].max()
    if not flagged_groups.empty:
        flagged_groups = flagged_groups.assign(_above_max_m15=True)
        observations = observations.merge(
            flagged_groups, on=["station_id", "timestamp"], how="left", validate="one_to_one",
        )
        excluded = observations["_above_max_m15"].eq(True)
        stats["station_windows_with_valid_values_discarded_above_max_m15"] = int(excluded.sum())
        observations = observations.loc[~excluded].drop(columns="_above_max_m15")
    before_mapping = len(observations)
    observations = observations.merge(mapping, on="station_id", how="inner", validate="many_to_one")
    stats["unmapped_station"] = before_mapping - len(observations)
    return observations, stats, flagged_groups


def process_year(args: argparse.Namespace, year: int, mapping: pd.DataFrame, parquet_files: list[Path]) -> None:
    radar_dir = args.radar_root / f"year={year}"
    output_dir = args.output_root / f"year={year}"
    timestamps_path = radar_dir / "radar_timestamps.npy"
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"{timestamps_path} ausente.")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"{output_dir} ausente; prepare a nova raiz de dataset antes de gerar targets.")
    sparse_path = output_dir / "targets_alertario_sparse.npz"
    metadata_path = output_dir / "targets_alertario_metadata.json"
    if (sparse_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} ja possui targets; use uma nova raiz ou --overwrite.")

    timestamps = [normalize_timestamp(value) for value in np.load(timestamps_path, allow_pickle=False)]
    timestamp_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    if not timestamps:
        raise ValueError(f"{timestamps_path} não contém timestamps.")
    offset = pd.Timedelta(minutes=args.observation_to_radar_offset_minutes)
    observation_start = min(timestamps) - offset
    observation_stop = max(timestamps) - offset + pd.Timedelta(minutes=15)
    observations, stats, flagged_groups = load_observations_in_interval(
        parquet_files, observation_start, observation_stop, mapping, args.max_m15_mm,
    )
    if not flagged_groups.empty:
        mapped_station_ids = set(mapping["station_id"].astype(int))
        flagged_groups = flagged_groups[
            flagged_groups["station_id"].astype(int).isin(mapped_station_ids)
        ].copy()
        flagged_groups["radar_timestamp"] = flagged_groups["timestamp"] + offset
        stats["station_windows_excluded_above_max_m15"] = int(
            flagged_groups["radar_timestamp"].isin(timestamp_index).sum()
        )
    observations["radar_timestamp"] = observations["timestamp"] + offset
    observations["frame"] = observations["radar_timestamp"].map(timestamp_index)
    missing_timestamps = int(observations["frame"].isna().sum())
    observations = observations.dropna(subset=["frame"]).copy()
    observations["frame"] = observations["frame"].astype(np.int32)
    observations = observations.sort_values(["frame", "station_id"])
    observed_frames = np.sort(observations["frame"].unique())
    first_observed_frame = int(observed_frames[0]) if len(observed_frames) else None
    last_observed_frame = int(observed_frames[-1]) if len(observed_frames) else None
    leading_unobserved = first_observed_frame if first_observed_frame is not None else len(timestamps)
    trailing_unobserved = (
        len(timestamps) - last_observed_frame - 1
        if last_observed_frame is not None else len(timestamps)
    )
    interior_unobserved = (
        last_observed_frame - first_observed_frame + 1 - len(observed_frames)
        if first_observed_frame is not None else 0
    )

    np.savez_compressed(
        sparse_path,
        frame=observations["frame"].to_numpy(np.int32),
        row=observations["row"].to_numpy(np.uint16),
        column=observations["column"].to_numpy(np.uint16),
        station_id=observations["station_id"].to_numpy(np.int32),
        value=np.log1p(observations["m15"].to_numpy(np.float32)).astype(np.float32),
    )
    mapping_sha256 = hashlib.sha256(args.mapping.read_bytes()).hexdigest()
    metadata = {
        "source": "alertario",
        "year": year,
        "target": "m15",
        "target_unit": "log1p(mm/15min)",
        "target_transform": "log1p",
        "original_target_unit": "mm/15min",
        "height": args.height,
        "width": args.width,
        "channels": 1,
        "temporal_resolution_minutes": 15,
        "observation_timestamp_to_radar_offset_minutes": args.observation_to_radar_offset_minutes,
        "observation_timestamp_convention": "configured_by_cli",
        "shape": [len(timestamps), args.height, args.width, 1],
        "format": "sparse",
        "sparse_file": sparse_path.name,
        "sparse_fields": {"frame": "int32", "row": "uint16", "column": "uint16", "station_id": "int32", "value": "float32"},
        "observation_count": int(len(observations)),
        "observation_coverage": {
            "radar_frame_count": len(timestamps),
            "observed_frame_count": int(len(observed_frames)),
            "first_observed_frame": first_observed_frame,
            "last_observed_frame": last_observed_frame,
            "first_observed_radar_timestamp": (
                timestamps[first_observed_frame].isoformat()
                if first_observed_frame is not None else None
            ),
            "last_observed_radar_timestamp": (
                timestamps[last_observed_frame].isoformat()
                if last_observed_frame is not None else None
            ),
            "leading_radar_frames_without_observation": leading_unobserved,
            "interior_radar_frames_without_observation": int(interior_unobserved),
            "trailing_radar_frames_without_observation": trailing_unobserved,
        },
        "sample_eligibility": {
            "require_observation_in_each_output_frame": True,
        },
        "station_provenance": True,
        "mapping_file": str(args.mapping),
        "mapping_sha256": mapping_sha256,
        "quality_control": (
            "m15 missing, sentinel -99.99, and negative values excluded; "
            "when configured, station windows containing m15 above the upper bound are excluded as missing"
        ),
        "quality_control_parameters": {
            "max_m15_mm": args.max_m15_mm,
            "above_max_action": (
                "exclude_station_window_as_missing" if args.max_m15_mm is not None else "disabled"
            ),
        },
        "quality_control_counts": {**stats, "timestamp_absent": missing_timestamps},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[{year}] observations={len(observations)} output={output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("--year-end deve ser maior ou igual a --year-start.")
    if args.max_m15_mm is not None and (not np.isfinite(args.max_m15_mm) or args.max_m15_mm <= 0):
        raise ValueError("--max-m15-mm deve ser finito e maior que zero.")
    parquet_files = sorted(args.alertario_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {args.alertario_root}")
    mapping = load_mapping(args)
    for year in range(args.year_start, args.year_end + 1):
        process_year(args, year, mapping, parquet_files)


if __name__ == "__main__":
    main()
