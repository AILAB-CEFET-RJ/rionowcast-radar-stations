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


def load_year_observations(parquet_files: list[Path], year: int, mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    stop = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    frames = []
    stats = {"rows_scanned": 0, "invalid_datetime": 0, "missing_m15": 0, "sentinel_m15": 0,
             "negative_m15": 0, "unmapped_station": 0}
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
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["station_id", "timestamp", "m15", "row", "column"]), stats
    observations = pd.concat(frames, ignore_index=True)
    observations["station_id"] = observations["station_id"].astype(int)
    observations["timestamp"] = observations["timestamp"].dt.floor("15min")
    observations = observations.groupby(["station_id", "timestamp"], as_index=False)["m15"].max()
    before_mapping = len(observations)
    observations = observations.merge(mapping, on="station_id", how="inner", validate="many_to_one")
    stats["unmapped_station"] = before_mapping - len(observations)
    return observations, stats


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

    timestamps = [normalize_timestamp(value) for value in np.load(timestamps_path, allow_pickle=True)]
    timestamp_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    observations, stats = load_year_observations(parquet_files, year, mapping)
    observations["frame"] = observations["timestamp"].map(timestamp_index)
    missing_timestamps = int(observations["frame"].isna().sum())
    observations = observations.dropna(subset=["frame"]).copy()
    observations["frame"] = observations["frame"].astype(np.int32)
    observations = observations.sort_values(["frame", "station_id"])

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
        "shape": [len(timestamps), args.height, args.width, 1],
        "format": "sparse",
        "sparse_file": sparse_path.name,
        "sparse_fields": {"frame": "int32", "row": "uint16", "column": "uint16", "station_id": "int32", "value": "float32"},
        "observation_count": int(len(observations)),
        "station_provenance": True,
        "mapping_file": str(args.mapping),
        "mapping_sha256": mapping_sha256,
        "quality_control": "m15 missing, sentinel -99.99, and negative values excluded",
        "quality_control_counts": {**stats, "timestamp_absent": missing_timestamps},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[{year}] observations={len(observations)} output={output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("--year-end deve ser maior ou igual a --year-start.")
    parquet_files = sorted(args.alertario_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {args.alertario_root}")
    mapping = load_mapping(args)
    for year in range(args.year_start, args.year_end + 1):
        process_year(args, year, mapping, parquet_files)


if __name__ == "__main__":
    main()
