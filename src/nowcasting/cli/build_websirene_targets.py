#!/usr/bin/env python3
"""Build sparse, QC-approved WebSirene targets on the radar grid."""
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
        description="Gera targets WebSirene esparsos, auditados e alinhados ao radar."
    )
    parser.add_argument("--qc-root", type=Path, required=True,
                        help="Raiz produzida por nowcasting-audit-websirene.")
    parser.add_argument("--station-whitelist", type=Path, required=True,
                        help="CSV de estações aprovadas na auditoria.")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True,
                        help="Dataset auditável contendo year=<ano>/radar_timestamps.npy.")
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height-orig", type=int, default=656)
    parser.add_argument("--width-orig", type=int, default=654)
    parser.add_argument(
        "--observation-to-radar-offset-minutes", type=int, required=True,
        help=("Offset que converte timestamp_15min WebSirene ao início UTC da janela de radar. "
              "Deve ser escolhido pela auditoria temporal entre redes."),
    )
    parser.add_argument("--cell-aggregation", choices=("median", "mean", "max"), default="median",
                        help="Agregação para múltiplas estações na mesma célula/instante.")
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
    mapping["station_id"] = pd.to_numeric(mapping["station_id"], errors="raise").astype(int)
    if mapping["station_id"].duplicated().any():
        raise ValueError(f"{args.mapping}: station_id duplicado.")
    rows, columns = direct_resampled_pixels(
        mapping["pixel_i"].to_numpy(float), mapping["pixel_j"].to_numpy(float),
        height_orig=args.height_orig, width_orig=args.width_orig,
        height=args.height, width=args.width,
    )
    mapping["row"] = rows
    mapping["column"] = columns
    return mapping[["station_id", "row", "column"]]


def load_observations(
    qc_root: Path, station_ids: set[int], year: int, mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frames = []
    stats = {"files_found": 0, "rows_read": 0, "rows_accepted": 0, "stations_without_mapping": 0}
    for station_id in sorted(station_ids):
        path = qc_root / f"station_id={station_id}" / f"year={year}" / "data.parquet"
        if not path.is_file():
            continue
        stats["files_found"] += 1
        frame = pd.read_parquet(path, columns=["station_id", "timestamp_15min", "m15", "qc_status"])
        stats["rows_read"] += len(frame)
        frame = frame.loc[frame["qc_status"] == "accepted", ["station_id", "timestamp_15min", "m15"]].copy()
        stats["rows_accepted"] += len(frame)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["station_id", "timestamp", "m15", "row", "column"]), stats

    observations = pd.concat(frames, ignore_index=True)
    observations["station_id"] = pd.to_numeric(observations["station_id"], errors="raise").astype(int)
    observations["timestamp"] = pd.to_datetime(observations.pop("timestamp_15min"), utc=True, errors="coerce")
    observations["m15"] = pd.to_numeric(observations["m15"], errors="coerce")
    observations = observations.dropna(subset=["timestamp", "m15"])
    # A série bruta tem leituras de cinco minutos; para uma mesma estação/janela,
    # preservamos o maior acumulado m15 antes de agregar espacialmente.
    observations = observations.groupby(["station_id", "timestamp"], as_index=False)["m15"].max()
    before_mapping = len(observations)
    observations = observations.merge(mapping, on="station_id", how="inner", validate="many_to_one")
    stats["stations_without_mapping"] = before_mapping - len(observations)
    return observations, stats


def aggregate_cells(observations: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(columns=["frame", "row", "column", "station_id", "station_count", "m15"])
    grouped = observations.groupby(["frame", "row", "column"], as_index=False)
    values = grouped["m15"].agg(aggregation).rename(columns={"m15": "m15"})
    provenance = grouped["station_id"].agg(
        station_count="nunique",
        station_id=lambda ids: int(ids.iloc[0]) if ids.nunique() == 1 else -1,
    ).reset_index()
    return values.merge(provenance, on=["frame", "row", "column"], validate="one_to_one")


def process_year(args: argparse.Namespace, year: int, mapping: pd.DataFrame, station_ids: set[int]) -> None:
    year_dir = args.radar_root / f"year={year}"
    timestamps_path = year_dir / "radar_timestamps.npy"
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"{timestamps_path} ausente.")
    sparse_path = year_dir / "targets_websirene_sparse.npz"
    metadata_path = year_dir / "targets_websirene_metadata.json"
    if (sparse_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{year_dir} já possui targets WebSirene; use --overwrite.")

    timestamps = [normalize_timestamp(value) for value in np.load(timestamps_path, allow_pickle=False)]
    timestamp_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    observations, stats = load_observations(args.qc_root, station_ids, year, mapping)
    offset = pd.Timedelta(minutes=args.observation_to_radar_offset_minutes)
    observations["radar_timestamp"] = observations["timestamp"] + offset
    observations["frame"] = observations["radar_timestamp"].map(timestamp_index)
    stats["timestamp_absent"] = int(observations["frame"].isna().sum())
    observations = observations.dropna(subset=["frame"]).copy()
    observations["frame"] = observations["frame"].astype(np.int32)
    cells = aggregate_cells(observations, args.cell_aggregation).sort_values(["frame", "row", "column"])

    np.savez_compressed(
        sparse_path,
        frame=cells["frame"].to_numpy(np.int32),
        row=cells["row"].to_numpy(np.uint16),
        column=cells["column"].to_numpy(np.uint16),
        station_id=cells["station_id"].to_numpy(np.int32),
        station_count=cells["station_count"].to_numpy(np.uint16),
        value=np.log1p(cells["m15"].to_numpy(np.float32)).astype(np.float32),
    )
    metadata = {
        "source": "websirene", "year": year, "target": "m15",
        "target_unit": "log1p(mm/15min)", "target_transform": "log1p",
        "original_target_unit": "mm/15min", "height": args.height, "width": args.width,
        "channels": 1, "temporal_resolution_minutes": 15,
        "observation_timestamp_to_radar_offset_minutes": args.observation_to_radar_offset_minutes,
        "shape": [len(timestamps), args.height, args.width, 1], "format": "sparse",
        "sparse_file": sparse_path.name,
        "sparse_fields": {"frame": "int32", "row": "uint16", "column": "uint16",
                          "station_id": "int32 (-1 when a cell aggregates stations)",
                          "station_count": "uint16", "value": "float32"},
        "observation_count": int(len(cells)), "source_station_observation_count": int(len(observations)),
        "station_provenance": True, "mapping_file": str(args.mapping),
        "mapping_sha256": hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
        "qc_root": str(args.qc_root), "station_whitelist": str(args.station_whitelist),
        "cell_aggregation": args.cell_aggregation,
        "quality_control": "only qc_status=accepted from whitelisted stations", "quality_control_counts": stats,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{year}] source_observations={len(observations)} cells={len(cells)} output={year_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("--year-end deve ser maior ou igual a --year-start.")
    if not args.station_whitelist.is_file():
        raise FileNotFoundError(f"Whitelist ausente: {args.station_whitelist}")
    whitelist = pd.read_csv(args.station_whitelist)
    if "station_id" not in whitelist:
        raise ValueError("--station-whitelist requer a coluna station_id.")
    station_ids = set(pd.to_numeric(whitelist["station_id"], errors="raise").astype(int))
    mapping = load_mapping(args)
    missing_mapping = station_ids - set(mapping["station_id"])
    if missing_mapping:
        raise ValueError(f"Whitelist contém estações ausentes no mapeamento: {sorted(missing_mapping)}")
    for year in range(args.year_start, args.year_end + 1):
        process_year(args, year, mapping, station_ids)


if __name__ == "__main__":
    main()
