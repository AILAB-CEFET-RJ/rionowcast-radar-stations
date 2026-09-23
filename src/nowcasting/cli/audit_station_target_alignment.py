#!/usr/bin/env python3
"""Audit the spatial agreement between station mapping and sparse targets."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.station_geometry import direct_resampled_pixels, sparse_target_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita os pixels AlertaRio do CSV contra targets esparsos anuais."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height-orig", type=int, default=656)
    parser.add_argument("--width-orig", type=int, default=654)
    return parser.parse_args()


def sparse_coordinate_counts(path: Path) -> Counter[tuple[int, int]]:
    with np.load(path) as sparse:
        rows = np.asarray(sparse["row"], dtype=np.int64)
        columns = np.asarray(sparse["column"], dtype=np.int64)
    return Counter(zip(rows.tolist(), columns.tolist()))


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("--year-end deve ser maior ou igual a --year-start.")
    mapping = pd.read_csv(args.mapping)
    required = {"station_id", "pixel_i", "pixel_j"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"{args.mapping}: colunas ausentes: {sorted(missing)}")
    mapping = mapping.dropna(subset=required).copy()
    if mapping["station_id"].duplicated().any():
        raise ValueError("O mapeamento possui station_id duplicado.")

    counts: Counter[tuple[int, int]] = Counter()
    source_shapes: set[tuple[int, int]] = set()
    target_shape: tuple[int, int] | None = None
    years_loaded = []
    for year in range(args.year_start, args.year_end + 1):
        metadata_path = args.dataset_root / f"year={year}" / "targets_alertario_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sparse_path = metadata_path.parent / metadata["sparse_file"]
        if not sparse_path.is_file():
            continue
        shape = tuple(metadata["shape"])
        current_target_shape = (int(shape[1]), int(shape[2]))
        if target_shape is not None and target_shape != current_target_shape:
            raise ValueError("Todos os anos precisam ter a mesma resolucao de target.")
        target_shape = current_target_shape
        source_shape = tuple(metadata.get("source_spatial_shape", current_target_shape))
        source_shapes.add((int(source_shape[0]), int(source_shape[1])))
        counts.update(sparse_coordinate_counts(sparse_path))
        years_loaded.append(year)
    if not years_loaded or target_shape is None:
        raise FileNotFoundError("Nenhum target AlertaRio esparso foi encontrado no intervalo informado.")
    if len(source_shapes) != 1:
        raise ValueError(f"source_spatial_shape inconsistente: {sorted(source_shapes)}")

    source_height, source_width = next(iter(source_shapes))
    direct_rows, direct_columns = direct_resampled_pixels(
        mapping["pixel_i"].to_numpy(float), mapping["pixel_j"].to_numpy(float),
        height_orig=args.height_orig, width_orig=args.width_orig,
        height=target_shape[0], width=target_shape[1],
    )
    pipeline_rows, pipeline_columns = sparse_target_pixels(
        mapping["pixel_i"].to_numpy(float), mapping["pixel_j"].to_numpy(float),
        height_orig=args.height_orig, width_orig=args.width_orig,
        source_height=source_height, source_width=source_width,
        height=target_shape[0], width=target_shape[1],
    )
    report = mapping[[column for column in ("station_id", "nome", "pixel_i", "pixel_j") if column in mapping]].copy()
    report["direct_row"] = direct_rows
    report["direct_column"] = direct_columns
    report["pipeline_row"] = pipeline_rows
    report["pipeline_column"] = pipeline_columns
    report["direct_observations"] = [counts[(int(row), int(column))] for row, column in zip(direct_rows, direct_columns)]
    report["pipeline_observations"] = [counts[(int(row), int(column))] for row, column in zip(pipeline_rows, pipeline_columns)]
    report["coordinates_match"] = (direct_rows == pipeline_rows) & (direct_columns == pipeline_columns)
    report["pipeline_coordinate_present"] = report["pipeline_observations"] > 0
    report["direct_coordinate_present"] = report["direct_observations"] > 0
    report["alignment_status"] = np.select(
        [
            report["coordinates_match"] & report["pipeline_coordinate_present"],
            ~report["coordinates_match"] & report["pipeline_coordinate_present"],
            ~report["pipeline_coordinate_present"],
        ],
        ["matching_coordinate_present", "requires_pipeline_transform", "missing_from_sparse_targets"],
        default="unknown",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "station_target_alignment.csv", index=False)
    summary = {
        "dataset_root": str(args.dataset_root),
        "years_loaded": years_loaded,
        "target_shape": list(target_shape),
        "source_spatial_shape": [source_height, source_width],
        "stations": int(len(report)),
        "coordinates_matching": int(report["coordinates_match"].sum()),
        "pipeline_coordinates_present": int(report["pipeline_coordinate_present"].sum()),
        "stations_requiring_pipeline_transform": int((~report["coordinates_match"] & report["pipeline_coordinate_present"]).sum()),
        "status_counts": report["alignment_status"].value_counts().to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Relatorio: {args.output_dir / 'station_target_alignment.csv'}", flush=True)


if __name__ == "__main__":
    main()
