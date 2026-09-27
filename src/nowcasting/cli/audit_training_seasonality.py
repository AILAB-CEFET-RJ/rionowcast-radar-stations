"""Audit seasonal class balance using the exact nowcasting training windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.dataset import RadarStationMemmapDataset, parse_years


CLASS_NAMES = ("weak_or_dry", "moderate", "strong", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita a sazonalidade das classes de chuva nas janelas reais de treinamento."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--years", required=True, help="Ex.: 2012-2021")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-source", choices=("alertario", "websirene"), default="alertario")
    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--t-out", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--thresholds", default="1.25,6.25,12.5")
    parser.add_argument("--crop-stations", action="store_true")
    parser.add_argument("--crop-margin-pixels", type=int, default=20)
    parser.add_argument("--station-mapping", type=Path)
    parser.add_argument("--mapping-height-orig", type=int, default=656)
    parser.add_argument("--mapping-width-orig", type=int, default=654)
    return parser.parse_args()


def parse_thresholds(value: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != 3 or tuple(sorted(values)) != values:
        raise ValueError("--thresholds deve conter três valores crescentes, por exemplo 1.25,6.25,12.5.")
    return values


def sample_forecast_timestamps(dataset: RadarStationMemmapDataset) -> list[pd.Timestamp]:
    by_year: dict[int, np.ndarray] = {}
    for year in dataset.years:
        metadata_path = dataset.radar_root / f"year={year}" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        timestamps = np.load(
            dataset.radar_root / f"year={year}" / metadata.get("timestamps_file", "radar_timestamps.npy"),
            allow_pickle=False,
        )
        by_year[year] = pd.to_datetime(timestamps, utc=True)
    return [pd.Timestamp(by_year[year][start + dataset.t_in]) for year, start in dataset.samples]


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    thresholds = parse_thresholds(args.thresholds)
    if args.crop_stations and args.station_mapping is None:
        raise ValueError("--crop-stations requer --station-mapping.")
    dataset = RadarStationMemmapDataset(
        args.dataset_root, years, t_in=args.t_in, t_out=args.t_out, stride=args.stride,
        target_source=args.target_source, split_name="seasonality-audit",
        crop_stations=args.crop_stations, crop_margin_pixels=args.crop_margin_pixels,
        station_mapping=args.station_mapping, mapping_height_orig=args.mapping_height_orig,
        mapping_width_orig=args.mapping_width_orig,
    )
    timestamps = sample_forecast_timestamps(dataset)
    classes = dataset.get_sample_classes(thresholds)
    records = pd.DataFrame({
        "forecast_timestamp_utc": timestamps,
        "class_index": classes,
    })
    records["year"] = records["forecast_timestamp_utc"].dt.year
    records["month"] = records["forecast_timestamp_utc"].dt.month
    records["class_name"] = [CLASS_NAMES[index] for index in classes]
    counts = (
        records.groupby(["year", "month", "class_index", "class_name"], as_index=False)
        .size().rename(columns={"size": "windows"})
    )
    month_totals = counts.groupby(["year", "month"], as_index=False)["windows"].sum().rename(columns={"windows": "month_windows"})
    counts = counts.merge(month_totals, on=["year", "month"], validate="many_to_one")
    counts["month_class_fraction"] = counts["windows"] / counts["month_windows"]
    overall = (
        records.groupby(["month", "class_index", "class_name"], as_index=False)
        .size().rename(columns={"size": "windows"})
    )
    overall_totals = overall.groupby("month", as_index=False)["windows"].sum().rename(columns={"windows": "month_windows"})
    overall = overall.merge(overall_totals, on="month", validate="many_to_one")
    overall["month_class_fraction"] = overall["windows"] / overall["month_windows"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output_dir / "class_counts_by_year_month.csv", index=False)
    overall.to_csv(args.output_dir / "class_counts_by_month.csv", index=False)
    summary = {
        "years": years,
        "samples": len(dataset),
        "thresholds_mm_15min": thresholds,
        "class_names": CLASS_NAMES,
        "window_definition": {"t_in": args.t_in, "t_out": args.t_out, "stride": args.stride},
        "crop": dataset.crop_metadata,
        "recommendation": "Use these counts to define optional seasonal sampler weights; preserve all months in canonical datasets and evaluation.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Auditoria sazonal concluída | amostras={len(dataset)} | saída={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
