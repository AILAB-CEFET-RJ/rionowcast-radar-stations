"""Seleciona candidatos secos e chuvosos para figuras qualitativas RGB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nowcasting.dataset import RadarStationMemmapDataset, parse_years
from nowcasting.radar_sequence_dataset import RadarFrameSequenceDataset
from nowcasting.station_dataset import load_station_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seleciona sequências de radar classificadas retrospectivamente pelas estações.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mapping-height-orig", type=int, default=654)
    parser.add_argument("--mapping-width-orig", type=int, default=656)
    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--t-out", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--wet-threshold-mm15", type=float, default=6.25)
    parser.add_argument("--candidates-per-class", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def target_timestamp(dataset: RadarFrameSequenceDataset, index: int) -> str:
    year, start = dataset.samples[index]
    data = dataset.year_data[year]
    metadata = data["metadata"]
    assert isinstance(metadata, dict)
    timestamps_path = dataset.radar_root / f"year={year}" / metadata.get("timestamps_file", "radar_timestamps.npy")
    timestamps = np.load(timestamps_path, allow_pickle=False)
    return str(timestamps[start + dataset.t_in])


def main() -> None:
    args = parse_args()
    if min(args.t_in, args.t_out, args.stride, args.candidates_per_class) <= 0 or args.wet_threshold_mm15 <= 0:
        raise ValueError("t-in, t-out, stride, candidates-per-class e wet-threshold-mm15 devem ser positivos.")
    years = parse_years(args.years)
    radar = RadarFrameSequenceDataset(
        args.dataset_root, years, t_in=args.t_in, t_out=args.t_out, stride=args.stride, split_name="showcase-radar",
    )
    stations = RadarStationMemmapDataset(
        args.dataset_root, years, t_in=args.t_in, t_out=args.t_out, stride=args.stride,
        target_source="alertario", split_name="showcase-stations",
    )
    if radar.samples != stations.samples:
        raise ValueError("Janelas de radar e targets de estações não coincidem; não é seguro selecionar showcases.")
    first_frames = radar.year_data[years[0]]["frames"]
    assert isinstance(first_frames, np.memmap)
    pixels, station_ids = load_station_pixels(
        args.mapping, first_frames.shape[1], first_frames.shape[2],
        height_orig=args.mapping_height_orig, width_orig=args.mapping_width_orig,
    )
    dry, wet = [], []
    for index in range(len(radar)):
        _, target, mask = stations[index]
        values = np.expm1(target.numpy()[0, :, pixels[:, 0], pixels[:, 1]])
        observed = mask.numpy()[0, :, pixels[:, 0], pixels[:, 1]].astype(bool)
        if not observed.all():
            continue
        maximum = float(values.max())
        _, future_rgb = radar[index]
        active_pixels = int((future_rgb.amax(dim=0) > 0).sum().item())
        row = {
            "example_index": index,
            "year": radar.samples[index][0],
            "target_timestamp_utc": target_timestamp(radar, index),
            "station_count": len(station_ids),
            "target_max_mm_15min": maximum,
            "target_active_rgb_pixels": active_pixels,
        }
        if maximum == 0.0:
            dry.append(row)
        elif maximum >= args.wet_threshold_mm15:
            wet.append(row)
    # Dry controls favor sparse future RGB. Wet candidates retain both a strong
    # station observation and substantial visual content for manual inspection.
    dry.sort(key=lambda row: (row["target_active_rgb_pixels"], row["target_timestamp_utc"]))
    wet.sort(key=lambda row: (-row["target_max_mm_15min"], -row["target_active_rgb_pixels"], row["target_timestamp_utc"]))
    result = {
        "scope": "retrospective qualitative selection only; not used for training or metrics",
        "years": years,
        "t_in": args.t_in,
        "t_out": args.t_out,
        "stride": args.stride,
        "dry_definition": "all mapped AlertaRio stations observed 0.0 mm/15 min in all future horizons",
        "wet_definition": f"all mapped stations valid and maximum future observation >= {args.wet_threshold_mm15:g} mm/15 min",
        "dry_candidates": dry[:args.candidates_per_class],
        "wet_candidates": wet[:args.candidates_per_class],
        "counts": {"dry_eligible": len(dry), "wet_eligible": len(wet)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Candidatos selecionados | secos={len(dry)} | chuvosos={len(wet)} | saída={args.output}", flush=True)


if __name__ == "__main__":
    main()
