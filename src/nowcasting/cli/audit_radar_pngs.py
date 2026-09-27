"""Audit timestamped Sumaré PNG archives before building radar memmaps."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from nowcasting.cli.build_radar_memmaps import (
    _calendar_buckets,
    build_time_buckets,
    collect_files_for_year,
)
from nowcasting.radar_capture import load_capture_config, load_reflectivity_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita cobertura e geometria dos PNGs do radar Sumaré.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--aggregate-minutes", type=int, default=15)
    parser.add_argument("--min-frames-per-window", type=int, default=6)
    parser.add_argument("--verify-images", action="store_true", help="Abre cada PNG selecionado para validar geometria e decodificação.")
    parser.add_argument("--preview-count", type=int, default=6)
    return parser.parse_args()


def _write_coverage(path: Path, year: int, minutes: int, buckets: dict[datetime, list], min_frames: int) -> dict[str, int]:
    counts = Counter()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("timestamp", "source_frames", "eligible"))
        writer.writeheader()
        for bucket in _calendar_buckets(year, minutes):
            source_frames = len(buckets.get(bucket, ()))
            eligible = int(source_frames >= min_frames)
            writer.writerow({"timestamp": bucket.isoformat(), "source_frames": source_frames, "eligible": eligible})
            counts["expected_windows"] += 1
            counts["eligible_windows"] += eligible
            counts[f"source_frames_{source_frames}"] += 1
    return dict(counts)


def _eligible_gaps(buckets: dict[datetime, list], min_frames: int, minutes: int) -> list[dict[str, object]]:
    eligible = [timestamp for timestamp, paths in sorted(buckets.items()) if len(paths) >= min_frames]
    gaps = []
    for previous, current in zip(eligible, eligible[1:]):
        elapsed = int((current - previous).total_seconds() // 60)
        if elapsed > minutes:
            gaps.append({"after": previous.isoformat(), "before": current.isoformat(), "missing_minutes": elapsed - minutes})
    return gaps


def _preview(items: list[tuple[datetime, Path]], config: dict, output_dir: Path, count: int) -> None:
    if not items or count <= 0:
        return
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    indexes = np.linspace(0, len(items) - 1, min(count, len(items)), dtype=int)
    for index in indexes:
        timestamp, path = items[int(index)]
        with Image.open(path) as image:
            original = image.convert("RGB")
        rgb = load_reflectivity_rgb(path, config)
        stem = preview_dir / f"{timestamp:%Y%m%dT%H%M}"
        original.save(stem.with_name(f"{stem.name}_original.png"))
        Image.fromarray(rgb, mode="RGB").save(stem.with_name(f"{stem.name}_crop.png"))
        output = config.get("output", {})
        size = (int(output.get("width", 128)), int(output.get("height", 128)))
        Image.fromarray(rgb, mode="RGB").resize(size, Image.NEAREST).save(
            stem.with_name(f"{stem.name}_{size[0]}x{size[1]}.png")
        )


def audit_year(args: argparse.Namespace, config: dict, year: int) -> dict[str, object]:
    year_dir = args.output_dir / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    items, source_report = collect_files_for_year(args.data_root, year)
    buckets = build_time_buckets(items, args.aggregate_minutes)
    coverage = _write_coverage(year_dir / "window_coverage.csv", year, args.aggregate_minutes, buckets, args.min_frames_per_window)
    verification = Counter()
    if args.verify_images:
        for _, path in items:
            try:
                load_reflectivity_rgb(path, config)
                verification["valid_images"] += 1
            except Exception:
                verification["invalid_images"] += 1
    _preview(items, config, year_dir, args.preview_count)
    result = {
        "year": year,
        "source_manifest": {**source_report, "unique_timestamps": len(items)},
        "coverage": coverage,
        "temporal_gaps_between_eligible_windows": _eligible_gaps(buckets, args.min_frames_per_window, args.aggregate_minutes),
        "image_verification": dict(verification),
    }
    (year_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{year}] únicos={len(items)} | elegíveis={coverage['eligible_windows']} | lacunas={len(result['temporal_gaps_between_eligible_windows'])}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.year_start > args.year_end:
        raise ValueError("year-start deve ser menor ou igual a year-end.")
    config = load_capture_config(args.capture_config)
    summaries = [audit_year(args, config, year) for year in range(args.year_start, args.year_end + 1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"years": summaries}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
