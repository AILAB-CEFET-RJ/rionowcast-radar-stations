"""Compare radar-echo characteristics for gauge-selected historical events."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nowcasting.cli.audit_historical_station_mapping import (
    aggregate_bucket,
    load_observations,
)
from nowcasting.cli.build_radar_memmaps import build_time_buckets, collect_files_for_year
from nowcasting.radar_capture import load_capture_config


def frame_statistics(frame: np.ndarray) -> dict[str, float | int | None]:
    """Summarize visible radar echo without assigning a physical RGB meaning."""
    active = frame.max(axis=2) > 0
    active_count = int(active.sum())
    result: dict[str, float | int | None] = {
        "active_pixels": active_count,
        "active_fraction": float(active.mean()),
        "max_channel_value": int(frame.max()),
        "mean_channel_value": float(frame.mean()),
        "active_rgb_colors": int(np.unique(frame[active], axis=0).shape[0]) if active_count else 0,
        "echo_row_min": None,
        "echo_row_max": None,
        "echo_column_min": None,
        "echo_column_max": None,
    }
    if active_count:
        rows, columns = np.where(active)
        result.update({
            "echo_row_min": int(rows.min()), "echo_row_max": int(rows.max()),
            "echo_column_min": int(columns.min()), "echo_column_max": int(columns.max()),
        })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara ecos de radar em eventos selecionados pela chuva das estações AlertaRio."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--alertario-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--aggregate-minutes", type=int, default=15)
    parser.add_argument("--min-frames-per-window", type=int, default=6)
    parser.add_argument("--min-rain-mm", type=float, default=1.25)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--radar-offset-minutes", type=int, default=-15)
    parser.add_argument("--preview-count", type=int, default=12)
    return parser.parse_args()


def audit_year(args: argparse.Namespace, config: dict, mapping: pd.DataFrame, year: int) -> tuple[list[dict], dict]:
    observations = load_observations(args.alertario_root, mapping, year)
    items, _ = collect_files_for_year(args.data_root, year)
    buckets = build_time_buckets(items, args.aggregate_minutes)
    maxima = observations.groupby("timestamp", as_index=False)["m15"].max().sort_values("m15", ascending=False)
    selected: list[tuple[object, object]] = []
    for event in maxima.itertuples(index=False):
        station_timestamp = event.timestamp.tz_localize(None).to_pydatetime()
        radar_timestamp = station_timestamp + timedelta(minutes=args.radar_offset_minutes)
        if event.m15 >= args.min_rain_mm and len(buckets.get(radar_timestamp, ())) >= args.min_frames_per_window:
            selected.append((station_timestamp, radar_timestamp))
        if len(selected) == args.max_events:
            break
    if not selected:
        raise ValueError(f"{year}: nenhum evento elegível.")

    previews = args.output_dir / f"year={year}" / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for ordinal, (station_timestamp, radar_timestamp) in enumerate(selected, start=1):
        frame = aggregate_bucket(buckets[radar_timestamp], config)
        event_observations = observations.loc[observations["timestamp"] == pd.Timestamp(station_timestamp, tz="UTC")]
        wet = event_observations["m15"] >= args.min_rain_mm
        record = {
            "year": year,
            "station_timestamp_utc": station_timestamp.isoformat(),
            "radar_timestamp_utc": radar_timestamp.isoformat(),
            "radar_frames": len(buckets[radar_timestamp]),
            "station_observations": int(len(event_observations)),
            "wet_stations": int(wet.sum()),
            "max_m15_mm_15min": float(event_observations["m15"].max()),
            "mean_m15_mm_15min": float(event_observations["m15"].mean()),
            **frame_statistics(frame),
        }
        records.append(record)
        if ordinal <= args.preview_count:
            Image.fromarray(frame, mode="RGB").resize(
                (frame.shape[1] * 2, frame.shape[0] * 2), Image.Resampling.NEAREST
            ).save(previews / f"{radar_timestamp:%Y%m%dT%H%M}_m15_{record['max_m15_mm_15min']:.2f}.png")
        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"[{year}] Eventos auditados {ordinal}/{len(selected)}", flush=True)

    metrics = pd.DataFrame(records)
    summary = {
        "year": year,
        "events": int(len(records)),
        "median_max_m15_mm_15min": float(metrics["max_m15_mm_15min"].median()),
        "median_active_pixels": float(metrics["active_pixels"].median()),
        "median_active_fraction": float(metrics["active_fraction"].median()),
        "events_without_visible_echo": int((metrics["active_pixels"] == 0).sum()),
        "mean_radar_frames": float(metrics["radar_frames"].mean()),
    }
    return records, summary


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start or args.max_events <= 0 or args.preview_count < 0:
        raise ValueError("Intervalo de anos ou limites de eventos inválidos.")
    config = load_capture_config(args.capture_config)
    mapping = pd.read_csv(args.mapping)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records, summaries = [], []
    for year in range(args.year_start, args.year_end + 1):
        records, summary = audit_year(args, config, mapping, year)
        all_records.extend(records)
        summaries.append(summary)
    with (args.output_dir / "event_echo_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_records[0].keys())
        writer.writeheader()
        writer.writerows(all_records)
    summary = {
        "capture_config": config.get("name"),
        "years": summaries,
        "radar_offset_minutes": args.radar_offset_minutes,
        "min_rain_mm_15min": args.min_rain_mm,
        "status": "diagnostic_only_no_physical_interpretation_of_rgb",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Auditoria comparativa concluída | saída={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
