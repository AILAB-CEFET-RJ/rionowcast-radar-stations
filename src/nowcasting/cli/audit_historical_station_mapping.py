"""Rank candidate station-to-radar transforms for historical Sumaré PNGs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.cli.audit_capture_mapping import render_overlay
from nowcasting.cli.build_radar_memmaps import build_time_buckets, collect_files_for_year
from nowcasting.radar_capture import load_capture_config, load_reflectivity_rgb
from nowcasting.radar_georeferencing import geographic_station_pixels, orientation_candidates, require_pixels_in_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita orientação do mapeamento AlertaRio nos PNGs históricos.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--alertario-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--aggregate-minutes", type=int, default=15)
    parser.add_argument("--min-frames-per-window", type=int, default=6)
    parser.add_argument("--min-rain-mm", type=float, default=1.25)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--neighborhood-radius", type=int, default=4)
    parser.add_argument(
        "--radar-offset-minutes", type=int, default=0,
        help="Deslocamento aplicado ao timestamp da estação para buscar a janela de radar.",
    )
    parser.add_argument("--wet-threshold-mm", type=float, default=1.25)
    return parser.parse_args()


def candidate_pixels(mapping: pd.DataFrame, height: int, width: int, legacy_height: int = 656, legacy_width: int = 654) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return identity, transpose, and reflection candidates in image coordinates."""
    rows = pd.to_numeric(mapping["pixel_i"], errors="raise").to_numpy(float)
    columns = pd.to_numeric(mapping["pixel_j"], errors="raise").to_numpy(float)

    def scale(values: np.ndarray, source: int, destination: int) -> np.ndarray:
        return np.clip(np.rint(values * destination / source).astype(np.int64), 0, destination - 1)

    direct_rows, direct_columns = scale(rows, legacy_height, height), scale(columns, legacy_width, width)
    transposed_rows, transposed_columns = scale(columns, legacy_width, height), scale(rows, legacy_height, width)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for prefix, base_rows, base_columns in (("direct", direct_rows, direct_columns), ("transpose", transposed_rows, transposed_columns)):
        for vertical, horizontal, suffix in ((False, False, ""), (True, False, "_flip_v"), (False, True, "_flip_h"), (True, True, "_flip_vh")):
            result[prefix + suffix] = (
                height - 1 - base_rows if vertical else base_rows,
                width - 1 - base_columns if horizontal else base_columns,
            )
    return result


def all_candidate_pixels(mapping: pd.DataFrame, height: int, width: int, config: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Include legacy-orientation controls and the physical geographic candidate."""
    candidates = candidate_pixels(mapping, height, width)
    georeferencing = config.get("historical_georeferencing")
    if georeferencing:
        rows, columns = geographic_station_pixels(mapping, height, width, georeferencing)
        require_pixels_in_bounds(rows, columns, height, width)
        geographic = orientation_candidates(rows, columns, height, width)
        for candidate_rows, candidate_columns in geographic.values():
            require_pixels_in_bounds(candidate_rows, candidate_columns, height, width)
        candidates.update(geographic)
    return candidates


def load_observations(root: Path, mapping: pd.DataFrame, year: int) -> pd.DataFrame:
    frames = []
    start, stop = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    for path in sorted(root.rglob("*.parquet")):
        frame = pd.read_parquet(path, columns=["estacao_id", "dia_utc", "m15"])
        frame.columns = ["station_id", "timestamp", "m15"]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.floor("15min")
        frame["m15"] = pd.to_numeric(frame["m15"], errors="coerce")
        frame = frame.loc[frame["timestamp"].ge(start) & frame["timestamp"].lt(stop) & frame["m15"].ge(0)]
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True)
    observations["station_id"] = observations["station_id"].astype(int)
    observations = observations.groupby(["station_id", "timestamp"], as_index=False)["m15"].max()
    return observations.merge(mapping, on="station_id", how="inner", validate="many_to_one")


def aggregate_bucket(captures: list[tuple[object, Path]], config: dict) -> np.ndarray:
    frames = [load_reflectivity_rgb(path, config) for _, path in captures]
    return np.maximum.reduce(frames)


def local_signal(frame: np.ndarray, rows: np.ndarray, columns: np.ndarray, radius: int) -> np.ndarray:
    signals = np.zeros(len(rows), dtype=np.float32)
    for index, (row, column) in enumerate(zip(rows, columns)):
        top, bottom = max(0, row - radius), min(frame.shape[0], row + radius + 1)
        left, right = max(0, column - radius), min(frame.shape[1], column + radius + 1)
        signals[index] = frame[top:bottom, left:right].max() / 255.0
    return signals


def summarize_scores(records: dict[str, list[tuple[float, float]]], wet_threshold: float) -> list[dict[str, object]]:
    summaries = []
    for candidate, values in records.items():
        rain = np.asarray([value[0] for value in values], dtype=np.float64)
        signal = np.asarray([value[1] for value in values], dtype=np.float64)
        correlation = float(np.corrcoef(rain, signal)[0, 1]) if len(values) > 1 and rain.std() and signal.std() else None
        rain_rank = pd.Series(rain).rank(method="average").to_numpy()
        signal_rank = pd.Series(signal).rank(method="average").to_numpy()
        rank_correlation = float(np.corrcoef(rain_rank, signal_rank)[0, 1]) if rain_rank.std() and signal_rank.std() else None
        wet, dry = rain >= wet_threshold, rain == 0
        wet_hit_rate = float((signal[wet] > 0).mean()) if wet.any() else None
        dry_hit_rate = float((signal[dry] > 0).mean()) if dry.any() else None
        summaries.append({
            "candidate": candidate,
            "observations": int(len(values)),
            "weighted_signal": float(np.average(signal, weights=np.maximum(rain, 1e-6))),
            "echo_hit_rate": float((signal > 0).mean()),
            "weighted_echo_hit_rate": float(np.average(signal > 0, weights=np.maximum(rain, 1e-6))),
            "rain_signal_correlation": correlation,
            "rain_signal_rank_correlation": rank_correlation,
            "wet_observations": int(wet.sum()),
            "wet_echo_hit_rate": wet_hit_rate,
            "dry_observations": int(dry.sum()),
            "dry_echo_hit_rate": dry_hit_rate,
            "wet_dry_hit_rate_delta": (wet_hit_rate - dry_hit_rate) if wet_hit_rate is not None and dry_hit_rate is not None else None,
        })
    return sorted(summaries, key=lambda item: item["weighted_signal"], reverse=True)


def main() -> None:
    args = parse_args()
    if args.max_events <= 0 or args.neighborhood_radius < 0:
        raise ValueError("max-events deve ser positivo e neighborhood-radius não pode ser negativo.")
    config = load_capture_config(args.capture_config)
    mapping = pd.read_csv(args.mapping)
    required = {"station_id", "pixel_i", "pixel_j"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"{args.mapping}: colunas exigidas: {sorted(required)}")
    mapping["station_id"] = mapping["station_id"].astype(int)
    source = config["source_image"]
    height, width = int(source["height"]), int(source["width"])
    candidates = all_candidate_pixels(mapping, height, width, config)
    observations = load_observations(args.alertario_root, mapping, args.year)
    items, _ = collect_files_for_year(args.data_root, args.year)
    buckets = build_time_buckets(items, args.aggregate_minutes)
    event_maxima = observations.groupby("timestamp", as_index=False)["m15"].max().sort_values("m15", ascending=False)
    selected: list[tuple[object, object]] = []
    for row in event_maxima.itertuples(index=False):
        station_timestamp = row.timestamp.tz_localize(None).to_pydatetime()
        radar_timestamp = station_timestamp + timedelta(minutes=args.radar_offset_minutes)
        if row.m15 >= args.min_rain_mm and len(buckets.get(radar_timestamp, ())) >= args.min_frames_per_window:
            selected.append((station_timestamp, radar_timestamp))
        if len(selected) == args.max_events:
            break
    if not selected:
        raise ValueError("Nenhum evento chuvoso possui uma janela de radar elegível.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[tuple[float, float]]] = defaultdict(list)
    strongest_frame = None
    strongest_timestamp = selected[0][1]
    event_rows = []
    for ordinal, (station_timestamp, radar_timestamp) in enumerate(selected, start=1):
        frame = aggregate_bucket(buckets[radar_timestamp], config)
        if radar_timestamp == strongest_timestamp:
            strongest_frame = frame
        event_observations = observations.loc[observations["timestamp"] == pd.Timestamp(station_timestamp, tz="UTC")]
        for name, (rows, columns) in candidates.items():
            by_station = dict(zip(mapping["station_id"], local_signal(frame, rows, columns, args.neighborhood_radius)))
            for observation in event_observations.itertuples(index=False):
                records[name].append((float(observation.m15), float(by_station[observation.station_id])))
        event_rows.append({"station_timestamp_utc": station_timestamp.isoformat(), "radar_timestamp_utc": radar_timestamp.isoformat(), "max_m15_mm_15min": float(event_observations["m15"].max()), "stations": int(len(event_observations)), "radar_frames": len(buckets[radar_timestamp])})
        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"Eventos auditados {ordinal}/{len(selected)}", flush=True)

    scores = summarize_scores(records, args.wet_threshold_mm)
    with (args.output_dir / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=scores[0].keys())
        writer.writeheader()
        writer.writerows(scores)
    for name, (rows, columns) in candidates.items():
        report = mapping.copy()
        report["candidate_row"] = rows
        report["candidate_column"] = columns
        report.to_csv(args.output_dir / f"mapping_{name}.csv", index=False)
        render_overlay(strongest_frame, mapping, rows, columns, show_labels=False).save(args.output_dir / f"overlay_{name}.png")
    summary = {
        "year": args.year, "source_shape": [height, width], "legacy_mapping_shape": [656, 654],
        "events": event_rows, "candidate_scores": scores,
        "recommended_candidate": scores[0]["candidate"],
        "recommendation_status": "requires_temporal_offset_holdout_and_meteorological_review",
        "neighborhood_radius_pixels": args.neighborhood_radius,
        "radar_offset_minutes": args.radar_offset_minutes,
        "wet_threshold_mm_15min": args.wet_threshold_mm,
        "historical_georeferencing": config.get("historical_georeferencing"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Auditoria concluída | candidato líder={scores[0]['candidate']} | saída={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
