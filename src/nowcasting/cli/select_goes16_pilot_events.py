"""Select wet events and matched dry controls for a non-predictive GOES pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.goes16 import ensure_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seleciona eventos C13 e controles secos a partir dos targets AlertaRio.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-start", type=int, default=2023)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--max-wet-events", type=int, default=24)
    parser.add_argument("--min-rain-mm", type=float, default=6.25)
    parser.add_argument("--min-separation-hours", type=float, default=24.0)
    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--t-out", type=int, default=5)
    return parser.parse_args()


def contiguous(timestamps: list[pd.Timestamp], start: int, length: int) -> bool:
    return all(timestamps[index + 1] - timestamps[index] == pd.Timedelta(minutes=15) for index in range(start, start + length - 1))


def candidates_for_year(root: Path, year: int, t_in: int, t_out: int) -> list[dict]:
    year_dir = root / f"year={year}"
    timestamps = [pd.Timestamp(ensure_utc(value)) for value in np.load(year_dir / "radar_timestamps.npy", allow_pickle=False)]
    metadata = json.loads((year_dir / "targets_alertario_metadata.json").read_text(encoding="utf-8"))
    with np.load(year_dir / metadata["sparse_file"]) as sparse:
        frames = np.asarray(sparse["frame"], dtype=np.int64)
        values = np.expm1(np.asarray(sparse["value"], dtype=np.float64))
        station_ids = np.asarray(sparse["station_id"], dtype=np.int64) if "station_id" in sparse.files else np.full(len(frames), -1, dtype=np.int64)
    maxima = np.full(len(timestamps), np.nan, dtype=np.float64)
    station_at_max = np.full(len(timestamps), -1, dtype=np.int64)
    for frame, value, station_id in zip(frames, values, station_ids):
        if value > maxima[frame] or not np.isfinite(maxima[frame]):
            maxima[frame], station_at_max[frame] = value, station_id
    result = []
    for frame, value in enumerate(maxima):
        if frame < t_in or frame + t_out > len(timestamps) or not np.isfinite(value):
            continue
        future_maxima = maxima[frame:frame + t_out]
        if not np.isfinite(future_maxima).all():
            continue
        sequence_start = frame - t_in
        if not contiguous(timestamps, sequence_start, t_in + t_out):
            continue
        result.append({"year": year, "target_frame_index": frame, "timestamp": timestamps[frame], "max_m15_mm_15min": float(value), "max_m15_future_mm_15min": float(future_maxima.max()), "station_id": int(station_at_max[frame])})
    return result


def separated(candidate: dict, selected: list[dict], minimum: pd.Timedelta) -> bool:
    return all(abs(candidate["timestamp"] - item["timestamp"]) >= minimum for item in selected)


def serializable(candidate: dict, kind: str, t_in: int, t_out: int) -> dict:
    timestamp = candidate["timestamp"]
    return {
        "kind": kind,
        "selection_uses_observed_target": True,
        "year": candidate["year"],
        "target_frame_index": candidate["target_frame_index"],
        "event_timestamp_utc": timestamp.isoformat(),
        "station_id_at_max": candidate["station_id"],
        "max_m15_mm_15min": candidate["max_m15_mm_15min"],
        "max_m15_future_mm_15min": candidate["max_m15_future_mm_15min"],
        "input_timestamps_utc": [(timestamp - pd.Timedelta(minutes=15 * offset)).isoformat() for offset in range(t_in, 0, -1)],
        "target_timestamps_utc": [(timestamp + pd.Timedelta(minutes=15 * offset)).isoformat() for offset in range(t_out)],
    }


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start or min(args.max_wet_events, args.t_in, args.t_out) <= 0:
        raise ValueError("Anos, quantidade de eventos e comprimentos temporais devem ser positivos.")
    if args.min_rain_mm < 0 or args.min_separation_hours < 0:
        raise ValueError("Limiar de chuva e separação temporal não podem ser negativos.")
    candidates = [item for year in range(args.year_start, args.year_end + 1) for item in candidates_for_year(args.dataset_root, year, args.t_in, args.t_out)]
    separation = pd.Timedelta(hours=args.min_separation_hours)
    wet: list[dict] = []
    for item in sorted(candidates, key=lambda value: value["max_m15_mm_15min"], reverse=True):
        if item["max_m15_mm_15min"] >= args.min_rain_mm and separated(item, wet, separation):
            wet.append(item)
        if len(wet) == args.max_wet_events:
            break
    if not wet:
        raise ValueError("Nenhum evento úmido elegível foi encontrado.")
    dry: list[dict] = []
    for event in wet:
        same_regime = [
            item for item in candidates
            if item["year"] == event["year"] and item["timestamp"].month == event["timestamp"].month
            and item["timestamp"].hour == event["timestamp"].hour
            and item["max_m15_future_mm_15min"] == 0.0
        ]
        for item in sorted(same_regime, key=lambda value: value["timestamp"]):
            if separated(item, wet + dry, separation):
                dry.append(item)
                break
    manifest = [serializable(item, "wet", args.t_in, args.t_out) for item in wet]
    manifest.extend(serializable(item, "dry_control", args.t_in, args.t_out) for item in dry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "events.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "scope": "event-audit-only; selection uses observed AlertaRio targets and must not be used as predictive evaluation",
        "wet_events": len(wet), "dry_controls": len(dry), "requested_wet_events": args.max_wet_events,
        "years": [args.year_start, args.year_end], "min_rain_mm_15min": args.min_rain_mm,
        "min_separation_hours": args.min_separation_hours, "t_in": args.t_in, "t_out": args.t_out,
        "dry_control_rule": "maximum m15 across every future horizon must equal zero",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Eventos GOES selecionados | úmidos={len(wet)} | controles secos={len(dry)} | saída={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
