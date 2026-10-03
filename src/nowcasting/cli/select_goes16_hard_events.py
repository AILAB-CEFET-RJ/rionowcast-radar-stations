"""Select radar-matched wet/dry events before spending storage on GOES scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.cli.select_goes16_pilot_events import candidates_for_year, separated, serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seleciona pares wet/dry com presença visual de eco de radar comparável."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-start", type=int, default=2023)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--min-rain-mm", type=float, default=1.25)
    parser.add_argument("--max-echo-difference", type=float, default=0.05)
    parser.add_argument("--min-separation-hours", type=float, default=24.0)
    parser.add_argument("--neighborhood-radius", type=int, default=2)
    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--t-out", type=int, default=5)
    return parser.parse_args()


def station_pixels(year_dir: Path) -> np.ndarray:
    metadata = json.loads((year_dir / "targets_alertario_metadata.json").read_text(encoding="utf-8"))
    with np.load(year_dir / metadata["sparse_file"]) as sparse:
        pixels = np.column_stack((np.asarray(sparse["row"]), np.asarray(sparse["column"])))
    return np.unique(pixels, axis=0)


def echo_fraction(frame: np.ndarray, pixels: np.ndarray, radius: int) -> float:
    """Mean non-black RGB fraction around all mapped station pixels."""
    echo = np.any(frame != 0, axis=-1).astype(np.int32)
    integral = np.pad(echo, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    values = []
    for row, column in pixels:
        top, bottom = max(0, int(row) - radius), min(echo.shape[0], int(row) + radius + 1)
        left, right = max(0, int(column) - radius), min(echo.shape[1], int(column) + radius + 1)
        total = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
        values.append(total / ((bottom - top) * (right - left)))
    return float(np.mean(values))


def load_year_scores(root: Path, year: int, candidates: list[dict], radius: int) -> list[dict]:
    year_dir = root / f"year={year}"
    metadata = json.loads((year_dir / "metadata.json").read_text(encoding="utf-8"))
    radar = np.memmap(
        year_dir / metadata.get("frames_file", "radar_frames.dat"), dtype=np.dtype(metadata["dtype"]),
        mode="r", shape=tuple(metadata["shape"]),
    )
    pixels = station_pixels(year_dir)
    for candidate in candidates:
        candidate["radar_echo_fraction"] = echo_fraction(radar[candidate["target_frame_index"] - 1], pixels, radius)
    return candidates


def select_pairs(wet: list[dict], dry: list[dict], maximum_difference: float, separation_hours: float, maximum_pairs: int) -> list[tuple[dict, dict]]:
    selected: list[dict] = []
    pairs: list[tuple[dict, dict]] = []
    separation = pd.Timedelta(hours=separation_hours)
    for wet_event in sorted(wet, key=lambda item: item["max_m15_mm_15min"], reverse=True):
        candidates = [
            dry_event for dry_event in dry
            if dry_event["year"] == wet_event["year"]
            and dry_event["timestamp"].month == wet_event["timestamp"].month
            and dry_event["timestamp"].hour == wet_event["timestamp"].hour
            and abs(dry_event["radar_echo_fraction"] - wet_event["radar_echo_fraction"]) <= maximum_difference
            and separated(dry_event, selected, separation)
            and separated(wet_event, selected, separation)
        ]
        if not candidates:
            continue
        dry_event = min(candidates, key=lambda item: (abs(item["radar_echo_fraction"] - wet_event["radar_echo_fraction"]), item["timestamp"]))
        selected.extend((wet_event, dry_event))
        pairs.append((wet_event, dry_event))
        if len(pairs) == maximum_pairs:
            break
    return pairs


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start or min(args.max_pairs, args.t_in, args.t_out) <= 0:
        raise ValueError("Anos, pares e comprimentos temporais devem ser positivos.")
    if args.min_rain_mm < 0 or args.max_echo_difference < 0 or args.neighborhood_radius < 0:
        raise ValueError("Limiar de chuva, diferença de eco e raio devem ser não negativos.")
    all_candidates = []
    for year in range(args.year_start, args.year_end + 1):
        candidates = candidates_for_year(args.dataset_root, year, args.t_in, args.t_out)
        all_candidates.extend(load_year_scores(args.dataset_root, year, candidates, args.neighborhood_radius))
    wet = [item for item in all_candidates if item["max_m15_mm_15min"] >= args.min_rain_mm]
    dry = [item for item in all_candidates if item["max_m15_future_mm_15min"] == 0.0]
    pairs = select_pairs(wet, dry, args.max_echo_difference, args.min_separation_hours, args.max_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, rows = [], []
    for index, (wet_event, dry_event) in enumerate(pairs, start=1):
        pair_id = f"hard-{index:03d}"
        for kind, event, counterpart in (("wet", wet_event, dry_event), ("dry_control", dry_event, wet_event)):
            record = serializable(event, kind, args.t_in, args.t_out)
            record["pair_id"] = pair_id
            record["matched_event_timestamp_utc"] = counterpart["timestamp"].isoformat()
            record["radar_echo_fraction"] = event["radar_echo_fraction"]
            manifest.append(record)
            rows.append({"pair_id": pair_id, "kind": kind, "timestamp_utc": event["timestamp"].isoformat(), "radar_echo_fraction": event["radar_echo_fraction"], "max_m15_mm_15min": event["max_m15_mm_15min"], "max_m15_future_mm_15min": event["max_m15_future_mm_15min"], "echo_difference": abs(wet_event["radar_echo_fraction"] - dry_event["radar_echo_fraction"])})
    pd.DataFrame(rows).to_csv(args.output_dir / "matched_pairs.csv", index=False)
    (args.output_dir / "events.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    candidate_frame = pd.DataFrame(all_candidates)
    summary = {
        "scope": "selection for a conditional signal audit; not predictive evaluation",
        "candidate_events": len(all_candidates), "wet_candidates": len(wet), "dry_candidates_all_future_horizons": len(dry),
        "pairs": len(pairs), "requested_pairs": args.max_pairs, "min_rain_mm_15min": args.min_rain_mm,
        "max_echo_difference": args.max_echo_difference, "min_separation_hours": args.min_separation_hours,
        "neighborhood_radius": args.neighborhood_radius, "t_in": args.t_in, "t_out": args.t_out,
        "radar_metric": "mean fraction of non-black RGB pixels around mapped stations; visual echo-presence proxy, not reflectivity",
    }
    if not candidate_frame.empty:
        candidate_frame.to_csv(args.output_dir / "candidates.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Pares difíceis selecionados | pares={len(pairs)}/{args.max_pairs} | saída={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
