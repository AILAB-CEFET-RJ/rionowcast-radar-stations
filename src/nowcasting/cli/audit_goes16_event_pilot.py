"""Audit causal C13 event-pilot artefacts against sparse AlertaRio targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nowcasting.goes16 import ensure_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita causalidade e sinal C13 do piloto por eventos.")
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighborhood-radius", type=int, default=2)
    parser.add_argument("--max-figures", type=int, default=12)
    parser.add_argument(
        "--figure-kind",
        choices=("wet", "dry_control", "all"),
        default="wet",
        help="Tipo de evento a ilustrar; não altera as métricas calculadas sobre todos os eventos.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        help="Diretório das figuras; padrão: <output-dir>/figures.",
    )
    return parser.parse_args()


def local_mean(frame: np.ndarray, row: int, column: int, radius: int) -> float:
    top, bottom = max(0, row - radius), min(frame.shape[0], row + radius + 1)
    left, right = max(0, column - radius), min(frame.shape[1], column + radius + 1)
    return float(np.nanmean(frame[top:bottom, left:right]))


def local_echo_fraction(frame: np.ndarray, row: int, column: int, radius: int) -> float:
    """Fraction of non-black radar RGB pixels around a station.

    Historical radar RGB is a visual product, so this is an echo-presence proxy,
    not a physical reflectivity measurement.
    """
    top, bottom = max(0, row - radius), min(frame.shape[0], row + radius + 1)
    left, right = max(0, column - radius), min(frame.shape[1], column + radius + 1)
    return float(np.any(frame[top:bottom, left:right] != 0, axis=-1).mean())


def target_records(dataset_root: Path, year: int, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    year_dir = dataset_root / f"year={year}"
    metadata = json.loads((year_dir / "targets_alertario_metadata.json").read_text(encoding="utf-8"))
    with np.load(year_dir / metadata["sparse_file"]) as sparse:
        mask = np.asarray(sparse["frame"]) == frame
        station_ids = np.asarray(sparse["station_id"])[mask] if "station_id" in sparse.files else np.full(int(mask.sum()), -1)
        return np.asarray(sparse["row"])[mask], np.asarray(sparse["column"])[mask], np.expm1(np.asarray(sparse["value"])[mask]), station_ids


def load_radar_year(dataset_root: Path, year: int) -> tuple[np.memmap, dict[object, int]]:
    year_dir = dataset_root / f"year={year}"
    metadata = json.loads((year_dir / "metadata.json").read_text(encoding="utf-8"))
    shape = tuple(metadata["shape"])
    frames = np.memmap(
        year_dir / metadata.get("frames_file", "radar_frames.dat"),
        dtype=np.dtype(metadata["dtype"]), mode="r", shape=shape,
    )
    timestamps = np.load(year_dir / metadata.get("timestamps_file", "radar_timestamps.npy"), allow_pickle=False)
    return frames, {ensure_utc(timestamp): index for index, timestamp in enumerate(timestamps)}


def radar_input_frames(
    dataset_root: Path, event: dict, cache: dict[int, tuple[np.memmap, dict[object, int]]],
) -> np.ndarray:
    year = int(event["year"])
    if year not in cache:
        cache[year] = load_radar_year(dataset_root, year)
    frames, frame_index = cache[year]
    indices = []
    for timestamp in event["input_timestamps_utc"]:
        index = frame_index.get(ensure_utc(timestamp))
        if index is None:
            raise ValueError(f"{event['kind']} {event['event_timestamp_utc']}: frame de radar ausente em {timestamp}.")
        indices.append(index)
    return np.asarray(frames[indices], dtype=np.uint8)


def render_event(
    path: Path, goes_frames: np.ndarray, radar_frames: np.ndarray, rows: np.ndarray,
    columns: np.ndarray, rainfall: np.ndarray, title: str,
) -> None:
    figure, axes = plt.subplots(2, len(goes_frames), figsize=(3.1 * len(goes_frames), 6.2), constrained_layout=True)
    for index in range(len(goes_frames)):
        goes_axis, radar_axis = axes[0, index], axes[1, index]
        image = goes_axis.imshow(goes_frames[index, :, :, 0], cmap="turbo", vmin=180, vmax=300)
        radar_axis.imshow(radar_frames[index])
        for axis in (goes_axis, radar_axis):
            axis.scatter(columns, rows, s=12 + 15 * np.sqrt(np.maximum(rainfall, 0)), facecolors="none", edgecolors="white", linewidths=0.7)
            axis.set_xticks([])
            axis.set_yticks([])
        goes_axis.set_title(f"Entrada {index + 1}")
    axes[0, 0].set_ylabel("GOES C13")
    axes[1, 0].set_ylabel("Radar RGB")
    figure.colorbar(image, ax=axes[0].tolist(), label="C13 (K)", shrink=0.8)
    figure.suptitle(title)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.neighborhood_radius < 0 or args.max_figures < 0:
        raise ValueError("--neighborhood-radius e --max-figures não podem ser negativos.")
    event_dirs = sorted(
        (path for path in (args.pilot_root / "events").glob("*") if (path / "metadata.json").is_file()),
        key=lambda path: (not path.name.startswith("wet-"), path.name),
    )
    if not event_dirs:
        raise FileNotFoundError(f"Nenhum evento publicado em {args.pilot_root / 'events'}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.figures_dir or args.output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    for path in figures_dir.glob("*.png"):
        path.unlink()
    records, event_rows, causality_errors, rendered, radar_cache = [], [], 0, 0, {}
    for event_dir in event_dirs:
        metadata = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
        event = metadata["event"]
        frames = np.load(event_dir / metadata["frames_file"], allow_pickle=False)
        available = np.load(event_dir / metadata["availability_file"], allow_pickle=False).astype(bool)
        source_times = np.load(event_dir / metadata["source_timestamps_file"], allow_pickle=False)
        input_times = [ensure_utc(value) for value in event["input_timestamps_utc"]]
        causal = [not source or ensure_utc(source) <= radar for source, radar in zip(source_times, input_times)]
        causality_errors += int(len(causal) - sum(causal))
        rows, columns, rainfall, station_ids = target_records(args.dataset_root, int(event["year"]), int(event["target_frame_index"]))
        radar_frames = radar_input_frames(args.dataset_root, event, radar_cache)
        for row, column, value, station_id in zip(rows, columns, rainfall, station_ids):
            c13 = local_mean(frames[-1, :, :, 0], int(row), int(column), args.neighborhood_radius) if bool(available[-1]) else np.nan
            radar_echo = local_echo_fraction(radar_frames[-1], int(row), int(column), args.neighborhood_radius)
            records.append({"event_id": event_dir.name, "pair_id": event.get("pair_id"), "kind": event["kind"], "complete": bool(metadata["complete"]), "year": event["year"], "target_timestamp_utc": event["event_timestamp_utc"], "station_id": int(station_id), "row": int(row), "column": int(column), "m15_mm_15min": float(value), "c13_kelvin": c13, "radar_echo_fraction": radar_echo})
        event_rows.append({"event_id": event_dir.name, "pair_id": event.get("pair_id"), "kind": event["kind"], "complete": bool(metadata["complete"]), "available_inputs": int(available.sum()), "causal_inputs": int(sum(causal)), "target_max_m15_mm_15min": event["max_m15_mm_15min"], "station_observations": int(len(rows))})
        selected_for_figure = args.figure_kind == "all" or event["kind"] == args.figure_kind
        if rendered < args.max_figures and metadata["complete"] and selected_for_figure:
            render_event(figures_dir / f"{event_dir.name}.png", frames, radar_frames, rows, columns, rainfall, f"{event['kind']} | {event['event_timestamp_utc']} | max m15={event['max_m15_mm_15min']:.2f}")
            rendered += 1
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "station_signal.csv", index=False)
    pd.DataFrame(event_rows).to_csv(args.output_dir / "events.csv", index=False)
    valid = frame.loc[frame["complete"]].dropna(subset=["c13_kelvin"])
    wet, dry = valid.loc[valid["kind"] == "wet"], valid.loc[valid["kind"] == "dry_control"]
    correlation = float(wet["m15_mm_15min"].corr(-wet["c13_kelvin"], method="spearman")) if len(wet) > 1 else None
    summary = {"scope": "event-audit-only; not predictive evaluation", "events": len(event_rows), "complete_events": int(sum(row["complete"] for row in event_rows)), "station_records": len(frame), "complete_station_records": len(valid), "excluded_incomplete_events": int(sum(not row["complete"] for row in event_rows)), "causality_errors": causality_errors, "wet_c13_mean_kelvin": float(wet["c13_kelvin"].mean()) if len(wet) else None, "dry_c13_mean_kelvin": float(dry["c13_kelvin"].mean()) if len(dry) else None, "wet_radar_echo_fraction": float(wet["radar_echo_fraction"].mean()) if len(wet) else None, "dry_radar_echo_fraction": float(dry["radar_echo_fraction"].mean()) if len(dry) else None, "wet_coldness_m15_spearman": correlation, "radar_metric": "fraction of non-black RGB pixels in station neighborhood; visual echo-presence proxy, not reflectivity", "figure_kind": args.figure_kind, "figures_dir": str(figures_dir), "figures": rendered}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Auditoria C13 concluída | eventos={summary['events']} | completos={summary['complete_events']} | erros causais={causality_errors} | saída={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
