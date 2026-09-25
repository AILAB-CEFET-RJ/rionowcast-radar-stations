from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from nowcasting.operational_inference import aggregate_capture_bucket
from nowcasting.radar_capture import crop_bounds, load_capture_config
from nowcasting.station_geometry import direct_resampled_pixels


def station_pixels(
    mapping: pd.DataFrame,
    height: int,
    width: int,
    source_height: int = 656,
    source_width: int = 654,
) -> tuple[np.ndarray, np.ndarray]:
    required = {"station_id", "pixel_i", "pixel_j"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapeamento: colunas ausentes: {sorted(missing)}")
    rows, columns = direct_resampled_pixels(
        mapping["pixel_i"].to_numpy(float),
        mapping["pixel_j"].to_numpy(float),
        height_orig=source_height,
        width_orig=source_width,
        height=height,
        width=width,
    )
    return rows, columns


def render_overlay(frame: np.ndarray, mapping: pd.DataFrame, rows: np.ndarray, columns: np.ndarray, show_labels: bool) -> Image.Image:
    scale = 5
    canvas = Image.fromarray(frame, mode="RGB").resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(canvas)
    for index, (row, column) in enumerate(zip(rows, columns)):
        x, y = int(column * scale), int(row * scale)
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red", outline="white")
        if show_labels:
            label = str(mapping.iloc[index].get("nome", mapping.iloc[index]["station_id"]))
            draw.text((x + 5, y + 3), label, fill="white", stroke_width=1, stroke_fill="black")
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sobrepõe o mapeamento AlertaRio transformado a uma captura real do radar."
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--show-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = json.loads(args.events.read_text(encoding="utf-8"))
    if not 0 <= args.event_index < len(events):
        raise IndexError(f"--event-index deve estar entre 0 e {len(events) - 1}.")
    event = events[args.event_index]
    bucket = event["input"][-1]
    capture_config = load_capture_config(args.capture_config)
    paths = [args.capture_root / relative for relative in bucket["png_files"]]
    frame = aggregate_capture_bucket(paths, capture_config, args.width, args.height)
    mapping = pd.read_csv(args.mapping)
    left, top, right, bottom = crop_bounds(capture_config)
    rows, columns = station_pixels(
        mapping, args.height, args.width, bottom - top, right - left
    )
    if len(set(zip(rows.tolist(), columns.tolist()))) != len(mapping):
        raise ValueError("Transformação gerou estações no mesmo pixel de destino.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_overlay(frame, mapping, rows, columns, args.show_labels).save(args.output_dir / "station_overlay.png")
    report = mapping[[column for column in ("station_id", "nome", "latitude", "longitude") if column in mapping]].copy()
    report["row_128"] = rows
    report["column_128"] = columns
    report.to_csv(args.output_dir / "station_pixels_128.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "event_timestamp_utc": event["event_timestamp_utc"],
                "event_station": event["station_name"],
                "event_m15_mm_15min": event["m15_mm_15min"],
                "input_bucket_timestamp_utc": bucket["timestamp_utc"],
                "station_count": len(mapping),
                "capture_crop_raw_pixels": {"left": left, "top": top, "right_exclusive": right, "bottom_exclusive": bottom},
                "mapping_status": capture_config["station_mapping_transform"]["status"],
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Auditoria visual salva em: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
