from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def extract_timestamp(file_path: Path) -> datetime:
    """Parse the timestamp convention used by the Sumaré PNG captures."""
    parts = file_path.stem.rstrip("_").split("_")
    if len(parts) != 5:
        raise ValueError(f"Nome fora do padrão esperado: {file_path.name}")
    return datetime(*map(int, parts))


def evenly_spaced(items: list[dict], count: int) -> list[dict]:
    if not items:
        return []
    count = min(count, len(items))
    indexes = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[index] for index in np.unique(indexes)]


def rgb_and_alpha(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba[:, :, :3], rgba[:, :, 3]


def sample_image_stats(path: Path, black_threshold: int) -> dict:
    rgb, alpha = rgb_and_alpha(path)
    visible = alpha > 0
    nonblack = visible & np.any(rgb > black_threshold, axis=2)
    total = nonblack.size
    return {
        "nonblack_pixels": int(nonblack.sum()),
        "nonblack_fraction": float(nonblack.mean()),
        "opaque_fraction": float(visible.mean()),
        "nonblack_mask": nonblack,
    }


def select_contact_records(records: list[dict], count: int) -> list[dict]:
    ordered = sorted(records, key=lambda record: record["nonblack_pixels"])
    return evenly_spaced(ordered, count)


def save_contact_sheet(records: list[dict], output_path: Path) -> None:
    if not records:
        return

    thumbnail_width = 320
    thumbnail_height = 240
    label_height = 42
    columns = min(3, len(records))
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * thumbnail_width, rows * (thumbnail_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, record in enumerate(records):
        with Image.open(record["path"]) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((thumbnail_width, thumbnail_height), Image.Resampling.LANCZOS)

        x = (index % columns) * thumbnail_width
        y = (index // columns) * (thumbnail_height + label_height)
        canvas.paste(thumbnail, (x, y))
        draw.rectangle((x, y + thumbnail_height, x + thumbnail_width, y + thumbnail_height + label_height), fill="white")
        draw.text((x + 5, y + thumbnail_height + 4), record["timestamp"].strftime("%Y-%m-%d %H:%M"), fill="black")
        draw.text(
            (x + 5, y + thumbnail_height + 20),
            f"pixels não pretos: {record['nonblack_fraction']:.1%}",
            fill="black",
        )

    canvas.save(output_path)


def save_layout_persistence(records: list[dict], output_path: Path) -> dict:
    if not records:
        return {"sampled_images_same_size": 0}

    reference_size = records[0]["size"]
    compatible = [record for record in records if record["size"] == reference_size]
    presence = np.zeros((reference_size[1], reference_size[0]), dtype=np.uint16)

    for record in compatible:
        presence += record["nonblack_mask"]

    frequency = presence / len(compatible)
    persistence = np.round(frequency * 255).astype(np.uint8)
    Image.fromarray(persistence, mode="L").save(output_path)

    static = frequency >= 0.95
    rows, columns = np.where(static)
    bbox = None
    if len(rows):
        bbox = {
            "left": int(columns.min()),
            "top": int(rows.min()),
            "right_exclusive": int(columns.max() + 1),
            "bottom_exclusive": int(rows.max() + 1),
        }

    return {
        "sampled_images_same_size": len(compatible),
        "reference_size": list(reference_size),
        "persistent_nonblack_pixels_at_least_95_percent": int(static.sum()),
        "persistent_nonblack_bbox_at_least_95_percent": bbox,
    }


def inspect_capture(
    input_root: Path,
    output_dir: Path,
    sample_count: int = 192,
    contact_count: int = 6,
    black_threshold: int = 3,
) -> dict:
    png_paths = sorted(input_root.rglob("*.png"))
    total_files = sum(1 for path in input_root.rglob("*") if path.is_file())
    records: list[dict] = []
    invalid_images = 0
    invalid_timestamps = 0
    size_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()

    for index, path in enumerate(png_paths, start=1):
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                size = image.size
                mode = image.mode
        except Exception:
            invalid_images += 1
            continue

        try:
            timestamp = extract_timestamp(path)
        except ValueError:
            invalid_timestamps += 1
            continue

        records.append({"path": path, "timestamp": timestamp, "size": size, "mode": mode})
        size_counts[f"{size[0]}x{size[1]}"] += 1
        mode_counts[mode] += 1

        if index % 5000 == 0 or index == len(png_paths):
            print(f"Cabeçalhos lidos: {index}/{len(png_paths)}", flush=True)

    records.sort(key=lambda record: record["timestamp"])
    sampled = evenly_spaced(records, sample_count)

    for index, record in enumerate(sampled, start=1):
        stats = sample_image_stats(record["path"], black_threshold)
        record.update(stats)
        print(f"Amostras analisadas: {index}/{len(sampled)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sample_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "timestamp", "relative_path", "width", "height", "mode",
                "nonblack_pixels", "nonblack_fraction", "opaque_fraction",
            ),
        )
        writer.writeheader()
        for record in sampled:
            writer.writerow(
                {
                    "timestamp": record["timestamp"].isoformat(),
                    "relative_path": record["path"].relative_to(input_root),
                    "width": record["size"][0],
                    "height": record["size"][1],
                    "mode": record["mode"],
                    "nonblack_pixels": record["nonblack_pixels"],
                    "nonblack_fraction": record["nonblack_fraction"],
                    "opaque_fraction": record["opaque_fraction"],
                }
            )

    save_contact_sheet(select_contact_records(sampled, contact_count), output_dir / "contact_sheet.png")
    persistence = save_layout_persistence(sampled, output_dir / "layout_persistence.png")
    summary = {
        "input_root": str(input_root),
        "total_files": total_files,
        "png_files": len(png_paths),
        "valid_pngs_with_timestamp": len(records),
        "invalid_images": invalid_images,
        "invalid_timestamps": invalid_timestamps,
        "timestamp_start": records[0]["timestamp"].isoformat() if records else None,
        "timestamp_end": records[-1]["timestamp"].isoformat() if records else None,
        "image_sizes": dict(size_counts),
        "image_modes": dict(mode_counts),
        "sample_count": len(sampled),
        "black_threshold": black_threshold,
        "sample_nonblack_fraction": {
            "min": min((record["nonblack_fraction"] for record in sampled), default=None),
            "median": float(np.median([record["nonblack_fraction"] for record in sampled])) if sampled else None,
            "max": max((record["nonblack_fraction"] for record in sampled), default=None),
        },
        "artifacts": {
            "sample_stats": csv_path.name,
            "contact_sheet": "contact_sheet.png",
            "layout_persistence": "layout_persistence.png",
        },
        "persistence": persistence,
        "next_step": (
            "Use contact_sheet.png and layout_persistence.png to define and validate "
            "the canonical reflectivity crop before generating a training dataset."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita a geometria e os overlays das capturas PNG do radar Sumaré."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=192)
    parser.add_argument("--contact-count", type=int, default=6)
    parser.add_argument("--black-threshold", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count < 1 or args.contact_count < 1:
        raise ValueError("--sample-count e --contact-count devem ser positivos.")
    summary = inspect_capture(
        input_root=args.input_root,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        contact_count=args.contact_count,
        black_threshold=args.black_threshold,
    )
    print(
        "Auditoria concluída | "
        f"PNGs válidos={summary['valid_pngs_with_timestamp']} | "
        f"saída={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
