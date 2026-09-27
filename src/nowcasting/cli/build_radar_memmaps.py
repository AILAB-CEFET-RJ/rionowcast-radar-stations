"""Build versioned radar memmaps from timestamped Sumaré PNG captures.

The output keeps only usable 15-minute composites, but records every expected
time bucket in ``window_coverage.csv``. Consumers can therefore distinguish a
missing interval from two adjacent samples. Audited datasets set
``enforce_timestamp_continuity`` in metadata; the training dataset uses it to
avoid windows spanning a gap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from nowcasting.radar_capture import load_capture_config, load_reflectivity_rgb
from nowcasting.radar_timestamps import local_filename_timestamp_to_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera memmaps auditáveis do radar Sumaré.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--aggregate-minutes", type=int, default=15)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--min-frames-per-window", type=int, default=6)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument(
        "--aggregation", choices=("rgb-max", "latest"), default="rgb-max",
        help="Agregação de imagem. rgb-max preserva a série legada, mas não é grandeza física.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Substitui somente diretórios anuais já finalizados.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Pula anos já publicados e processa somente os anos ausentes.",
    )
    parser.add_argument(
        "--restart-partial", action="store_true",
        help="Remove e reconstrói um ano parcial; requer --resume.",
    )
    return parser.parse_args()


def extract_timestamp(path: Path) -> datetime:
    parts = path.stem.rstrip("_").split("_")
    if len(parts) != 5:
        raise ValueError(f"Nome fora do padrão YYYY_MM_DD_HH_MM: {path.name}")
    return datetime(*map(int, parts))


def _score_file(path: Path) -> int:
    """Prefer the capture without the duplicate trailing underscore suffix."""
    return 0 if path.stem.endswith("_") else 1


def collect_files_for_year(
    data_root: Path, year: int, start_date: date | None = None, end_date: date | None = None,
) -> tuple[list[tuple[datetime, Path]], dict[str, int]]:
    best: dict[datetime, Path] = {}
    report = {"png_files": 0, "invalid_filename": 0, "duplicate_timestamps": 0}
    current = start_date or date(year, 1, 1)
    end = end_date or date(year, 12, 31)
    if current.year != year or end.year != year or current > end:
        raise ValueError("O intervalo de datas deve pertencer ao ano solicitado.")
    while current <= end:
        if current.day == 1:
            print(f"[{year}] Lendo mês {current.month:02d}...", flush=True)
        directory = data_root / f"{current:%Y}" / f"{current:%m}" / f"{current:%d}"
        if directory.is_dir():
            for path in sorted(directory.glob("*.png")):
                report["png_files"] += 1
                try:
                    timestamp = extract_timestamp(path)
                except (TypeError, ValueError):
                    report["invalid_filename"] += 1
                    continue
                if timestamp in best:
                    report["duplicate_timestamps"] += 1
                    if _score_file(path) <= _score_file(best[timestamp]):
                        continue
                best[timestamp] = path
        current += timedelta(days=1)
    return sorted(best.items()), report


def collect_files_for_utc_year(
    data_root: Path, year: int, source_timezone: str,
) -> tuple[list[tuple[datetime, Path]], dict[str, int]]:
    """Collect local-calendar files which belong to one UTC output year."""
    best: dict[datetime, Path] = {}
    report = {"png_files": 0, "invalid_filename": 0, "duplicate_timestamps": 0, "source_local_years": [year - 1, year]}
    local_periods = (
        (year - 1, date(year - 1, 12, 31), date(year - 1, 12, 31)),
        (year, date(year, 1, 1), date(year, 12, 31)),
    )
    for local_year, start_date, end_date in local_periods:
        local_items, local_report = collect_files_for_year(data_root, local_year, start_date, end_date)
        for key in ("png_files", "invalid_filename", "duplicate_timestamps"):
            report[key] += local_report[key]
        for local_timestamp, path in local_items:
            timestamp = local_filename_timestamp_to_utc(local_timestamp, source_timezone)
            if timestamp.year != year:
                continue
            if timestamp in best:
                report["duplicate_timestamps"] += 1
                if _score_file(path) <= _score_file(best[timestamp]):
                    continue
            best[timestamp] = path
    return sorted(best.items()), report


def build_time_buckets(items: list[tuple[datetime, Path]], minutes: int) -> dict[datetime, list[tuple[datetime, Path]]]:
    buckets: dict[datetime, list[tuple[datetime, Path]]] = defaultdict(list)
    for timestamp, path in items:
        bucket = timestamp.replace(minute=(timestamp.minute // minutes) * minutes, second=0, microsecond=0)
        buckets[bucket].append((timestamp, path))
    return buckets


def load_radar_image_rgb(path: Path, width: int, height: int, capture_config: dict) -> np.ndarray:
    rgb = load_reflectivity_rgb(path, capture_config)
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize((width, height), Image.NEAREST), dtype=np.uint8)


def _calendar_buckets(year: int, minutes: int):
    current = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    while current < end:
        yield current
        current += timedelta(minutes=minutes)


def _config_digest(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _atomic_publish(partial: Path, final: Path, overwrite: bool) -> None:
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"Destino já existe: {final}. Use --overwrite após validar o conteúdo.")
        backup = final.with_name(f"{final.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(final, backup)
        try:
            os.replace(partial, final)
        except Exception:
            os.replace(backup, final)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(partial, final)


def prepare_year_resume(
    output_root: Path, year: int, resume: bool, restart_partial: bool,
) -> bool:
    """Apply annual resume policy and return whether the year must be skipped."""
    final_dir = output_root / f"year={year}"
    partial_dir = output_root / f"year={year}.partial"
    if resume and final_dir.exists():
        print(f"[{year}] já finalizado; pulando por --resume.", flush=True)
        return True
    if restart_partial and partial_dir.exists():
        if not partial_dir.is_dir():
            raise FileExistsError(f"Parcial não é diretório: {partial_dir}")
        print(f"[{year}] removendo parcial para reconstrução solicitada: {partial_dir}", flush=True)
        shutil.rmtree(partial_dir)
    return False


def process_year(
    year: int, data_root: Path, output_root: Path, aggregate_minutes: int,
    height: int, width: int, min_frames_per_window: int, capture_config: dict,
    config_digest: str, aggregation: str, overwrite: bool, source_timezone: str | None = None,
) -> None:
    final_dir = output_root / f"year={year}"
    partial_dir = output_root / f"year={year}.partial"
    if partial_dir.exists():
        raise FileExistsError(f"Geração parcial existente: {partial_dir}. Remova-a somente após inspeção.")
    partial_dir.mkdir(parents=True)
    try:
        if source_timezone:
            items, source_report = collect_files_for_utc_year(data_root, year, source_timezone)
        else:
            items, source_report = collect_files_for_year(data_root, year)
        buckets = build_time_buckets(items, aggregate_minutes)
        coverage: dict[datetime, dict[str, int]] = {
            bucket: {"source_frames": len(values), "valid_frames": 0, "usable": 0}
            for bucket, values in buckets.items()
        }
        candidates = [(bucket, values) for bucket, values in sorted(buckets.items()) if len(values) >= min_frames_per_window]
        print(f"[{year}] PNGs únicos={len(items)} | candidatos={len(candidates)}", flush=True)
        frames_path = partial_dir / "radar_frames.dat"
        frames = np.memmap(frames_path, dtype=np.uint8, mode="w+", shape=(len(candidates), height, width, 3))
        timestamps: list[str] = []
        invalid_images = 0
        written = 0
        for index, (bucket, captures) in enumerate(candidates):
            usable: list[tuple[datetime, np.ndarray]] = []
            for timestamp, path in captures:
                try:
                    usable.append((timestamp, load_radar_image_rgb(path, width, height, capture_config)))
                except Exception as error:
                    invalid_images += 1
                    if invalid_images <= 20:
                        print(f"[{year}] imagem inválida: {path} | {error}", flush=True)
            coverage[bucket] = {"source_frames": len(captures), "valid_frames": len(usable), "usable": int(len(usable) >= min_frames_per_window)}
            if len(usable) < min_frames_per_window:
                continue
            arrays = [frame for _, frame in usable]
            frames[written] = np.maximum.reduce(arrays) if aggregation == "rgb-max" else arrays[-1]
            timestamps.append(bucket.isoformat())
            written += 1
            if (index + 1) % 1000 == 0:
                print(f"[{year}] processadas={index + 1}/{len(candidates)} | gravadas={written} | inválidas={invalid_images}", flush=True)
        frames.flush()
        del frames
        with frames_path.open("r+b") as file:
            file.truncate(written * height * width * 3)
        np.save(partial_dir / "radar_timestamps.npy", np.asarray(timestamps, dtype="<U19"))
        with (partial_dir / "window_coverage.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("timestamp", "source_frames", "valid_frames", "usable"))
            writer.writeheader()
            for bucket in _calendar_buckets(year, aggregate_minutes):
                row = coverage.get(bucket, {"source_frames": 0, "valid_frames": 0, "usable": 0})
                writer.writerow({"timestamp": bucket.isoformat(), **row})
        expected_windows = sum(1 for _ in _calendar_buckets(year, aggregate_minutes))
        metadata = {
            "dataset_contract": "radar_sumare_audited_v3",
            "year": year,
            "aggregate_minutes": aggregate_minutes,
            "aggregation_method": aggregation,
            "aggregation_warning": "rgb-max is an RGB-image aggregation, not a physical reflectivity aggregation.",
            "min_frames_per_window": min_frames_per_window,
            "height": height, "width": width, "channels": 3, "dtype": "uint8",
            "shape": [written, height, width, 3],
            "frames_file": "radar_frames.dat", "timestamps_file": "radar_timestamps.npy",
            "coverage_file": "window_coverage.csv",
            "capture_preprocessing": capture_config,
            "capture_config_sha256": config_digest,
            "timestamp_alignment": {
                "png_filename_timezone": source_timezone,
                "radar_timestamp_convention": "window_start_utc" if source_timezone else "source_filename_naive",
            },
            "enforce_timestamp_continuity": True,
            "source_manifest": {**source_report, "unique_timestamps": len(items), "expected_time_buckets": expected_windows, "written_time_buckets": written, "invalid_images": invalid_images},
        }
        (partial_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _atomic_publish(partial_dir, final_dir, overwrite)
        print(f"[{year}] concluído | frames={written} | destino={final_dir}", flush=True)
    except Exception:
        print(f"[{year}] falhou; resultado parcial preservado em {partial_dir}", flush=True)
        raise


def main() -> None:
    args = parse_args()
    if args.year_start > args.year_end or args.aggregate_minutes <= 0 or args.min_frames_per_window <= 0:
        raise ValueError("Intervalo de anos, aggregate-minutes e min-frames-per-window devem ser positivos.")
    if args.resume and args.overwrite:
        raise ValueError("--resume e --overwrite são mutuamente exclusivos.")
    if args.restart_partial and not args.resume:
        raise ValueError("--restart-partial requer --resume.")
    config = load_capture_config(args.capture_config)
    temporal = config.get("temporal_alignment", {})
    source_timezone = temporal.get("png_filename_timezone")
    if config.get("name") == "radar_sumare_historical_audited" and not source_timezone:
        raise ValueError("O contrato histórico auditável deve declarar temporal_alignment.png_filename_timezone.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    digest = _config_digest(args.capture_config)
    print(f"Contrato auditável | config_sha256={digest} | agregação={args.aggregation}", flush=True)
    for year in range(args.year_start, args.year_end + 1):
        if prepare_year_resume(args.output_root, year, args.resume, args.restart_partial):
            continue
        process_year(year, args.data_root, args.output_root, args.aggregate_minutes, args.height, args.width,
                     args.min_frames_per_window, config, digest, args.aggregation, args.overwrite, source_timezone)


if __name__ == "__main__":
    main()
