"""Build causal GOES ABI tensors aligned one-to-one with audited radar frames."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from nowcasting.goes16 import (
    config_digest,
    ensure_utc,
    load_goes_config,
    parse_goes_filename_bounds,
    read_and_reproject_scene,
    select_causal_scene,
    target_grid_latlon,
)
from nowcasting.radar_capture import load_capture_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera memmaps GOES-16 causalmente alinhados ao dataset de radar.")
    parser.add_argument("--goes-root", type=Path, required=True, help="Raiz dos NetCDFs baixados pelo downloader GOES.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset de radar que receberá os artefatos GOES.")
    parser.add_argument("--goes-config", type=Path, required=True)
    parser.add_argument("--radar-config", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def timestamps_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<U32").tobytes()).hexdigest()


def collect_scenes(root: Path, year: int, channels: list[str]) -> tuple[dict[str, list[tuple]], dict[str, int]]:
    scenes = {channel: [] for channel in channels}
    report = {"netcdf_files": 0, "invalid_filename": 0, "outside_year": 0, "duplicate_scenes": 0}
    seen: set[tuple[str, object]] = set()
    for path in sorted(root.rglob("*.nc")):
        report["netcdf_files"] += 1
        try:
            channel, start, end = parse_goes_filename_bounds(path)
        except ValueError:
            report["invalid_filename"] += 1
            continue
        if channel not in scenes:
            continue
        if start.year != year:
            report["outside_year"] += 1
            continue
        key = (channel, start, end)
        if key in seen:
            report["duplicate_scenes"] += 1
            continue
        seen.add(key)
        scenes[channel].append((start, end, path))
    return scenes, report


def publish(partial: Path, final: Path, overwrite: bool) -> None:
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final}: artefatos GOES já existem; use --overwrite após inspeção.")
        backup = final.with_name(final.name + ".previous")
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


def build_year(args: argparse.Namespace, goes_config: dict, radar_config: dict, year: int) -> None:
    year_dir = args.dataset_root / f"year={year}"
    metadata_path = year_dir / "metadata.json"
    timestamps_path = year_dir / "radar_timestamps.npy"
    if not metadata_path.is_file() or not timestamps_path.is_file():
        raise FileNotFoundError(f"{year}: metadata.json ou radar_timestamps.npy ausentes em {year_dir}.")
    final = year_dir / "goes16"
    partial = year_dir / "goes16.partial"
    if args.resume and final.is_dir():
        print(f"[{year}] GOES já finalizado; pulando por --resume.", flush=True)
        return
    if partial.exists():
        raise FileExistsError(f"{partial}: geração GOES parcial existente; revise ou remova antes de repetir.")
    radar_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    radar_timestamps = np.load(timestamps_path, allow_pickle=False)
    radar_times = [ensure_utc(value) for value in radar_timestamps]
    shape = tuple(radar_metadata["shape"])
    if len(shape) != 4 or shape[1] != goes_config["target_grid"]["height"] or shape[2] != goes_config["target_grid"]["width"]:
        raise ValueError(f"{year}: grade de radar {shape} incompatível com a configuração GOES.")
    scenes, source_report = collect_scenes(args.goes_root, year, goes_config["channels"])
    latitude, longitude = target_grid_latlon(radar_config, shape[1], shape[2])
    partial.mkdir(parents=True)
    try:
        channels = list(goes_config["channels"])
        frames_path = partial / "goes_frames.dat"
        frames = np.memmap(frames_path, dtype=np.float32, mode="w+", shape=(len(radar_times), shape[1], shape[2], len(channels)))
        frames[:] = np.nan
        available = np.zeros(len(radar_times), dtype=np.uint8)
        source_times = np.full((len(radar_times), len(channels)), "", dtype="<U32")
        selected_count = {channel: 0 for channel in channels}
        read_failures = 0
        maximum_age = float(goes_config["temporal_alignment"]["maximum_scene_age_minutes"])
        for index, radar_time in enumerate(radar_times):
            selected = [select_causal_scene(scenes[channel], radar_time, maximum_age) for channel in channels]
            if any(value is None for value in selected):
                continue
            try:
                for channel_index, (_, scene_end, path) in enumerate(selected):
                    frames[index, :, :, channel_index] = read_and_reproject_scene(path, latitude, longitude)
                    source_times[index, channel_index] = scene_end.isoformat()
                    selected_count[channels[channel_index]] += 1
                if np.isfinite(frames[index]).all():
                    available[index] = 1
                else:
                    frames[index] = np.nan
            except Exception as error:
                frames[index] = np.nan
                read_failures += 1
                if read_failures <= 20:
                    print(f"[{year}] cena GOES inválida em {radar_time.isoformat()}: {error}", flush=True)
            if (index + 1) % 1000 == 0:
                print(f"[{year}] GOES processadas={index + 1}/{len(radar_times)} | disponíveis={int(available.sum())} | falhas={read_failures}", flush=True)
        frames.flush()
        del frames
        np.save(partial / "goes_available.npy", available)
        np.save(partial / "goes_source_timestamps.npy", source_times)
        metadata = {
            "dataset_contract": "goes16_abi_audited_v2",
            "year": year,
            "shape": [len(radar_times), shape[1], shape[2], len(channels)],
            "dtype": "float32",
            "frames_file": "goes_frames.dat",
            "availability_file": "goes_available.npy",
            "source_timestamps_file": "goes_source_timestamps.npy",
            "source_timestamp_semantics": "scene end UTC",
            "channels": channels,
            "normalization": goes_config["storage"]["normalization"],
            "radar_timestamps_file": "../radar_timestamps.npy",
            "radar_timestamps_sha256": timestamps_digest(radar_timestamps),
            "causality": goes_config["temporal_alignment"],
            "target_grid": goes_config["target_grid"],
            "goes_config": goes_config,
            "goes_config_sha256": config_digest(args.goes_config),
            "radar_config_sha256": config_digest(args.radar_config),
            "source_manifest": {**source_report, "scenes_per_channel": {channel: len(scenes[channel]) for channel in channels}, "selected_per_channel": selected_count, "read_failures": read_failures, "available_frames": int(available.sum())},
        }
        (partial / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        publish(partial, final, args.overwrite)
    except Exception:
        raise
    print(f"[{year}] GOES concluído | disponíveis={int(available.sum())}/{len(available)} | destino={final}", flush=True)


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("--year-end deve ser maior ou igual a --year-start.")
    if args.resume and args.overwrite:
        raise ValueError("--resume e --overwrite são mutuamente exclusivos.")
    goes_config = load_goes_config(args.goes_config)
    radar_config = load_capture_config(args.radar_config)
    for year in range(args.year_start, args.year_end + 1):
        build_year(args, goes_config, radar_config, year)


if __name__ == "__main__":
    main()
