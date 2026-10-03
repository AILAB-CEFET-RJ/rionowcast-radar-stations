"""Build a small C13 event pilot without retaining GOES Full Disk files."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import timedelta
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
    parser = argparse.ArgumentParser(description="Gera o piloto C13 por eventos, baixando e removendo cada Full Disk imediatamente.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--goes-config", type=Path, required=True)
    parser.add_argument("--radar-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def hour_prefix(config: dict, timestamp) -> str:
    timestamp = ensure_utc(timestamp)
    source = config["source"]
    return f"{source['bucket']}/{source['product']}/{timestamp.year}/{timestamp.timetuple().tm_yday:03d}/{timestamp.hour:02d}"


def scenes_near(fs, config: dict, timestamp, listing_cache: dict[str, list[tuple]]) -> list[tuple]:
    timestamp = ensure_utc(timestamp)
    candidates = []
    for anchor in (timestamp - timedelta(hours=1), timestamp):
        prefix = hour_prefix(config, anchor)
        if prefix not in listing_cache:
            try:
                objects = fs.ls(prefix, detail=False)
            except FileNotFoundError:
                # O bucket não materializa prefixos para horas sem cenas.
                objects = []
            except Exception as error:
                raise RuntimeError(f"Falha ao listar {prefix}: {error}") from error
            entries = []
            for remote in objects:
                try:
                    channel, scene_start, scene_end = parse_goes_filename_bounds(remote)
                except ValueError:
                    continue
                if channel in config["channels"]:
                    entries.append((scene_start, scene_end, Path(remote)))
            listing_cache[prefix] = entries
        candidates.extend(listing_cache[prefix])
    return candidates


def atomic_publish(partial: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"{final} já existe.")
    os.replace(partial, final)


def event_id(event: dict) -> str:
    return f"{event['kind']}-{event['year']}-{int(event['target_frame_index']):05d}"


def main() -> None:
    args = parse_args()
    try:
        import s3fs
    except ImportError as error:
        raise RuntimeError("Piloto GOES streaming requer s3fs; instale requirements.txt.") from error
    config = load_goes_config(args.goes_config)
    if len(config["channels"]) != 1 or config["channels"][0] != "C13":
        raise ValueError("O construtor do piloto V1 requer somente o canal C13.")
    radar_config = load_capture_config(args.radar_config)
    grid = config["target_grid"]
    latitude, longitude = target_grid_latlon(radar_config, int(grid["height"]), int(grid["width"]))
    events = json.loads(args.events.read_text(encoding="utf-8"))
    if not events:
        raise ValueError("Manifesto de eventos vazio.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    listing_cache: dict[str, list[tuple]] = {}
    frame_cache: dict[str, tuple[np.ndarray, object, object]] = {}
    source_rows = []
    for ordinal, event in enumerate(events, start=1):
        identifier = event_id(event)
        final_dir = args.output_dir / "events" / identifier
        if args.resume and (final_dir / "metadata.json").is_file():
            print(f"[{ordinal}/{len(events)}] {identifier}: já publicado; pulando.", flush=True)
            continue
        partial_dir = final_dir.with_name(final_dir.name + ".partial")
        if partial_dir.exists():
            raise FileExistsError(f"{partial_dir} existe; revise antes de reiniciar o piloto.")
        print(f"[{ordinal}/{len(events)}] {identifier}: iniciando...", flush=True)
        input_times = [ensure_utc(value) for value in event["input_timestamps_utc"]]
        partial_dir.mkdir(parents=True)
        frames = np.full((len(input_times), int(grid["height"]), int(grid["width"]), 1), np.nan, dtype=np.float32)
        available = np.zeros(len(input_times), dtype=np.uint8)
        source_times = np.full(len(input_times), "", dtype="<U32")
        remote_keys = [""] * len(input_times)
        errors = []
        with tempfile.TemporaryDirectory(prefix="rionowcast-goes-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            for index, radar_time in enumerate(input_times):
                try:
                    selected = select_causal_scene(
                        scenes_near(fs, config, radar_time, listing_cache), radar_time,
                        float(config["temporal_alignment"]["maximum_scene_age_minutes"]),
                    )
                    if selected is None:
                        errors.append({"input_index": index, "timestamp_utc": radar_time.isoformat(), "error": "no_causal_scene"})
                        continue
                    scene_start, scene_end, remote = selected
                    remote_text = str(remote)
                    if remote_text not in frame_cache:
                        local = temporary_root / Path(remote_text).name
                        try:
                            fs.get(remote_text, str(local))
                            frame_cache[remote_text] = (read_and_reproject_scene(local, latitude, longitude), scene_start, scene_end)
                        finally:
                            local.unlink(missing_ok=True)
                    frame, cached_start, cached_end = frame_cache[remote_text]
                    if cached_start != scene_start or cached_end != scene_end:
                        raise ValueError("Cache GOES com timestamp inconsistente.")
                    frames[index, :, :, 0] = frame
                    if np.isfinite(frame).all():
                        available[index] = 1
                    else:
                        errors.append({"input_index": index, "timestamp_utc": radar_time.isoformat(), "error": "nonfinite_cmi"})
                    source_times[index], remote_keys[index] = scene_end.isoformat(), remote_text
                    source_rows.append({"event_id": identifier, "input_index": index, "radar_timestamp_utc": radar_time.isoformat(), "scene_start_timestamp_utc": scene_start.isoformat(), "scene_end_timestamp_utc": scene_end.isoformat(), "remote": remote_text, "age_minutes": (radar_time - scene_end).total_seconds() / 60.0, "available": int(available[index])})
                except Exception as error:
                    errors.append({"input_index": index, "timestamp_utc": radar_time.isoformat(), "error": str(error)})
        np.save(partial_dir / "c13_frames.npy", frames)
        np.save(partial_dir / "availability.npy", available)
        np.save(partial_dir / "source_timestamps.npy", source_times)
        metadata = {
            "event": event,
            "shape": list(frames.shape),
            "dtype": "float32",
            "channels": ["C13"],
            "frames_file": "c13_frames.npy",
            "availability_file": "availability.npy",
            "source_timestamps_file": "source_timestamps.npy",
            "source_timestamp_semantics": "scene end UTC",
            "remote_keys": remote_keys,
            "complete": bool(available.all()),
            "errors": errors,
            "goes_config_sha256": config_digest(args.goes_config),
            "radar_config_sha256": config_digest(args.radar_config),
            "causality": config["temporal_alignment"],
            "normalization": config["storage"]["normalization"],
            "storage_policy": "Full Disk NetCDF is held only in a temporary directory and removed after reprojection.",
        }
        (partial_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        atomic_publish(partial_dir, final_dir)
        print(f"[{ordinal}/{len(events)}] {identifier}: C13 disponíveis={int(available.sum())}/{len(available)}", flush=True)
    source_path = args.output_dir / "source_manifest.jsonl"
    with source_path.open("a", encoding="utf-8") as file:
        for row in source_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "events_manifest": str(args.events), "goes_config_sha256": config_digest(args.goes_config),
        "radar_config_sha256": config_digest(args.radar_config), "cached_unique_scenes": len(frame_cache),
        "source_manifest": str(source_path), "storage_policy": "streaming temporary Full Disk files",
    }
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Piloto C13 concluído | cenas únicas={len(frame_cache)} | saída={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
