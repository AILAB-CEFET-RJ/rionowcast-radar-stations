"""Build resumable, causal GOES memmaps from public NOAA S3."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path

import numpy as np

from nowcasting.cli.build_goes16_event_pilot import scenes_near
from nowcasting.goes16 import (
    AbiSampler,
    build_abi_sampler,
    config_digest,
    ensure_utc,
    load_goes_config,
    parse_goes_filename_bounds,
    read_and_reproject_scene_with_sampler,
    select_causal_scene,
    target_grid_latlon,
)
from nowcasting.radar_capture import load_capture_config


STATE_FILE = "generation_state.json"
COMPLETED_FILE = "goes_completed.npy"


def select_causal_scenes_by_channel(
    candidates: list[tuple], channels: list[str], radar_time, maximum_age_minutes: float,
) -> list[tuple | None]:
    """Select one strictly causal ABI scene for every requested channel."""
    candidates_by_channel: dict[str, list[tuple]] = {channel: [] for channel in channels}
    for scene_start, scene_end, remote in candidates:
        channel, _, _ = parse_goes_filename_bounds(remote)
        if channel in candidates_by_channel:
            candidates_by_channel[channel].append((scene_start, scene_end, remote))
    return [
        select_causal_scene(candidates_by_channel[channel], radar_time, maximum_age_minutes)
        for channel in channels
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera memmaps GOES por streaming S3, sem reter Full Disk NetCDF.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--goes-config", type=Path, required=True)
    parser.add_argument("--radar-config", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def digest_timestamps(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<U32").tobytes()).hexdigest()


def in_scope(timestamp, start: date | None, end: date | None) -> bool:
    day = ensure_utc(timestamp).date()
    return (start is None or day >= start) and (end is None or day <= end)


def publish(partial: Path, final: Path, overwrite: bool) -> None:
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final}: artefatos GOES já existem; use --overwrite após inspeção.")
        previous = final.with_name(final.name + ".previous")
        if previous.exists():
            raise FileExistsError(f"{previous}: backup anterior existe; revise antes de sobrescrever.")
        os.replace(final, previous)
        try:
            os.replace(partial, final)
        except Exception:
            os.replace(previous, final)
            raise
        shutil.rmtree(previous)
    else:
        os.replace(partial, final)


def scope_metadata(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "start_date": args.start_date.isoformat() if args.start_date else None,
        "end_date": args.end_date.isoformat() if args.end_date else None,
    }


def state_contract(args: argparse.Namespace, raw_timestamps: np.ndarray, shape: tuple[int, ...], channels: list[str]) -> dict:
    return {
        "version": 2,
        "goes_config_sha256": config_digest(args.goes_config),
        "radar_config_sha256": config_digest(args.radar_config),
        "radar_timestamps_sha256": digest_timestamps(raw_timestamps),
        "shape": list(shape),
        "channels": channels,
        "coverage_scope": scope_metadata(args),
        "source_timestamp_semantics": "scene end UTC",
    }


def write_state(partial: Path, contract: dict) -> None:
    (partial / STATE_FILE).write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def open_or_create_partial(
    args: argparse.Namespace,
    partial: Path,
    contract: dict,
    frame_shape: tuple[int, int, int, int],
    requested: np.ndarray,
) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray, bool]:
    frames_path = partial / "goes_frames.dat"
    available_path = partial / "goes_available.npy"
    source_path = partial / "goes_source_timestamps.npy"
    completed_path = partial / COMPLETED_FILE
    resumed = partial.exists()
    if resumed:
        if not args.resume:
            raise FileExistsError(f"{partial}: use --resume para continuar ou revise os artefatos.")
        state_path = partial / STATE_FILE
        if not state_path.is_file():
            raise ValueError(f"{partial}: estado de geração ausente; não é seguro retomar.")
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise ValueError(f"{partial}: contrato diverge da execução solicitada; não é seguro retomar.")
        required = (frames_path, available_path, source_path, completed_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{partial}: artefatos de retomada ausentes: {', '.join(missing)}")
        frames = np.memmap(frames_path, dtype=np.float32, mode="r+", shape=frame_shape)
        available = np.load(available_path, allow_pickle=False).astype(np.uint8, copy=False)
        source_times = np.load(source_path, allow_pickle=False)
        completed = np.load(completed_path, allow_pickle=False).astype(np.uint8, copy=False)
        if len(available) != frame_shape[0] or source_times.shape != (frame_shape[0], frame_shape[3]) or len(completed) != frame_shape[0]:
            raise ValueError(f"{partial}: shapes de retomada incompatíveis.")
        return frames, available, source_times, completed, True

    partial.mkdir(parents=True)
    frames = np.memmap(frames_path, dtype=np.float32, mode="w+", shape=frame_shape)
    frames[:] = np.nan
    available = np.zeros(frame_shape[0], dtype=np.uint8)
    source_times = np.full((frame_shape[0], frame_shape[3]), "", dtype="<U32")
    completed = np.zeros(frame_shape[0], dtype=np.uint8)
    completed[~requested] = 1
    np.save(available_path, available)
    np.save(source_path, source_times)
    np.save(completed_path, completed)
    (partial / "source_manifest.jsonl").touch()
    write_state(partial, contract)
    return frames, available, source_times, completed, False


def checkpoint(
    partial: Path,
    frames: np.memmap,
    available: np.ndarray,
    source_times: np.ndarray,
    completed: np.ndarray,
) -> None:
    frames.flush()
    np.save(partial / "goes_available.npy", available)
    np.save(partial / "goes_source_timestamps.npy", source_times)
    np.save(partial / COMPLETED_FILE, completed)


def download_remote(filesystem, remote: Path, destination: Path, retries: int) -> Path:
    for attempt in range(1, retries + 1):
        try:
            filesystem.get(str(remote), str(destination))
            return destination
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def report_progress(year: int, done: int, total: int, available: int, failures: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = done / elapsed
    remaining = (total - done) / rate if rate else float("inf")
    print(
        f"[{year}] processadas={done}/{total} | disponíveis={available} | falhas={failures} | "
        f"{rate:.2f} frame/s | ETA={remaining / 60:.1f} min",
        flush=True,
    )


def build_year(args: argparse.Namespace, config: dict, radar_config: dict, filesystem, year: int) -> None:
    year_dir = args.dataset_root / f"year={year}"
    metadata = json.loads((year_dir / "metadata.json").read_text(encoding="utf-8"))
    raw_timestamps = np.load(year_dir / "radar_timestamps.npy", allow_pickle=False)
    radar_times = [ensure_utc(value) for value in raw_timestamps]
    radar_shape = tuple(metadata["shape"])
    channels = list(config["channels"])
    if len(radar_shape) != 4 or radar_shape[1:3] != (config["target_grid"]["height"], config["target_grid"]["width"]):
        raise ValueError(f"{year}: grade de radar incompatível com GOES.")
    final, partial = year_dir / "goes16", year_dir / "goes16.partial"
    if args.resume and final.is_dir():
        print(f"[{year}] GOES já finalizado; pulando por --resume.", flush=True)
        return
    requested = np.array([in_scope(value, args.start_date, args.end_date) for value in radar_times], dtype=bool)
    contract = state_contract(args, raw_timestamps, (len(radar_times), radar_shape[1], radar_shape[2], len(channels)), channels)
    frames, available, source_times, completed, resumed = open_or_create_partial(
        args, partial, contract, tuple(contract["shape"]), requested,
    )
    lock_path = partial / ".stream.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"{year}: outra execução já está escrevendo em {partial}.") from error
    lock_handle.write(f"pid={os.getpid()}\n")
    lock_handle.flush()
    pending = [index for index in np.flatnonzero(requested & ~completed.astype(bool))]
    if not pending:
        print(f"[{year}] retomada sem frames pendentes.", flush=True)
    elif args.download_workers < 1 or args.checkpoint_interval < 1 or args.retries < 1:
        raise ValueError("--download-workers, --checkpoint-interval e --retries devem ser positivos.")

    latitude, longitude = target_grid_latlon(radar_config, radar_shape[1], radar_shape[2])
    listing_cache: dict[str, list[tuple]] = {}
    staging = partial / "downloads"
    staging.mkdir(exist_ok=True)
    for stale in staging.glob("*.nc"):
        stale.unlink()
    failures = 0
    processed_since_checkpoint = 0
    started = time.monotonic()
    sampler: AbiSampler | None = None

    def select(index: int):
        radar_time = radar_times[index]
        candidates = scenes_near(filesystem, config, radar_time, listing_cache)
        return select_causal_scenes_by_channel(
            candidates, channels, radar_time,
            float(config["temporal_alignment"]["maximum_scene_age_minutes"]),
        )

    def record(index: int, selected, values: list[np.ndarray] | None, error: Exception | None = None) -> None:
        nonlocal failures, processed_since_checkpoint
        radar_time = radar_times[index]
        if error is not None:
            failures += 1
            print(f"[{year}] frame {index} falhou: {error}", flush=True)
            return
        if any(item is None for item in selected):
            completed[index] = 1
        else:
            for channel_index, (scene_start, scene_end, remote) in enumerate(selected):
                frames[index, :, :, channel_index] = values[channel_index]
                source_times[index, channel_index] = scene_end.isoformat()
                with (partial / "source_manifest.jsonl").open("a", encoding="utf-8") as manifest:
                    manifest.write(json.dumps({
                        "frame": int(index), "radar_timestamp_utc": radar_time.isoformat(), "channel": channels[channel_index],
                        "scene_start_timestamp_utc": scene_start.isoformat(), "scene_end_timestamp_utc": scene_end.isoformat(),
                        "remote": str(remote), "age_minutes": (radar_time - scene_end).total_seconds() / 60.0,
                    }, ensure_ascii=False) + "\n")
            if all(np.isfinite(value).all() for value in values):
                available[index] = 1
                completed[index] = 1
            else:
                frames[index] = np.nan
                completed[index] = 1
        processed_since_checkpoint += 1

    # Build and validate the reusable spatial lookup from the first available scene.
    remaining: list[tuple[int, list]] = []
    for index in pending:
        selected = select(index)
        if any(item is None for item in selected):
            record(index, selected, [])
            continue
        if sampler is None:
            scene_start, scene_end, remote = selected[0]
            local = staging / f"{index:06d}-{Path(remote).name}"
            try:
                download_remote(filesystem, remote, local, args.retries)
                sampler = build_abi_sampler(local, latitude, longitude)
                values = [read_and_reproject_scene_with_sampler(local, sampler)]
                for _, _, other_remote in selected[1:]:
                    other_local = staging / f"{index:06d}-{Path(other_remote).name}"
                    try:
                        download_remote(filesystem, other_remote, other_local, args.retries)
                        values.append(read_and_reproject_scene_with_sampler(other_local, sampler))
                    finally:
                        other_local.unlink(missing_ok=True)
                record(index, selected, values)
            except Exception as error:
                record(index, selected, None, error)
            finally:
                local.unlink(missing_ok=True)
        else:
            remaining.append((index, selected))

    if sampler is not None and remaining:
        def download_task(index: int, selected) -> tuple[int, list, list[Path]]:
            local_paths = []
            try:
                for channel_index, (_, _, remote) in enumerate(selected):
                    local = staging / f"{index:06d}-{channel_index}-{Path(remote).name}"
                    local_paths.append(download_remote(filesystem, remote, local, args.retries))
                return index, selected, local_paths
            except Exception:
                for path in local_paths:
                    path.unlink(missing_ok=True)
                raise

        iterator = iter(remaining)
        futures: dict[Future, tuple[int, list]] = {}
        with ThreadPoolExecutor(max_workers=args.download_workers, thread_name_prefix="goes-download") as executor:
            def submit_one() -> bool:
                try:
                    index, selected = next(iterator)
                except StopIteration:
                    return False
                futures[executor.submit(download_task, index, selected)] = (index, selected)
                return True

            for _ in range(args.download_workers * 2):
                if not submit_one():
                    break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    fallback_index, fallback_selected = futures.pop(future)
                    try:
                        index, selected, local_paths = future.result()
                        try:
                            values = [read_and_reproject_scene_with_sampler(path, sampler) for path in local_paths]
                            record(index, selected, values)
                        finally:
                            for path in local_paths:
                                path.unlink(missing_ok=True)
                    except Exception as error:
                        record(fallback_index, fallback_selected, None, error)
                    submit_one()
                    if processed_since_checkpoint >= args.checkpoint_interval:
                        checkpoint(partial, frames, available, source_times, completed)
                        report_progress(year, int((requested & completed.astype(bool)).sum()), int(requested.sum()), int(available.sum()), failures, started)
                        processed_since_checkpoint = 0

    checkpoint(partial, frames, available, source_times, completed)
    pending_after = int((requested & ~completed.astype(bool)).sum())
    if pending_after:
        raise RuntimeError(f"[{year}] {pending_after} frames pendentes após falhas; execute novamente com --resume.")
    selected_count = int(available.sum())
    output_metadata = {
        "dataset_contract": "goes16_abi_audited_v2",
        "build_mode": "streaming_s3_resumable",
        "year": year,
        "shape": list(contract["shape"]),
        "dtype": "float32",
        "frames_file": "goes_frames.dat",
        "availability_file": "goes_available.npy",
        "source_timestamps_file": "goes_source_timestamps.npy",
        "source_timestamp_semantics": "scene end UTC",
        "source_manifest_file": "source_manifest.jsonl",
        "channels": channels,
        "normalization": config["storage"]["normalization"],
        "radar_timestamps_file": "../radar_timestamps.npy",
        "radar_timestamps_sha256": digest_timestamps(raw_timestamps),
        "causality": config["temporal_alignment"],
        "target_grid": config["target_grid"],
        "goes_config": config,
        "goes_config_sha256": config_digest(args.goes_config),
        "radar_config_sha256": config_digest(args.radar_config),
        "coverage_scope": scope_metadata(args),
        "source_manifest": {
            "selected_frames": selected_count,
            "available_frames": selected_count,
            "read_failures": failures,
            "storage_policy": "Full Disk NetCDF is downloaded to a bounded staging directory and removed after reprojection.",
        },
        "performance": {
            "download_workers": args.download_workers,
            "checkpoint_interval": args.checkpoint_interval,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    (partial / "metadata.json").write_text(json.dumps(output_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del frames
    shutil.rmtree(staging)
    lock_handle.close()
    lock_path.unlink(missing_ok=True)
    publish(partial, final, args.overwrite)
    print(f"[{year}] GOES streaming concluído | disponíveis={selected_count}/{len(available)} | destino={final}", flush=True)


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start or args.resume and args.overwrite:
        raise ValueError("Intervalo de anos inválido ou flags --resume/--overwrite incompatíveis.")
    if (args.start_date is None) != (args.end_date is None) or (args.start_date and args.end_date < args.start_date):
        raise ValueError("Informe --start-date e --end-date juntos, em ordem crescente.")
    if args.download_workers < 1 or args.checkpoint_interval < 1 or args.retries < 1:
        raise ValueError("--download-workers, --checkpoint-interval e --retries devem ser positivos.")
    try:
        import s3fs
    except ImportError as error:
        raise RuntimeError("Streaming GOES requer s3fs.") from error
    config, radar_config = load_goes_config(args.goes_config), load_capture_config(args.radar_config)
    filesystem = s3fs.S3FileSystem(anon=True)
    for year in range(args.year_start, args.year_end + 1):
        build_year(args, config, radar_config, filesystem, year)


if __name__ == "__main__":
    main()
