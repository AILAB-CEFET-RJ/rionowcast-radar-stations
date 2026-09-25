from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from nowcasting.radar_capture import parse_capture_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seleciona eventos fortes do AlertaRio com sequência completa de "
            "capturas do radar para demonstrações operacionais."
        )
    )
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--alertario-root", type=Path, required=True)
    parser.add_argument("--station-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-threshold", type=float, default=12.5)
    parser.add_argument("--max-m15", type=float, default=175.0)
    parser.add_argument("--aggregate-minutes", type=int, default=15)
    parser.add_argument("--min-frames-per-bucket", type=int, default=3)
    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--t-out", type=int, default=5)
    parser.add_argument("--min-separation-minutes", type=int, default=180)
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args()


def collect_capture_buckets(
    capture_root: Path,
    aggregate_minutes: int,
    min_frames_per_bucket: int,
) -> dict[pd.Timestamp, list[Path]]:
    buckets: dict[pd.Timestamp, list[Path]] = defaultdict(list)
    paths = sorted(capture_root.rglob("*.png"))
    for index, path in enumerate(paths, start=1):
        try:
            timestamp = parse_capture_timestamp(path)
        except ValueError:
            continue
        timestamp = timestamp.replace(
            minute=(timestamp.minute // aggregate_minutes) * aggregate_minutes,
            second=0,
            microsecond=0,
        )
        # The legacy memmap builder stores filename timestamps as UTC-naive values.
        # This explicit conversion preserves that convention during operational demos.
        buckets[pd.Timestamp(timestamp, tz="UTC")].append(path)
        if index % 10000 == 0 or index == len(paths):
            print(f"Capturas lidas: {index}/{len(paths)}", flush=True)
    return {
        timestamp: sorted(paths)
        for timestamp, paths in buckets.items()
        if len(paths) >= min_frames_per_bucket
    }


def load_alertario_events(
    alertario_root: Path,
    mapped_station_ids: set[int],
    event_threshold: float,
    max_m15: float,
) -> pd.DataFrame:
    events = []
    files = sorted(alertario_root.rglob("*.parquet"))
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=["estacao_id", "estacao", "dia_utc", "m15"])
        frame = frame.loc[frame["estacao_id"].isin(mapped_station_ids)].copy()
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["dia_utc"], utc=True, errors="coerce").dt.floor("15min")
        frame["m15"] = pd.to_numeric(frame["m15"], errors="coerce")
        frame = frame.loc[
            frame["timestamp"].notna()
            & frame["m15"].ge(event_threshold)
            & frame["m15"].le(max_m15),
            ["timestamp", "estacao_id", "estacao", "m15"],
        ]
        if not frame.empty:
            events.append(frame)
        if index % 25 == 0 or index == len(files):
            print(f"Parquets lidos: {index}/{len(files)}", flush=True)

    if not events:
        return pd.DataFrame(columns=["timestamp", "station_id", "station_name", "m15"])
    data = pd.concat(events, ignore_index=True)
    data = data.groupby(["timestamp", "estacao_id", "estacao"], as_index=False)["m15"].max()
    data = data.sort_values(["timestamp", "m15"], ascending=[True, False])
    data = data.drop_duplicates("timestamp", keep="first")
    return data.rename(columns={"estacao_id": "station_id", "estacao": "station_name"})


def build_sequence(
    event_timestamp: pd.Timestamp,
    buckets: dict[pd.Timestamp, list[Path]],
    aggregate_minutes: int,
    t_in: int,
    t_out: int,
) -> dict[str, list[pd.Timestamp]] | None:
    interval = pd.Timedelta(minutes=aggregate_minutes)
    input_timestamps = [event_timestamp - interval * offset for offset in range(t_in, 0, -1)]
    target_timestamps = [event_timestamp + interval * offset for offset in range(t_out)]
    required = input_timestamps + target_timestamps
    if any(timestamp not in buckets for timestamp in required):
        return None
    return {"input": input_timestamps, "target": target_timestamps}


def select_events(
    events: pd.DataFrame,
    buckets: dict[pd.Timestamp, list[Path]],
    *,
    aggregate_minutes: int,
    t_in: int,
    t_out: int,
    min_separation_minutes: int,
    limit: int,
) -> list[dict]:
    selected: list[dict] = []
    min_separation = pd.Timedelta(minutes=min_separation_minutes)
    for event in events.sort_values("m15", ascending=False).itertuples(index=False):
        sequence = build_sequence(event.timestamp, buckets, aggregate_minutes, t_in, t_out)
        if sequence is None:
            continue
        if any(abs(event.timestamp - chosen["event_timestamp"]) < min_separation for chosen in selected):
            continue
        selected.append(
            {
                "event_timestamp": event.timestamp,
                "station_id": int(event.station_id),
                "station_name": str(event.station_name),
                "m15": float(event.m15),
                "input_timestamps": sequence["input"],
                "target_timestamps": sequence["target"],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def serializable_event(event: dict, capture_root: Path, buckets: dict[pd.Timestamp, list[Path]]) -> dict:
    return {
        "event_timestamp_utc": event["event_timestamp"].isoformat(),
        "station_id": event["station_id"],
        "station_name": event["station_name"],
        "m15_mm_15min": event["m15"],
        "input": [
            {
                "timestamp_utc": timestamp.isoformat(),
                "png_files": [str(path.relative_to(capture_root)) for path in buckets[timestamp]],
            }
            for timestamp in event["input_timestamps"]
        ],
        "target": [
            {
                "timestamp_utc": timestamp.isoformat(),
                "png_files": [str(path.relative_to(capture_root)) for path in buckets[timestamp]],
            }
            for timestamp in event["target_timestamps"]
        ],
    }


def main() -> None:
    args = parse_args()
    if min(args.aggregate_minutes, args.min_frames_per_bucket, args.t_in, args.t_out, args.limit) <= 0:
        raise ValueError("Parâmetros temporais e --limit devem ser positivos.")
    mapping = pd.read_csv(args.station_mapping, usecols=["station_id"])
    station_ids = set(mapping["station_id"].dropna().astype(int))
    buckets = collect_capture_buckets(
        args.capture_root, args.aggregate_minutes, args.min_frames_per_bucket
    )
    events = load_alertario_events(
        args.alertario_root, station_ids, args.event_threshold, args.max_m15
    )
    selected = select_events(
        events,
        buckets,
        aggregate_minutes=args.aggregate_minutes,
        t_in=args.t_in,
        t_out=args.t_out,
        min_separation_minutes=args.min_separation_minutes,
        limit=args.limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [serializable_event(event, args.capture_root, buckets) for event in selected]
    (args.output_dir / "events.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "event_timestamp_utc": event["event_timestamp"].isoformat(),
                "station_id": event["station_id"],
                "station_name": event["station_name"],
                "m15_mm_15min": event["m15"],
                "input_start_utc": event["input_timestamps"][0].isoformat(),
                "input_end_utc": event["input_timestamps"][-1].isoformat(),
                "target_start_utc": event["target_timestamps"][0].isoformat(),
                "target_end_utc": event["target_timestamps"][-1].isoformat(),
            }
            for event in selected
        ]
    ).to_csv(args.output_dir / "events.csv", index=False)
    summary = {
        "capture_root": str(args.capture_root),
        "alertario_root": str(args.alertario_root),
        "mapped_stations": len(station_ids),
        "valid_capture_buckets": len(buckets),
        "event_threshold_mm_15min": args.event_threshold,
        "max_m15_mm_15min": args.max_m15,
        "input_frames": args.t_in,
        "target_frames": args.t_out,
        "selected_events": len(selected),
        "timestamp_convention": "PNG filename timestamps are interpreted as UTC to match the legacy radar memmaps.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Eventos elegíveis={len(events)} | selecionados={len(selected)} | saída={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
