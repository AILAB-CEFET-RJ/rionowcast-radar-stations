#!/usr/bin/env python3
"""Compare audited WebSirene readings with nearby AlertaRio stations."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nowcasting.websirene_qc import nearest_reference_stations, paired_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara observacoes WebSirene aprovadas com estacoes AlertaRio proximas."
    )
    parser.add_argument("--websirene-root", type=Path, required=True)
    parser.add_argument("--alertario-root", type=Path, required=True)
    parser.add_argument("--websirene-mapping", type=Path, required=True)
    parser.add_argument("--alertario-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--max-distance-km", type=float, default=5.0)
    parser.add_argument("--min-pairs", type=int, default=100)
    return parser.parse_args()


def load_websirene(root: Path, station_ids: set[int], year_start: int, year_end: int) -> pd.DataFrame:
    frames = []
    for station_id in sorted(station_ids):
        for year in range(year_start, year_end + 1):
            path = root / f"station_id={station_id}" / f"year={year}" / "data.parquet"
            if not path.is_file():
                continue
            frame = pd.read_parquet(path, columns=["timestamp_15min", "m15", "qc_status"])
            frame = frame.loc[frame["qc_status"] == "accepted", ["timestamp_15min", "m15"]].copy()
            frame["station_id"] = station_id
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["station_id", "timestamp_15min", "websirene_m15"])
    observations = pd.concat(frames, ignore_index=True)
    observations["timestamp_15min"] = pd.to_datetime(observations["timestamp_15min"], utc=True)
    observations = observations.groupby(["station_id", "timestamp_15min"], as_index=False)["m15"].max()
    return observations.rename(columns={"m15": "websirene_m15"})


def load_alertario(root: Path, station_ids: set, year_start: int, year_end: int) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("*.parquet")):
        frame = pd.read_parquet(path, columns=["estacao_id", "dia_utc", "m15"])
        frame = frame.loc[frame["estacao_id"].isin(station_ids)].copy()
        if frame.empty:
            continue
        frame["timestamp_15min"] = pd.to_datetime(frame["dia_utc"], utc=True, errors="coerce").dt.floor("15min")
        frame = frame.loc[
            frame["timestamp_15min"].dt.year.between(year_start, year_end) & frame["m15"].notna(),
            ["estacao_id", "timestamp_15min", "m15"],
        ]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["station_id", "timestamp_15min", "reference_m15"])
    observations = pd.concat(frames, ignore_index=True)
    observations = observations.groupby(["estacao_id", "timestamp_15min"], as_index=False)["m15"].max()
    return observations.rename(columns={"estacao_id": "station_id", "m15": "reference_m15"})


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("--year-end deve ser maior ou igual a --year-start.")
    web_mapping = pd.read_csv(args.websirene_mapping)
    alert_mapping = pd.read_csv(args.alertario_mapping)
    pairs = nearest_reference_stations(web_mapping, alert_mapping, args.max_distance_km)
    if pairs.empty:
        raise RuntimeError("Nenhum par WebSirene-AlertaRio encontrado no raio informado.")

    web = load_websirene(args.websirene_root, set(pairs["websirene_station_id"]), args.year_start, args.year_end)
    reference = load_alertario(args.alertario_root, set(pairs["alertario_station_id"]), args.year_start, args.year_end)
    results = []
    for pair in pairs.itertuples(index=False):
        web_values = web.loc[web["station_id"] == pair.websirene_station_id].drop(columns="station_id")
        ref_values = reference.loc[reference["station_id"] == pair.alertario_station_id].drop(columns="station_id")
        aligned = web_values.merge(ref_values, on="timestamp_15min", how="inner")
        metrics = paired_metrics(aligned)
        results.append({
            "websirene_station_id": pair.websirene_station_id,
            "alertario_station_id": pair.alertario_station_id,
            "distance_km": pair.distance_km,
            **metrics,
            "meets_min_pairs": metrics["n"] >= args.min_pairs,
        })
    comparison = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "nearest_station_pairs.csv", index=False)
    comparison.to_csv(args.output_dir / "station_comparison.csv", index=False)
    print(f"Comparacao concluida: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
