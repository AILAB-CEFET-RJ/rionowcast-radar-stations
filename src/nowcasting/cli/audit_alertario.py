#!/usr/bin/env python3
"""Profile original AlertaRio Parquets before target generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita cobertura e valores ausentes nos Parquets originais do AlertaRio."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sentinel", type=float, default=-99.99)
    parser.add_argument("--suspect-m15", type=float, default=50.0)
    parser.add_argument("--reject-m15", type=float, default=175.0)
    return parser.parse_args()


def summarize_file(frame: pd.DataFrame, sentinel: float, suspect: float, reject: float) -> pd.DataFrame:
    required = {"estacao_id", "dia_utc", "m15"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    data = frame.rename(columns={"estacao_id": "station_id", "dia_utc": "timestamp"}).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data["m15"] = pd.to_numeric(data["m15"], errors="coerce")
    data["year"] = data["timestamp"].dt.year.fillna(-1).astype(int)
    data = data.dropna(subset=["station_id"]).copy()
    data["station_id"] = data["station_id"].astype(int)
    data["invalid_timestamp"] = data["timestamp"].isna()
    data["missing_m15"] = data["m15"].isna()
    data["sentinel_m15"] = data["m15"].eq(sentinel)
    data["negative_m15"] = data["m15"].lt(0) & ~data["sentinel_m15"]
    data["valid_m15"] = data["m15"].notna() & data["m15"].ge(0)
    data["suspect_m15"] = data["m15"].gt(suspect)
    data["reject_m15"] = data["m15"].gt(reject)
    data["timestamp_15min"] = data["timestamp"].dt.floor("15min")
    data["duplicate_exact_timestamp"] = data.duplicated(["station_id", "timestamp"], keep=False)
    data["multiple_rows_15min"] = data.duplicated(["station_id", "timestamp_15min"], keep=False)

    rows = []
    for (station_id, year), group in data.groupby(["station_id", "year"], sort=True):
        valid = group.loc[group["valid_m15"], "m15"]
        rows.append({
            "station_id": station_id,
            "year": year,
            "rows": int(len(group)),
            "invalid_timestamp": int(group["invalid_timestamp"].sum()),
            "missing_m15": int(group["missing_m15"].sum()),
            "sentinel_m15": int(group["sentinel_m15"].sum()),
            "negative_m15": int(group["negative_m15"].sum()),
            "valid_m15": int(group["valid_m15"].sum()),
            "suspect_m15": int(group["suspect_m15"].sum()),
            "reject_m15": int(group["reject_m15"].sum()),
            "duplicate_exact_timestamp": int(group["duplicate_exact_timestamp"].sum()),
            "multiple_rows_15min": int(group["multiple_rows_15min"].sum()),
            "first_timestamp": group["timestamp"].min(),
            "last_timestamp": group["timestamp"].max(),
            "min_valid_m15": valid.min() if not valid.empty else None,
            "max_valid_m15": valid.max() if not valid.empty else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    files = sorted(args.input_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {args.input_root}")
    summaries = []
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=["estacao_id", "dia_utc", "m15"])
        summaries.append(summarize_file(frame, args.sentinel, args.suspect_m15, args.reject_m15))
        if index == 1 or index % 25 == 0 or index == len(files):
            print(f"files {index}/{len(files)}", flush=True)
    station_year = pd.concat(summaries, ignore_index=True).groupby(["station_id", "year"], as_index=False).agg(
        rows=("rows", "sum"), invalid_timestamp=("invalid_timestamp", "sum"),
        missing_m15=("missing_m15", "sum"), sentinel_m15=("sentinel_m15", "sum"),
        negative_m15=("negative_m15", "sum"), valid_m15=("valid_m15", "sum"),
        suspect_m15=("suspect_m15", "sum"), reject_m15=("reject_m15", "sum"),
        duplicate_exact_timestamp=("duplicate_exact_timestamp", "sum"),
        multiple_rows_15min=("multiple_rows_15min", "sum"),
        first_timestamp=("first_timestamp", "min"), last_timestamp=("last_timestamp", "max"),
        min_valid_m15=("min_valid_m15", "min"), max_valid_m15=("max_valid_m15", "max"),
    )
    station = station_year.groupby("station_id", as_index=False).agg(
        rows=("rows", "sum"), valid_m15=("valid_m15", "sum"), missing_m15=("missing_m15", "sum"),
        sentinel_m15=("sentinel_m15", "sum"), negative_m15=("negative_m15", "sum"),
        suspect_m15=("suspect_m15", "sum"), reject_m15=("reject_m15", "sum"),
        years=("year", "nunique"), first_timestamp=("first_timestamp", "min"),
        last_timestamp=("last_timestamp", "max"), max_valid_m15=("max_valid_m15", "max"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_year.to_csv(args.output_dir / "station_year_summary.csv", index=False)
    station.to_csv(args.output_dir / "station_summary.csv", index=False)
    totals = {column: int(station_year[column].sum()) for column in (
        "rows", "invalid_timestamp", "missing_m15", "sentinel_m15", "negative_m15",
        "valid_m15", "suspect_m15", "reject_m15", "duplicate_exact_timestamp", "multiple_rows_15min",
    )}
    (args.output_dir / "summary.json").write_text(json.dumps({
        "input_root": str(args.input_root), "files": len(files), "stations": int(len(station)),
        "years": sorted(station_year["year"].unique().tolist()), "sentinel": args.sentinel,
        "suspect_m15": args.suspect_m15, "reject_m15": args.reject_m15, "totals": totals,
        "duplicate_scope": "within each input file; multiple_rows_15min represents expected temporal aggregation, not necessarily duplicated raw records",
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Auditoria concluida: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
