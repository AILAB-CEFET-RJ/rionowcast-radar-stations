#!/usr/bin/env python3
"""Exporta os maiores acumulados de 15 minutos do WebSirene por ano.

Exemplo:
    nowcasting-export-websirene-maxima \\
        --input-root data/datasets/raw/websirene \\
        --output-dir outputs/analysis/websirene-maximos \\
        --year-start 2012 --year-end 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta os maiores valores de m15 do WebSirene por ano."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Raiz com station_id=<id>/year=<ano>/data.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Diretorio para os CSVs gerados.",
    )
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--station-start", type=int, default=1)
    parser.add_argument("--station-end", type=int, default=83)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def read_station_year(path: Path, station_id: int) -> pd.DataFrame | None:
    if not path.exists():
        return None

    dataframe = pd.read_parquet(path)
    required_columns = {"observation_datetime", "m15"}
    if not required_columns.issubset(dataframe.columns):
        return None

    columns = [
        column
        for column in ("id", "nome", "observation_datetime", "m15")
        if column in dataframe.columns
    ]
    result = dataframe.loc[:, columns].copy()
    result["station_id"] = station_id
    result["observation_datetime"] = pd.to_datetime(
        result["observation_datetime"], utc=True, errors="coerce"
    )
    return result.dropna(subset=["observation_datetime", "m15"])


def export_year(args: argparse.Namespace, year: int) -> None:
    frames: list[pd.DataFrame] = []
    for station_id in range(args.station_start, args.station_end + 1):
        path = args.input_root / f"station_id={station_id}" / f"year={year}" / "data.parquet"
        dataframe = read_station_year(path, station_id)
        if dataframe is not None:
            frames.append(dataframe)

    if not frames:
        print(f"{year}: sem dados")
        return

    year_dataframe = pd.concat(frames, ignore_index=True)
    output_columns = [
        column
        for column in ("station_id", "id", "nome", "observation_datetime", "m15")
        if column in year_dataframe.columns
    ]
    top_dataframe = (
        year_dataframe.sort_values("m15", ascending=False)
        .head(args.top_n)
        .loc[:, output_columns]
    )
    output_path = args.output_dir / f"maximos_m15_{year}.csv"
    top_dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"{year}: observacoes={len(year_dataframe)} max_m15={year_dataframe['m15'].max()}")
    print(f"{year}: salvo em {output_path}")


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("--year-end deve ser maior ou igual a --year-start.")
    if args.station_end < args.station_start:
        raise ValueError("--station-end deve ser maior ou igual a --station-start.")
    if args.top_n < 1:
        raise ValueError("--top-n deve ser maior que zero.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in range(args.year_start, args.year_end + 1):
        export_year(args, year)


if __name__ == "__main__":
    main()
