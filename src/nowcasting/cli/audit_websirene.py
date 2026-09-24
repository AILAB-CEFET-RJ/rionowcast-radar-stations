#!/usr/bin/env python3
"""Audit WebSirene observations without modifying the raw source data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.websirene_qc import audit_observations, load_config, station_summary, write_qc_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita observacoes WebSirene e gera uma whitelist preliminar por estacao."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Raiz com station_id=<id>/year=<ano>/data.parquet.")
    parser.add_argument("--processed-root", type=Path, required=True, help="Destino dos Parquets auditados.")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Destino de tabelas e relatorios da auditoria.")
    parser.add_argument("--config", type=Path, help="JSON com limiares e criterios de aprovacao.")
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--station-start", type=int, default=1)
    parser.add_argument("--station-end", type=int, default=83)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescrever Parquets e relatorios da versao de auditoria indicada.",
    )
    return parser.parse_args()


def status_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["qc_status"].value_counts()
    return {status: int(counts.get(status, 0)) for status in ("accepted", "suspect", "rejected")}


def audit_file(source: Path, destination: Path, station_id: int, year: int, config: dict) -> pd.DataFrame:
    audited = audit_observations(pd.read_parquet(source), station_id, year, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audited.to_parquet(destination, index=False)
    return audited


def write_flagged_rows(path: Path, frame: pd.DataFrame, include_header: bool) -> bool:
    flagged = frame.loc[frame["qc_status"] != "accepted"].copy()
    if flagged.empty:
        if include_header:
            path.parent.mkdir(parents=True, exist_ok=True)
            flagged.to_csv(path, index=False, encoding="utf-8")
        return include_header
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(path, index=False, mode="a", header=include_header, encoding="utf-8")
    return False


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start or args.station_end < args.station_start:
        raise ValueError("Os limites finais devem ser maiores ou iguais aos iniciais.")
    if args.config and not args.config.is_file():
        raise FileNotFoundError(f"Configuracao nao encontrada: {args.config}")

    config = load_config(args.config)
    if args.processed_root.exists() and any(args.processed_root.rglob("data.parquet")) and not args.overwrite:
        raise FileExistsError(
            f"Ja existem Parquets auditados em {args.processed_root}. "
            "Use outro diretorio de versao ou --overwrite."
        )
    if args.analysis_dir.exists() and any(args.analysis_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"O diretorio de analise ja contem arquivos: {args.analysis_dir}. "
            "Use outro diretorio de versao ou --overwrite."
        )
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    flagged_path = args.analysis_dir / "flagged_observations.csv"
    if flagged_path.exists():
        flagged_path.unlink()

    summaries: list[pd.DataFrame] = []
    year_rows: list[dict[str, int]] = []
    flagged_header = True
    files_processed = 0
    total_status_counts = {"accepted": 0, "suspect": 0, "rejected": 0}
    for station_id in range(args.station_start, args.station_end + 1):
        station_frames: list[pd.DataFrame] = []
        for year in range(args.year_start, args.year_end + 1):
            source = args.input_root / f"station_id={station_id}" / f"year={year}" / "data.parquet"
            if not source.is_file():
                continue
            destination = args.processed_root / f"station_id={station_id}" / f"year={year}" / "data.parquet"
            audited = audit_file(source, destination, station_id, year, config)
            counts = status_counts(audited)
            for status, count in counts.items():
                total_status_counts[status] += count
            year_rows.append({"station_id": station_id, "year": year, "observations": len(audited), **counts})
            flagged_header = write_flagged_rows(flagged_path, audited, flagged_header)
            station_frames.append(audited)
            files_processed += 1
            print(
                f"station={station_id} year={year} observations={len(audited)} "
                f"accepted={counts['accepted']} suspect={counts['suspect']} rejected={counts['rejected']}",
                flush=True,
            )
        if station_frames:
            summaries.append(station_summary(pd.concat(station_frames, ignore_index=True), config))

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    yearly = pd.DataFrame(year_rows)
    summary.to_csv(args.analysis_dir / "station_summary.csv", index=False, encoding="utf-8")
    yearly.to_csv(args.analysis_dir / "year_summary.csv", index=False, encoding="utf-8")
    summary.loc[summary["station_status"] == "approved"].to_csv(
        args.analysis_dir / "station_whitelist.csv", index=False, encoding="utf-8"
    )
    write_qc_report(args.analysis_dir / "report.md", summary, config)
    (args.analysis_dir / "summary.json").write_text(
        json.dumps(
            {
                "qc_version": config["version"],
                "input_root": str(args.input_root),
                "processed_root": str(args.processed_root),
                "files_processed": files_processed,
                "stations_audited": int(len(summary)),
                "stations_approved": int((summary["station_status"] == "approved").sum()) if not summary.empty else 0,
                "observation_status_counts": total_status_counts,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Auditoria concluida: {args.analysis_dir}", flush=True)


if __name__ == "__main__":
    main()
