"""Audit coverage and causality of GOES tensors aligned to radar timestamps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from nowcasting.goes16 import ensure_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita artefatos GOES-16 já alinhados ao radar.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    return parser.parse_args()


def audit_year(root: Path, year: int) -> tuple[dict, list[dict]]:
    year_dir = root / f"year={year}"
    metadata = json.loads((year_dir / "goes16" / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("source_timestamp_semantics") != "scene end UTC":
        raise ValueError(
            f"{year}: auditoria estrita requer source_timestamp_semantics='scene end UTC'."
        )
    radar_times = [ensure_utc(value) for value in np.load(year_dir / "radar_timestamps.npy", allow_pickle=False)]
    available = np.load(year_dir / "goes16" / metadata["availability_file"], allow_pickle=False).astype(bool)
    source = np.load(year_dir / "goes16" / metadata["source_timestamps_file"], allow_pickle=False)
    if len(radar_times) != len(available) or source.shape[0] != len(radar_times):
        raise ValueError(f"{year}: comprimentos GOES e radar incompatíveis.")
    rows: list[dict] = []
    causal_errors = 0
    for index, radar_time in enumerate(radar_times):
        source_values = [ensure_utc(value) for value in source[index] if value]
        causal = all(value <= radar_time for value in source_values)
        causal_errors += int(not causal)
        rows.append({"year": year, "timestamp_utc": radar_time.isoformat(), "available": int(available[index]), "causal": int(causal), "source_end_timestamps_utc": ";".join(value.isoformat() for value in source_values)})
    scope = metadata.get("coverage_scope", {})
    start, end = scope.get("start_date"), scope.get("end_date")
    scope_mask = np.array([
        (start is None or timestamp.date().isoformat() >= start)
        and (end is None or timestamp.date().isoformat() <= end)
        for timestamp in radar_times
    ], dtype=bool)
    scoped_frames = int(scope_mask.sum())
    scoped_available = int(available[scope_mask].sum())
    return ({"year": year, "radar_frames": len(radar_times), "available_frames": int(available.sum()), "availability_fraction": float(available.mean()), "coverage_scope": scope, "scoped_radar_frames": scoped_frames, "scoped_available_frames": scoped_available, "scoped_availability_fraction": scoped_available / scoped_frames if scoped_frames else 0.0, "outside_scope_available_frames": int(available[~scope_mask].sum()), "causality_errors": causal_errors, "channels": metadata["channels"], "source_manifest": metadata["source_manifest"]}, rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, rows = [], []
    for year in range(args.year_start, args.year_end + 1):
        summary, year_rows = audit_year(args.dataset_root, year)
        summaries.append(summary)
        rows.extend(year_rows)
    with (args.output_dir / "coverage.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("year", "timestamp_utc", "available", "causal", "source_end_timestamps_utc"))
        writer.writeheader()
        writer.writerows(rows)
    result = {"years": summaries, "available_frames": sum(item["available_frames"] for item in summaries), "radar_frames": sum(item["radar_frames"] for item in summaries), "causality_errors": sum(item["causality_errors"] for item in summaries)}
    result["availability_fraction"] = result["available_frames"] / result["radar_frames"] if result["radar_frames"] else 0.0
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Auditoria GOES concluída | cobertura={result['availability_fraction']:.2%} | erros causais={result['causality_errors']} | saída={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
