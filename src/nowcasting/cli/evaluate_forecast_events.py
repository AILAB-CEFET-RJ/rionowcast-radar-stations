"""Evaluate station precipitation events from canonical forecast records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.cli.evaluate_forecast_records import named_path
from nowcasting.forecast_evaluation import align_records, event_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia detecção e antecedência de eventos por estação.")
    parser.add_argument("--forecast", action="append", type=named_path, required=True)
    parser.add_argument("--thresholds", default="6.25,12.5")
    parser.add_argument("--event-gap-minutes", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.event_gap_minutes <= 0:
        raise ValueError("--event-gap-minutes deve ser positivo.")
    thresholds = [float(value) for value in args.thresholds.split(",")]
    aligned = align_records({name: pd.read_parquet(path) for name, path in dict(args.forecast).items()})
    report = {name: {f"{threshold:g}": event_metrics(records, threshold, gap_minutes=args.event_gap_minutes)
                     for threshold in thresholds}
              for name, records in aligned.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"thresholds_mm_15min": thresholds, "experiments": report}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Avaliação por evento salva em {args.output}")


if __name__ == "__main__":
    main()
