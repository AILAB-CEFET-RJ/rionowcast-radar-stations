"""Select a horizon-wise validation ensemble and export canonical records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.forecast_ensemble import blend_horizons, select_horizon_weights
from nowcasting.forecast_evaluation import evaluate_records


def parse_weights(value: str) -> list[float]:
    weights = [float(item) for item in value.split(",")]
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise argparse.ArgumentTypeError("--candidate-weights deve conter valores entre 0 e 1.")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seleciona, na validação, um peso por horizonte para combinar dois forecast records.")
    parser.add_argument("--validation-baseline", type=Path, required=True)
    parser.add_argument("--validation-candidate", type=Path, required=True)
    parser.add_argument("--test-baseline", type=Path, required=True)
    parser.add_argument("--test-candidate", type=Path, required=True)
    parser.add_argument("--candidate-weights", type=parse_weights, default=parse_weights("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"))
    parser.add_argument("--selection-metric", choices=("mae", "rmse"), default="mae")
    parser.add_argument("--thresholds", default="1.25,6.25,12.5")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    validation_baseline = pd.read_parquet(args.validation_baseline)
    validation_candidate = pd.read_parquet(args.validation_candidate)
    test_baseline = pd.read_parquet(args.test_baseline)
    test_candidate = pd.read_parquet(args.test_candidate)
    weights, candidates = select_horizon_weights(
        validation_baseline, validation_candidate, args.candidate_weights, metric=args.selection_metric)
    validation = blend_horizons(validation_baseline, validation_candidate, weights, experiment_id=args.experiment_id)
    test = blend_horizons(test_baseline, test_candidate, weights, experiment_id=args.experiment_id)
    for output, records in ((args.validation_output, validation), (args.test_output, test)):
        output.parent.mkdir(parents=True, exist_ok=True)
        records.to_parquet(output, index=False)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    summary = {
        "selection_split": "validation",
        "report_split": "test",
        "experiment_id": args.experiment_id,
        "sources": {
            "validation_baseline": str(args.validation_baseline),
            "validation_candidate": str(args.validation_candidate),
            "test_baseline": str(args.test_baseline),
            "test_candidate": str(args.test_candidate),
        },
        "selection_metric": args.selection_metric,
        "candidate_weights": args.candidate_weights,
        "selected_candidate_weight_by_horizon": weights,
        "validation_candidates": candidates,
        "validation_metrics": evaluate_records(validation, thresholds),
        "test_metrics": evaluate_records(test, thresholds),
        "validation_output": str(args.validation_output),
        "test_output": str(args.test_output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    formatted = ", ".join(f"T+{horizon * 15}={weight:g}" for horizon, weight in weights.items())
    print(f"Ensemble exportado | pesos da candidata: {formatted} | saída={args.test_output}")


if __name__ == "__main__":
    main()
