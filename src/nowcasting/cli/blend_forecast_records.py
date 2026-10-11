"""Select a horizon-wise validation ensemble and export canonical records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.forecast_ensemble import blend_horizons, select_global_alert_weight, select_horizon_weights
from nowcasting.forecast_evaluation import evaluate_records


def parse_weights(value: str) -> list[float]:
    weights = [float(item) for item in value.split(",")]
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise argparse.ArgumentTypeError("--candidate-weights deve conter valores entre 0 e 1.")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seleciona, na validação, pesos para combinar dois forecast records.")
    parser.add_argument("--validation-baseline", type=Path, required=True)
    parser.add_argument("--validation-candidate", type=Path, required=True)
    parser.add_argument("--test-baseline", type=Path, required=True)
    parser.add_argument("--test-candidate", type=Path, required=True)
    parser.add_argument("--candidate-weights", type=parse_weights, default=parse_weights("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"))
    parser.add_argument("--selection-objective", choices=("horizon-mae", "horizon-rmse", "municipal-csi"), default="horizon-mae")
    parser.add_argument("--selection-metric", choices=("mae", "rmse"), default=None,
                        help="Alias legado para --selection-objective horizon-mae ou horizon-rmse.")
    parser.add_argument("--thresholds", default="1.25,6.25,12.5")
    parser.add_argument("--decision-thresholds", default="0,0.25,0.5,0.75,1,1.25,2,3,4,6.25,8,10,12.5,16")
    parser.add_argument("--minimum-pod", type=float, default=None,
                        help="Aplicável a municipal-csi; descarte pesos com POD municipal abaixo deste valor.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    validation_baseline = pd.read_parquet(args.validation_baseline)
    validation_candidate = pd.read_parquet(args.validation_candidate)
    test_baseline = pd.read_parquet(args.test_baseline)
    test_candidate = pd.read_parquet(args.test_candidate)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    objective = args.selection_objective
    if args.selection_metric is not None:
        if objective != "horizon-mae":
            raise ValueError("--selection-metric não pode ser combinado com --selection-objective.")
        objective = f"horizon-{args.selection_metric}"
    if objective == "municipal-csi":
        selected_weight, candidates = select_global_alert_weight(
            validation_baseline, validation_candidate, args.candidate_weights,
            observed_thresholds=thresholds,
            decision_thresholds=[float(value) for value in args.decision_thresholds.split(",")],
            minimum_pod=args.minimum_pod,
        )
        horizons = {int(value) for value in validation_baseline["horizon"].unique()}
        weights = {horizon: selected_weight for horizon in sorted(horizons)}
    else:
        metric = objective.removeprefix("horizon-")
        weights, candidates = select_horizon_weights(
            validation_baseline, validation_candidate, args.candidate_weights, metric=metric)
    validation = blend_horizons(validation_baseline, validation_candidate, weights, experiment_id=args.experiment_id)
    test = blend_horizons(test_baseline, test_candidate, weights, experiment_id=args.experiment_id)
    for output, records in ((args.validation_output, validation), (args.test_output, test)):
        output.parent.mkdir(parents=True, exist_ok=True)
        records.to_parquet(output, index=False)
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
        "selection_objective": objective,
        "minimum_pod": args.minimum_pod,
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
