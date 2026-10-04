"""Select alert decision thresholds on validation, then report independent test performance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.cli.evaluate_forecast_records import named_path
from nowcasting.forecast_evaluation import align_records, categorical_with_decision_threshold, select_decision_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra limiares de alerta somente na validação.")
    parser.add_argument("--validation-forecast", action="append", type=named_path, required=True)
    parser.add_argument("--test-forecast", action="append", type=named_path, required=True)
    parser.add_argument("--observed-thresholds", default="1.25,6.25,12.5")
    parser.add_argument("--decision-thresholds", default="0,0.25,0.5,0.75,1,1.25,2,3,4,6.25,8,10,12.5,16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = align_records({name: pd.read_parquet(path) for name, path in dict(args.validation_forecast).items()})
    test = align_records({name: pd.read_parquet(path) for name, path in dict(args.test_forecast).items()})
    if set(validation) != set(test):
        raise ValueError("Validação e teste devem conter os mesmos nomes de experimento.")
    observed_thresholds = [float(value) for value in args.observed_thresholds.split(",")]
    candidates = [float(value) for value in args.decision_thresholds.split(",")]
    output = {"selection_split": "validation", "report_split": "test", "experiments": {}}
    for name in validation:
        output["experiments"][name] = {}
        for threshold in observed_thresholds:
            selected, validation_metric = select_decision_threshold(validation[name], threshold, candidates)
            valid_test = test[name].loc[test[name]["is_observed"]]
            raw = categorical_with_decision_threshold(valid_test["predicted_mm_15min"].to_numpy(float), valid_test["observed_mm_15min"].to_numpy(float), threshold, threshold)
            calibrated = categorical_with_decision_threshold(valid_test["predicted_mm_15min"].to_numpy(float), valid_test["observed_mm_15min"].to_numpy(float), threshold, selected)
            output["experiments"][name][f"{threshold:g}"] = {"selected_decision_threshold": selected,
                                                                  "validation_metric": validation_metric,
                                                                  "test_raw": raw, "test_calibrated": calibrated}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Calibração de alerta salva em {args.output}")


if __name__ == "__main__":
    main()
