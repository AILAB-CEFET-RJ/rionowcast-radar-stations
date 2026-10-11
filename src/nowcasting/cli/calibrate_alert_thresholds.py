"""Select alert decision thresholds on validation, then report independent test performance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.cli.evaluate_forecast_records import named_path
from nowcasting.forecast_evaluation import align_records, categorical_records, select_decision_threshold, validate_records


def load_forecasts(named_paths: list[tuple[str, Path]]) -> dict[str, pd.DataFrame]:
    """Load one or more canonical record files, aligning only when needed."""
    records = {name: pd.read_parquet(path) for name, path in named_paths}
    if len(records) == 1:
        name, frame = next(iter(records.items()))
        return {name: validate_records(frame, name)}
    return align_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra limiares de alerta somente na validação.")
    parser.add_argument("--validation-forecast", action="append", type=named_path, required=True)
    parser.add_argument("--test-forecast", action="append", type=named_path, required=True)
    parser.add_argument("--observed-thresholds", default="1.25,6.25,12.5")
    parser.add_argument("--decision-thresholds", default="0,0.25,0.5,0.75,1,1.25,2,3,4,6.25,8,10,12.5,16")
    parser.add_argument("--scope", choices=("station", "municipal"), default="station",
                        help="Unidade usada para selecionar e reportar o limiar de alerta.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = load_forecasts(args.validation_forecast)
    test = load_forecasts(args.test_forecast)
    if set(validation) != set(test):
        raise ValueError("Validação e teste devem conter os mesmos nomes de experimento.")
    observed_thresholds = [float(value) for value in args.observed_thresholds.split(",")]
    candidates = [float(value) for value in args.decision_thresholds.split(",")]
    output = {"selection_split": "validation", "report_split": "test", "scope": args.scope, "experiments": {}}
    for name in validation:
        output["experiments"][name] = {}
        for threshold in observed_thresholds:
            selected, validation_metric = select_decision_threshold(
                validation[name], threshold, candidates, scope=args.scope)
            raw = categorical_records(test[name], threshold, threshold, scope=args.scope)
            calibrated = categorical_records(test[name], threshold, selected, scope=args.scope)
            output["experiments"][name][f"{threshold:g}"] = {"selected_decision_threshold": selected,
                                                                  "validation_metric": validation_metric,
                                                                  "test_raw": raw, "test_calibrated": calibrated}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Calibração de alerta salva em {args.output}")


if __name__ == "__main__":
    main()
