"""Evaluate station precipitation events from canonical forecast records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.cli.evaluate_forecast_records import named_path
from nowcasting.forecast_evaluation import align_records, event_metrics


def calibrated_threshold(calibration: dict, experiment: str, observed_threshold: float) -> float:
    """Read a threshold selected on validation for one experiment and event definition."""
    try:
        value = calibration["experiments"][experiment][f"{observed_threshold:g}"]["selected_decision_threshold"]
    except KeyError as error:
        raise ValueError(
            f"Calibração não contém limiar para experimento={experiment!r}, "
            f"observado={observed_threshold:g}."
        ) from error
    value = float(value)
    if value < 0:
        raise ValueError("Limiar de decisão calibrado não pode ser negativo.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia detecção, antecedência e falsos alertas por evento.")
    parser.add_argument("--forecast", action="append", type=named_path, required=True)
    parser.add_argument("--thresholds", default="6.25,12.5")
    parser.add_argument("--event-gap-minutes", type=int, default=15)
    parser.add_argument("--scope", choices=("station", "municipal", "both"), default="both",
                        help="Estação, máximo municipal entre estações, ou ambos.")
    parser.add_argument(
        "--calibration", type=Path,
        help="alert_calibration.json: reporta eventos brutos e com limiares escolhidos na validação.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.event_gap_minutes <= 0:
        raise ValueError("--event-gap-minutes deve ser positivo.")
    thresholds = [float(value) for value in args.thresholds.split(",")]
    named = dict(args.forecast)
    if len(named) != len(args.forecast):
        raise ValueError("Nomes de --forecast devem ser únicos.")
    aligned = align_records({name: pd.read_parquet(path) for name, path in named.items()})
    scopes = ("station", "municipal") if args.scope == "both" else (args.scope,)
    calibration = (json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None)
    report = {}
    for name, records in aligned.items():
        report[name] = {}
        for scope in scopes:
            report[name][scope] = {}
            for threshold in thresholds:
                raw = event_metrics(records, threshold, gap_minutes=args.event_gap_minutes, scope=scope)
                if calibration is None:
                    report[name][scope][f"{threshold:g}"] = raw
                    continue
                decision = calibrated_threshold(calibration, name, threshold)
                report[name][scope][f"{threshold:g}"] = {
                    "raw": raw,
                    "calibrated": event_metrics(
                        records, threshold, decision_threshold=decision,
                        gap_minutes=args.event_gap_minutes, scope=scope,
                    ),
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"thresholds_mm_15min": thresholds, "scopes": scopes, "experiments": report}
    if calibration is not None:
        output["calibration"] = {
            "path": str(args.calibration),
            "selection_split": calibration.get("selection_split"),
            "report_split": calibration.get("report_split"),
        }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Avaliação por evento salva em {args.output}")


if __name__ == "__main__":
    main()
