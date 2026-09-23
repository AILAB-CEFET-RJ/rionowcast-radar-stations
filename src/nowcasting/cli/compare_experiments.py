#!/usr/bin/env python3
"""Consolida resultados de experimentos de radar e somente-estações."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMPARABLE_CONFIG = ("train_years", "val_years", "test_years", "step", "stride")


def parse_experiment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NOME=DIRETORIO.")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Use NOME=DIRETORIO.")
    return name, Path(path)


def load_experiment(name: str, directory: Path) -> dict:
    configuration_path, summary_path = directory / "configuration.json", directory / "summary.json"
    if not configuration_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"{name}: configuration.json ou summary.json ausente em {directory}")
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "results" in summary:
        if len(summary["results"]) != 1:
            raise ValueError(f"{name}: comparação atual requer exatamente uma iteração.")
        result = summary["results"][0]
    else:
        result = summary
    metrics = result.get("test_metrics")
    if not metrics:
        raise ValueError(f"{name}: métricas de teste ausentes.")
    return {"name": name, "directory": str(directory), "configuration": configuration, "metrics": metrics}


def validate(experiments: list[dict]) -> list[str]:
    reference = experiments[0]["configuration"]
    warnings: list[str] = []
    for experiment in experiments[1:]:
        current = experiment["configuration"]
        differences = [key for key in COMPARABLE_CONFIG if reference.get(key) != current.get(key)]
        if differences:
            warnings.append(
                f"{experiment['name']}: configuração diferente em {', '.join(differences)}; "
                "a comparação não é estritamente pareada."
            )
    counts = {experiment["name"]: experiment["metrics"]["global"]["n"] for experiment in experiments}
    if len(set(counts.values())) != 1:
        warnings.append(f"Número de observações de teste difere: {counts}.")
    return warnings


def value(metric: dict, key: str) -> str:
    current = metric.get(key)
    return "-" if current is None else (str(current) if key == "n" else f"{current:.6f}")


def table(experiments: list[dict], selector, title: str) -> list[str]:
    lines = [f"## {title}", "", "| Experimento | n | RMSE | MAE | Bias |", "|---|---:|---:|---:|---:|"]
    for experiment in experiments:
        metric = selector(experiment["metrics"])
        lines.append("| " + experiment["name"] + " | " + " | ".join(value(metric, key) for key in ("n", "rmse", "mae", "bias")) + " |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara summaries de experimentos de nowcasting.")
    parser.add_argument("--experiment", action="append", type=parse_experiment, required=True,
                        help="NOME=DIRETORIO; repetir para cada experimento.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    experiments = [load_experiment(name, directory) for name, directory in args.experiment]
    warnings = validate(experiments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {"experiments": experiments, "warnings": warnings}
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = ["# Comparação de Experimentos", ""]
    if warnings:
        lines.extend(["## Avisos", *[f"- {warning}" for warning in warnings], ""])
    lines.extend(table(experiments, lambda metrics: metrics["global"], "Global"))
    horizon_count = len(experiments[0]["metrics"].get("horizons", []))
    for index in range(horizon_count):
        lines.extend(table(experiments, lambda metrics, index=index: metrics["horizons"][index], f"Horizonte T+{index + 1}"))
    for intensity in ("weak", "moderate", "strong", "extreme"):
        lines.extend(table(experiments, lambda metrics, intensity=intensity: metrics["intensity"][intensity], f"Intensidade: {intensity}"))

    if all("stations" in experiment["metrics"] for experiment in experiments):
        station_ids = sorted(set.intersection(*[set(experiment["metrics"]["stations"]) for experiment in experiments]))
        lines.extend(["## Por Estação", ""])
        for station_id in station_ids:
            lines.extend(table(experiments, lambda metrics, station_id=station_id: metrics["stations"][station_id]["global"], f"Estação {station_id}"))
    else:
        lines.extend(["## Por Estação", "", "Métricas por estação ainda não estão disponíveis para todos os experimentos informados.", ""])

    (args.output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparação salva em {args.output_dir}")


if __name__ == "__main__":
    main()
