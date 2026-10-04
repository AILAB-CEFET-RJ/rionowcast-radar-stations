"""Compare canonical station-forecast records without retraining models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nowcasting.forecast_evaluation import align_records, evaluate_records, paired_daily_bootstrap


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NOME=ARQUIVO.parquet.")
    name, path = value.split("=", 1)
    return name, Path(path)


def performance_diagram(metrics: dict[str, dict], threshold: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, report in metrics.items():
        item = report["thresholds"][threshold]["global"]
        if item["pod"] is not None and item["success_ratio"] is not None:
            axis.scatter(item["success_ratio"], item["pod"], label=name, s=60)
            axis.annotate(name, (item["success_ratio"], item["pod"]), xytext=(5, 4), textcoords="offset points")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Razão de sucesso (1 - FAR)", ylabel="POD",
             title=f"Diagrama de desempenho: limiar {threshold} mm/15 min")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia previsões canônicas por estação, limiar e antecedência.")
    parser.add_argument("--forecast", action="append", type=named_path, required=True, help="NOME=ARQUIVO.parquet; repetir.")
    parser.add_argument("--baseline", required=True, help="Nome do experimento de referência, tipicamente B1.")
    parser.add_argument("--thresholds", default="1.25,6.25,12.5")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261004)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if not thresholds or any(value <= 0 for value in thresholds) or sorted(thresholds) != thresholds:
        raise ValueError("--thresholds deve conter valores positivos em ordem crescente.")
    named = dict(args.forecast)
    if len(named) != len(args.forecast) or args.baseline not in named:
        raise ValueError("Nomes de --forecast devem ser únicos e incluir --baseline.")
    aligned = align_records({name: pd.read_parquet(path) for name, path in named.items()})
    reports = {name: evaluate_records(records, thresholds) for name, records in aligned.items()}
    bootstrap = {name: paired_daily_bootstrap(aligned[args.baseline], records, thresholds,
                                               replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
                 for name, records in aligned.items() if name != args.baseline}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"baseline": args.baseline, "thresholds_mm_15min": thresholds,
               "records": {name: int(len(records)) for name, records in aligned.items()},
               "valid_records": {name: int(records["is_observed"].sum()) for name, records in aligned.items()},
               "metrics": reports, "paired_daily_bootstrap": bootstrap,
               "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                             "grouping": "target local day in America/Sao_Paulo"}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = []
    for name, report in reports.items():
        rows.append({"experiment": name, "scope": "global", **report["global"]})
        for horizon, item in report["horizons"].items():
            rows.append({"experiment": name, "scope": f"horizon_{horizon}", **item})
    pd.DataFrame(rows).to_csv(args.output_dir / "continuous_metrics.csv", index=False)
    for threshold in thresholds:
        performance_diagram(reports, f"{threshold:g}", args.output_dir / f"performance_diagram_{threshold:g}mm15.png")
    print(f"Avaliação salva em {args.output_dir}")


if __name__ == "__main__":
    main()
