"""Compare canonical station-forecast records without retraining models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nowcasting.forecast_evaluation import (
    INTENSITY_BINS, align_records, evaluate_records, paired_daily_bootstrap, skill_scores,
)


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NOME=ARQUIVO.parquet.")
    name, path = value.split("=", 1)
    return name, Path(path)


def performance_diagram(metrics: dict[str, dict], threshold: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    success_ratio, pod = np.meshgrid(np.linspace(0.01, 1.0, 200), np.linspace(0.01, 1.0, 200))
    csi = 1.0 / (1.0 / success_ratio + 1.0 / pod - 1.0)
    contours = axis.contour(success_ratio, pod, csi, levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            colors="0.7", linewidths=0.7)
    axis.clabel(contours, inline=True, fontsize=7, fmt="CSI %.1f")
    for bias in (0.25, 0.5, 1.0, 2.0, 4.0):
        x = np.linspace(max(0.01, 0.01 / bias), min(1.0, 1.0 / bias), 100)
        axis.plot(x, bias * x, color="0.82", linestyle="--", linewidth=0.7)
        axis.text(x[-1], bias * x[-1], f"B={bias:g}", color="0.5", fontsize=7, ha="right", va="bottom")
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


def _number(value: object, digits: int = 4) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _write_tabular_reports(summary: dict, output_dir: Path) -> None:
    """Write compact human-readable reports alongside the complete JSON artifact."""
    metrics = summary["metrics"]
    baseline = summary["baseline"]
    markdown = ["# Avaliação auditável de previsões", "", f"Baseline: `{baseline}`.", "",
                "## Métricas globais", "", "| Experimento | N | RMSE | MAE | Viés |",
                "|---|---:|---:|---:|---:|"]
    latex = ["% Gerado automaticamente por nowcasting-evaluate-forecast-records.",
             "\\begin{table}[htbp]", "\\centering", "\\caption{Métricas globais de previsão.}",
             "\\begin{tabular}{lrrrr}", "\\toprule", "Experimento & N & RMSE & MAE & Viés \\\\", "\\midrule"]
    for name, report in metrics.items():
        item = report["global"]
        markdown.append("| {} | {} | {} | {} | {} |".format(
            name, item["n"], _number(item["rmse"]), _number(item["mae"]), _number(item["bias"])))
        latex.append("{} & {} & {} & {} & {} \\\\".format(
            name.replace("_", "\\_"), item["n"], _number(item["rmse"]), _number(item["mae"]), _number(item["bias"])))
    markdown.extend(["", "## Skill contra a baseline", "", "| Experimento | Skill MAE | Skill RMSE |",
                     "|---|---:|---:|"])
    for name, score in summary["skill_scores_against_baseline"].items():
        item = score["global"]
        markdown.append(f"| {name} | {_number(item['mae_skill'])} | {_number(item['rmse_skill'])} |")
    markdown.extend(["", "## Métricas categóricas globais", ""])
    for threshold in summary["thresholds_mm_15min"]:
        key = f"{threshold:g}"
        markdown.extend([f"### Limiar {key} mm/15 min", "",
                         "| Experimento | POD | FAR | CSI | Viés de frequência | ETS | HSS |",
                         "|---|---:|---:|---:|---:|---:|---:|"])
        for name, report in metrics.items():
            item = report["thresholds"][key]["global"]
            markdown.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                name, *[_number(item[column]) for column in ("pod", "far", "csi", "frequency_bias", "ets", "hss")]))
        markdown.append("")
    markdown.extend(["## Incerteza", "",
                     "Intervalos de 95% são obtidos por bootstrap pareado por dia local (America/Sao_Paulo). "
                     "Os valores completos, inclusive por horizonte e intensidade, estão em `summary.json` e `bootstrap_metrics.csv`.", ""])
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\label{tab:forecast-global-metrics}", "\\end{table}", ""])
    (output_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    (output_dir / "report.tex").write_text("\n".join(latex), encoding="utf-8")


def _write_csv_artifacts(reports: dict[str, dict], skills: dict[str, dict], bootstrap: dict[str, dict], output_dir: Path) -> None:
    continuous_rows, categorical_rows, skill_rows, bootstrap_rows = [], [], [], []
    for name, report in reports.items():
        continuous_rows.append({"experiment": name, "scope": "global", **report["global"]})
        for horizon, item in report["horizons"].items():
            continuous_rows.append({"experiment": name, "scope": f"horizon_{horizon}", **item})
        for intensity, item in report["intensity"].items():
            continuous_rows.append({"experiment": name, "scope": f"intensity_{intensity}", **item})
        for station, station_report in report["stations"].items():
            continuous_rows.append({"experiment": name, "scope": "station_global", "station_id": station,
                                    **station_report["global"]})
            for horizon, item in station_report["horizons"].items():
                continuous_rows.append({"experiment": name, "scope": f"station_horizon_{horizon}",
                                        "station_id": station, **item})
        for threshold, threshold_report in report["thresholds"].items():
            categorical_rows.append({"experiment": name, "scope": "global", "threshold_mm_15min": threshold,
                                     **threshold_report["global"]})
            for horizon, item in threshold_report["horizons"].items():
                categorical_rows.append({"experiment": name, "scope": f"horizon_{horizon}",
                                         "threshold_mm_15min": threshold, **item})
    for name, report in skills.items():
        skill_rows.append({"experiment": name, "scope": "global", **report["global"]})
        for horizon, item in report["horizons"].items():
            skill_rows.append({"experiment": name, "scope": f"horizon_{horizon}", **item})
    for name, report in bootstrap.items():
        for scope, values in [("global", report["global"]),
                              *[(f"horizon_{horizon}", value) for horizon, value in report["horizons"].items()],
                              *[(f"intensity_{intensity}", value) for intensity, value in report["intensity"].items()]]:
            for metric, item in values.items():
                bootstrap_rows.append({"experiment": name, "scope": scope, "metric": metric,
                                       "estimate": item["estimate"], "ci95_low": item["ci95"][0],
                                       "ci95_high": item["ci95"][1]})
    pd.DataFrame(continuous_rows).to_csv(output_dir / "continuous_metrics.csv", index=False)
    pd.DataFrame(categorical_rows).to_csv(output_dir / "categorical_metrics.csv", index=False)
    pd.DataFrame(skill_rows).to_csv(output_dir / "skill_scores.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "bootstrap_metrics.csv", index=False)


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
    bootstrap, skills = {}, {}
    horizons = sorted({int(value) for report in reports.values() for value in report["horizons"]})
    for name, records in aligned.items():
        if name == args.baseline:
            continue
        skills[name] = skill_scores(reports[args.baseline], reports[name])
        bootstrap[name] = {
            "global": paired_daily_bootstrap(aligned[args.baseline], records, thresholds,
                                               replicates=args.bootstrap_replicates, seed=args.bootstrap_seed),
            "horizons": {str(horizon): paired_daily_bootstrap(
                aligned[args.baseline], records, thresholds, replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + horizon, horizon=horizon,
            ) for horizon in horizons},
            "intensity": {intensity: paired_daily_bootstrap(
                aligned[args.baseline], records, thresholds, replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + 100 + position, intensity=(low, high),
            ) for position, (intensity, low, high) in enumerate(INTENSITY_BINS)},
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"baseline": args.baseline, "thresholds_mm_15min": thresholds,
               "records": {name: int(len(records)) for name, records in aligned.items()},
               "valid_records": {name: int(records["is_observed"].sum()) for name, records in aligned.items()},
               "metrics": reports, "skill_scores_against_baseline": skills,
               "paired_daily_bootstrap": bootstrap,
               "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                             "grouping": "target local day in America/Sao_Paulo"}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv_artifacts(reports, skills, bootstrap, args.output_dir)
    _write_tabular_reports(summary, args.output_dir)
    for threshold in thresholds:
        performance_diagram(reports, f"{threshold:g}", args.output_dir / f"performance_diagram_{threshold:g}mm15.png")
    print(f"Avaliação salva em {args.output_dir}")


if __name__ == "__main__":
    main()
