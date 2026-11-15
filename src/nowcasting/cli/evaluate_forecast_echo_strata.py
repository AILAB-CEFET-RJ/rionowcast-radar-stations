"""Evaluate canonical forecasts separately with weak and present local RGB echo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.forecast_evaluation import (
    align_records,
    evaluate_records,
    paired_daily_bootstrap,
    skill_scores,
    stratify_by_radar_echo,
)


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NOME=ARQUIVO.parquet.")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia previsões por estrato de eco RGB local no último frame de entrada."
    )
    parser.add_argument("--forecast", action="append", type=named_path, required=True,
                        help="NOME=ARQUIVO.parquet; repetir.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--thresholds", default="1.25,6.25,12.5")
    parser.add_argument("--low-echo-max", type=float, default=0.05)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    named = dict(args.forecast)
    if len(named) != len(args.forecast) or args.baseline not in named:
        raise ValueError("Nomes de --forecast devem ser únicos e incluir --baseline.")

    raw = {name: pd.read_parquet(path) for name, path in named.items()}
    aligned = align_records(raw)
    strata = {name: stratify_by_radar_echo(frame, args.low_echo_max) for name, frame in raw.items()}
    result: dict[str, object] = {
        "baseline": args.baseline,
        "thresholds_mm_15min": thresholds,
        "low_echo_max": args.low_echo_max,
        "echo_definition": "fraction of non-black RGB pixels in radius-two station neighborhood at last input frame",
        "strata": {},
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                      "grouping": "target local day in America/Sao_Paulo"},
    }
    for label in ("low_echo", "echo_present"):
        selected = {name: value[label] for name, value in strata.items()}
        # The context is derived solely from the common radar input. Check it
        # explicitly before comparing predictions inside a stratum.
        reference = selected[args.baseline]["radar_echo_fraction"].reset_index(drop=True)
        for name, frame in selected.items():
            if not reference.equals(frame["radar_echo_fraction"].reset_index(drop=True)):
                raise ValueError(f"{name}: contexto RGB diferente da baseline no estrato {label}.")
        selected_aligned = align_records(selected)
        reports = {name: evaluate_records(frame, thresholds) for name, frame in selected_aligned.items()}
        bootstrap, skills = {}, {}
        for name, frame in selected_aligned.items():
            if name == args.baseline:
                continue
            skills[name] = skill_scores(reports[args.baseline], reports[name])
            bootstrap[name] = paired_daily_bootstrap(
                selected_aligned[args.baseline], frame, thresholds,
                replicates=args.bootstrap_replicates, seed=args.bootstrap_seed,
            )
        result["strata"][label] = {
            "records": {name: int(len(frame)) for name, frame in selected_aligned.items()},
            "valid_records": {name: int(frame["is_observed"].sum()) for name, frame in selected_aligned.items()},
            "metrics": reports,
            "skill_scores_against_baseline": skills,
            "paired_daily_bootstrap": bootstrap,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Avaliação estratificada por eco salva em {args.output}")


if __name__ == "__main__":
    main()
