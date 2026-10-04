"""Aggregate independent seed evaluations without mixing their forecast records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use SEED=CAMINHO/summary.json.")
    name, path = value.split("=", 1)
    return name, Path(path)


def _leaves(value: object, path: tuple[str, ...] = ()) -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, child in value.items():
            result.update(_leaves(child, (*path, str(key))))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        return {".".join(path): float(value)}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida métricas de múltiplas seeds já avaliadas.")
    parser.add_argument("--summary", action="append", type=named_path, required=True,
                        help="SEED=CAMINHO/summary.json; repetir para cada seed.")
    parser.add_argument("--experiment", required=True, help="Nome do experimento dentro de metrics, por exemplo C3.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    named = dict(args.summary)
    if len(named) != len(args.summary):
        raise ValueError("Nomes de --summary devem ser únicos.")
    flat: dict[str, dict[str, float]] = {}
    for seed, path in named.items():
        summary = json.loads(path.read_text(encoding="utf-8"))
        try:
            metrics = summary["metrics"][args.experiment]
        except KeyError as error:
            raise ValueError(f"{path}: experimento {args.experiment!r} não encontrado em metrics.") from error
        flat[seed] = _leaves(metrics)
    common = set.intersection(*(set(values) for values in flat.values()))
    rows = []
    for metric in sorted(common):
        values = np.asarray([flat[seed][metric] for seed in named], dtype=float)
        rows.append({"metric": metric, "seeds": len(values), "mean": float(values.mean()),
                     "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                     "min": float(values.min()), "max": float(values.max())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "seed_aggregate.csv", index=False)
    payload = {"experiment": args.experiment, "seed_summaries": {seed: str(path) for seed, path in named.items()},
               "metrics_common_to_all_seeds": rows}
    (args.output_dir / "seed_aggregate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Agregação de {len(named)} seeds salva em {args.output_dir}")


if __name__ == "__main__":
    main()
