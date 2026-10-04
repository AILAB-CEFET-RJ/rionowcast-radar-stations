"""Generate a transparent shell plan for repeated training seeds."""
from __future__ import annotations

import argparse
from pathlib import Path


def _seeds(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seeds deve conter inteiros separados por vírgula.") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--seeds deve ser não vazio e sem repetição.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera um script auditável para executar múltiplas seeds.")
    parser.add_argument("--seeds", type=_seeds, required=True, help="Ex.: 101,202,303")
    parser.add_argument("--run-prefix", required=True, help="Prefixo de --run-name de cada seed.")
    parser.add_argument("--command-template", required=True,
                        help="Comando com campos {seed} e {run_name}; o comando não é executado por esta ferramenta.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        lines = [args.command_template.format(seed=seed, run_name=f"{args.run_prefix}-seed-{seed}")
                 for seed in args.seeds]
    except KeyError as error:
        raise ValueError(f"Campo não reconhecido no template: {error}. Use apenas {{seed}} e {{run_name}}.") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = ["#!/usr/bin/env bash", "# Gerado por nowcasting-plan-seed-runs.", "set -euo pipefail", ""]
    content.extend(lines)
    args.output.write_text("\n".join(content) + "\n", encoding="utf-8")
    args.output.chmod(args.output.stat().st_mode | 0o111)
    print(f"Plano de {len(lines)} seeds salvo em {args.output}")


if __name__ == "__main__":
    main()
