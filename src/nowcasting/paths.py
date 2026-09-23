"""Localizacao de recursos do projeto quando executado a partir do pacote."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Retorna a raiz do checkout, independentemente do diretorio atual."""
    candidates = (Path.cwd().resolve(), Path(__file__).resolve())
    for start in candidates:
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists() and (candidate / "src" / "nowcasting").is_dir():
                return candidate
    raise RuntimeError(
        "Nao foi possivel localizar a raiz do projeto. Execute a partir de um checkout do repositorio."
    )
