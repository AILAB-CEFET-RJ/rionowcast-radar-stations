"""Inventory and safely stage monthly Sumaré radar archives.

The source delivery is a collection of ``YYYY_MM.tar.gz`` archives containing
``YYYY/MM/DD/YYYY_MM_DD_HH_MM.png`` captures. This command makes the source
manifest explicit before the normal PNG audit and memmap build steps consume
the files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath


ARCHIVE_PATTERN = re.compile(r"(?P<year>\d{4})_(?P<month>0[1-9]|1[0-2])\.tar\.gz$")
PNG_PATTERN = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}\.png$")


@dataclass(frozen=True)
class Archive:
    path: Path
    year: int
    month: int


def png_timestamp(path: PurePosixPath, archive: Archive) -> datetime:
    year, month, day, hour, minute = map(int, path.stem.split("_"))
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError as error:
        raise ValueError(f"Timestamp de PNG invalido em {archive.path.name}: {path}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventaria e extrai com seguranca os arquivos mensais do radar Sumare."
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extrai os PNGs sob --raw-root depois de validar cada arquivo.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Aceita arquivos ja extraidos somente quando o tamanho coincide com o membro do tar.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_archives(root: Path, year_start: int, year_end: int) -> list[Archive]:
    if year_start > year_end:
        raise ValueError("--year-start nao pode ser maior que --year-end.")
    if not root.is_dir():
        raise FileNotFoundError(f"Diretorio de arquivos ausente: {root}")
    archives: list[Archive] = []
    for path in sorted(root.glob("*.tar.gz")):
        match = ARCHIVE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        year = int(match["year"])
        if year_start <= year <= year_end:
            archives.append(Archive(path=path, year=year, month=int(match["month"])))
    if not archives:
        raise FileNotFoundError(
            f"Nenhum arquivo YYYY_MM.tar.gz para {year_start}--{year_end} em {root}."
        )
    keys = [(archive.year, archive.month) for archive in archives]
    if len(keys) != len(set(keys)):
        raise ValueError("Ha mais de um arquivo para o mesmo ano e mes.")
    return archives


def _relative_member_path(member: tarfile.TarInfo, archive: Archive) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Membro inseguro em {archive.path.name}: {member.name}")
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        raise ValueError(f"Membro nao regular em {archive.path.name}: {member.name}")
    if member.isdir():
        return path
    if not member.isfile():
        raise ValueError(f"Tipo de membro nao suportado em {archive.path.name}: {member.name}")
    if len(path.parts) != 4 or path.parts[0] != str(archive.year) or path.parts[1] != f"{archive.month:02d}":
        raise ValueError(f"Layout inesperado em {archive.path.name}: {member.name}")
    if not PNG_PATTERN.fullmatch(path.name):
        raise ValueError(f"Nome de PNG inesperado em {archive.path.name}: {member.name}")
    png_timestamp(path, archive)
    return path


def inspect_archive(archive: Archive) -> tuple[list[tarfile.TarInfo], dict[str, object]]:
    members: list[tarfile.TarInfo] = []
    paths: set[PurePosixPath] = set()
    png_paths: list[str] = []
    directory_date_mismatches = 0
    mismatch_examples: list[str] = []
    archive_month_mismatches = 0
    archive_month_mismatch_examples: list[str] = []
    with tarfile.open(archive.path, mode="r:gz") as tar:
        for member in tar:
            relative = _relative_member_path(member, archive)
            if member.isfile():
                if relative in paths:
                    raise ValueError(f"Membro duplicado em {archive.path.name}: {member.name}")
                paths.add(relative)
                members.append(member)
                png_paths.append(member.name)
                timestamp = png_timestamp(relative, archive)
                if timestamp.strftime("%Y/%m/%d") != "/".join(relative.parts[:3]):
                    directory_date_mismatches += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append(member.name)
                if (timestamp.year, timestamp.month) != (archive.year, archive.month):
                    archive_month_mismatches += 1
                    if len(archive_month_mismatch_examples) < 5:
                        archive_month_mismatch_examples.append(member.name)
    if not members:
        raise ValueError(f"Arquivo sem PNGs: {archive.path}")
    return members, {
        "archive": archive.path.name,
        "year": archive.year,
        "month": archive.month,
        "bytes": archive.path.stat().st_size,
        "sha256": sha256_file(archive.path),
        "png_members": len(members),
        "first_png": min(png_paths),
        "last_png": max(png_paths),
        "directory_date_mismatches": directory_date_mismatches,
        "directory_date_mismatch_examples": mismatch_examples,
        "archive_month_mismatches": archive_month_mismatches,
        "archive_month_mismatch_examples": archive_month_mismatch_examples,
    }


def extract_archive(archive: Archive, members: list[tarfile.TarInfo], raw_root: Path, resume: bool) -> dict[str, int]:
    written = 0
    skipped = 0
    with tarfile.open(archive.path, mode="r:gz") as tar:
        for member in members:
            relative = _relative_member_path(member, archive)
            destination = raw_root.joinpath(*relative.parts)
            if destination.exists():
                if resume and destination.is_file() and destination.stat().st_size == member.size:
                    skipped += 1
                    continue
                raise FileExistsError(
                    f"Destino ja existe: {destination}. Use uma raiz V4 vazia ou --resume apos inspecao."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Nao foi possivel ler {member.name} em {archive.path}")
            with source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            written += 1
    return {"extracted_pngs": written, "resumed_pngs": skipped}


def write_manifest(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.partial")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    archives = discover_archives(args.archive_root, args.year_start, args.year_end)
    reports: list[dict[str, object]] = []
    for archive in archives:
        members, report = inspect_archive(archive)
        if args.extract:
            report.update(extract_archive(archive, members, args.raw_root, args.resume))
        reports.append(report)
        print(
            f"{archive.path.name} | PNGs={report['png_members']} | sha256={report['sha256']}"
            + (f" | extraidos={report['extracted_pngs']}" if args.extract else ""),
            flush=True,
        )
    write_manifest(args.output_manifest, {
        "format": "radar_sumare_monthly_archives_v1",
        "archive_root": str(args.archive_root),
        "raw_root": str(args.raw_root),
        "year_start": args.year_start,
        "year_end": args.year_end,
        "extracted": bool(args.extract),
        "archives": reports,
        "totals": {
            "archives": len(reports),
            "png_members": sum(int(report["png_members"]) for report in reports),
            "archive_bytes": sum(int(report["bytes"]) for report in reports),
            "extracted_pngs": sum(int(report.get("extracted_pngs", 0)) for report in reports),
        },
    })
    print(f"Manifesto salvo em {args.output_manifest}", flush=True)


if __name__ == "__main__":
    main()
