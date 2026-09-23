"""Cria uma versao espacialmente reduzida de um dataset memmap anual.

O utilitario permite testar resolucoes menores sem depender dos PNGs brutos
do radar. Os frames sao reamostrados por vizinho mais proximo, o mesmo metodo
usado pelo gerador de frames; os targets e mascaras esparsos sao remapeados
para a nova grade.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


TARGET_SOURCES = {
    "websirene": ("Y_all.dat", "M_all.dat", "targets_metadata.json"),
    "alertario": (
        "Y_alertario.dat",
        "M_alertario.dat",
        "targets_alertario_metadata.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduz espacialmente um dataset memmap de radar, targets e "
            "mascaras, sem recarregar as imagens PNG originais."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument(
        "--target-source",
        choices=("alertario", "websirene", "both"),
        default="both",
        help="Conjunto de targets a copiar e remapear (padrao: both).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Numero de instantes processados por bloco (padrao: 64).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Intervalo, em frames, entre mensagens de progresso (padrao: 1000).",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=4, ensure_ascii=False)


def nearest_indices(source_size: int, output_size: int) -> np.ndarray:
    """Indices de entrada equivalentes a reduzir por vizinho mais proximo."""
    indices = ((np.arange(output_size) + 0.5) * source_size / output_size).astype(
        np.int64
    )
    return np.clip(indices, 0, source_size - 1)


def ensure_new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"O arquivo de destino ja existe: {path}. "
            "Escolha outro --output-root para evitar sobrescrever dados."
        )


def should_log_progress(start: int, stop: int, total: int, interval: int) -> bool:
    return start == 0 or stop == total or start // interval != (stop - 1) // interval


def downsample_radar(
    source_dir: Path, output_dir: Path, height: int, width: int, chunk_size: int,
    progress_interval: int = 1000,
) -> tuple[dict, tuple[int, int, int, int]]:
    metadata_path = source_dir / "metadata.json"
    metadata = read_json(metadata_path)
    shape = tuple(metadata["shape"])

    if len(shape) != 4 or shape[3] != 3:
        raise ValueError(f"Shape de radar invalido em {metadata_path}: {shape}")

    n_frames, source_height, source_width, channels = shape
    if height > source_height or width > source_width:
        raise ValueError(
            "Este utilitario apenas reduz a resolucao. "
            f"Origem: {source_height}x{source_width}; destino: {height}x{width}."
        )

    source_path = source_dir / metadata.get("frames_file", "radar_frames.dat")
    output_path = output_dir / "radar_frames.dat"
    ensure_new_path(output_path)

    source = np.memmap(
        source_path, dtype=np.dtype(metadata["dtype"]), mode="r", shape=shape
    )
    output_shape = (n_frames, height, width, channels)
    output = np.memmap(output_path, dtype=source.dtype, mode="w+", shape=output_shape)

    row_indices = nearest_indices(source_height, height)
    col_indices = nearest_indices(source_width, width)
    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        output[start:stop] = source[start:stop, row_indices][:, :, col_indices, :]
        if should_log_progress(start, stop, n_frames, progress_interval):
            print(f"[radar] frames {stop}/{n_frames}", flush=True)

    output.flush()
    del output
    del source

    timestamps_file = metadata.get("timestamps_file", "radar_timestamps.npy")
    shutil.copy2(source_dir / timestamps_file, output_dir / timestamps_file)
    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "height": height,
            "width": width,
            "shape": list(output_shape),
            "frames_file": output_path.name,
            "timestamps_file": timestamps_file,
            "spatial_resampling": "nearest",
            "source_spatial_shape": [source_height, source_width],
        }
    )
    write_json(output_dir / "metadata.json", output_metadata)
    return output_metadata, shape


def downsample_targets(source_dir: Path, output_dir: Path, source_name: str,
                       source_radar_shape: tuple[int, int, int, int], height: int,
                       width: int, chunk_size: int,
                       progress_interval: int = 1000) -> None:
    y_name, m_name, metadata_name = TARGET_SOURCES[source_name]
    source_metadata_path = source_dir / metadata_name
    if not source_metadata_path.exists():
        print(f"[{source_name}] metadados ausentes; ignorando.", flush=True)
        return

    metadata = read_json(source_metadata_path)
    shape = tuple(metadata["shape"])
    if len(shape) != 4 or shape[3] != 1:
        raise ValueError(f"Shape de target invalido em {source_metadata_path}: {shape}")
    if shape[:3] != source_radar_shape[:3]:
        raise ValueError(
            f"{source_name}: target {shape} nao esta alinhado ao radar "
            f"{source_radar_shape}."
        )

    output_y_path = output_dir / y_name
    output_m_path = output_dir / m_name
    output_metadata_path = output_dir / metadata_name
    for path in (output_y_path, output_m_path, output_metadata_path):
        ensure_new_path(path)

    source_y = np.memmap(
        source_dir / metadata.get("Y_file", y_name),
        dtype=np.dtype(metadata["Y_dtype"]), mode="r", shape=shape
    )
    source_m = np.memmap(
        source_dir / metadata.get("M_file", m_name),
        dtype=np.dtype(metadata["M_dtype"]), mode="r", shape=shape
    )
    output_shape = (shape[0], height, width, 1)
    output_y = np.memmap(output_y_path, dtype=source_y.dtype, mode="w+", shape=output_shape)
    output_m = np.memmap(output_m_path, dtype=source_m.dtype, mode="w+", shape=output_shape)
    output_y[:] = 0
    output_m[:] = 0

    # Inverte o mapeamento de amostragem: cada pixel de origem vai ao pixel
    # mais proximo da grade de destino. Isso preserva todas as observacoes.
    source_height, source_width = shape[1:3]
    target_rows = np.rint(np.arange(source_height) * (height - 1) /
                          max(source_height - 1, 1)).astype(np.int64)
    target_cols = np.rint(np.arange(source_width) * (width - 1) /
                          max(source_width - 1, 1)).astype(np.int64)

    for start in range(0, shape[0], chunk_size):
        stop = min(start + chunk_size, shape[0])
        mask = source_m[start:stop, :, :, 0]
        local_t, source_i, source_j = np.nonzero(mask)
        if len(local_t):
            target_i = target_rows[source_i]
            target_j = target_cols[source_j]
            values = source_y[start:stop, :, :, 0][local_t, source_i, source_j]
            np.maximum.at(output_y, (start + local_t, target_i, target_j, 0), values)
            output_m[start + local_t, target_i, target_j, 0] = 1
        if should_log_progress(start, stop, shape[0], progress_interval):
            print(f"[{source_name}] frames {stop}/{shape[0]}", flush=True)

    output_y.flush()
    output_m.flush()
    del output_y
    del output_m
    del source_y
    del source_m

    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "height": height,
            "width": width,
            "shape": list(output_shape),
            "Y_file": y_name,
            "M_file": m_name,
            "spatial_resampling": "sparse-nearest",
            "source_spatial_shape": [source_height, source_width],
        }
    )
    write_json(output_metadata_path, output_metadata)


def process_year(args: argparse.Namespace, year: int) -> None:
    source_dir = args.source_root / f"year={year}"
    if not source_dir.is_dir():
        print(f"[{year}] diretorio de origem ausente; ignorando.", flush=True)
        return

    output_dir = args.output_root / f"year={year}"
    if output_dir.exists():
        raise FileExistsError(
            f"O diretorio de destino ja existe: {output_dir}. "
            "Use um --output-root novo."
        )
    output_dir.mkdir(parents=True)

    print(f"\n=== ANO {year} ===", flush=True)
    _, source_radar_shape = downsample_radar(
        source_dir, output_dir, args.height, args.width, args.chunk_size,
        args.progress_interval,
    )
    sources = TARGET_SOURCES if args.target_source == "both" else (args.target_source,)
    for source_name in sources:
        downsample_targets(
            source_dir, output_dir, source_name, source_radar_shape,
            args.height, args.width, args.chunk_size, args.progress_interval,
        )


def main() -> None:
    args = parse_args()
    if args.year_start > args.year_end:
        raise ValueError("--year-start nao pode ser maior que --year-end.")
    if args.height <= 0 or args.width <= 0 or args.chunk_size <= 0 or args.progress_interval <= 0:
        raise ValueError("--height, --width, --chunk-size e --progress-interval devem ser positivos.")
    if args.output_root.resolve() == args.source_root.resolve():
        raise ValueError("--output-root deve ser diferente de --source-root.")

    for year in range(args.year_start, args.year_end + 1):
        process_year(args, year)


if __name__ == "__main__":
    main()
