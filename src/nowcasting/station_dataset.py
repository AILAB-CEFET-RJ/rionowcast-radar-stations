"""Dataset temporal de estações, construído a partir dos targets esparsos."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_station_pixels(
    mapping_path: str | Path, height: int, width: int, *, height_orig: int = 656, width_orig: int = 654,
) -> tuple[np.ndarray, list[int]]:
    """Converte o mapeamento no grid original para pixels da resolução atual."""
    mapping_path = Path(mapping_path)
    with mapping_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"station_id", "pixel_i", "pixel_j"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{mapping_path}: esperadas colunas {sorted(required)}.")

    station_ids: list[int] = []
    pixels: list[tuple[int, int]] = []
    for row in rows:
        station_ids.append(int(row["station_id"]))
        pixels.append((
            int(np.rint(float(row["pixel_i"]) * height / height_orig)),
            int(np.rint(float(row["pixel_j"]) * width / width_orig)),
        ))
    pixel_array = np.asarray(pixels, dtype=np.int64)
    if ((pixel_array[:, 0] < 0) | (pixel_array[:, 0] >= height) |
            (pixel_array[:, 1] < 0) | (pixel_array[:, 1] >= width)).any():
        raise ValueError(f"{mapping_path}: estação fora da grade {height}x{width}.")
    if len(set(map(tuple, pixel_array))) != len(pixel_array):
        raise ValueError("Duas ou mais estações ocupam o mesmo pixel nesta resolução.")
    return pixel_array, station_ids


class StationSequenceDataset(Dataset):
    """Histórico de chuva/máscara nas estações e alvos futuros nas mesmas estações.

    `inputs` contém dois canais por estação: valor transformado por ``log1p`` e
    indicador de observação. Os targets seguem a mesma unidade dos memmaps.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        years: list[int],
        *,
        mapping: str | Path,
        t_in: int = 5,
        t_out: int = 5,
        stride: int = 5,
        mapping_height_orig: int = 656,
        mapping_width_orig: int = 654,
        split_name: str = "dataset",
    ):
        self.dataset_root = Path(dataset_root)
        self.years = sorted(set(years))
        self.t_in, self.t_out, self.stride = t_in, t_out, stride
        self.mapping = Path(mapping)
        self.mapping_height_orig, self.mapping_width_orig = mapping_height_orig, mapping_width_orig
        self.split_name = split_name
        self.samples: list[tuple[int, int]] = []
        self.year_data: dict[int, dict[str, np.ndarray]] = {}
        self.station_ids: list[int] = []
        self.station_pixels: np.ndarray | None = None
        if not self.years or min(t_in, t_out, stride) <= 0:
            raise ValueError("years, t_in, t_out e stride devem ser positivos.")

        for year in self.years:
            self._load_year(year)
        if not self.samples:
            raise ValueError(f"{split_name}: nenhuma janela disponível.")
        print(f"[{split_name}] Total de amostras: {len(self.samples)}", flush=True)

    def _load_year(self, year: int) -> None:
        year_dir = self.dataset_root / f"year={year}"
        metadata_path = year_dir / "metadata.json"
        target_metadata_path = year_dir / "targets_alertario_metadata.json"
        if not metadata_path.is_file() or not target_metadata_path.is_file():
            raise FileNotFoundError(f"{year}: metadata do radar ou AlertaRio ausente.")
        radar_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
        shape = tuple(radar_metadata["shape"])
        if target_metadata.get("format") != "sparse":
            raise ValueError("StationSequenceDataset requer targets AlertaRio no formato esparso.")
        if self.station_pixels is None:
            self.station_pixels, self.station_ids = load_station_pixels(
                self.mapping, shape[1], shape[2], height_orig=self.mapping_height_orig,
                width_orig=self.mapping_width_orig,
            )
        elif tuple(shape[1:3]) != tuple(self.year_data[self.years[0]]["shape"]):
            raise ValueError("Todos os anos devem ter a mesma resolução espacial.")

        sparse_path = year_dir / target_metadata["sparse_file"]
        with np.load(sparse_path) as sparse:
            frame = np.asarray(sparse["frame"], dtype=np.int64)
            rows = np.asarray(sparse["row"], dtype=np.int64)
            columns = np.asarray(sparse["column"], dtype=np.int64)
            value = np.asarray(sparse["value"], dtype=np.float32)
            station_id = (
                np.asarray(sparse["station_id"], dtype=np.int64)
                if "station_id" in sparse.files else None
            )
        if not (len(frame) == len(rows) == len(columns) == len(value)):
            raise ValueError(f"{year}: arrays esparsos com comprimentos diferentes.")
        if station_id is not None and len(station_id) != len(frame):
            raise ValueError(f"{year}: station_id esparso com comprimento invalido.")

        station_index = {tuple(pixel): index for index, pixel in enumerate(self.station_pixels)}
        station_id_index = {station_id: index for index, station_id in enumerate(self.station_ids)}
        values = np.zeros((shape[0], len(self.station_ids)), dtype=np.float32)
        masks = np.zeros((shape[0], len(self.station_ids)), dtype=np.float32)
        if station_id is not None:
            for current_frame, current_station_id, current_value in zip(frame, station_id, value):
                index = station_id_index.get(int(current_station_id))
                if index is not None:
                    values[current_frame, index] = max(values[current_frame, index], current_value)
                    masks[current_frame, index] = 1.0
        else:
            for current_frame, row, column, current_value in zip(frame, rows, columns, value):
                index = station_index.get((int(row), int(column)))
                if index is not None:
                    values[current_frame, index] = current_value
                    masks[current_frame, index] = 1.0
        self.year_data[year] = {"values": values, "masks": masks, "shape": np.asarray(shape[1:3])}

        n_possible = shape[0] - (self.t_in + self.t_out) + 1
        if n_possible <= 0:
            raise ValueError(f"{year}: frames insuficientes para as sequências configuradas.")
        self.samples.extend((year, start) for start in range(0, n_possible, self.stride))
        print(
            f"[{self.split_name}] Ano {year} carregado | estações={len(self.station_ids)} | "
            f"frames={shape[0]} | amostras={len(range(0, n_possible, self.stride))}", flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        year, start = self.samples[index]
        data = self.year_data[year]
        x_end, y_end = start + self.t_in, start + self.t_in + self.t_out
        values, masks = data["values"], data["masks"]
        inputs = np.stack((values[start:x_end], masks[start:x_end]), axis=-1)
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(values[x_end:y_end]),
            torch.from_numpy(masks[x_end:y_end]),
        )
