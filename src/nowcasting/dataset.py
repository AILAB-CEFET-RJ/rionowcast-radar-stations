from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


TARGET_METADATA_FILES = {
    "websirene": "targets_metadata.json",
    "alertario": "targets_alertario_metadata.json",
}


def parse_years(value: str) -> list[int]:
    """Converte `2012-2014,2016` em uma lista ordenada de anos únicos."""
    years: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Intervalo de anos inválido: {item}")
            years.update(range(start, end + 1))
        else:
            years.add(int(item))
    if not years:
        raise ValueError("É necessário informar pelo menos um ano.")
    return sorted(years)


class RadarStationMemmapDataset(Dataset):
    """Janelas temporais de radar e targets esparsos de estações.

    A classe não divide os dados internamente. Cada instância representa
    exatamente os anos recebidos, o que evita vazamento entre splits temporais.
    """

    def __init__(
        self,
        radar_root: str | Path,
        years: list[int],
        *,
        t_in: int = 5,
        t_out: int = 5,
        stride: int = 5,
        target_source: str = "alertario",
        split_name: str = "dataset",
        crop_stations: bool = False,
        crop_margin_pixels: int = 20,
        station_mapping: str | Path | None = None,
        mapping_height_orig: int = 656,
        mapping_width_orig: int = 654,
    ):
        self.radar_root = Path(radar_root)
        self.years = sorted(set(years))
        self.t_in = t_in
        self.t_out = t_out
        self.stride = stride
        self.target_source = target_source.lower()
        self.split_name = split_name
        self.crop_stations = crop_stations
        self.crop_margin_pixels = crop_margin_pixels
        self.station_mapping = Path(station_mapping) if station_mapping else None
        self.mapping_height_orig = mapping_height_orig
        self.mapping_width_orig = mapping_width_orig
        self.crop_bounds: tuple[int, int, int, int] | None = None
        self.crop_metadata: dict[str, object] = {"enabled": False}
        self.year_data: dict[int, dict[str, np.memmap]] = {}
        self.samples: list[tuple[int, int]] = []
        self._sample_class_cache: dict[tuple[float, ...], np.ndarray] = {}

        if self.target_source not in TARGET_METADATA_FILES:
            raise ValueError(
                f"Fonte inválida: {target_source}. "
                f"Opções: {sorted(TARGET_METADATA_FILES)}"
            )
        if not self.years:
            raise ValueError("O split não contém anos.")
        if min(t_in, t_out, stride) <= 0:
            raise ValueError("t_in, t_out e stride devem ser positivos.")
        if crop_margin_pixels < 0:
            raise ValueError("crop_margin_pixels não pode ser negativo.")
        if crop_stations and self.target_source != "alertario":
            raise ValueError("--crop-stations é suportado apenas para targets do AlertaRio.")
        if crop_stations and not self.station_mapping:
            raise ValueError("crop_stations requer station_mapping.")

        for year in self.years:
            self._load_year(year)

        if not self.samples:
            raise ValueError(f"{split_name}: nenhuma janela disponível.")
        print(f"[{split_name}] Total de amostras: {len(self.samples)}", flush=True)

    def _configure_station_crop(self, height: int, width: int) -> None:
        if not self.crop_stations or self.crop_bounds is not None:
            return
        if not self.station_mapping or not self.station_mapping.is_file():
            raise FileNotFoundError(f"Mapeamento de estações não encontrado: {self.station_mapping}")

        with self.station_mapping.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        required = {"station_id", "pixel_i", "pixel_j"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(
                f"{self.station_mapping}: esperadas colunas {sorted(required)}."
            )

        station_pixels = []
        for row in rows:
            try:
                source_row, source_column = float(row["pixel_i"]), float(row["pixel_j"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{self.station_mapping}: pixel inválido para estação {row.get('station_id')!r}.") from error
            station_pixels.append((
                int(np.rint(source_row * height / self.mapping_height_orig)),
                int(np.rint(source_column * width / self.mapping_width_orig)),
            ))
        if not station_pixels:
            raise ValueError(f"{self.station_mapping}: nenhuma estação mapeada.")

        mapped = np.asarray(station_pixels, dtype=np.int64)
        if ((mapped[:, 0] < 0) | (mapped[:, 0] >= height) |
                (mapped[:, 1] < 0) | (mapped[:, 1] >= width)).any():
            raise ValueError(f"{self.station_mapping}: estação fora da grade {height}x{width}.")
        top = max(0, int(mapped[:, 0].min()) - self.crop_margin_pixels)
        bottom = min(height, int(mapped[:, 0].max()) + self.crop_margin_pixels + 1)
        left = max(0, int(mapped[:, 1].min()) - self.crop_margin_pixels)
        right = min(width, int(mapped[:, 1].max()) + self.crop_margin_pixels + 1)
        self.crop_bounds = (top, bottom, left, right)
        self.crop_metadata = {
            "enabled": True,
            "mapping": str(self.station_mapping),
            "margin_pixels": self.crop_margin_pixels,
            "source_shape": [height, width],
            "bounds": {"row_start": top, "row_stop": bottom, "column_start": left, "column_stop": right},
            "shape": [bottom - top, right - left],
            "station_count": len(station_pixels),
        }
        print(f"[{self.split_name}] Crop estações | bounds=({top}:{bottom}, {left}:{right}) | "
              f"shape=({bottom - top}, {right - left}) | margem={self.crop_margin_pixels}", flush=True)

    def _crop_slices(self) -> tuple[slice, slice]:
        if self.crop_bounds is None:
            return slice(None), slice(None)
        top, bottom, left, right = self.crop_bounds
        return slice(top, bottom), slice(left, right)

    def _load_year(self, year: int) -> None:
        year_dir = self.radar_root / f"year={year}"
        metadata_path = year_dir / "metadata.json"
        target_metadata_path = year_dir / TARGET_METADATA_FILES[self.target_source]
        required = (metadata_path, year_dir / "radar_frames.dat", target_metadata_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{self.split_name}: arquivos ausentes para {year}: {', '.join(missing)}"
            )

        with metadata_path.open(encoding="utf-8") as file:
            radar_metadata = json.load(file)
        with target_metadata_path.open(encoding="utf-8") as file:
            target_metadata = json.load(file)

        radar_shape = tuple(radar_metadata["shape"])
        target_shape = tuple(target_metadata["shape"])
        if len(radar_shape) != 4 or radar_shape[-1] != 3:
            raise ValueError(f"{year}: shape de radar inválido: {radar_shape}")
        if len(target_shape) != 4 or target_shape[-1] != 1:
            raise ValueError(f"{year}: shape de target inválido: {target_shape}")
        if radar_shape[:3] != target_shape[:3]:
            raise ValueError(
                f"{year}: radar {radar_shape} e target {target_shape} não estão alinhados."
            )
        self._configure_station_crop(radar_shape[1], radar_shape[2])

        frames = np.memmap(
            year_dir / radar_metadata.get("frames_file", "radar_frames.dat"),
            dtype=np.dtype(radar_metadata["dtype"]), mode="r", shape=radar_shape,
        )
        if target_metadata.get("format") == "sparse":
            sparse_path = year_dir / target_metadata["sparse_file"]
            if not sparse_path.is_file():
                raise FileNotFoundError(f"{year}: target esparso ausente: {sparse_path}")
            with np.load(sparse_path) as sparse:
                target = {name: np.asarray(sparse[name]) for name in ("frame", "row", "column", "value")}
            if not (len(target["frame"]) == len(target["row"]) == len(target["column"]) == len(target["value"])):
                raise ValueError(f"{year}: arrays esparsos com comprimentos diferentes.")
            if len(target["frame"]) and (target["frame"].min() < 0 or target["frame"].max() >= radar_shape[0]
                                       or target["row"].min() < 0 or target["row"].max() >= radar_shape[1]
                                       or target["column"].min() < 0 or target["column"].max() >= radar_shape[2]):
                raise ValueError(f"{year}: indices esparsos fora da grade.")
            self.year_data[year] = {"frames": frames, "sparse": target}
        else:
            targets = np.memmap(year_dir / target_metadata["Y_file"], dtype=np.dtype(target_metadata["Y_dtype"]), mode="r", shape=target_shape)
            masks = np.memmap(year_dir / target_metadata["M_file"], dtype=np.dtype(target_metadata["M_dtype"]), mode="r", shape=target_shape)
            self.year_data[year] = {"frames": frames, "targets": targets, "masks": masks}

        n_possible = len(frames) - (self.t_in + self.t_out) + 1
        if n_possible <= 0:
            raise ValueError(f"{year}: frames insuficientes para as sequências configuradas.")
        self.samples.extend((year, start) for start in range(0, n_possible, self.stride))
        print(
            f"[{self.split_name}] Ano {year} carregado | fonte={self.target_source} | "
            f"frames={len(frames)} | shape={radar_shape} | "
            f"amostras={len(range(0, n_possible, self.stride))}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        year, start = self.samples[index]
        data = self.year_data[year]
        x_end = start + self.t_in
        y_end = x_end + self.t_out
        row_slice, column_slice = self._crop_slices()
        x = np.array(data["frames"][start:x_end, row_slice, column_slice], dtype=np.float32) / 255.0
        if "sparse" in data:
            sparse = data["sparse"]
            left = np.searchsorted(sparse["frame"], x_end, side="left")
            right = np.searchsorted(sparse["frame"], y_end, side="left")
            y = np.zeros((self.t_out, x.shape[1], x.shape[2], 1), dtype=np.float32)
            m = np.zeros_like(y)
            frames = sparse["frame"][left:right] - x_end
            rows, columns = sparse["row"][left:right], sparse["column"][left:right]
            if self.crop_bounds is not None:
                top, bottom, crop_left, crop_right = self.crop_bounds
                inside = ((rows >= top) & (rows < bottom) &
                          (columns >= crop_left) & (columns < crop_right))
                frames, rows, columns = frames[inside], rows[inside] - top, columns[inside] - crop_left
            if self.crop_bounds is not None:
                values = sparse["value"][left:right][inside]
            else:
                values = sparse["value"][left:right]
            y[frames, rows, columns, 0] = values
            m[frames, rows, columns, 0] = 1.0
        else:
            y = np.array(data["targets"][x_end:y_end, row_slice, column_slice], dtype=np.float32)
            m = np.array(data["masks"][x_end:y_end, row_slice, column_slice], dtype=np.float32)
        return (
            torch.from_numpy(x).permute(3, 0, 1, 2),
            torch.from_numpy(y).permute(3, 0, 1, 2),
            torch.from_numpy(m).permute(3, 0, 1, 2),
        )

    def get_balanced_sample_weights(
        self, thresholds: tuple[float, float, float] = (1.25, 6.25, 12.5)
    ) -> tuple[np.ndarray, np.ndarray]:
        thresholds = tuple(float(value) for value in thresholds)
        if thresholds not in self._sample_class_cache:
            classes = np.zeros(len(self.samples), dtype=np.int64)
            for index, (year, start) in enumerate(self.samples):
                data = self.year_data[year]
                y_start = start + self.t_in
                if "sparse" in data:
                    sparse = data["sparse"]
                    left = np.searchsorted(sparse["frame"], y_start, side="left")
                    right = np.searchsorted(sparse["frame"], y_start + self.t_out, side="left")
                    values = sparse["value"][left:right]
                    if self.crop_bounds is not None:
                        top, bottom, crop_left, crop_right = self.crop_bounds
                        rows, columns = sparse["row"][left:right], sparse["column"][left:right]
                        values = values[(rows >= top) & (rows < bottom) &
                                        (columns >= crop_left) & (columns < crop_right)]
                    maximum = np.expm1(values).max() if len(values) else None
                else:
                    row_slice, column_slice = self._crop_slices()
                    target = np.array(data["targets"][y_start:y_start + self.t_out, row_slice, column_slice])
                    mask = np.array(data["masks"][y_start:y_start + self.t_out, row_slice, column_slice]) > 0
                    maximum = np.expm1(target[mask]).max() if mask.any() else None
                if maximum is not None:
                    classes[index] = np.searchsorted(thresholds, maximum, side="right")
            self._sample_class_cache[thresholds] = classes

        classes = self._sample_class_cache[thresholds]
        counts = np.bincount(classes, minlength=4).astype(np.int64)
        by_class = np.zeros(4, dtype=np.float64)
        by_class[counts > 0] = 1.0 / counts[counts > 0]
        return by_class[classes], counts
