"""Datasets densos para previsão de sequências RGB do radar.

Este módulo trata o produto RGB do radar como imagem. Não converte cores em
refletividade nem em taxa de precipitação.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RadarFrameSequenceDataset(Dataset):
    """Retorna cinco frames RGB passados e cinco frames RGB futuros.

    Os anos recebidos já definem o split temporal. Janelas que atravessam
    lacunas do acervo de radar são descartadas.
    """

    def __init__(
        self,
        radar_root: str | Path,
        years: list[int],
        *,
        t_in: int = 5,
        t_out: int = 5,
        stride: int = 5,
        split_name: str = "dataset",
    ):
        if not years:
            raise ValueError("O split não contém anos.")
        if min(t_in, t_out, stride) <= 0:
            raise ValueError("t_in, t_out e stride devem ser positivos.")
        self.radar_root = Path(radar_root)
        self.years = sorted(set(years))
        self.t_in, self.t_out, self.stride = t_in, t_out, stride
        self.split_name = split_name
        self.year_data: dict[int, dict[str, object]] = {}
        self.samples: list[tuple[int, int]] = []
        for year in self.years:
            self._load_year(year)
        if not self.samples:
            raise ValueError(f"{split_name}: nenhuma janela disponível.")
        print(f"[{split_name}] Total de sequências RGB: {len(self.samples)}", flush=True)

    def _load_year(self, year: int) -> None:
        year_dir = self.radar_root / f"year={year}"
        metadata_path = year_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"{self.split_name}: metadata ausente para {year}: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shape = tuple(metadata.get("shape", ()))
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"{year}: shape RGB de radar inválido: {shape}")
        frames_path = year_dir / metadata.get("frames_file", "radar_frames.dat")
        if not frames_path.is_file():
            raise FileNotFoundError(f"{year}: frames de radar ausentes: {frames_path}")
        frames = np.memmap(frames_path, dtype=np.dtype(metadata.get("dtype", "uint8")), mode="r", shape=shape)
        starts: list[int] = list(range(0, len(frames) - self.t_in - self.t_out + 1, self.stride))
        if metadata.get("enforce_timestamp_continuity", False):
            timestamps_path = year_dir / metadata.get("timestamps_file", "radar_timestamps.npy")
            if not timestamps_path.is_file():
                raise FileNotFoundError(f"{year}: timestamps obrigatórios ausentes: {timestamps_path}")
            timestamps = np.load(timestamps_path, allow_pickle=False)
            if len(timestamps) != len(frames):
                raise ValueError(f"{year}: timestamps e frames têm comprimentos diferentes.")
            minutes = np.asarray(timestamps, dtype="datetime64[m]").astype(np.int64)
            expected = int(metadata.get("aggregate_minutes", 15))
            discontinuities = np.diff(minutes) != expected
            prefix = np.concatenate(([0], np.cumsum(discontinuities, dtype=np.int64)))
            sequence_length = self.t_in + self.t_out
            before = len(starts)
            starts = [start for start in starts if prefix[start] == prefix[start + sequence_length - 1]]
            discarded = before - len(starts)
            if discarded:
                print(f"[{self.split_name}] Ano {year} | descartadas {discarded} sequências que cruzam lacunas do radar", flush=True)
        self.year_data[year] = {"frames": frames, "metadata": metadata}
        self.samples.extend((year, start) for start in starts)
        print(f"[{self.split_name}] Ano {year} carregado | frames={len(frames)} | shape={shape} | sequências={len(starts)}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        year, start = self.samples[index]
        frames = self.year_data[year]["frames"]
        assert isinstance(frames, np.memmap)
        middle = start + self.t_in
        x = np.array(frames[start:middle], dtype=np.float32) / 255.0
        y = np.array(frames[middle:middle + self.t_out], dtype=np.float32) / 255.0
        return torch.from_numpy(x).permute(3, 0, 1, 2), torch.from_numpy(y).permute(3, 0, 1, 2)
