"""Compatibilidade para o caminho antigo do loader memmap.

O módulo canônico é ``nowcasting.dataset``.
"""

from nowcasting.dataset import RadarStationMemmapDataset, parse_years

__all__ = ["RadarStationMemmapDataset", "parse_years"]
