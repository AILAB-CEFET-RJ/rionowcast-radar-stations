"""Compatibilidade para imports antigos.

O loader canônico está em ``src/nowcasting/dataset.py``. Novos scripts devem
importá-lo como ``from nowcasting.dataset import RadarStationMemmapDataset``.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.dataset import RadarStationMemmapDataset, parse_years

__all__ = ["RadarStationMemmapDataset", "parse_years"]
