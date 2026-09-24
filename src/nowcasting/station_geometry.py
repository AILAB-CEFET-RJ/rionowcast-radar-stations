"""Coordinate transforms shared by station datasets and alignment audits."""
from __future__ import annotations

import numpy as np


def direct_resampled_pixels(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    height_orig: int,
    width_orig: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map original-grid coordinates directly to an output grid."""
    target_rows = np.rint(rows * height / height_orig).astype(np.int64)
    target_columns = np.rint(columns * width / width_orig).astype(np.int64)
    return np.clip(target_rows, 0, height - 1), np.clip(target_columns, 0, width - 1)


def sparse_target_pixels(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    height_orig: int,
    width_orig: int,
    source_height: int,
    source_width: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the generator's original-grid -> dense -> sparse transform."""
    source_rows = np.rint(rows * source_height / height_orig).astype(np.int64)
    source_columns = np.rint(columns * source_width / width_orig).astype(np.int64)
    target_rows = np.rint(source_rows * (height - 1) / max(source_height - 1, 1)).astype(np.int64)
    target_columns = np.rint(source_columns * (width - 1) / max(source_width - 1, 1)).astype(np.int64)
    return np.clip(target_rows, 0, height - 1), np.clip(target_columns, 0, width - 1)
