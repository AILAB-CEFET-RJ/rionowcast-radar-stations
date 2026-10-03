"""Geographic candidates for the historical Radar Sumaré bitmap."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_008.8


def geographic_station_pixels(
    mapping: pd.DataFrame,
    source_height: int,
    source_width: int,
    georeferencing: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Map station WGS84 coordinates to a north-up, radar-centred bitmap.

    The historical image product is assumed to cover a square whose centre is
    the radar location and whose edge is ``range_km`` from that centre.  The
    assumption is explicit in the capture config and must be validated by the
    station/radar alignment audit before it is used to generate targets.
    """
    required = {"latitude", "longitude"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapeamento: colunas ausentes para georreferenciamento: {sorted(missing)}")
    if source_height <= 1 or source_width <= 1:
        raise ValueError("A geometria da imagem deve ter ao menos dois pixels por eixo.")

    radar = georeferencing["radar_center_wgs84"]
    range_m = float(georeferencing["display_range_km"]) * 1_000.0
    if range_m <= 0:
        raise ValueError("display_range_km deve ser positivo.")

    longitude = pd.to_numeric(mapping["longitude"], errors="raise").to_numpy(float)
    latitude = pd.to_numeric(mapping["latitude"], errors="raise").to_numpy(float)
    latitude_rad = np.deg2rad(latitude)
    longitude_rad = np.deg2rad(longitude)
    radar_latitude_rad = np.deg2rad(float(radar["latitude"]))
    radar_longitude_rad = np.deg2rad(float(radar["longitude"]))
    delta_longitude = longitude_rad - radar_longitude_rad
    cos_c = (
        np.sin(radar_latitude_rad) * np.sin(latitude_rad)
        + np.cos(radar_latitude_rad) * np.cos(latitude_rad) * np.cos(delta_longitude)
    )
    central_angle = np.arccos(np.clip(cos_c, -1.0, 1.0))
    scale = np.ones_like(central_angle)
    nonzero = central_angle > 1e-12
    scale[nonzero] = central_angle[nonzero] / np.sin(central_angle[nonzero])
    east_m = EARTH_RADIUS_M * scale * np.cos(latitude_rad) * np.sin(delta_longitude)
    north_m = EARTH_RADIUS_M * scale * (
        np.cos(radar_latitude_rad) * np.sin(latitude_rad)
        - np.sin(radar_latitude_rad) * np.cos(latitude_rad) * np.cos(delta_longitude)
    )

    center_row = (source_height - 1) / 2.0
    center_column = (source_width - 1) / 2.0
    meters_per_column = range_m / center_column
    meters_per_row = range_m / center_row
    rows = np.rint(center_row - np.asarray(north_m) / meters_per_row).astype(np.int64)
    columns = np.rint(center_column + np.asarray(east_m) / meters_per_column).astype(np.int64)
    return rows, columns


def require_pixels_in_bounds(rows: np.ndarray, columns: np.ndarray, height: int, width: int) -> None:
    invalid = (rows < 0) | (rows >= height) | (columns < 0) | (columns >= width)
    if invalid.any():
        count = int(invalid.sum())
        raise ValueError(f"Georreferenciamento colocou {count} estação(ões) fora do bitmap {height}x{width}.")


def orientation_candidates(
    rows: np.ndarray, columns: np.ndarray, height: int, width: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the eight image-axis conventions for one physical geometry.

    A raw PNG has no georeferencing metadata.  These variants preserve the
    radar-centred scale while testing whether rows/columns were exchanged or
    reflected during rendering.  The proportional rescale makes transpose
    valid for the 654x656 non-square bitmap.
    """
    def transpose(values: np.ndarray, source: int, destination: int) -> np.ndarray:
        return np.rint(values * (destination - 1) / (source - 1)).astype(np.int64)

    transposed_rows = transpose(columns, width, height)
    transposed_columns = transpose(rows, height, width)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for prefix, base_rows, base_columns in (
        ("geographic_aeqd", rows, columns),
        ("geographic_aeqd_transpose", transposed_rows, transposed_columns),
    ):
        for vertical, horizontal, suffix in (
            (False, False, "_north_up"),
            (True, False, "_flip_v"),
            (False, True, "_flip_h"),
            (True, True, "_flip_vh"),
        ):
            result[prefix + suffix] = (
                height - 1 - base_rows if vertical else base_rows,
                width - 1 - base_columns if horizontal else base_columns,
            )
    return result
