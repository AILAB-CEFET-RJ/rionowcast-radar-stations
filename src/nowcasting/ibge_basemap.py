"""Preparation of local IBGE municipal boundaries for operational maps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import shapefile


def radar_extent(
    geographic_grid_path: Path,
    margin_degrees: float = 0.0,
    frame_height: int | None = None,
    frame_width: int | None = None,
    crop_bounds: tuple[int, int, int, int] | None = None,
) -> tuple[float, float, float, float]:
    """Return ``min_lon, min_lat, max_lon, max_lat`` for a radar grid or its ROI."""
    if margin_degrees < 0:
        raise ValueError("margin_degrees não pode ser negativa.")
    with np.load(geographic_grid_path) as grid:
        latitude = grid["lat"]
        longitude = grid["lon"]
    if frame_height is not None or frame_width is not None:
        if frame_height is None or frame_width is None or min(frame_height, frame_width) <= 0:
            raise ValueError("frame_height e frame_width devem ser positivos quando a ROI for informada.")
        row_indices = np.clip(
            ((np.arange(frame_height) + 0.5) * latitude.shape[0] / frame_height).astype(int),
            0,
            latitude.shape[0] - 1,
        )
        column_indices = np.clip(
            ((np.arange(frame_width) + 0.5) * longitude.shape[1] / frame_width).astype(int),
            0,
            longitude.shape[1] - 1,
        )
        latitude = latitude[row_indices][:, column_indices]
        longitude = longitude[row_indices][:, column_indices]
    if crop_bounds is not None:
        top, bottom, left, right = crop_bounds
        if not (0 <= top < bottom <= latitude.shape[0] and 0 <= left < right <= latitude.shape[1]):
            raise ValueError(f"crop_bounds inválidos para a grade: {crop_bounds}")
        latitude = latitude[top:bottom, left:right]
        longitude = longitude[top:bottom, left:right]
    return (
        float(longitude.min()) - margin_degrees,
        float(latitude.min()) - margin_degrees,
        float(longitude.max()) + margin_degrees,
        float(latitude.max()) + margin_degrees,
    )


def bbox_intersects(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    """Return whether two ``min_lon, min_lat, max_lon, max_lat`` boxes intersect."""
    left_min_lon, left_min_lat, left_max_lon, left_max_lat = left
    right_min_lon, right_min_lat, right_max_lon, right_max_lat = right
    return not (
        left_max_lon < right_min_lon
        or right_max_lon < left_min_lon
        or left_max_lat < right_min_lat
        or right_max_lat < left_min_lat
    )


def build_municipal_basemap(
    shapefile_path: Path,
    geographic_grid_path: Path,
    output_path: Path,
    margin_degrees: float = 0.1,
    frame_height: int | None = None,
    frame_width: int | None = None,
    crop_bounds: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Filter IBGE municipal polygons to the radar context and write GeoJSON."""
    if not shapefile_path.is_file():
        raise FileNotFoundError(f"Shapefile não encontrado: {shapefile_path}")
    if not geographic_grid_path.is_file():
        raise FileNotFoundError(f"Grade geográfica não encontrada: {geographic_grid_path}")
    extent = radar_extent(
        geographic_grid_path, margin_degrees, frame_height, frame_width, crop_bounds
    )
    # The IBGE 2024 RJ .cpg declares Windows-1252; names such as "Valença"
    # fail when the DBF is forced through UTF-8.
    reader = shapefile.Reader(str(shapefile_path), encoding="1252")
    features = []
    for item in reader.iterShapeRecords():
        shape_bbox = tuple(float(value) for value in item.shape.bbox)
        if not bbox_intersects(extent, shape_bbox):
            continue
        record = item.record.as_dict()
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "CD_MUN": str(record.get("CD_MUN", "")),
                    "NM_MUN": str(record.get("NM_MUN", "")),
                },
                "geometry": item.shape.__geo_interface__,
            }
        )
    if not features:
        raise ValueError("Nenhum município do shapefile intersecta a extensão do radar.")
    collection = {
        "type": "FeatureCollection",
        "source": "IBGE Malha Municipal 2024",
        "radar_extent": {
            "min_longitude": extent[0],
            "min_latitude": extent[1],
            "max_longitude": extent[2],
            "max_latitude": extent[3],
        },
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collection, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "source": "IBGE Malha Municipal 2024",
        "municipalities": len(features),
        "extent": extent,
        "crop_bounds": list(crop_bounds) if crop_bounds is not None else None,
        "output": str(output_path),
    }
