"""Shared, causal GOES-16 ABI utilities for the audited nowcasting pipeline."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


GOES_FILENAME = re.compile(
    r"-M\d(?P<channel>C\d{2})_G16_s(?P<start>\d{13,14})_e(?P<end>\d{13,14})_c\d+\.nc$"
)


def load_goes_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "version", "source", "channels", "temporal_alignment", "target_grid", "storage"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"{path}: campos obrigatórios ausentes: {sorted(missing)}")
    channels = config["channels"]
    if not channels or any(not re.fullmatch(r"C\d{2}", str(channel)) for channel in channels):
        raise ValueError(f"{path}: channels deve conter identificadores ABI como C13.")
    if len(set(channels)) != len(channels):
        raise ValueError(f"{path}: channels contém duplicatas.")
    if config["temporal_alignment"].get("selection") != "latest scene whose end timestamp is at or before the radar window start":
        raise ValueError(f"{path}: somente seleção causal latest-at-or-before é suportada.")
    if float(config["temporal_alignment"].get("maximum_scene_age_minutes", 0)) <= 0:
        raise ValueError(f"{path}: maximum_scene_age_minutes deve ser positivo.")
    if config["storage"].get("missing_value") != "NaN":
        raise ValueError(f"{path}: v1 exige NaN para valores GOES ausentes.")
    return config


def config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_goes_filename(path: str | Path) -> tuple[str, datetime]:
    """Return ABI channel and UTC scene-start time from a NOAA CMIPF filename."""
    channel, start, _ = parse_goes_filename_bounds(path)
    return channel, start


def parse_goes_filename_bounds(path: str | Path) -> tuple[str, datetime, datetime]:
    """Return ABI channel plus UTC scene start/end times from a NOAA filename."""
    match = GOES_FILENAME.search(Path(path).name)
    if not match:
        raise ValueError(f"Nome GOES ABI-L2-CMIPF inválido: {Path(path).name}")
    start_text, end_text = match.group("start"), match.group("end")
    # NOAA uses YYYYJJJHHMMSS (occasionally a tenths-of-second digit is appended).
    start = datetime.strptime(start_text[:13], "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_text[:13], "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
    if end < start:
        raise ValueError(f"Nome GOES ABI-L2-CMIPF inválido: término anterior ao início em {Path(path).name}")
    return match.group("channel"), start, end


def ensure_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, np.datetime64):
        timestamp = datetime.fromtimestamp(
            value.astype("datetime64[ns]").astype(np.int64) / 1_000_000_000,
            tz=timezone.utc,
        )
    else:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def select_causal_scene(
    candidates: list[tuple[datetime, datetime, Path]], target: datetime, maximum_age_minutes: float,
) -> tuple[datetime, datetime, Path] | None:
    """Select the latest fully acquired scene available at ``target``."""
    target = ensure_utc(target)
    eligible = [(start, end, path) for start, end, path in candidates if end <= target]
    if not eligible:
        return None
    selected = max(eligible, key=lambda item: item[1])
    if target - selected[1] > timedelta(minutes=maximum_age_minutes):
        return None
    return selected


def target_grid_latlon(radar_config: dict[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the radar-equivalent AEQD grid as WGS84 latitude/longitude arrays."""
    try:
        from pyproj import CRS, Transformer
    except ImportError as error:  # pragma: no cover - dependency declared by the project
        raise RuntimeError("A reprojeção GOES requer pyproj.") from error
    geo = radar_config["historical_georeferencing"]
    radar = geo["radar_center_wgs84"]
    radius = float(geo["display_range_km"]) * 1_000.0
    # Pixel centres cover the same radar-centred square represented by the bitmap.
    east = np.linspace(-radius, radius, width, dtype=np.float64)
    north = np.linspace(radius, -radius, height, dtype=np.float64)
    east_grid, north_grid = np.meshgrid(east, north)
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={float(radar['latitude'])} +lon_0={float(radar['longitude'])} "
        "+R=6371008.8 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(east_grid, north_grid)
    return np.asarray(latitude), np.asarray(longitude)


def _nearest_indices(coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Nearest-neighbour indices for a monotonic one-dimensional coordinate."""
    if coordinates.ndim != 1 or len(coordinates) < 2:
        raise ValueError("Coordenada ABI inválida.")
    reverse = coordinates[0] > coordinates[-1]
    ordered = coordinates[::-1] if reverse else coordinates
    indexes = np.searchsorted(ordered, values)
    indexes = np.clip(indexes, 1, len(ordered) - 1)
    previous, following = ordered[indexes - 1], ordered[indexes]
    indexes -= np.abs(values - previous) <= np.abs(values - following)
    return len(coordinates) - 1 - indexes if reverse else indexes


@dataclass(frozen=True)
class AbiSampler:
    """Reusable nearest-neighbour lookup from the radar grid to an ABI grid."""

    projection_digest: str
    x_digest: str
    y_digest: str
    x_index: np.ndarray
    y_index: np.ndarray


def _scene_geometry(source) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    projection = source.variables.get("goes_imager_projection")
    if projection is None:
        raise ValueError("goes_imager_projection ausente.")
    attrs = {name: projection.getncattr(name) for name in projection.ncattrs()}
    if "x" not in source.variables or "y" not in source.variables:
        raise ValueError("esperado x e y no NetCDF CMIPF.")
    return attrs, np.asarray(source.variables["x"][:]), np.asarray(source.variables["y"][:])


def _geometry_digest(value: object) -> str:
    if isinstance(value, np.ndarray):
        payload = np.ascontiguousarray(value).view(np.uint8).tobytes()
    else:
        payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_abi_sampler(path: Path, latitude: np.ndarray, longitude: np.ndarray) -> AbiSampler:
    """Precompute radar-grid lookup indices from one representative ABI scene."""
    try:
        import netCDF4
        from pyproj import CRS, Transformer
    except ImportError as error:
        raise RuntimeError("Leitura GOES requer netCDF4 e pyproj; instale requirements.txt.") from error
    with netCDF4.Dataset(path) as source:
        if "CMI" not in source.variables:
            raise ValueError(f"{path}: esperado CMI no NetCDF CMIPF.")
        attrs, x, y = _scene_geometry(source)
        height = float(attrs["perspective_point_height"])
        geos = CRS.from_cf(attrs)
        transformer = Transformer.from_crs("EPSG:4326", geos, always_xy=True)
        x_m, y_m = transformer.transform(longitude, latitude)
        x_index = _nearest_indices(x, np.asarray(x_m) / height)
        y_index = _nearest_indices(y, np.asarray(y_m) / height)
        return AbiSampler(
            projection_digest=_geometry_digest(attrs),
            x_digest=_geometry_digest(x),
            y_digest=_geometry_digest(y),
            x_index=x_index,
            y_index=y_index,
        )


def read_and_reproject_scene_with_sampler(path: Path, sampler: AbiSampler) -> np.ndarray:
    """Read one CMI field using a previously validated ABI spatial lookup."""
    try:
        import netCDF4
    except ImportError as error:
        raise RuntimeError("Leitura GOES requer netCDF4; instale requirements.txt.") from error
    with netCDF4.Dataset(path) as source:
        if "CMI" not in source.variables:
            raise ValueError(f"{path}: esperado CMI no NetCDF CMIPF.")
        attrs, x, y = _scene_geometry(source)
        if (
            _geometry_digest(attrs) != sampler.projection_digest
            or _geometry_digest(x) != sampler.x_digest
            or _geometry_digest(y) != sampler.y_digest
        ):
            raise ValueError(f"{path}: geometria ABI difere da cena usada para construir o sampler.")
        raw_cmi = source.variables["CMI"][:]
        cmi = np.asarray(raw_cmi.filled(np.nan) if np.ma.isMaskedArray(raw_cmi) else raw_cmi, dtype=np.float32)
        values = cmi[sampler.y_index, sampler.x_index]
        fill_value = getattr(source.variables["CMI"], "_FillValue", None)
        if fill_value is not None:
            values = np.where(values == fill_value, np.nan, values)
        return np.asarray(values, dtype=np.float32)


def read_and_reproject_scene(path: Path, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Read one CMIPF scene, building a sampler for single-scene callers."""
    sampler = build_abi_sampler(path, latitude, longitude)
    return read_and_reproject_scene_with_sampler(path, sampler)
