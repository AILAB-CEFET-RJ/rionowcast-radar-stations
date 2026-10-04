"""Causal optical-flow utilities for the RGB radar baseline."""
from __future__ import annotations

import numpy as np


def visual_echo(frames_rgb: np.ndarray) -> np.ndarray:
    """Return a non-physical scalar echo proxy from RGB radar frames.

    The maximum channel preserves colored echo boundaries without claiming that
    RGB values are calibrated reflectivity.
    """
    frames = np.asarray(frames_rgb, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Esperados frames RGB com shape (tempo, altura, largura, 3).")
    if not np.isfinite(frames).all():
        raise ValueError("Frames RGB contêm valores não finitos.")
    scale = 255.0 if frames.max(initial=0.0) > 1.0 else 1.0
    return np.clip(frames.max(axis=-1) / scale, 0.0, 1.0)


def extrapolate_visual_echo(frames_rgb: np.ndarray, horizons: int) -> tuple[np.ndarray, bool]:
    """Extrapolate the final visual-echo proxy using only past RGB frames.

    Returns ``(forecast, fallback)``. A zero-motion fallback is retained for
    nearly empty fields or optional-dependency numerical failures and must be
    reported by the caller.
    """
    if horizons <= 0:
        raise ValueError("horizons deve ser positivo.")
    echo = visual_echo(frames_rgb)
    try:
        from pysteps import motion, nowcasts
    except ImportError as error:
        raise RuntimeError("Fluxo óptico requer pysteps e opencv-python-headless; instale requirements.txt.") from error
    try:
        velocity = motion.get_method("LK")(echo)
        forecast = nowcasts.get_method("extrapolation")(echo[-1], velocity, horizons)
        forecast = np.asarray(forecast, dtype=np.float32)
        if forecast.shape != (horizons, *echo.shape[1:]):
            raise ValueError("Extrapolação óptica inválida.")
        # Pixels advectados para fora do domínio não têm eco observado.
        forecast = np.nan_to_num(forecast, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(forecast, 0.0, 1.0), False
    except Exception:
        return np.repeat(echo[-1: ], horizons, axis=0).astype(np.float32), True


def station_echo_features(
    forecast: np.ndarray, station_pixels: np.ndarray, radius: int,
) -> np.ndarray:
    """Extract mean, maximum and nonzero fraction around each station."""
    values = np.asarray(forecast, dtype=np.float32)
    pixels = np.asarray(station_pixels, dtype=np.int64)
    if values.ndim != 3 or radius < 0:
        raise ValueError("Forecast ou raio de vizinhança inválido.")
    result = np.empty((values.shape[0], len(pixels), 3), dtype=np.float32)
    height, width = values.shape[1:]
    for station, (row, column) in enumerate(pixels):
        top, bottom = max(0, row - radius), min(height, row + radius + 1)
        left, right = max(0, column - radius), min(width, column + radius + 1)
        window = values[:, top:bottom, left:right]
        result[:, station, 0] = window.mean(axis=(1, 2))
        result[:, station, 1] = window.max(axis=(1, 2))
        result[:, station, 2] = (window > 0).mean(axis=(1, 2))
    return result


def station_persistence_from_history(
    values: np.ndarray, masks: np.ndarray, horizons: int,
) -> tuple[np.ndarray, int]:
    """Repeat the last observed station value in a causal input history.

    ``values`` and ``masks`` have shape ``(time, station)`` and values are in
    the target's log1p space. Stations with no observed input receive zero,
    matching the B1 persistence convention.
    """
    history = np.asarray(values, dtype=np.float32)
    observed = np.asarray(masks, dtype=bool)
    if history.ndim != 2 or observed.shape != history.shape or horizons <= 0:
        raise ValueError("Histórico de estações, máscara ou horizontes inválidos.")
    positions = np.where(observed, np.arange(history.shape[0])[:, None], -1).max(axis=0)
    result = np.zeros(history.shape[1], dtype=np.float32)
    available = positions >= 0
    result[available] = history[positions[available], np.flatnonzero(available)]
    return np.broadcast_to(result, (horizons, len(result))).copy(), int((~available).sum())
