from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nowcasting.radar_capture import load_reflectivity_rgb


def aggregate_capture_bucket(
    paths: list[Path],
    capture_config: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    """Aggregate PNG captures into one 15-minute RGB radar frame."""
    if not paths:
        raise ValueError("Bucket sem PNGs para agregação.")
    aggregate = None
    for path in paths:
        rgb = load_reflectivity_rgb(path, capture_config)
        resized = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize((width, height), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        aggregate = resized if aggregate is None else np.maximum(aggregate, resized)
    return aggregate


def apply_model_crop(frames: np.ndarray, configuration: dict[str, Any]) -> np.ndarray:
    """Apply the station-region crop recorded by a trained model, when enabled."""
    crop = configuration.get("crop", {"enabled": False})
    if not crop.get("enabled", False):
        return frames
    bounds = crop["bounds"]
    return frames[
        :,
        int(bounds["row_start"]):int(bounds["row_stop"]),
        int(bounds["column_start"]):int(bounds["column_stop"]),
        :,
    ]


def prepare_radar_input(
    event: dict[str, Any],
    capture_root: Path,
    capture_config: dict[str, Any],
    configuration: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    """Build the normalized ``[1, 3, T, H, W]`` tensor expected by STConvS2S."""
    input_buckets = event["input"]
    expected_frames = int(configuration.get("step", 5))
    if len(input_buckets) != expected_frames:
        raise ValueError(
            f"Manifesto possui {len(input_buckets)} buckets de entrada; "
            f"o checkpoint espera {expected_frames}."
        )
    frames = []
    for bucket in input_buckets:
        paths = [capture_root / relative_path for relative_path in bucket["png_files"]]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"PNG ausente no manifesto: {missing[0]}")
        frames.append(aggregate_capture_bucket(paths, capture_config, width, height))
    stacked = apply_model_crop(np.stack(frames), configuration).astype(np.float32) / 255.0
    return np.transpose(stacked, (3, 0, 1, 2))[None, ...]


def validate_checkpoint_configuration(
    configuration: dict[str, Any], capture_config: dict[str, Any],
) -> None:
    """Reject checkpoints that were trained with incompatible radar geometry."""
    trained_capture = configuration.get("radar_capture_preprocessing")
    if trained_capture is None:
        raise ValueError(
            "Checkpoint legado sem radar_capture_preprocessing. Ele não pode ser "
            "aplicado às capturas recortadas para inferência operacional."
        )
    if trained_capture != capture_config:
        raise ValueError(
            "A configuração de captura do checkpoint difere da configuração de inferência."
        )
    if configuration.get("input_stations", False):
        raise ValueError(
            "Inferência com --input-stations ainda requer as observações defasadas "
            "das estações e não é suportada por este CLI inicial."
        )
