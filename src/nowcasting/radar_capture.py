from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_capture_timestamp(path: Path) -> datetime:
    """Parse the ``YYYY_MM_DD_HH_MM`` timestamp encoded in a capture filename."""
    parts = path.stem.rstrip("_").split("_")
    if len(parts) != 5:
        raise ValueError(f"Nome fora do padrão esperado: {path.name}")
    return datetime(*map(int, parts))


def load_capture_config(path: Path) -> dict[str, Any]:
    """Load and validate a versioned geometry specification for PNG captures."""
    config = json.loads(path.read_text(encoding="utf-8"))
    source = config.get("source_image", {})
    crop = config.get("reflectivity_crop", {})
    required_source = ("width", "height")
    required_crop = ("left", "top", "right_exclusive", "bottom_exclusive")

    if any(key not in source for key in required_source) or any(
        key not in crop for key in required_crop
    ):
        raise ValueError(f"Configuração de captura inválida: {path}")

    width, height = int(source["width"]), int(source["height"])
    left, top = int(crop["left"]), int(crop["top"])
    right, bottom = int(crop["right_exclusive"]), int(crop["bottom_exclusive"])
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Recorte fora dos limites da imagem: {path}")
    return config


def crop_bounds(config: dict[str, Any]) -> tuple[int, int, int, int]:
    crop = config["reflectivity_crop"]
    return (
        int(crop["left"]),
        int(crop["top"]),
        int(crop["right_exclusive"]),
        int(crop["bottom_exclusive"]),
    )


def load_reflectivity_rgb(
    path: Path,
    capture_config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Read a PNG capture, optionally crop overlays, and preserve transparent pixels as black."""
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)

    if capture_config is not None:
        expected = capture_config["source_image"]
        expected_size = (int(expected["height"]), int(expected["width"]))
        if rgba.shape[:2] != expected_size:
            raise ValueError(
                f"Geometria inesperada em {path}: {rgba.shape[1]}x{rgba.shape[0]}; "
                f"esperado {expected_size[1]}x{expected_size[0]}"
            )
        left, top, right, bottom = crop_bounds(capture_config)
        rgba = rgba[top:bottom, left:right]

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3] > 0
    return rgb * alpha[:, :, None]
