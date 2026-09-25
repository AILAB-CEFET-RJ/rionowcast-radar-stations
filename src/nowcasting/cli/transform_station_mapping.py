from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nowcasting.radar_capture import crop_bounds, load_capture_config


def transform_mapping(mapping: pd.DataFrame, capture_config: dict) -> pd.DataFrame:
    required = {"station_id", "pixel_i", "pixel_j"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapeamento: colunas ausentes: {sorted(missing)}")
    transform = capture_config.get("station_mapping_transform")
    if not transform:
        raise ValueError("A configuração não declara station_mapping_transform.")
    legacy = transform["legacy_grid"]
    result = mapping.copy()
    result["legacy_pixel_i"] = pd.to_numeric(result["pixel_i"], errors="raise")
    result["legacy_pixel_j"] = pd.to_numeric(result["pixel_j"], errors="raise")
    if (
        result["legacy_pixel_i"].lt(0).any()
        or result["legacy_pixel_i"].ge(int(legacy["height"])).any()
        or result["legacy_pixel_j"].lt(0).any()
        or result["legacy_pixel_j"].ge(int(legacy["width"])).any()
    ):
        raise ValueError("Mapeamento legado possui pixels fora de sua grade declarada.")

    left, top, right, bottom = crop_bounds(capture_config)
    crop_height, crop_width = bottom - top, right - left
    row_offset = int(transform["row_offset"])
    column_offset = int(transform["column_offset"])
    result["pixel_i"] = result["legacy_pixel_i"] + row_offset
    result["pixel_j"] = result["legacy_pixel_j"] + column_offset
    if (
        result["pixel_i"].lt(0).any()
        or result["pixel_i"].ge(crop_height).any()
        or result["pixel_j"].lt(0).any()
        or result["pixel_j"].ge(crop_width).any()
    ):
        raise ValueError("Transformação colocou estação fora do recorte canônico.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transforma o mapeamento AlertaRio para o recorte canônico do radar."
    )
    parser.add_argument("--input-mapping", type=Path, required=True)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_capture_config(args.capture_config)
    mapping = pd.read_csv(args.input_mapping)
    transformed = transform_mapping(mapping, config)
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(args.output_mapping, index=False)
    print(
        f"Mapeamento transformado | estações={len(transformed)} | "
        f"saída={args.output_mapping}",
        flush=True,
    )


if __name__ == "__main__":
    main()
