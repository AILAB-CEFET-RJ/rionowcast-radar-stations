"""Create a documented geographic station mapping for historical Sumaré PNGs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.radar_capture import load_capture_config
from nowcasting.radar_georeferencing import geographic_station_pixels, require_pixels_in_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera mapeamento AlertaRio geográfico candidato para o bitmap histórico do Radar Sumaré."
    )
    parser.add_argument("--input-mapping", type=Path, required=True)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_capture_config(args.capture_config)
    georeferencing = config.get("historical_georeferencing")
    if not georeferencing:
        raise ValueError("A configuração não declara historical_georeferencing.")
    source = config["source_image"]
    height, width = int(source["height"]), int(source["width"])
    mapping = pd.read_csv(args.input_mapping)
    rows, columns = geographic_station_pixels(mapping, height, width, georeferencing)
    require_pixels_in_bounds(rows, columns, height, width)

    result = mapping.copy()
    result["pixel_i"] = rows
    result["pixel_j"] = columns
    result["mapping_method"] = "aeqd_radar_centered_north_up"
    result["mapping_config_version"] = config["version"]
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_mapping, index=False)
    metadata = {
        "output_mapping": str(args.output_mapping),
        "stations": int(len(result)),
        "source_shape": [height, width],
        "historical_georeferencing": georeferencing,
        "status": georeferencing.get("status", "candidate_requires_station_radar_alignment_validation"),
    }
    args.output_mapping.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Mapeamento geográfico candidato | estações={len(result)} | saída={args.output_mapping}", flush=True)


if __name__ == "__main__":
    main()
