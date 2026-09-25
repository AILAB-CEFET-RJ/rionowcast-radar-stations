from __future__ import annotations

import argparse
import json
from pathlib import Path

from nowcasting.ibge_basemap import build_municipal_basemap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte a malha municipal do IBGE em GeoJSON local para o produto operacional."
    )
    parser.add_argument("--shapefile", type=Path, required=True)
    parser.add_argument("--geographic-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin-degrees", type=float, default=0.1)
    parser.add_argument("--frame-height", type=int)
    parser.add_argument("--frame-width", type=int)
    parser.add_argument(
        "--crop-bounds", nargs=4, type=int, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="ROI na grade do radar reduzida; requer --frame-height e --frame-width.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.frame_height is None) != (args.frame_width is None):
        raise ValueError("--frame-height e --frame-width devem ser informados juntos.")
    if args.crop_bounds is not None and args.frame_height is None:
        raise ValueError("--crop-bounds requer --frame-height e --frame-width.")
    summary = build_municipal_basemap(
        args.shapefile,
        args.geographic_grid,
        args.output,
        args.margin_degrees,
        args.frame_height,
        args.frame_width,
        tuple(args.crop_bounds) if args.crop_bounds is not None else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
