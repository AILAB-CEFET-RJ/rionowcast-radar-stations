from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nowcasting.operational_mock import run_mock
from nowcasting.paths import project_root
from nowcasting.radar_capture import load_capture_config


PROJECT_ROOT = project_root()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera uma demonstracao operacional por persistencia nas estacoes, sem usar modelo treinado."
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--alertario-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-height", type=int, default=128)
    parser.add_argument("--frame-width", type=int, default=128)
    parser.add_argument("--max-m15", type=float, default=175.0)
    parser.add_argument("--crop-stations", action="store_true", help="Recorta o painel raster para a ROI das estações.")
    parser.add_argument("--crop-margin-pixels", type=int, default=20)
    parser.add_argument(
        "--geographic-grid", type=Path, default=PROJECT_ROOT / "data" / "sumare_radar_latlon_grid.npz",
        help="Grade lat/lon usada para criar o mapa Folium.",
    )
    parser.add_argument("--show-station-names", action="store_true")
    parser.add_argument(
        "--map-tiles", choices=("none", "openstreetmap"), default="none",
        help="Base cartografica. 'none' evita requisicoes a provedores externos e e o padrao da demo.",
    )
    parser.add_argument(
        "--municipal-basemap", type=Path,
        help="GeoJSON local derivado da malha municipal do IBGE para compor o mapa.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = json.loads(args.events.read_text(encoding="utf-8"))
    if not 0 <= args.event_index < len(events):
        raise IndexError(f"--event-index deve estar entre 0 e {len(events) - 1}.")
    summary = run_mock(
        events[args.event_index],
        args.capture_root,
        load_capture_config(args.capture_config),
        args.alertario_root,
        pd.read_csv(args.mapping),
        args.output_dir,
        args.frame_width,
        args.frame_height,
        args.max_m15,
        args.crop_stations,
        args.crop_margin_pixels,
        args.geographic_grid,
        args.show_station_names,
        args.map_tiles,
        args.municipal_basemap,
    )
    print(
        f"Mock concluido | estações={summary['stations_with_persistence_forecast']} | "
        f"saida={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
