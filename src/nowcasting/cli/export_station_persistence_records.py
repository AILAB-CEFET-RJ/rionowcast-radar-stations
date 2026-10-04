"""Export B1 station persistence in the canonical forecast-record contract."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nowcasting.dataset import RadarStationMemmapDataset, parse_years
from nowcasting.forecast_records import export_flat_station_forecasts
from nowcasting.optical_flow import station_persistence_from_history
from nowcasting.station_dataset import load_station_pixels


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta previsões B1 de persistência no formato canônico.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mapping-height-orig", type=int, default=654)
    parser.add_argument("--mapping-width-orig", type=int, default=656)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="B1-persistence")
    args = parser.parse_args()
    years = parse_years(args.years)
    dataset = RadarStationMemmapDataset(args.dataset_root, years, target_source="alertario", t_in=args.step,
                                        t_out=args.step, stride=args.stride, input_stations=True, split_name="export")
    first = dataset.year_data[years[0]]["frames"]
    pixels, station_ids = load_station_pixels(args.mapping, first.shape[1], first.shape[2],
                                              height_orig=args.mapping_height_orig, width_orig=args.mapping_width_orig)
    predictions = []
    for index in range(len(dataset)):
        x, target, _ = dataset[index]
        values = x[3, :, pixels[:, 0], pixels[:, 1]].numpy()
        masks = x[4, :, pixels[:, 0], pixels[:, 1]].numpy()
        repeated, _ = station_persistence_from_history(values, masks, target.shape[1])
        predictions.append(repeated.reshape(-1))
    summary = export_flat_station_forecasts(dataset, pixels, station_ids, np.concatenate(predictions), args.output,
                                            experiment_id=args.experiment_id)
    print(f"B1 exportado | linhas={summary['rows']} | saída={args.output}")


if __name__ == "__main__":
    main()
