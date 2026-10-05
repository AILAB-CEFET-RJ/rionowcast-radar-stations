"""Export canonical station forecasts from a trained STConvS2S checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nowcasting.cli.train import model_class
from nowcasting.dataset import RadarStationMemmapDataset
from nowcasting.forecast_records import export_flat_station_forecasts
from nowcasting.residual_persistence import build_forecaster
from nowcasting.station_dataset import load_station_pixels


def station_pixels(dataset: RadarStationMemmapDataset, configuration: dict) -> tuple[np.ndarray, list[int]]:
    year = dataset.years[0]
    frames = dataset.year_data[year]["frames"]
    pixels, ids = load_station_pixels(configuration["station_mapping"], frames.shape[1], frames.shape[2],
                                      height_orig=int(configuration["mapping_height_orig"]),
                                      width_orig=int(configuration["mapping_width_orig"]))
    if dataset.crop_bounds is not None:
        top, bottom, left, right = dataset.crop_bounds
        inside = ((pixels[:, 0] >= top) & (pixels[:, 0] < bottom) &
                  (pixels[:, 1] >= left) & (pixels[:, 1] < right))
        if not inside.all():
            raise ValueError("O crop do experimento excluiu estações do mapeamento.")
        pixels = pixels.copy()
        pixels[:, 0] -= top
        pixels[:, 1] -= left
    return pixels, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta previsões de checkpoint STConvS2S no contrato canônico.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cuda", default="0")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size deve ser positivo.")
    configuration = json.loads((args.experiment_dir / "configuration.json").read_text(encoding="utf-8"))
    required = ("dataset_root", "test_years", "step", "stride", "model", "num_layers", "hidden_dim",
                "kernel_size", "stconvs2s_root", "target_source", "station_mapping", "mapping_height_orig",
                "mapping_width_orig", "crop_stations", "crop_margin_pixels", "input_stations", "input_goes")
    missing = [name for name in required if name not in configuration]
    if missing:
        raise ValueError(f"configuration.json não contém: {', '.join(missing)}")
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    dataset = RadarStationMemmapDataset(
        configuration["dataset_root"], configuration["test_years"], t_in=int(configuration["step"]),
        t_out=int(configuration["step"]), stride=int(configuration["stride"] or configuration["step"]),
        target_source=configuration["target_source"], split_name="export", crop_stations=configuration["crop_stations"],
        crop_margin_pixels=int(configuration["crop_margin_pixels"]), input_stations=configuration["input_stations"],
        input_goes=configuration["input_goes"], station_mapping=configuration["station_mapping"],
        mapping_height_orig=int(configuration["mapping_height_orig"]), mapping_width_orig=int(configuration["mapping_width_orig"]),
    )
    pixels, ids = station_pixels(dataset, configuration)
    sample_x, sample_y, _ = dataset[0]
    constructor = model_class(Path(configuration["stconvs2s_root"]), configuration["model"])
    model = build_forecaster(
        constructor, sample_x, sample_y, num_layers=int(configuration["num_layers"]),
        hidden_dim=int(configuration["hidden_dim"]), kernel_size=int(configuration["kernel_size"]),
        device=device, step=int(configuration["step"]),
        forecast_formulation=configuration.get("forecast_formulation", "direct"),
        input_stations=bool(configuration["input_stations"]),
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            batch = [dataset[index][0] for index in range(start, min(start + args.batch_size, len(dataset)))]
            output = model(torch.stack(batch).to(device))[:, 0, :, pixels[:, 0], pixels[:, 1]]
            predictions.append(output.cpu().numpy().reshape(-1))
            if start + len(batch) == len(dataset) or (start // args.batch_size + 1) % 100 == 0:
                print(f"Exportadas {start + len(batch)}/{len(dataset)} sequências", flush=True)
    summary = export_flat_station_forecasts(dataset, pixels, ids, np.concatenate(predictions), args.output,
                                            experiment_id=args.experiment_id or args.experiment_dir.name)
    print(f"Checkpoint exportado | linhas={summary['rows']} | saída={args.output}")


if __name__ == "__main__":
    main()
