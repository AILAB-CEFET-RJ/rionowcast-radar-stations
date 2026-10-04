"""Evaluate B2b: optical flow blended with causal station persistence."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from nowcasting.cli.evaluate_optical_flow import (
    PROJECT_ROOT, fit_models, metrics, predict, validate_splits,
)
from nowcasting.dataset import RadarStationMemmapDataset, parse_years
from nowcasting.optical_flow import (
    extrapolate_visual_echo, station_echo_features, station_persistence_from_history,
)
from nowcasting.station_dataset import load_station_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline causal B2b: fluxo óptico RGB + persistência das estações.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "configs" / "mapeamento_pixel_estacao_alertario_historical_v1.csv")
    parser.add_argument("--mapping-height-orig", type=int, default=654)
    parser.add_argument("--mapping-width-orig", type=int, default=656)
    parser.add_argument("--train-years", required=True)
    parser.add_argument("--val-years", required=True)
    parser.add_argument("--test-years", required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--neighborhood-radius", type=int, default=2)
    parser.add_argument("--ridge-alphas", default="0.1,1,10")
    parser.add_argument("--blend-weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
                        help="Pesos do B2a; 0 equivale a persistência e 1 a fluxo óptico puro.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limita amostras por split; somente para smoke tests.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "experiments")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def rows_from_dataset(
    dataset: RadarStationMemmapDataset, station_pixels: np.ndarray, radius: int, max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    features, targets, masks, persistence = [], [], [], []
    fallbacks = missing_history = 0
    limit = min(len(dataset), max_samples) if max_samples else len(dataset)
    station_indexes = np.arange(len(station_pixels), dtype=np.float32)
    for index in range(limit):
        x, target, mask = dataset[index]
        horizons = target.shape[1]
        rgb = x[:3].permute(1, 2, 3, 0).numpy()
        forecast, fallback = extrapolate_visual_echo(rgb, horizons)
        local = station_echo_features(forecast, station_pixels, radius)
        station_column = np.broadcast_to(station_indexes, local.shape[:2])[..., None]
        features.append(np.concatenate((local, station_column), axis=-1).reshape(-1, 4))
        targets.append(target.numpy()[0][:, station_pixels[:, 0], station_pixels[:, 1]].reshape(-1))
        masks.append(mask.numpy()[0][:, station_pixels[:, 0], station_pixels[:, 1]].reshape(-1))
        history_values = x[3, :, station_pixels[:, 0], station_pixels[:, 1]].numpy()
        history_masks = x[4, :, station_pixels[:, 0], station_pixels[:, 1]].numpy()
        repeated, missing = station_persistence_from_history(history_values, history_masks, horizons)
        persistence.append(repeated.reshape(-1))
        fallbacks += int(fallback)
        missing_history += missing
        if (index + 1) % 250 == 0 or index + 1 == limit:
            print(
                f"[{dataset.split_name}] fluxo processado={index + 1}/{limit} | "
                f"fallbacks={fallbacks} | históricos ausentes={missing_history}", flush=True,
            )
    return (
        np.concatenate(features), np.concatenate(targets), np.concatenate(masks).astype(bool),
        np.concatenate(persistence), fallbacks, missing_history,
    )


def parse_positive_numbers(value: str, name: str, *, allow_zero: bool = False) -> list[float]:
    result = [float(item) for item in value.split(",")]
    if not result or any(item < 0 if allow_zero else item <= 0 for item in result):
        raise ValueError(f"{name} contém valor inválido.")
    return result


def blend(optical_log: np.ndarray, persistence_log: np.ndarray, optical_weight: float) -> np.ndarray:
    optical = np.maximum(np.expm1(optical_log), 0.0)
    persistence = np.maximum(np.expm1(persistence_log), 0.0)
    return np.log1p(optical_weight * optical + (1.0 - optical_weight) * persistence)


def main() -> None:
    args = parse_args()
    if min(args.step, args.stride) <= 0 or args.neighborhood_radius < 0 or args.max_samples == 0:
        raise ValueError("step, stride e max-samples devem ser positivos; raio não pode ser negativo.")
    train_years, val_years, test_years = map(parse_years, (args.train_years, args.val_years, args.test_years))
    validate_splits(train_years, val_years, test_years)
    alphas = parse_positive_numbers(args.ridge_alphas, "--ridge-alphas")
    weights = parse_positive_numbers(args.blend_weights, "--blend-weights", allow_zero=True)
    if any(weight > 1 for weight in weights):
        raise ValueError("--blend-weights deve permanecer entre 0 e 1.")
    common = dict(target_source="alertario", t_in=args.step, t_out=args.step, stride=args.stride, input_stations=True)
    datasets = [RadarStationMemmapDataset(args.dataset_root, years, split_name=name, **common)
                for name, years in (("train", train_years), ("val", val_years), ("test", test_years))]
    first = datasets[0].year_data[train_years[0]]["frames"]
    pixels, station_ids = load_station_pixels(args.mapping, first.shape[1], first.shape[2],
                                              height_orig=args.mapping_height_orig, width_orig=args.mapping_width_orig)
    rows = [rows_from_dataset(dataset, pixels, args.neighborhood_radius, args.max_samples) for dataset in datasets]
    train_x, train_y, train_m, _, train_fallbacks, train_missing = rows[0]
    val_x, val_y, val_m, val_persistence, val_fallbacks, val_missing = rows[1]
    test_x, test_y, test_m, test_persistence, test_fallbacks, test_missing = rows[2]
    candidates = []
    for alpha in alphas:
        fitted = fit_models(train_x, train_y, train_m, args.step, len(station_ids), alpha)
        val_optical = predict(fitted, val_x, args.step, len(station_ids))
        for weight in weights:
            validation = metrics(blend(val_optical, val_persistence, weight), val_y, val_m, args.step, len(station_ids))
            candidates.append((alpha, weight, fitted, validation))
            print(f"alpha={alpha:g} | peso_fluxo={weight:.2f} | val_mae={validation['global']['mae']:.6f}", flush=True)
    alpha, weight, fitted, validation = min(candidates, key=lambda item: item[3]["global"]["mae"])
    test_optical = predict(fitted, test_x, args.step, len(station_ids))
    test_metrics = metrics(blend(test_optical, test_persistence, weight), test_y, test_m, args.step, len(station_ids))
    run_dir = args.output_dir / (args.run_name or f"B2b-optical-flow-persistence-{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({"train_years": train_years, "val_years": val_years, "test_years": test_years,
                          "station_ids": station_ids, "pysteps_version": version("pysteps"),
                          "opencv_python_headless_version": version("opencv-python-headless"),
                          "motion": "pysteps Lucas-Kanade",
                          "extrapolation": "pysteps semilagrangian",
                          "blend": "weight * optical-flow precipitation + (1 - weight) * B1 persistence precipitation"})
    (run_dir / "configuration.json").write_text(json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    joblib.dump(fitted, run_dir / "ridge_readouts.joblib")
    summary = {"model": "optical-flow-rgb-persistence-blend", "selected_ridge_alpha": alpha,
               "selected_optical_flow_weight": weight, "validation_candidates": [
                   {"alpha": item[0], "optical_flow_weight": item[1], "metrics": item[3]} for item in candidates],
               "validation_metrics": validation, "test_metrics": test_metrics,
               "flow_fallbacks": {"train": train_fallbacks, "val": val_fallbacks, "test": test_fallbacks},
               "stations_without_input_history": {"train": train_missing, "val": val_missing, "test": test_missing}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Complete | alpha={alpha:g} | peso_fluxo={weight:.2f} | test={test_metrics['global']}", flush=True)


if __name__ == "__main__":
    main()
