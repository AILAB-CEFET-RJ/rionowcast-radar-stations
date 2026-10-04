"""Evaluate a causal optical-flow plus calibrated station-readout baseline."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nowcasting.dataset import RadarStationMemmapDataset, parse_years
from nowcasting.optical_flow import extrapolate_visual_echo, station_echo_features
from nowcasting.paths import project_root
from nowcasting.station_dataset import load_station_pixels


PROJECT_ROOT = project_root()
BINS = (("weak", 0.0, 1.25), ("moderate", 1.25, 6.25), ("strong", 6.25, 12.5), ("extreme", 12.5, float("inf")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline causal: fluxo óptico do radar RGB + leitura Ridge nas estações.")
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
    parser.add_argument("--max-samples", type=int, default=None, help="Limita amostras por split; somente para smoke tests.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "experiments")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def validate_splits(train: list[int], val: list[int], test: list[int]) -> None:
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Os splits temporais não podem compartilhar anos.")


def rows_from_dataset(dataset: RadarStationMemmapDataset, station_pixels: np.ndarray, radius: int, max_samples: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    features, targets, masks, fallbacks = [], [], [], 0
    limit = min(len(dataset), max_samples) if max_samples else len(dataset)
    station_indexes = np.arange(len(station_pixels), dtype=np.float32)
    for index in range(limit):
        x, target, mask = dataset[index]
        rgb = x[:3].permute(1, 2, 3, 0).numpy()
        forecast, fallback = extrapolate_visual_echo(rgb, target.shape[1])
        local = station_echo_features(forecast, station_pixels, radius)
        horizons, stations, _ = local.shape
        station_column = np.broadcast_to(station_indexes, (horizons, stations))[..., None]
        features.append(np.concatenate((local, station_column), axis=-1).reshape(-1, 4))
        targets.append(target.numpy()[0][:, station_pixels[:, 0], station_pixels[:, 1]].reshape(-1))
        masks.append(mask.numpy()[0][:, station_pixels[:, 0], station_pixels[:, 1]].reshape(-1))
        fallbacks += int(fallback)
        if (index + 1) % 250 == 0 or index + 1 == limit:
            print(f"[{dataset.split_name}] fluxo processado={index + 1}/{limit} | fallbacks={fallbacks}", flush=True)
    return np.concatenate(features), np.concatenate(targets), np.concatenate(masks).astype(bool), fallbacks


def model(alpha: float) -> Pipeline:
    preprocess = ColumnTransformer([
        ("echo", StandardScaler(), [0, 1, 2]),
        ("station", OneHotEncoder(handle_unknown="ignore"), [3]),
    ])
    return Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=alpha, solver="lsqr"))])


def fit_models(features: np.ndarray, targets: np.ndarray, masks: np.ndarray, horizons: int, stations: int, alpha: float) -> list[Pipeline]:
    models = []
    for horizon in range(horizons):
        selector = np.arange(horizon * stations, len(targets), horizons * stations)
        expanded = (selector[:, None] + np.arange(stations)).reshape(-1)
        valid = masks[expanded]
        current = model(alpha)
        current.fit(features[expanded][valid], targets[expanded][valid])
        models.append(current)
    return models


def predict(models: list[Pipeline], features: np.ndarray, horizons: int, stations: int) -> np.ndarray:
    result = np.empty(len(features), dtype=np.float32)
    for horizon, current in enumerate(models):
        selector = np.arange(horizon * stations, len(features), horizons * stations)
        expanded = (selector[:, None] + np.arange(stations)).reshape(-1)
        result[expanded] = current.predict(features[expanded])
    return result


def metrics(predicted_log: np.ndarray, target_log: np.ndarray, mask: np.ndarray, horizons: int, stations: int) -> dict:
    predicted, observed = np.maximum(np.expm1(predicted_log), 0.0), np.expm1(target_log)
    error = predicted - observed
    def summarize(selection: np.ndarray) -> dict:
        values = error[selection]
        return {"n": int(len(values)), "rmse": float(np.sqrt(np.mean(values ** 2))) if len(values) else None, "mae": float(np.mean(np.abs(values))) if len(values) else None, "bias": float(np.mean(values)) if len(values) else None}
    result = {"global": summarize(mask), "horizons": [], "intensity": {}}
    layout = np.arange(len(mask)).reshape(-1, horizons, stations)
    for horizon in range(horizons):
        horizon_mask = np.zeros(len(mask), dtype=bool)
        horizon_mask[layout[:, horizon, :].reshape(-1)] = True
        result["horizons"].append(summarize(mask & horizon_mask))
    for name, low, high in BINS:
        result["intensity"][name] = summarize(mask & (observed >= low) & (observed < high))
    return result


def main() -> None:
    args = parse_args()
    if min(args.step, args.stride) <= 0 or args.neighborhood_radius < 0 or args.max_samples == 0:
        raise ValueError("step, stride e max-samples devem ser positivos; raio não pode ser negativo.")
    train_years, val_years, test_years = map(parse_years, (args.train_years, args.val_years, args.test_years))
    validate_splits(train_years, val_years, test_years)
    common = dict(target_source="alertario", t_in=args.step, t_out=args.step, stride=args.stride)
    datasets = [RadarStationMemmapDataset(args.dataset_root, years, split_name=name, **common) for name, years in (("train", train_years), ("val", val_years), ("test", test_years))]
    first = datasets[0].year_data[train_years[0]]["frames"]
    pixels, station_ids = load_station_pixels(args.mapping, first.shape[1], first.shape[2], height_orig=args.mapping_height_orig, width_orig=args.mapping_width_orig)
    rows = [rows_from_dataset(dataset, pixels, args.neighborhood_radius, args.max_samples) for dataset in datasets]
    train_x, train_y, train_m, train_fallbacks = rows[0]
    val_x, val_y, val_m, val_fallbacks = rows[1]
    test_x, test_y, test_m, test_fallbacks = rows[2]
    alphas = [float(value) for value in args.ridge_alphas.split(",")]
    if not alphas or any(value <= 0 for value in alphas):
        raise ValueError("--ridge-alphas deve conter valores positivos.")
    candidates = []
    for alpha in alphas:
        fitted = fit_models(train_x, train_y, train_m, args.step, len(station_ids), alpha)
        validation = metrics(predict(fitted, val_x, args.step, len(station_ids)), val_y, val_m, args.step, len(station_ids))
        candidates.append((alpha, fitted, validation))
        print(f"alpha={alpha:g} | val_mae={validation['global']['mae']:.6f}", flush=True)
    alpha, fitted, validation = min(candidates, key=lambda item: item[2]["global"]["mae"])
    test_metrics = metrics(predict(fitted, test_x, args.step, len(station_ids)), test_y, test_m, args.step, len(station_ids))
    run_dir = args.output_dir / (args.run_name or f"B2a-optical-flow-{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({"train_years": train_years, "val_years": val_years, "test_years": test_years, "station_ids": station_ids, "pysteps_version": version("pysteps"), "motion": "pysteps Lucas-Kanade", "extrapolation": "pysteps semilagrangian", "echo_proxy": "max(R,G,B)/255; visual, not calibrated reflectivity"})
    (run_dir / "configuration.json").write_text(json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    joblib.dump(fitted, run_dir / "ridge_readouts.joblib")
    summary = {"model": "optical-flow-rgb-calibrated-readout", "selected_ridge_alpha": alpha, "validation_candidates": [{"alpha": item[0], "metrics": item[2]} for item in candidates], "validation_metrics": validation, "test_metrics": test_metrics, "flow_fallbacks": {"train": train_fallbacks, "val": val_fallbacks, "test": test_fallbacks}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Complete | alpha={alpha:g} | test={test_metrics['global']}", flush=True)


if __name__ == "__main__":
    main()
