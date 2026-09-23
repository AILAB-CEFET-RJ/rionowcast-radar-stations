#!/usr/bin/env python3
"""Treina e avalia baselines que usam somente o histórico das estações."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.dataset import parse_years
from nowcasting.losses import MaskedHuberLoss, MaskedMAELoss, WeightedMaskedHuberLoss, WeightedMaskedMAELoss
from nowcasting.station_dataset import StationSequenceDataset
from nowcasting.station_model import StationMLP


BINS = (("weak", 0.0, 1.25), ("moderate", 1.25, 6.25), ("strong", 6.25, 12.5), ("extreme", 12.5, float("inf")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline temporal somente com estações AlertaRio.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "data" / "mapeamento_pixel_estacao_alertario.csv")
    parser.add_argument("--train-years", required=True)
    parser.add_argument("--val-years", required=True)
    parser.add_argument("--test-years", required=True)
    parser.add_argument("--model", choices=("mlp", "persistence"), default="mlp")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--loss", choices=("masked-mae", "masked-huber", "weighted-mae", "weighted-huber"), default="masked-huber")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--loss-weights", default="1,5,10,20")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "experiments")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def validate_splits(train: list[int], val: list[int], test: list[int]) -> None:
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Os splits temporais não podem compartilhar anos.")


def criterion(args: argparse.Namespace, weights: tuple[float, ...]):
    if args.loss == "masked-mae":
        return MaskedMAELoss()
    if args.loss == "masked-huber":
        return MaskedHuberLoss(args.huber_delta)
    if args.loss == "weighted-mae":
        return WeightedMaskedMAELoss(weights)
    return WeightedMaskedHuberLoss(args.huber_delta, weights)


def loss_input(values: torch.Tensor) -> torch.Tensor:
    return values.unsqueeze(1).unsqueeze(-1)


def persistence(inputs: torch.Tensor, t_out: int) -> torch.Tensor:
    """Repete a última observação disponível por estação; ausências viram zero."""
    values, masks = inputs[..., 0], inputs[..., 1] > 0
    positions = torch.arange(values.shape[1], device=values.device).view(1, -1, 1)
    last_positions = torch.where(masks, positions, torch.full_like(positions, -1)).amax(dim=1)
    gather_positions = last_positions.clamp_min(0).unsqueeze(1)
    last_values = values.gather(1, gather_positions).squeeze(1)
    last_values = torch.where(last_positions >= 0, last_values, torch.zeros_like(last_values))
    return last_values.unsqueeze(1).expand(-1, t_out, -1)


def empty_stats(horizons: int, station_ids: list[int]) -> dict:
    return {"global": [0.0, 0.0, 0.0, 0], "horizons": [[0.0, 0.0, 0.0, 0] for _ in range(horizons)],
            "intensity": {name: [0.0, 0.0, 0.0, 0] for name, _, _ in BINS},
            "stations": {str(station_id): {"global": [0.0, 0.0, 0.0, 0],
                                             "horizons": [[0.0, 0.0, 0.0, 0] for _ in range(horizons)]}
                         for station_id in station_ids}}


def add_stats(destination: list, errors: torch.Tensor) -> None:
    if errors.numel():
        destination[0] += errors.square().sum().item()
        destination[1] += errors.abs().sum().item()
        destination[2] += errors.sum().item()
        destination[3] += errors.numel()


def update_stats(stats: dict, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 station_ids: list[int]) -> None:
    predicted, observed = torch.clamp(torch.expm1(output), min=0.0), torch.expm1(target)
    valid, error = mask > 0, predicted - observed
    add_stats(stats["global"], error[valid])
    for horizon, destination in enumerate(stats["horizons"]):
        add_stats(destination, error[:, horizon][valid[:, horizon]])
    for name, low, high in BINS:
        selection = valid & (observed >= low) & (observed < high)
        add_stats(stats["intensity"][name], error[selection])
    for index, station_id in enumerate(station_ids):
        station = stats["stations"][str(station_id)]
        add_stats(station["global"], error[:, :, index][valid[:, :, index]])
        for horizon, destination in enumerate(station["horizons"]):
            add_stats(destination, error[:, horizon, index][valid[:, horizon, index]])


def finalize(item: list) -> dict:
    se, ae, bias, n = item
    return {"n": n, "rmse": (se / n) ** 0.5 if n else None, "mae": ae / n if n else None, "bias": bias / n if n else None}


def evaluate(model, loader, loss_fn, device: torch.device, station_ids: list[int], *, use_persistence: bool = False) -> tuple[float, dict]:
    if model is not None:
        model.eval()
    total_loss, stats = 0.0, None
    with torch.no_grad():
        for inputs, target, mask in loader:
            inputs, target, mask = inputs.to(device), target.to(device), mask.to(device)
            output = persistence(inputs, target.shape[1]) if use_persistence else model(inputs)
            total_loss += loss_fn(loss_input(output), loss_input(target), loss_input(mask)).item()
            stats = stats or empty_stats(target.shape[1], station_ids)
            update_stats(stats, output, target, mask, station_ids)
    return total_loss / len(loader), {"global": finalize(stats["global"]), "horizons": [finalize(item) for item in stats["horizons"]],
                                       "intensity": {name: finalize(item) for name, item in stats["intensity"].items()},
                                       "stations": {station_id: {"global": finalize(values["global"]),
                                                                  "horizons": [finalize(item) for item in values["horizons"]]}
                                                    for station_id, values in stats["stations"].items()}}


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.cuda))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if min(args.batch_size, args.epochs, args.patience, args.hidden_dim, args.step) <= 0 or args.workers < 0:
        raise ValueError("batch-size, epochs, patience, hidden-dim e step devem ser positivos.")
    weights = tuple(float(item) for item in args.loss_weights.split(","))
    if len(weights) != 4 or any(item <= 0 for item in weights):
        raise ValueError("--loss-weights deve conter quatro valores positivos.")
    train_years, val_years, test_years = map(parse_years, (args.train_years, args.val_years, args.test_years))
    validate_splits(train_years, val_years, test_years)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    stride = args.stride or args.step
    common = {"mapping": args.mapping, "stride": stride}
    train = StationSequenceDataset(args.dataset_root, train_years, split_name="train", **common)
    val = StationSequenceDataset(args.dataset_root, val_years, split_name="val", **common)
    test = StationSequenceDataset(args.dataset_root, test_years, split_name="test", **common)
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=index == 0, num_workers=args.workers)
               for index, dataset in enumerate((train, val, test))]
    run_dir = args.output_dir / (args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"train_years": train_years, "val_years": val_years, "test_years": test_years,
                   "station_count": len(train.station_ids), "station_ids": train.station_ids})
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    loss_fn = criterion(args, weights)
    if args.model == "persistence":
        _, metrics = evaluate(None, loaders[2], loss_fn, device, test.station_ids, use_persistence=True)
        result = {"model": "persistence", "test_metrics": metrics}
    else:
        model = StationMLP(len(train.station_ids), hidden_dim=args.hidden_dim).to(device)
        optimizer, best_val, best_state, stalled, history = torch.optim.Adam(model.parameters(), lr=args.learning_rate), float("inf"), None, 0, []
        for epoch in range(1, args.epochs + 1):
            model.train(); total_loss = 0.0
            for inputs, target, mask in loaders[0]:
                inputs, target, mask = inputs.to(device), target.to(device), mask.to(device)
                optimizer.zero_grad(); output = model(inputs)
                current_loss = loss_fn(loss_input(output), loss_input(target), loss_input(mask))
                current_loss.backward(); optimizer.step(); total_loss += current_loss.item()
            val_loss, _ = evaluate(model, loaders[1], loss_fn, device, val.station_ids)
            history.append({"epoch": epoch, "loss": total_loss / len(loaders[0]), "val_loss": val_loss})
            if val_loss < best_val:
                best_val, best_state, stalled = val_loss, {key: value.detach().cpu() for key, value in model.state_dict().items()}, 0
                status = "new best"
            else:
                stalled += 1; status = "no improvement"
            print(f"epoch {epoch}/{args.epochs} | loss={history[-1]['loss']:.6f} | val_loss={val_loss:.6f} | {status} | patience={stalled}/{args.patience}", flush=True)
            if stalled >= args.patience:
                break
        model.load_state_dict(best_state)
        _, metrics = evaluate(model, loaders[2], loss_fn, device, test.station_ids)
        result = {"model": "mlp", "best_epoch": min(history, key=lambda item: item["val_loss"])["epoch"], "history": history, "test_metrics": metrics}
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Complete | test={result['test_metrics']['global']}", flush=True)


if __name__ == "__main__":
    main()
