#!/usr/bin/env python3
"""Treina STConvS2S no projeto Radar Sumaré + estações.

O repositório stconvs2s é usado exclusivamente como dependência dos modelos.
Dataset, splits temporais, losses e métricas de precipitação pertencem a este
repositório.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.dataset import RadarStationMemmapDataset, parse_years
from nowcasting.losses import (
    MaskedHuberLoss,
    MaskedMAELoss,
    WeightedMaskedHuberLoss,
    WeightedMaskedMAELoss,
)


PRECIPITATION_BINS = (
    ("weak", 0.0, 1.25),
    ("moderate", 1.25, 6.25),
    ("strong", 6.25, 12.5),
    ("extreme", 12.5, float("inf")),
)


class DistributedWeightedSampler(Sampler[int]):
    """Global weighted draws, deterministically sharded across DDP ranks."""
    def __init__(self, weights, num_samples, num_replicas, rank, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.num_replicas, self.rank, self.seed, self.epoch = num_replicas, rank, seed, 0
        self.total_size = ((num_samples + num_replicas - 1) // num_replicas) * num_replicas

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.total_size // self.num_replicas
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        return iter(indices[self.rank:self.total_size:self.num_replicas].tolist())


def distributed_context(enabled: bool):
    if not enabled:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def is_main(rank: int) -> bool:
    return rank == 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento específico Radar Sumaré + AlertaRio/WebSirene."
    )
    parser.add_argument(
        "--stconvs2s-root",
        type=Path,
        default=PROJECT_ROOT / "external" / "stconvs2s",
        help="Clone limpo e fixado do repositório da arquitetura (submódulo por padrão).",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-years", required=True)
    parser.add_argument("--val-years", required=True)
    parser.add_argument("--test-years", required=True)
    parser.add_argument("--target-source", choices=("alertario", "websirene"),
                        default="alertario")
    parser.add_argument("--model", choices=("stconvs2s-c", "stconvs2s-r"),
                        default="stconvs2s-c")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Numero de microbatches acumulados antes de cada optimizer.step().",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--log-interval", type=int, default=250,
        help="Exibe progresso de treino a cada N batches; 0 desativa.",
    )
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Salto entre janelas do dataset; usa --step quando omitido.",
    )
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--loss", choices=("masked-mae", "masked-huber", "weighted-mae", "weighted-huber"),
                        default="masked-mae")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--loss-weights", default="1,5,10,20")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--sampler-thresholds", default="1.25,6.25,12.5")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--distributed", action="store_true", help="Usa DDP; iniciar com torchrun.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "experiments")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def parse_floats(value: str, expected: int, option: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"{option} deve conter números separados por vírgula.") from error
    if len(values) != expected or any(item <= 0 for item in values):
        raise ValueError(f"{option} deve conter {expected} valores positivos.")
    if tuple(sorted(values)) != values and option == "--sampler-thresholds":
        raise ValueError("--sampler-thresholds deve estar em ordem crescente.")
    return values


def validate_splits(train: list[int], val: list[int], test: list[int]) -> None:
    groups = {"train": set(train), "val": set(val), "test": set(test)}
    overlaps = [
        f"{left}/{right}: {sorted(groups[left] & groups[right])}"
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        if groups[left] & groups[right]
    ]
    if overlaps:
        raise ValueError("Anos sobrepostos entre splits: " + "; ".join(overlaps))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def core_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_class(core_root: Path, model_name: str):
    if not (core_root / "model" / "stconvs2s.py").is_file():
        raise FileNotFoundError(f"Não foi encontrado model/stconvs2s.py em {core_root}")
    sys.path.insert(0, str(core_root))
    from model.stconvs2s import STConvS2S_C, STConvS2S_R
    return {"stconvs2s-c": STConvS2S_C, "stconvs2s-r": STConvS2S_R}[model_name]


def criterion_from_args(args: argparse.Namespace, weights: tuple[float, ...]):
    if args.loss == "masked-mae":
        return MaskedMAELoss()
    if args.loss == "masked-huber":
        return MaskedHuberLoss(args.huber_delta)
    if args.loss == "weighted-mae":
        return WeightedMaskedMAELoss(weights)
    return WeightedMaskedHuberLoss(args.huber_delta, weights)


def empty_stats(horizons: int) -> dict:
    return {
        "global": {"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0},
        "horizons": [{"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0} for _ in range(horizons)],
        "intensity": {name: {"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0}
                      for name, _, _ in PRECIPITATION_BINS},
    }


def update_stats(stats: dict, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    prediction = torch.clamp(torch.expm1(output), min=0.0)
    observed = torch.expm1(target)
    valid = mask > 0
    error = prediction - observed

    def accumulate(destination: dict, selection: torch.Tensor, values_source: torch.Tensor = error) -> None:
        if not selection.any():
            return
        values = values_source[selection]
        destination["se"] += values.square().sum().item()
        destination["ae"] += values.abs().sum().item()
        destination["bias"] += values.sum().item()
        destination["n"] += int(selection.sum().item())

    accumulate(stats["global"], valid)
    for index, destination in enumerate(stats["horizons"]):
        accumulate(destination, valid[:, :, index], error[:, :, index])
    for name, low, high in PRECIPITATION_BINS:
        selection = valid & (observed >= low) & (observed < high)
        accumulate(stats["intensity"][name], selection)


def finalized_stats(stats: dict) -> dict:
    def finalize(value: dict) -> dict:
        n = value["n"]
        if n == 0:
            return {"n": 0, "rmse": None, "mae": None, "bias": None}
        return {"n": n, "rmse": (value["se"] / n) ** 0.5,
                "mae": value["ae"] / n, "bias": value["bias"] / n}
    return {
        "global": finalize(stats["global"]),
        "horizons": [finalize(item) for item in stats["horizons"]],
        "intensity": {name: finalize(item) for name, item in stats["intensity"].items()},
    }


def synchronize_stats(stats: dict, device: torch.device) -> None:
    if not dist.is_initialized() or stats is None:
        return
    leaves = [stats["global"], *stats["horizons"], *stats["intensity"].values()]
    for value in leaves:
        totals = torch.tensor([value["se"], value["ae"], value["bias"], value["n"]], device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        value.update(se=totals[0].item(), ae=totals[1].item(), bias=totals[2].item(), n=int(totals[3].item()))


def evaluate(model, loader, criterion, device, collect_metrics: bool = False):
    model.eval()
    total_loss = 0.0
    stats = None
    with torch.no_grad():
        for inputs, target, mask in loader:
            inputs, target, mask = inputs.to(device), target.to(device), mask.to(device)
            output = model(inputs)
            total_loss += criterion(output, target, mask).item()
            if collect_metrics:
                if stats is None:
                    stats = empty_stats(target.shape[2])
                update_stats(stats, output, target, mask)
    if not len(loader):
        raise ValueError("DataLoader vazio.")
    loss_parts = torch.tensor([total_loss, len(loader)], device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_parts, op=dist.ReduceOp.SUM)
    synchronize_stats(stats, device)
    return (loss_parts[0] / loss_parts[1]).item(), finalized_stats(stats) if stats else None


def train_one_iteration(args, model_type, device, datasets, run_dir: Path, iteration: int,
                        weights: tuple[float, ...], thresholds: tuple[float, ...], rank=0, world_size=1) -> dict:
    seed = args.seed + iteration * 10 + rank
    set_seed(seed)
    train_dataset, val_dataset, test_dataset = datasets
    sampler = None
    if args.balanced_sampler:
        sample_weights, counts = train_dataset.get_balanced_sample_weights(thresholds)
        sampler = (DistributedWeightedSampler(sample_weights, len(sample_weights), world_size, rank, args.seed + iteration * 10)
                   if args.distributed else WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True))
        if is_main(rank): print(f"Balanced sampler | class_counts={counts.tolist()}", flush=True)
    elif args.distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers}
    train_loader = DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, sampler=DistributedSampler(val_dataset, world_size, rank, shuffle=False) if args.distributed else None, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, sampler=DistributedSampler(test_dataset, world_size, rank, shuffle=False) if args.distributed else None, **loader_args)

    sample_x, sample_y, _ = train_dataset[0]
    model = model_type(
        (1, *sample_x.shape), args.num_layers, args.hidden_dim, args.kernel_size,
        device, 0.0, args.step, output_channels=sample_y.shape[0],
    ).to(device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[device.index])
    criterion = criterion_from_args(args, weights)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9, eps=1e-6)
    if is_main(rank): print(
        f"Training batches | microbatch={args.batch_size} | "
        f"accumulation={args.gradient_accumulation_steps} | "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps}",
        flush=True,
    )
    checkpoint_path = run_dir / f"iteration_{iteration + 1}_best.pt"
    history = []
    best_val = float("inf")
    best_epoch = 0
    stalled = 0
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        losses = []
        epoch_started = time.monotonic()
        total_batches = len(train_loader)
        for batch_index, (inputs, target, mask) in enumerate(train_loader, start=1):
            if (batch_index - 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()
                group_size = min(
                    args.gradient_accumulation_steps,
                    total_batches - batch_index + 1,
                )
            inputs, target, mask = inputs.to(device), target.to(device), mask.to(device)
            loss = criterion(model(inputs), target, mask)
            (loss / group_size).backward()
            if batch_index % args.gradient_accumulation_steps == 0 or batch_index == total_batches:
                optimizer.step()
            losses.append(loss.item())
            if args.log_interval and (batch_index % args.log_interval == 0 or batch_index == total_batches):
                elapsed = time.monotonic() - epoch_started
                rate = batch_index / elapsed if elapsed else 0.0
                remaining = (total_batches - batch_index) / rate if rate else 0.0
                if is_main(rank): print(
                    f"Iteration {iteration + 1} | epoch {epoch}/{args.epochs} | "
                    f"batch {batch_index}/{total_batches} ({100 * batch_index / total_batches:.1f}%) | "
                    f"loss={np.mean(losses):.6f} | {rate:.2f} batch/s | "
                    f"ETA={remaining / 60:.1f} min",
                    flush=True,
                )
        train_loss = float(np.mean(losses))
        val_loss, _ = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val, best_epoch, stalled = val_loss, epoch, 0
            if is_main(rank): torch.save({"model_state_dict": (model.module if args.distributed else model).state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "configuration": vars(args)}, checkpoint_path)
            stopping_status = f"new best | patience=0/{args.patience}"
        else:
            stalled += 1
            stopping_status = (
                f"no improvement | best={best_val:.6f} at epoch {best_epoch} | "
                f"patience={stalled}/{args.patience}"
            )
        if is_main(rank): print(f"Iteration {iteration + 1} | epoch {epoch}/{args.epochs} | "
              f"loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"{stopping_status}", flush=True)
        if stalled >= args.patience:
            if is_main(rank): print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.", flush=True)
            break

    # Checkpoints are created in this run and include trusted configuration
    # metadata in addition to tensors; PyTorch 2.6 defaults to weights_only.
    if args.distributed: dist.barrier()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    (model.module if args.distributed else model).load_state_dict(state["model_state_dict"])
    _, metrics = evaluate(model, test_loader, criterion, device, collect_metrics=True)
    elapsed = time.monotonic() - started
    result = {"seed": seed, "best_epoch": best_epoch, "best_val_loss": best_val,
              "elapsed_seconds": elapsed, "checkpoint": checkpoint_path.name,
              "test_metrics": metrics, "history": history}
    if is_main(rank):
        with (run_dir / f"iteration_{iteration + 1}.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        print(f"Iteration {iteration + 1} complete | test={metrics['global']}", flush=True)
    return result


def main() -> None:
    args = parse_arguments()
    if not args.distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    train_years, val_years, test_years = map(parse_years, (args.train_years, args.val_years, args.test_years))
    validate_splits(train_years, val_years, test_years)
    weights = parse_floats(args.loss_weights, 4, "--loss-weights")
    thresholds = parse_floats(args.sampler_thresholds, 3, "--sampler-thresholds")
    if (args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0
            or args.iterations <= 0 or args.step <= 0
            or args.gradient_accumulation_steps <= 0 or args.log_interval < 0
            or (args.stride is not None and args.stride <= 0)):
        raise ValueError("epochs, patience, batch-size, iterations, step, stride e gradient-accumulation-steps devem ser positivos; log-interval nao pode ser negativo.")

    rank, world_size, device = distributed_context(args.distributed)
    model_type = model_class(args.stconvs2s_root.resolve(), args.model)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    if is_main(rank):
        run_dir.mkdir(parents=True, exist_ok=False)
    if args.distributed:
        dist.barrier()
    datasets = (
        RadarStationMemmapDataset(args.dataset_root, train_years, stride=args.stride or args.step,
                                  target_source=args.target_source, split_name="train"),
        RadarStationMemmapDataset(args.dataset_root, val_years, stride=args.stride or args.step,
                                  target_source=args.target_source, split_name="val"),
        RadarStationMemmapDataset(args.dataset_root, test_years, stride=args.stride or args.step,
                                  target_source=args.target_source, split_name="test"),
    )
    configuration = vars(args) | {
        "device": str(device), "train_years": train_years, "val_years": val_years,
        "test_years": test_years, "stconvs2s_commit": core_commit(args.stconvs2s_root),
    }
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in configuration.items()}
    configuration["world_size"] = world_size
    if is_main(rank):
        with (run_dir / "configuration.json").open("w", encoding="utf-8") as file:
            json.dump(configuration, file, indent=2)
    results = [train_one_iteration(args, model_type, device, datasets, run_dir, index, weights, thresholds, rank, world_size)
               for index in range(args.iterations)]
    summary = {"iterations": len(results), "results": results}
    if is_main(rank):
        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
