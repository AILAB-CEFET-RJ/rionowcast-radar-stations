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
import signal
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


from nowcasting.dataset import RadarStationMemmapDataset, parse_years
from nowcasting.losses import (
    MaskedHuberLoss,
    MaskedMAELoss,
    WeightedMaskedHuberLoss,
    WeightedMaskedMAELoss,
)
from nowcasting.station_dataset import load_station_pixels
from nowcasting.paths import project_root


PROJECT_ROOT = project_root()


PRECIPITATION_BINS = (
    ("weak", 0.0, 1.25),
    ("moderate", 1.25, 6.25),
    ("strong", 6.25, 12.5),
    ("extreme", 12.5, float("inf")),
)

CHECKPOINT_VERSION = 1
RESUME_CONFIG_KEYS = (
    "dataset_root", "model", "num_layers", "hidden_dim", "kernel_size", "step", "stride",
    "target_source", "loss", "huber_delta", "loss_weights", "sampler_thresholds",
    "balanced_sampler", "batch_size", "gradient_accumulation_steps", "learning_rate",
    "seed", "distributed", "world_size", "train_years", "val_years", "test_years",
    "stconvs2s_commit", "crop_stations", "crop_margin_pixels", "station_mapping",
    "mapping_height_orig", "mapping_width_orig", "crop",
)


class StopRequested:
    """Defers SIGINT/SIGTERM handling until the current batch completes."""
    def __init__(self):
        self.requested = False

    def __call__(self, signum, _frame):
        self.requested = True
        print(f"Received signal {signum}; stopping after the current batch.", flush=True)


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
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


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
    parser.add_argument(
        "--crop-stations", action="store_true",
        help="Recorta radar e targets para o retângulo das estações AlertaRio com margem.",
    )
    parser.add_argument(
        "--crop-margin-pixels", type=int, default=20,
        help="Margem espacial do crop de estações, em pixels da grade do dataset.",
    )
    parser.add_argument(
        "--station-mapping", type=Path,
        default=PROJECT_ROOT / "data" / "mapeamento_pixel_estacao_alertario.csv",
        help="CSV com pixel_i/pixel_j das estações no grid original do radar.",
    )
    parser.add_argument("--mapping-height-orig", type=int, default=656)
    parser.add_argument("--mapping-width-orig", type=int, default=654)
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
        "--pin-memory", action=argparse.BooleanOptionalAction, default=None,
        help="Usa memoria fixada para acelerar transferencias para CUDA (padrao: ativado com CUDA).",
    )
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=False,
        help="Mantem workers do DataLoader entre epocas; requer --workers maior que zero.",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="Lotes preparados antecipadamente por worker; requer --workers maior que zero.",
    )
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
    parser.add_argument(
        "--resume", type=Path,
        help="Checkpoint iteration_N_last.pt para retomar a partir da proxima epoca.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1,
        help="Salva iteration_N_last.pt a cada N epocas concluidas.",
    )
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


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])


def atomic_torch_save(state: dict, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def checkpoint_state(model, optimizer, iteration: int, completed_epoch: int, best_val: float,
                     best_epoch: int, stalled: int, history: list, configuration: dict,
                     rng_states: list[dict]) -> dict:
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "completed_epoch": completed_epoch,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "stalled_epochs": stalled,
        "history": history,
        "configuration": configuration,
        "rng_states": rng_states,
    }


def collect_rng_states(local_state: dict) -> list[dict]:
    if not dist.is_initialized():
        return [local_state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, local_state)
    return states


def validate_resume_checkpoint(state: dict, configuration: dict, world_size: int) -> None:
    if state.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Versao de checkpoint incompativel para retomada.")
    saved = state.get("configuration", {})
    differences = [
        f"{key}: salvo={saved.get(key)!r}, atual={configuration.get(key)!r}"
        for key in RESUME_CONFIG_KEYS
        if saved.get(key) != configuration.get(key)
    ]
    if differences:
        raise ValueError("Checkpoint incompativel com a configuracao atual: " + "; ".join(differences))
    rng_states = state.get("rng_states", [])
    if len(rng_states) != world_size:
        raise ValueError(
            f"Checkpoint foi criado com {len(rng_states)} rank(s), mas a execucao atual usa {world_size}."
        )


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


def empty_stats(horizons: int, station_locations: list[tuple[int, int, int]] | None = None) -> dict:
    stats = {
        "global": {"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0},
        "horizons": [{"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0} for _ in range(horizons)],
        "intensity": {name: {"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0}
                      for name, _, _ in PRECIPITATION_BINS},
    }
    if station_locations:
        stats["stations"] = {
            str(station_id): {
                "global": {"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0},
                "horizons": [{"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0} for _ in range(horizons)],
            }
            for station_id, _, _ in station_locations
        }
    return stats


def update_stats(stats: dict, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 station_locations: list[tuple[int, int, int]] | None = None) -> None:
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
    if station_locations:
        for station_id, row, column in station_locations:
            station = stats["stations"][str(station_id)]
            station_valid = valid[:, :, :, row, column]
            station_error = error[:, :, :, row, column]
            accumulate(station["global"], station_valid, station_error)
            for index, destination in enumerate(station["horizons"]):
                accumulate(destination, station_valid[:, :, index], station_error[:, :, index])


def finalized_stats(stats: dict) -> dict:
    def finalize(value: dict) -> dict:
        n = value["n"]
        if n == 0:
            return {"n": 0, "rmse": None, "mae": None, "bias": None}
        return {"n": n, "rmse": (value["se"] / n) ** 0.5,
                "mae": value["ae"] / n, "bias": value["bias"] / n}
    result = {
        "global": finalize(stats["global"]),
        "horizons": [finalize(item) for item in stats["horizons"]],
        "intensity": {name: finalize(item) for name, item in stats["intensity"].items()},
    }
    if "stations" in stats:
        result["stations"] = {
            station_id: {"global": finalize(values["global"]),
                         "horizons": [finalize(item) for item in values["horizons"]]}
            for station_id, values in stats["stations"].items()
        }
    return result


def synchronize_stats(stats: dict, device: torch.device) -> None:
    if not dist.is_initialized() or stats is None:
        return
    leaves = [stats["global"], *stats["horizons"], *stats["intensity"].values()]
    if "stations" in stats:
        for station in stats["stations"].values():
            leaves.extend((station["global"], *station["horizons"]))
    for value in leaves:
        totals = torch.tensor([value["se"], value["ae"], value["bias"], value["n"]], device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        value.update(se=totals[0].item(), ae=totals[1].item(), bias=totals[2].item(), n=int(totals[3].item()))


def loader_options(args: argparse.Namespace) -> dict:
    """Build DataLoader options valid for both single-process and DDP runs."""
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
    }
    if args.workers:
        options["persistent_workers"] = args.persistent_workers
        options["prefetch_factor"] = args.prefetch_factor
    return options


def station_locations(dataset: RadarStationMemmapDataset, args: argparse.Namespace) -> list[tuple[int, int, int]]:
    """Retorna IDs e coordenadas das estações no sistema de referência do dataset."""
    first_year = dataset.years[0]
    frames = dataset.year_data[first_year]["frames"]
    pixels, station_ids = load_station_pixels(
        args.station_mapping, frames.shape[1], frames.shape[2],
        height_orig=args.mapping_height_orig, width_orig=args.mapping_width_orig,
    )
    if dataset.crop_bounds is not None:
        top, bottom, left, right = dataset.crop_bounds
        inside = ((pixels[:, 0] >= top) & (pixels[:, 0] < bottom) &
                  (pixels[:, 1] >= left) & (pixels[:, 1] < right))
        if not inside.all():
            raise ValueError("O crop excluiu uma estação do mapeamento.")
        pixels = pixels.copy()
        pixels[:, 0] -= top
        pixels[:, 1] -= left
    return [(station_id, int(row), int(column)) for station_id, (row, column) in zip(station_ids, pixels)]


def evaluate(model, loader, criterion, device, collect_metrics: bool = False,
             station_locations: list[tuple[int, int, int]] | None = None):
    model.eval()
    total_loss = 0.0
    stats = None
    with torch.no_grad():
        for inputs, target, mask in loader:
            inputs = inputs.to(device, non_blocking=loader.pin_memory)
            target = target.to(device, non_blocking=loader.pin_memory)
            mask = mask.to(device, non_blocking=loader.pin_memory)
            output = model(inputs)
            total_loss += criterion(output, target, mask).item()
            if collect_metrics:
                if stats is None:
                    stats = empty_stats(target.shape[2], station_locations)
                update_stats(stats, output, target, mask, station_locations)
    if not len(loader):
        raise ValueError("DataLoader vazio.")
    loss_parts = torch.tensor([total_loss, len(loader)], device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_parts, op=dist.ReduceOp.SUM)
    synchronize_stats(stats, device)
    return (loss_parts[0] / loss_parts[1]).item(), finalized_stats(stats) if stats else None


def train_one_iteration(args, model_type, device, datasets, run_dir: Path, iteration: int,
                        weights: tuple[float, ...], thresholds: tuple[float, ...], configuration: dict,
                        resume_state: dict | None = None, rank=0, world_size=1) -> dict:
    seed = args.seed + iteration * 10 + rank
    if resume_state is None:
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

    loader_args = loader_options(args)
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
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps * world_size}\n"
        f"DataLoader | workers={args.workers} | pin_memory={args.pin_memory} | "
        f"persistent_workers={args.persistent_workers} | "
        f"prefetch_factor={args.prefetch_factor if args.workers else 'n/a'}",
        flush=True,
    )
    best_checkpoint_path = run_dir / f"iteration_{iteration + 1}_best.pt"
    last_checkpoint_path = run_dir / f"iteration_{iteration + 1}_last.pt"
    history = []
    best_val = float("inf")
    best_epoch = 0
    stalled = 0
    start_epoch = 1
    if resume_state is not None:
        base_model = model.module if args.distributed else model
        base_model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        restore_rng_state(resume_state["rng_states"][rank])
        history = resume_state["history"]
        best_val = resume_state["best_val_loss"]
        best_epoch = resume_state["best_epoch"]
        stalled = resume_state["stalled_epochs"]
        start_epoch = resume_state["completed_epoch"] + 1
        if is_main(rank): print(
            f"Resuming iteration {iteration + 1} from epoch {start_epoch}; "
            f"best={best_val:.6f} at epoch {best_epoch} | patience={stalled}/{args.patience}",
            flush=True,
        )
    started = time.monotonic()
    stop_requested = StopRequested()
    previous_sigint = signal.signal(signal.SIGINT, stop_requested)
    previous_sigterm = signal.signal(signal.SIGTERM, stop_requested)
    interrupted = False

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        losses = []
        epoch_started = time.monotonic()
        total_batches = len(train_loader)
        data_wait_seconds = 0.0
        train_iterator = iter(train_loader)
        for batch_index in range(1, total_batches + 1):
            data_started = time.monotonic()
            inputs, target, mask = next(train_iterator)
            data_wait_seconds += time.monotonic() - data_started
            if (batch_index - 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()
                group_size = min(
                    args.gradient_accumulation_steps,
                    total_batches - batch_index + 1,
                )
            inputs = inputs.to(device, non_blocking=args.pin_memory)
            target = target.to(device, non_blocking=args.pin_memory)
            mask = mask.to(device, non_blocking=args.pin_memory)
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
            stop_flag = torch.tensor(int(stop_requested.requested), device=device)
            if dist.is_initialized():
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if stop_flag.item():
                interrupted = True
                break
        if interrupted:
            if is_main(rank): print(
                f"Interrupted during epoch {epoch}; last completed checkpoint remains "
                f"{last_checkpoint_path.name}.", flush=True,
            )
            break
        # CUDA kernels are asynchronous. Synchronize once per epoch so the
        # remainder after DataLoader wait time represents actual train work.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.monotonic() - epoch_started
        compute_seconds = max(0.0, epoch_seconds - data_wait_seconds)
        train_loss = float(np.mean(losses))
        val_loss, _ = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "data_wait_seconds": data_wait_seconds,
            "compute_seconds": compute_seconds,
            "epoch_seconds": epoch_seconds,
        })
        is_new_best = val_loss < best_val
        if is_new_best:
            best_val, best_epoch, stalled = val_loss, epoch, 0
            stopping_status = f"new best | patience=0/{args.patience}"
        else:
            stalled += 1
            stopping_status = (
                f"no improvement | best={best_val:.6f} at epoch {best_epoch} | "
                f"patience={stalled}/{args.patience}"
            )
        if is_main(rank): print(f"Iteration {iteration + 1} | epoch {epoch}/{args.epochs} | "
              f"loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"{stopping_status} | data_wait={data_wait_seconds:.1f}s | "
              f"compute={compute_seconds:.1f}s", flush=True)
        rng_states = collect_rng_states(capture_rng_state())
        state = checkpoint_state(model, optimizer, iteration, epoch, best_val, best_epoch, stalled,
                                 history, configuration, rng_states)
        if is_main(rank):
            if is_new_best:
                atomic_torch_save(state, best_checkpoint_path)
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs or stalled >= args.patience:
                atomic_torch_save(state, last_checkpoint_path)
        if stalled >= args.patience:
            if is_main(rank): print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.", flush=True)
            break

    signal.signal(signal.SIGINT, previous_sigint)
    signal.signal(signal.SIGTERM, previous_sigterm)

    if interrupted:
        return {
            "seed": seed, "interrupted": True, "last_completed_epoch": start_epoch - 1 if not history else history[-1]["epoch"],
            "checkpoint": last_checkpoint_path.name if last_checkpoint_path.exists() else None,
            "history": history,
        }

    # Checkpoints are created in this run and include trusted configuration
    # metadata in addition to tensors; PyTorch 2.6 defaults to weights_only.
    if args.distributed: dist.barrier()
    state = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    (model.module if args.distributed else model).load_state_dict(state["model_state_dict"])
    locations = station_locations(test_dataset, args)
    _, metrics = evaluate(model, test_loader, criterion, device, collect_metrics=True,
                          station_locations=locations)
    elapsed = time.monotonic() - started
    result = {"seed": seed, "best_epoch": best_epoch, "best_val_loss": best_val,
              "elapsed_seconds": elapsed, "checkpoint": best_checkpoint_path.name,
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
    if (args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0 or args.workers < 0
            or args.iterations <= 0 or args.step <= 0
            or args.gradient_accumulation_steps <= 0 or args.log_interval < 0
            or args.prefetch_factor <= 0 or args.checkpoint_every <= 0
            or args.crop_margin_pixels < 0 or args.mapping_height_orig <= 0 or args.mapping_width_orig <= 0
            or (args.stride is not None and args.stride <= 0)):
        raise ValueError("Parâmetros de treino e dimensões do mapeamento devem ser positivos; workers, log-interval e crop-margin-pixels não podem ser negativos.")
    if args.persistent_workers and not args.workers:
        raise ValueError("--persistent-workers requer --workers maior que zero.")

    rank, world_size, device = distributed_context(args.distributed)
    if args.pin_memory is None:
        args.pin_memory = device.type == "cuda"
    model_type = model_class(args.stconvs2s_root.resolve(), args.model)
    configuration = vars(args) | {
        "device": str(device), "train_years": train_years, "val_years": val_years,
        "test_years": test_years, "stconvs2s_commit": core_commit(args.stconvs2s_root),
    }
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in configuration.items()}
    configuration["world_size"] = world_size
    resume_state = None
    if args.resume:
        if args.iterations != 1:
            raise ValueError("--resume suporta apenas --iterations 1 na versao atual.")
        resume_path = args.resume.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint de retomada nao encontrado: {resume_path}")
        run_dir = resume_path.parent
        if args.run_name and args.run_name != run_dir.name:
            raise ValueError("--run-name deve corresponder ao diretorio do checkpoint ao usar --resume.")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_state.get("iteration") != 0:
            raise ValueError("--resume suporta somente checkpoints da primeira iteracao na versao atual.")
        if is_main(rank):
            print(f"Resuming from checkpoint: {resume_path}", flush=True)
    else:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.output_dir / run_name
        if is_main(rank):
            run_dir.mkdir(parents=True, exist_ok=False)
    if args.distributed:
        dist.barrier()
    datasets = (
        RadarStationMemmapDataset(args.dataset_root, train_years, stride=args.stride or args.step,
                                  target_source=args.target_source, split_name="train",
                                  crop_stations=args.crop_stations,
                                  crop_margin_pixels=args.crop_margin_pixels,
                                  station_mapping=args.station_mapping,
                                  mapping_height_orig=args.mapping_height_orig,
                                  mapping_width_orig=args.mapping_width_orig),
        RadarStationMemmapDataset(args.dataset_root, val_years, stride=args.stride or args.step,
                                  target_source=args.target_source, split_name="val",
                                  crop_stations=args.crop_stations,
                                  crop_margin_pixels=args.crop_margin_pixels,
                                  station_mapping=args.station_mapping,
                                  mapping_height_orig=args.mapping_height_orig,
                                  mapping_width_orig=args.mapping_width_orig),
        RadarStationMemmapDataset(args.dataset_root, test_years, stride=args.stride or args.step,
                                  target_source=args.target_source, split_name="test",
                                  crop_stations=args.crop_stations,
                                  crop_margin_pixels=args.crop_margin_pixels,
                                  station_mapping=args.station_mapping,
                                  mapping_height_orig=args.mapping_height_orig,
                                  mapping_width_orig=args.mapping_width_orig),
    )
    crop_metadata = datasets[0].crop_metadata
    if any(dataset.crop_metadata != crop_metadata for dataset in datasets[1:]):
        raise ValueError("O crop calculado difere entre os splits.")
    configuration["crop"] = crop_metadata
    if resume_state is not None:
        validate_resume_checkpoint(resume_state, configuration, world_size)
    if is_main(rank) and resume_state is None:
        with (run_dir / "configuration.json").open("w", encoding="utf-8") as file:
            json.dump(configuration, file, indent=2)
    results = [train_one_iteration(args, model_type, device, datasets, run_dir, index, weights, thresholds,
                                   configuration, resume_state, rank, world_size)
               for index in range(args.iterations)]
    summary = {"iterations": len(results), "results": results}
    if is_main(rank):
        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
