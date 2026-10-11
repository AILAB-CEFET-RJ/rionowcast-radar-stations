"""Treina STConvS2S para prever frames RGB futuros do radar do Sumaré."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from nowcasting.dataset import parse_years
from nowcasting.paths import project_root
from nowcasting.radar_sequence_dataset import RadarFrameSequenceDataset


PROJECT_ROOT = project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STConvS2S para previsão de frames RGB do radar, sem targets de estações.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-years", required=True)
    parser.add_argument("--val-years", required=True)
    parser.add_argument("--test-years", required=True)
    parser.add_argument("--stconvs2s-root", type=Path, default=PROJECT_ROOT / "external" / "stconvs2s")
    parser.add_argument("--model", choices=("stconvs2s-c", "stconvs2s-r"), default="stconvs2s-c")
    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--t-out", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--loss", choices=("mae", "huber"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--active-threshold", type=float, default=0.0,
                        help="Pixel é eco visual se max(R,G,B) for maior que este limiar normalizado.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "experiments")
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def validate_splits(train: list[int], val: list[int], test: list[int]) -> None:
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Os splits temporais não podem compartilhar anos.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def core_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_constructor(root: Path, name: str):
    if not (root / "model" / "stconvs2s.py").is_file():
        raise FileNotFoundError(f"STConvS2S não encontrado em {root}")
    sys.path.insert(0, str(root))
    from model.stconvs2s import STConvS2S_C, STConvS2S_R
    return {"stconvs2s-c": STConvS2S_C, "stconvs2s-r": STConvS2S_R}[name]


def loss_value(output: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.loss == "mae":
        return functional.l1_loss(output, target)
    return functional.huber_loss(output, target, delta=args.huber_delta)


def empty_metrics(horizons: int) -> dict:
    return {"global": {"se": 0.0, "ae": 0.0, "n": 0}, "active_echo": {"se": 0.0, "ae": 0.0, "n": 0},
            "horizons": [{"se": 0.0, "ae": 0.0, "n": 0} for _ in range(horizons)]}


def accumulate(destination: dict, error: torch.Tensor, selection: torch.Tensor) -> None:
    values = error[selection]
    destination["se"] += values.square().sum().item()
    destination["ae"] += values.abs().sum().item()
    destination["n"] += int(values.numel())


def finalize(metrics: dict) -> dict:
    def result(values: dict) -> dict:
        if not values["n"]:
            return {"n": 0, "rmse": None, "mae": None}
        return {"n": values["n"], "rmse": (values["se"] / values["n"]) ** 0.5, "mae": values["ae"] / values["n"]}
    return {"global": result(metrics["global"]), "active_echo": result(metrics["active_echo"]),
            "horizons": [result(item) for item in metrics["horizons"]]}


def predict_persistence(inputs: torch.Tensor, horizons: int) -> torch.Tensor:
    return inputs[:, :, -1:].expand(-1, -1, horizons, -1, -1)


def evaluate(model: torch.nn.Module | None, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> tuple[dict, dict]:
    """Return metrics for the network (or persistence if model is None) and persistence."""
    if model is not None:
        model.eval()
    network_stats = persistence_stats = None
    with torch.no_grad():
        for inputs, target in loader:
            inputs, target = inputs.to(device, non_blocking=True), target.to(device, non_blocking=True)
            persistence = predict_persistence(inputs, target.shape[2])
            output = persistence if model is None else torch.clamp(model(inputs), 0.0, 1.0)
            if network_stats is None:
                network_stats, persistence_stats = empty_metrics(target.shape[2]), empty_metrics(target.shape[2])
            active = target.amax(dim=1, keepdim=True) > args.active_threshold
            for prediction, destination in ((output, network_stats), (persistence, persistence_stats)):
                error = prediction - target
                accumulate(destination["global"], error, torch.ones_like(error, dtype=torch.bool))
                accumulate(destination["active_echo"], error, active.expand_as(error))
                for horizon, values in enumerate(destination["horizons"]):
                    accumulate(values, error[:, :, horizon], torch.ones_like(error[:, :, horizon], dtype=torch.bool))
    assert network_stats is not None and persistence_stats is not None
    return finalize(network_stats), finalize(persistence_stats)


def render_example(model: torch.nn.Module, dataset: RadarFrameSequenceDataset, device: torch.device, output_path: Path) -> int:
    """Write a compact qualitative comparison using the test sample with most future echo."""
    candidate_count = min(len(dataset), 200)
    index = max(range(candidate_count), key=lambda item: float(dataset[item][1].max()))
    inputs, target = dataset[index]
    with torch.no_grad():
        prediction = torch.clamp(model(inputs[None].to(device)), 0.0, 1.0)[0].cpu()
    persistence = predict_persistence(inputs[None], target.shape[1])[0]
    figure, axes = plt.subplots(4, target.shape[1], figsize=(3 * target.shape[1], 10), squeeze=False)
    rows = (("Último frame de entrada", inputs[:, -1]), ("Persistência", persistence),
            ("STConvS2S", prediction), ("Frame observado", target))
    for row, (label, image) in enumerate(rows):
        for horizon in range(target.shape[1]):
            frame = image if image.ndim == 3 else image[:, horizon]
            axes[row, horizon].imshow(frame.permute(1, 2, 0).numpy())
            axes[row, horizon].axis("off")
            if row == 0:
                axes[row, horizon].set_title(f"T+{15 * (horizon + 1)} min")
        axes[row, 0].set_ylabel(label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return index


def main() -> None:
    args = parse_args()
    if min(args.t_in, args.t_out, args.stride, args.epochs, args.patience, args.batch_size) <= 0:
        raise ValueError("t-in, t-out, stride, epochs, patience e batch-size devem ser positivos.")
    if args.workers < 0 or args.active_threshold < 0 or args.active_threshold > 1:
        raise ValueError("workers deve ser não negativo e active-threshold deve estar entre 0 e 1.")
    train_years, val_years, test_years = map(parse_years, (args.train_years, args.val_years, args.test_years))
    validate_splits(train_years, val_years, test_years)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    kwargs = dict(t_in=args.t_in, t_out=args.t_out, stride=args.stride)
    train_data = RadarFrameSequenceDataset(args.dataset_root, train_years, split_name="train", **kwargs)
    val_data = RadarFrameSequenceDataset(args.dataset_root, val_years, split_name="val", **kwargs)
    test_data = RadarFrameSequenceDataset(args.dataset_root, test_years, split_name="test", **kwargs)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)
    sample_x, sample_y = train_data[0]
    constructor = model_constructor(args.stconvs2s_root, args.model)
    model = constructor((1, *sample_x.shape), args.num_layers, args.hidden_dim, args.kernel_size, device, 0.0,
                        args.t_out, output_channels=sample_y.shape[0]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9, eps=1e-6)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({"train_years": train_years, "val_years": val_years, "test_years": test_years,
                          "stconvs2s_commit": core_commit(args.stconvs2s_root), "task": "radar-rgb-frame-forecast",
                          "data_semantics": "RGB visual radar product; not calibrated reflectivity or precipitation"})
    (run_dir / "configuration.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Input/output | RGB={tuple(sample_x.shape)} -> {tuple(sample_y.shape)} | device={device}", flush=True)
    best_active_mae, best_epoch, stalled, history = float("inf"), 0, 0, []
    best_path = run_dir / "best.pt"
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch, (inputs, target) in enumerate(train_loader, start=1):
            inputs, target = inputs.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_value(model(inputs), target, args)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.log_interval and (batch % args.log_interval == 0 or batch == len(train_loader)):
                print(f"Epoch {epoch}/{args.epochs} | batch {batch}/{len(train_loader)} | loss={np.mean(losses):.6f}", flush=True)
        val_network, val_persistence = evaluate(model, val_loader, device, args)
        active_mae = val_network["active_echo"]["mae"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": val_network})
        if active_mae < best_active_mae:
            best_active_mae, best_epoch, stalled = active_mae, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation": val_network,
                        "configuration": configuration}, best_path)
            status = "improved"
        else:
            stalled += 1
            status = f"no improvement ({stalled}/{args.patience})"
        print(f"Epoch {epoch} | val active-echo MAE={active_mae:.6f} | {status}", flush=True)
        if stalled >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.", flush=True)
            break
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    test_network, test_persistence = evaluate(model, test_loader, device, args)
    example_index = render_example(model, test_data, device, run_dir / "qualitative_test_example.png")
    summary = {"model": args.model, "best_epoch": best_epoch, "best_validation_active_echo_mae": best_active_mae,
               "elapsed_seconds": time.monotonic() - started, "test_metrics": test_network,
               "persistence_test_metrics": test_persistence, "qualitative_test_example_index": example_index,
               "history": history}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Complete | test RGB MAE={test_network['global']['mae']:.6f} | active-echo MAE={test_network['active_echo']['mae']:.6f}", flush=True)


if __name__ == "__main__":
    main()
