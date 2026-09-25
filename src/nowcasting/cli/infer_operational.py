from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nowcasting.cli.train import model_class
from nowcasting.operational_inference import (
    prepare_radar_input,
    validate_checkpoint_configuration,
)
from nowcasting.paths import project_root
from nowcasting.radar_capture import load_capture_config


PROJECT_ROOT = project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa inferência retrospectiva em uma sequência de capturas do radar Sumaré."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True, help="events.json produzido pelo seletor operacional.")
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--capture-config", type=Path, required=True)
    parser.add_argument("--frame-height", type=int, default=128)
    parser.add_argument("--frame-width", type=int, default=128)
    parser.add_argument("--stconvs2s-root", type=Path, default=PROJECT_ROOT / "external" / "stconvs2s")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda", default="0")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state or "configuration" not in state:
        raise ValueError("Checkpoint sem model_state_dict/configuration.")
    return state


def main() -> None:
    args = parse_args()
    if min(args.frame_height, args.frame_width) <= 0:
        raise ValueError("--frame-height e --frame-width devem ser positivos.")
    events = json.loads(args.events.read_text(encoding="utf-8"))
    if not 0 <= args.event_index < len(events):
        raise IndexError(f"--event-index deve estar entre 0 e {len(events) - 1}.")
    state = load_checkpoint(args.checkpoint)
    configuration = state["configuration"]
    capture_config = load_capture_config(args.capture_config)
    validate_checkpoint_configuration(configuration, capture_config)
    model_input = prepare_radar_input(
        events[args.event_index], args.capture_root, capture_config, configuration,
        args.frame_width, args.frame_height,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.cuda))
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")
    model_type = model_class(args.stconvs2s_root.resolve(), configuration["model"])
    model = model_type(
        tuple(model_input.shape),
        int(configuration["num_layers"]),
        int(configuration["hidden_dim"]),
        int(configuration["kernel_size"]),
        device,
        0.0,
        int(configuration["step"]),
        output_channels=1,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction_log1p = model(torch.from_numpy(model_input).to(device)).cpu().numpy()[0, 0]
    prediction_mm = np.maximum(np.expm1(prediction_log1p), 0.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event = events[args.event_index]
    np.savez_compressed(
        args.output_dir / "forecast.npz",
        prediction_log1p=prediction_log1p.astype(np.float32),
        prediction_mm_15min=prediction_mm.astype(np.float32),
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "event": event,
                "frame_shape": list(model_input.shape),
                "forecast_shape": list(prediction_mm.shape),
                "unit": "mm/15min",
                "capture_config": capture_config,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Inferência concluída: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
