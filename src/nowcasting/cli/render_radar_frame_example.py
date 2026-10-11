"""Regenera figuras qualitativas de um experimento RGB já treinado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nowcasting.cli.train_radar_frames import model_constructor, render_example
from nowcasting.radar_sequence_dataset import RadarFrameSequenceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera figuras por horizonte a partir de um checkpoint RGB já treinado.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--example-index", type=int, default=None,
                        help="Índice de teste. Por padrão reutiliza o índice registrado no summary.json.")
    parser.add_argument("--output-dir", type=Path,
                        help="Diretório das figuras; padrão: o diretório do experimento.")
    parser.add_argument("--output-prefix", default="qualitative_test",
                        help="Prefixo dos cinco PNGs, antes de _T+15 até _T+75.")
    parser.add_argument("--title", help="Título comum exibido nas cinco figuras.")
    parser.add_argument("--cuda", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration_path = args.run_dir / "configuration.json"
    checkpoint_path = args.run_dir / "best.pt"
    if not configuration_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("run-dir deve conter configuration.json e best.pt.")
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    years = [int(year) for year in configuration["test_years"]]
    dataset = RadarFrameSequenceDataset(
        args.dataset_root, years, t_in=int(configuration["t_in"]), t_out=int(configuration["t_out"]),
        stride=int(configuration["stride"]), split_name="render",
    )
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    sample_x, sample_y = dataset[0]
    root = Path(configuration["stconvs2s_root"])
    constructor = model_constructor(root, configuration["model"])
    model = constructor(
        (1, *sample_x.shape), int(configuration["num_layers"]), int(configuration["hidden_dim"]),
        int(configuration["kernel_size"]), device, 0.0, int(configuration["t_out"]),
        output_channels=sample_y.shape[0],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    example_index = args.example_index if args.example_index is not None else summary.get("qualitative_test_example_index")
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index, files = render_example(
        model, dataset, device, output_dir, example_index=example_index,
        filename_prefix=args.output_prefix, title=args.title,
    )
    if output_dir == args.run_dir:
        summary.update({"qualitative_test_example_index": index, "qualitative_figures": files})
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Figuras qualitativas regeneradas | índice={index} | arquivos={', '.join(files)}", flush=True)


if __name__ == "__main__":
    main()
