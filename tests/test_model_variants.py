from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.cli import train as train_cli
from nowcasting.cli.train import model_class, model_parameter_counts
from nowcasting.residual_persistence import build_forecaster


def create_year(root: Path, year: int) -> None:
    year_dir = root / f"year={year}"
    year_dir.mkdir(parents=True)
    radar_shape = (12, 2, 2, 3)
    target_shape = (12, 2, 2, 1)
    radar = np.memmap(year_dir / "radar_frames.dat", dtype=np.uint8, mode="w+", shape=radar_shape)
    target = np.memmap(year_dir / "Y_alertario.dat", dtype=np.float32, mode="w+", shape=target_shape)
    mask = np.memmap(year_dir / "M_alertario.dat", dtype=np.uint8, mode="w+", shape=target_shape)
    radar[:] = 0
    target[:] = 0
    mask[:] = 0
    target[:, 0, 0, 0] = np.log1p(1.0)
    mask[:, 0, 0, 0] = 1
    radar.flush()
    target.flush()
    mask.flush()
    (year_dir / "metadata.json").write_text(
        json.dumps({"shape": list(radar_shape), "dtype": "uint8"}), encoding="utf-8",
    )
    (year_dir / "targets_alertario_metadata.json").write_text(
        json.dumps({
            "shape": list(target_shape), "Y_dtype": "float32", "M_dtype": "uint8",
            "Y_file": "Y_alertario.dat", "M_file": "M_alertario.dat",
        }),
        encoding="utf-8",
    )


class ModelVariantTests(unittest.TestCase):
    def build(self, name: str):
        constructor = model_class(PROJECT_ROOT / "external" / "stconvs2s", name)
        return constructor(
            (1, 3, 5, 8, 8),
            num_layers=1,
            hidden_dim=8,
            kernel_size=3,
            device=torch.device("cpu"),
            dropout_rate=0.0,
            step=5,
            output_channels=1,
        )

    def test_capacity_variant_is_available_and_preserves_shape(self):
        model = self.build("stconvs2s-c-cap")
        output = model(torch.randn(2, 3, 5, 8, 8))
        self.assertEqual(output.shape, (2, 1, 5, 8, 8))

    def test_capacity_variant_adds_expected_parameter_budget(self):
        torch.manual_seed(11)
        baseline = self.build("stconvs2s-c")
        torch.manual_seed(11)
        capacity = self.build("stconvs2s-c-cap")

        baseline_counts = model_parameter_counts(baseline)
        capacity_counts = model_parameter_counts(capacity)
        channels = 8
        expected_control_parameters = 4 * channels ** 2 + 5 * channels + 1

        self.assertEqual(
            capacity_counts["trainable"] - baseline_counts["trainable"],
            expected_control_parameters,
        )
        self.assertEqual(capacity_counts["total"], capacity_counts["trainable"])

    def test_capacity_variant_works_in_residual_persistence_formulation(self):
        sample_x = torch.zeros((5, 5, 8, 8))
        sample_y = torch.zeros((1, 5, 8, 8))
        constructor = model_class(
            PROJECT_ROOT / "external" / "stconvs2s", "stconvs2s-c-cap",
        )
        model = build_forecaster(
            constructor, sample_x, sample_y, num_layers=1, hidden_dim=8,
            kernel_size=3, device=torch.device("cpu"), step=5,
            forecast_formulation="residual-persistence", input_stations=True,
        )

        output = model(torch.zeros((2, 5, 5, 8, 8)))
        self.assertEqual(output.shape, (2, 1, 5, 8, 8))

    def test_attention_variant_is_available_and_preserves_shape(self):
        model = self.build("stconvs2s-c-attn")
        output = model(torch.randn(2, 3, 5, 8, 8))
        self.assertEqual(output.shape, (2, 1, 5, 8, 8))

    def test_attention_variant_matches_capacity_control_budget(self):
        channels = 32

        def build(name: str):
            constructor = model_class(PROJECT_ROOT / "external" / "stconvs2s", name)
            return constructor(
                (1, 3, 5, 8, 8), num_layers=1, hidden_dim=channels,
                kernel_size=3, device=torch.device("cpu"), dropout_rate=0.0,
                step=5, output_channels=1,
            )

        baseline = model_parameter_counts(build("stconvs2s-c"))["trainable"]
        capacity = model_parameter_counts(build("stconvs2s-c-cap"))["trainable"] - baseline
        attention = model_parameter_counts(build("stconvs2s-c-attn"))["trainable"] - baseline

        self.assertEqual(capacity, 4 * channels ** 2 + 5 * channels + 1)
        self.assertEqual(attention, 4 * channels ** 2 + 6 * channels + 1 + 5 * channels)
        self.assertLessEqual(abs(attention - capacity) / capacity, 0.05)

    def test_attention_variant_works_in_residual_persistence_formulation(self):
        sample_x = torch.zeros((5, 5, 8, 8))
        sample_y = torch.zeros((1, 5, 8, 8))
        constructor = model_class(
            PROJECT_ROOT / "external" / "stconvs2s", "stconvs2s-c-attn",
        )
        model = build_forecaster(
            constructor, sample_x, sample_y, num_layers=1, hidden_dim=8,
            kernel_size=3, device=torch.device("cpu"), step=5,
            forecast_formulation="residual-persistence", input_stations=True,
        )

        output = model(torch.zeros((2, 5, 5, 8, 8)))
        self.assertEqual(output.shape, (2, 1, 5, 8, 8))

    def test_training_cli_records_capacity_variant_and_parameter_counts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_root = root / "dataset"
            output_root = root / "experiments"
            create_year(dataset_root, 2020)
            create_year(dataset_root, 2021)
            arguments = [
                "nowcasting-train",
                "--stconvs2s-root", str(PROJECT_ROOT / "external" / "stconvs2s"),
                "--dataset-root", str(dataset_root),
                "--train-years", "2020", "--val-years", "2021", "--test-years", "2022",
                "--input-stations", "--forecast-formulation", "residual-persistence",
                "--model", "stconvs2s-c-cap", "--num-layers", "1", "--hidden-dim", "4",
                "--kernel-size", "3", "--batch-size", "1", "--epochs", "1", "--patience", "1",
                "--step", "5", "--stride", "5", "--workers", "0", "--log-interval", "0",
                "--loss", "masked-huber", "--skip-test", "--output-dir", str(output_root),
                "--run-name", "a0-cap-smoke", "--seed", "1000",
            ]
            with mock.patch.object(sys, "argv", arguments):
                train_cli.main()

            run_dir = output_root / "a0-cap-smoke"
            configuration = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(configuration["model"], "stconvs2s-c-cap")
            self.assertGreater(configuration["model_parameters"]["trainable"], 0)
            self.assertEqual(summary["model_parameters"], configuration["model_parameters"])
            self.assertTrue((run_dir / "iteration_1_best.pt").is_file())

    def test_training_cli_records_attention_variant_and_parameter_counts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_root = root / "dataset"
            output_root = root / "experiments"
            create_year(dataset_root, 2020)
            create_year(dataset_root, 2021)
            arguments = [
                "nowcasting-train",
                "--stconvs2s-root", str(PROJECT_ROOT / "external" / "stconvs2s"),
                "--dataset-root", str(dataset_root),
                "--train-years", "2020", "--val-years", "2021", "--test-years", "2022",
                "--input-stations", "--forecast-formulation", "residual-persistence",
                "--model", "stconvs2s-c-attn", "--num-layers", "1", "--hidden-dim", "4",
                "--kernel-size", "3", "--batch-size", "1", "--epochs", "1", "--patience", "1",
                "--step", "5", "--stride", "5", "--workers", "0", "--log-interval", "0",
                "--loss", "masked-huber", "--skip-test", "--output-dir", str(output_root),
                "--run-name", "a1-attn-smoke", "--seed", "1000",
            ]
            with mock.patch.object(sys, "argv", arguments):
                train_cli.main()

            run_dir = output_root / "a1-attn-smoke"
            configuration = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(configuration["model"], "stconvs2s-c-attn")
            self.assertGreater(configuration["model_parameters"]["trainable"], 0)
            self.assertEqual(summary["model_parameters"], configuration["model_parameters"])
            self.assertIsNone(summary["results"][0]["peak_cuda_memory_allocated_bytes"])
            self.assertIsNone(summary["results"][0]["peak_cuda_memory_reserved_bytes"])
            self.assertTrue((run_dir / "iteration_1_best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
