from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.residual_persistence import (
    ResidualPersistenceForecaster,
    build_forecaster,
    persistence_from_station_channels,
)
from nowcasting.cli.train import (
    empty_stats,
    finalized_stats,
    update_persistence_history_stats,
)


class ZeroCore(nn.Module):
    def __init__(self, output_channels: int, horizons: int):
        super().__init__()
        self.output_channels, self.horizons = output_channels, horizons
        self.last_input_channels: int | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.last_input_channels = inputs.shape[1]
        return torch.zeros(
            (inputs.shape[0], self.output_channels, self.horizons, *inputs.shape[-2:]),
            dtype=inputs.dtype, device=inputs.device,
        )


class ResidualPersistenceTests(unittest.TestCase):
    def test_uses_last_valid_value_and_zero_for_missing_history(self):
        values = torch.tensor([[[[0.2, 0.5]], [[0.8, 0.9]], [[0.4, 0.7]]]])
        masks = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]]])
        output = persistence_from_station_channels(values, masks, horizons=2)
        expected = torch.tensor([[[[[0.8, 0.0]], [[0.8, 0.0]]]]])
        self.assertTrue(torch.equal(output, expected))

    def test_zero_correction_reproduces_b1_exactly(self):
        core = ZeroCore(output_channels=1, horizons=2)
        model = ResidualPersistenceForecaster(core, nonstation_channels=3, horizons=2)
        inputs = torch.zeros((1, 5, 3, 1, 2))
        inputs[0, 3, :, 0, 0] = torch.tensor([0.1, 0.4, 0.9])
        inputs[0, 4, :, 0, 0] = torch.tensor([1.0, 1.0, 0.0])
        inputs[0, 3, :, 0, 1] = torch.tensor([0.3, 0.5, 0.8])
        inputs[0, 4, :, 0, 1] = torch.tensor([0.0, 1.0, 1.0])
        output = model(inputs)
        expected = torch.tensor([[[[[0.4, 0.8]], [[0.4, 0.8]]]]])
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(core.last_input_channels, 3)

    def test_residual_requires_value_and_mask_channels(self):
        model = ResidualPersistenceForecaster(ZeroCore(1, 2), nonstation_channels=3, horizons=2)
        with self.assertRaisesRegex(ValueError, "canais"):
            model(torch.zeros((1, 4, 3, 2, 2)))

    def test_builder_rejects_residual_without_station_history(self):
        sample_x = torch.zeros((3, 2, 2, 2))
        sample_y = torch.zeros((1, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "input-stations"):
            build_forecaster(
                lambda *_args, **_kwargs: ZeroCore(1, 2), sample_x, sample_y, num_layers=1,
                hidden_dim=1, kernel_size=1, device=torch.device("cpu"), step=2,
                forecast_formulation="residual-persistence", input_stations=False,
            )

    def test_history_diagnostic_counts_each_valid_target_horizon(self):
        stats = empty_stats(horizons=2, residual_persistence=True)
        inputs = torch.zeros((1, 5, 3, 1, 2))
        inputs[0, -1, 1, 0, 0] = 1.0
        mask = torch.ones((1, 1, 2, 1, 2))
        update_persistence_history_stats(stats, inputs, mask)
        result = finalized_stats(stats)["persistence_history"]
        self.assertEqual(result["valid_pairs"], 4)
        self.assertEqual(result["pairs_without_history"], 2)
        self.assertEqual(result["fraction_without_history"], 0.5)


if __name__ == "__main__":
    unittest.main()
