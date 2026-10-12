from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.residual_persistence import (
    ConditionalLateFusionForecaster,
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


class ConstantCore(ZeroCore):
    def __init__(self, output_channels: int, horizons: int, value: float):
        super().__init__(output_channels, horizons)
        self.value = value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.last_input_channels = inputs.shape[1]
        return torch.full(
            (inputs.shape[0], self.output_channels, self.horizons, *inputs.shape[-2:]), self.value,
            dtype=inputs.dtype, device=inputs.device,
        )


class FirstInputCore(ZeroCore):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.last_input_channels = inputs.shape[1]
        return inputs[:, :1, :1].expand(-1, self.output_channels, self.horizons, -1, -1)


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

    def test_conditional_late_fusion_gates_c13_more_without_rgb_echo(self):
        radar_core = ZeroCore(1, 2)
        goes_core = ConstantCore(1, 2, 1.0)
        model = ConditionalLateFusionForecaster(
            radar_core, goes_core, radar_channels=3, goes_channels=1, horizons=2,
        )
        inputs = torch.zeros((1, 6, 3, 1, 2))
        inputs[0, 0, -1, 0, 1] = 1.0  # Echo only at the second spatial pixel.
        gate = model.goes_gate(inputs[:, :3])
        output = model(inputs)
        self.assertGreater(gate[0, 0, 0, 0, 0].item(), gate[0, 0, 0, 0, 1].item())
        self.assertEqual(radar_core.last_input_channels, 3)
        self.assertEqual(goes_core.last_input_channels, 1)
        self.assertGreater(output[0, 0, 0, 0, 0].item(), output[0, 0, 0, 0, 1].item())

    def test_builder_rejects_conditional_fusion_without_goes(self):
        sample_x = torch.zeros((5, 2, 2, 2))
        sample_y = torch.zeros((1, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "input-goes"):
            build_forecaster(
                lambda *_args, **_kwargs: ZeroCore(1, 2), sample_x, sample_y, num_layers=1,
                hidden_dim=1, kernel_size=1, device=torch.device("cpu"), step=2,
                forecast_formulation="conditional-late-fusion", input_stations=True, input_goes=False,
            )

    def test_conditional_capacity_control_zeros_c13_values_but_keeps_the_branch(self):
        inputs = torch.zeros((1, 6, 2, 1, 1))
        inputs[:, 3] = 2.0
        with_c13 = ConditionalLateFusionForecaster(
            ZeroCore(1, 2), FirstInputCore(1, 2), radar_channels=3, goes_channels=1, horizons=2,
        )
        control = ConditionalLateFusionForecaster(
            ZeroCore(1, 2), FirstInputCore(1, 2), radar_channels=3, goes_channels=1, horizons=2,
            goes_branch_control=True,
        )
        self.assertGreater(with_c13(inputs).abs().max().item(), 0.0)
        self.assertEqual(control(inputs).abs().max().item(), 0.0)

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
