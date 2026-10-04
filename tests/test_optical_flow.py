from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.optical_flow import extrapolate_visual_echo, station_echo_features, visual_echo
from nowcasting.cli.evaluate_optical_flow import metrics


class OpticalFlowTests(unittest.TestCase):
    def test_visual_echo_uses_rgb_input_only(self):
        frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)
        frames[1, 1, 2, 1] = 255
        echo = visual_echo(frames)
        self.assertEqual(echo.shape, (2, 3, 4))
        self.assertEqual(float(echo[1, 1, 2]), 1.0)

    def test_extrapolation_returns_requested_causal_horizons(self):
        frames = np.zeros((5, 128, 128, 3), dtype=np.uint8)
        for step in range(5):
            frames[step, 20:60, 10 + step * 3:50 + step * 3, 1] = 255
        forecast, fallback = extrapolate_visual_echo(frames, 3)
        self.assertFalse(fallback)
        self.assertEqual(forecast.shape, (3, 128, 128))
        self.assertTrue(np.isfinite(forecast).all())

    def test_station_features_respect_neighborhood(self):
        forecast = np.zeros((2, 5, 5), dtype=np.float32)
        forecast[:, 2, 2] = (0.5, 1.0)
        features = station_echo_features(forecast, np.array([[2, 2]]), 0)
        np.testing.assert_allclose(features[:, 0], [[0.5, 0.5, 1.0], [1.0, 1.0, 1.0]])

    def test_metrics_groups_each_horizon_without_dropping_masked_values(self):
        predicted = np.log1p(np.array([0.0, 1.0, 2.0, 3.0]))
        observed = np.log1p(np.array([0.0, 0.0, 2.0, 0.0]))
        result = metrics(predicted, observed, np.array([True, False, True, True]), horizons=2, stations=1)
        self.assertEqual(result["global"]["n"], 3)
        self.assertEqual(result["horizons"][0]["n"], 2)
        self.assertEqual(result["horizons"][1]["n"], 1)


if __name__ == "__main__":
    unittest.main()
