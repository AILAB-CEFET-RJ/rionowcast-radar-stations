from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from nowcasting.operational_mock import (
    build_persistence_forecast,
    crop_station_region,
    render_station_forecast_map,
)


class OperationalMockTests(unittest.TestCase):
    def test_repeats_latest_input_observation_at_each_target_horizon(self) -> None:
        input_times = [pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:15:00Z")]
        target_times = [pd.Timestamp("2024-01-01T00:30:00Z"), pd.Timestamp("2024-01-01T00:45:00Z")]
        observations = pd.DataFrame(
            {
                "timestamp": input_times + target_times,
                "station_id": [1, 1, 1, 1],
                "station_name": ["A"] * 4,
                "m15": [1.0, 3.0, 5.0, 7.0],
            }
        )
        stations = pd.DataFrame(
            {"station_id": [1], "latitude": [-22.9], "longitude": [-43.2], "row": [5], "column": [6]}
        )

        forecast = build_persistence_forecast(observations, input_times, target_times, stations)

        self.assertEqual(forecast["mock_prediction_mm_15min"].tolist(), [3.0, 3.0])
        self.assertEqual(forecast["source_timestamp"].tolist(), [input_times[-1].isoformat()] * 2)

    def test_crop_station_region_reindexes_station_pixels(self) -> None:
        frames = np.zeros((5, 10, 12, 3), dtype=np.uint8)
        stations = pd.DataFrame(
            {
                "station_id": [1, 2],
                "row": [3, 7],
                "column": [4, 9],
            }
        )

        cropped_frames, cropped_stations, bounds = crop_station_region(frames, stations, margin_pixels=1)

        self.assertEqual(bounds, (2, 9, 3, 11))
        self.assertEqual(cropped_frames.shape, (5, 7, 8, 3))
        self.assertEqual(cropped_stations[["row", "column"]].values.tolist(), [[1, 1], [5, 6]])

    def test_default_map_has_no_external_tile_provider(self) -> None:
        stations = pd.DataFrame(
            {"station_id": [1], "nome": ["A"], "latitude": [-22.9], "longitude": [-43.2]}
        )
        forecast = pd.DataFrame(
            {
                "station_id": [1], "station_name": ["A"], "latitude": [-22.9], "longitude": [-43.2],
                "source_timestamp": ["2024-01-01T00:00:00+00:00"],
                "target_timestamp": ["2024-01-01T00:15:00+00:00"],
                "mock_prediction_mm_15min": [2.0],
            }
        )
        event = {"target": [{"timestamp_utc": "2024-01-01T00:15:00+00:00"}]}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            grid_path = root / "grid.npz"
            output_path = root / "product.html"
            lat, lon = np.mgrid[-23:-22:4j, -44:-43:5j]
            np.savez(grid_path, lat=lat, lon=lon)

            render_station_forecast_map(
                forecast, stations, event, grid_path, 4, 5, None, output_path, False, "none", None
            )

            html = output_path.read_text(encoding="utf-8")
        self.assertNotIn("openstreetmap.org", html)
        self.assertIn("Mock T+15 min", html)


if __name__ == "__main__":
    unittest.main()
