from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.radar_sequence_dataset import RadarFrameSequenceDataset


class RadarFrameSequenceDatasetTests(unittest.TestCase):
    def create_year(self, root: Path) -> None:
        directory = root / "year=2024"
        directory.mkdir()
        shape = (8, 2, 3, 3)
        frames = np.memmap(directory / "radar_frames.dat", dtype=np.uint8, mode="w+", shape=shape)
        frames[:] = np.arange(shape[0], dtype=np.uint8)[:, None, None, None]
        frames.flush()
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=15 * index) for index in range(shape[0])]
        timestamps[5:] = [value + timedelta(minutes=15) for value in timestamps[5:]]
        np.save(directory / "radar_timestamps.npy", np.asarray([value.isoformat() for value in timestamps], dtype="<U19"))
        (directory / "metadata.json").write_text(json.dumps({"shape": list(shape), "dtype": "uint8",
            "frames_file": "radar_frames.dat", "timestamps_file": "radar_timestamps.npy", "aggregate_minutes": 15,
            "enforce_timestamp_continuity": True}), encoding="utf-8")

    def test_returns_rgb_past_and_future_and_excludes_gap_crossing_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_year(root)
            dataset = RadarFrameSequenceDataset(root, [2024], t_in=2, t_out=2, stride=1)
            self.assertEqual(len(dataset), 2)
            x, y = dataset[0]
            self.assertEqual(tuple(x.shape), (3, 2, 2, 3))
            self.assertEqual(tuple(y.shape), (3, 2, 2, 3))
            self.assertEqual(x[0, 0, 0, 0].item(), 0.0)
            self.assertAlmostEqual(y[0, 0, 0, 0].item(), 2 / 255)


if __name__ == "__main__":
    unittest.main()
