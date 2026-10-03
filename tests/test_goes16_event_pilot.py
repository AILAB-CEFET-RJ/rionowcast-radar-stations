from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.cli.build_goes16_event_pilot import scenes_near
from nowcasting.cli.select_goes16_pilot_events import candidates_for_year
from nowcasting.goes16 import select_causal_scene


class FakeS3:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def ls(self, prefix, detail=False):
        self.calls.append(prefix)
        if prefix not in self.objects:
            raise FileNotFoundError(prefix)
        return self.objects[prefix]


class Goes16EventPilotTests(unittest.TestCase):
    def test_event_candidates_require_full_contiguous_sequence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset" / "year=2024"
            root.mkdir(parents=True)
            timestamps = np.array([f"2024-01-01T{index // 4:02d}:{(index % 4) * 15:02d}:00" for index in range(12)], dtype="<U19")
            np.save(root / "radar_timestamps.npy", timestamps)
            np.savez(root / "targets_alertario_sparse.npz", frame=np.arange(5, 10, dtype=np.int32), row=np.zeros(5, dtype=np.uint16), column=np.ones(5, dtype=np.uint16), station_id=np.full(5, 7, dtype=np.int32), value=np.log1p(np.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)))
            (root / "targets_alertario_metadata.json").write_text(json.dumps({"sparse_file": "targets_alertario_sparse.npz"}), encoding="utf-8")
            candidates = candidates_for_year(root.parent, 2024, 5, 5)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["target_frame_index"], 5)
            self.assertEqual(candidates[0]["station_id"], 7)
            self.assertAlmostEqual(candidates[0]["max_m15_future_mm_15min"], 10.0, places=5)

    def test_streaming_listing_uses_current_and_previous_hour_only(self):
        config = {"source": {"bucket": "noaa-goes16", "product": "ABI-L2-CMIPF"}, "channels": ["C13"]}
        previous = "noaa-goes16/ABI-L2-CMIPF/2024/001/00/OR_ABI-L2-CMIPF-M6C13_G16_s20240010050200_e20240010059527_c20240010100004.nc"
        current = "noaa-goes16/ABI-L2-CMIPF/2024/001/01/OR_ABI-L2-CMIPF-M6C13_G16_s20240010100200_e20240010109527_c20240010110004.nc"
        fs = FakeS3({"noaa-goes16/ABI-L2-CMIPF/2024/001/00": [previous], "noaa-goes16/ABI-L2-CMIPF/2024/001/01": [current]})
        target = datetime(2024, 1, 1, 1, 5, tzinfo=timezone.utc)
        scenes = scenes_near(fs, config, target, {})
        selected = select_causal_scene(scenes, target, 20)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], datetime(2024, 1, 1, 0, 50, 20, tzinfo=timezone.utc))
        self.assertEqual(selected[1], datetime(2024, 1, 1, 0, 59, 52, tzinfo=timezone.utc))
        self.assertEqual(len(fs.calls), 2)

    def test_missing_hour_prefix_is_an_empty_candidate_list(self):
        config = {"source": {"bucket": "noaa-goes16", "product": "ABI-L2-CMIPF"}, "channels": ["C13"]}
        target = datetime(2024, 1, 1, 1, 5, tzinfo=timezone.utc)
        self.assertEqual(scenes_near(FakeS3({}), config, target, {}), [])


if __name__ == "__main__":
    unittest.main()
