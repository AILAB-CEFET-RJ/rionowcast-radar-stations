from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.goes16 import parse_goes_filename, parse_goes_filename_bounds, select_causal_scene
from nowcasting.cli.stream_goes16_memmaps import select_causal_scenes_by_channel


class Goes16Tests(unittest.TestCase):
    def test_parse_noaa_cmipf_filename(self):
        channel, timestamp = parse_goes_filename(
            "OR_ABI-L2-CMIPF-M6C13_G16_s20231230000123_e20231230010190_c20231230010270.nc"
        )
        self.assertEqual(channel, "C13")
        self.assertEqual(timestamp, datetime(2023, 5, 3, 0, 0, 12, tzinfo=timezone.utc))

    def test_parse_noaa_cmipf_end_timestamp(self):
        _, start, end = parse_goes_filename_bounds(
            "OR_ABI-L2-CMIPF-M6C13_G16_s20231230000123_e20231230010190_c20231230010270.nc"
        )
        self.assertEqual(start, datetime(2023, 5, 3, 0, 0, 12, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2023, 5, 3, 0, 10, 19, tzinfo=timezone.utc))

    def test_causal_scene_requires_acquisition_to_end_before_target(self):
        start = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 0, 10, tzinfo=timezone.utc)
        target = datetime(2023, 1, 1, 0, 15, tzinfo=timezone.utc)
        crossing_end = datetime(2023, 1, 1, 0, 20, tzinfo=timezone.utc)
        selected = select_causal_scene(
            [(start, end, Path("complete.nc")), (end, crossing_end, Path("crossing.nc"))], target, 20,
        )
        self.assertEqual(selected, (start, end, Path("complete.nc")))
        self.assertIsNone(select_causal_scene([(start, end, Path("complete.nc"))], target, 4))

    def test_selects_each_channel_independently(self):
        target = datetime(2023, 1, 1, 0, 15, tzinfo=timezone.utc)
        candidates = [
            (datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), datetime(2023, 1, 1, 0, 9, tzinfo=timezone.utc), Path("OR_ABI-L2-CMIPF-M6C13_G16_s20230010000000_e20230010009000_c20230010009100.nc")),
            (datetime(2023, 1, 1, 0, 1, tzinfo=timezone.utc), datetime(2023, 1, 1, 0, 10, tzinfo=timezone.utc), Path("OR_ABI-L2-CMIPF-M6C14_G16_s20230010001000_e20230010010000_c20230010010100.nc")),
        ]
        selected = select_causal_scenes_by_channel(candidates, ["C13", "C14"], target, 20)
        self.assertTrue(selected[0][2].name.startswith("OR_ABI-L2-CMIPF-M6C13"))
        self.assertTrue(selected[1][2].name.startswith("OR_ABI-L2-CMIPF-M6C14"))


if __name__ == "__main__":
    unittest.main()
