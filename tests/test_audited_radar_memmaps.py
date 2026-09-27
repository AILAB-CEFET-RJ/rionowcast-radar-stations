from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.cli.build_radar_memmaps import collect_files_for_utc_year, prepare_year_resume, process_year
from nowcasting.dataset import RadarStationMemmapDataset
from nowcasting.radar_timestamps import local_filename_timestamp_to_utc


class AuditedRadarMemmapTests(unittest.TestCase):
    def test_annual_resume_skips_finalized_year_and_restarts_only_partial_year(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            finalized = root / "year=2023"
            finalized.mkdir()
            self.assertTrue(prepare_year_resume(root, 2023, resume=True, restart_partial=False))

            partial = root / "year=2024.partial"
            partial.mkdir()
            (partial / "incomplete").write_text("partial", encoding="utf-8")
            self.assertFalse(prepare_year_resume(root, 2024, resume=True, restart_partial=True))
            self.assertFalse(partial.exists())

    def test_historical_png_timezone_preserves_brazilian_dst(self) -> None:
        self.assertEqual(
            local_filename_timestamp_to_utc(datetime(2024, 1, 1, 12, 0), "America/Sao_Paulo"),
            datetime(2024, 1, 1, 15, 0),
        )
        self.assertEqual(
            local_filename_timestamp_to_utc(datetime(2018, 1, 1, 12, 0), "America/Sao_Paulo"),
            datetime(2018, 1, 1, 14, 0),
        )

    def test_utc_collection_includes_previous_local_calendar_year(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "2023" / "12" / "31"
            directory.mkdir(parents=True)
            Image.new("RGBA", (1, 1)).save(directory / "2023_12_31_23_00.png")
            items, report = collect_files_for_utc_year(root, 2024, "America/Sao_Paulo")
            self.assertEqual([timestamp for timestamp, _ in items], [datetime(2024, 1, 1, 2, 0)])
            self.assertEqual(report["source_local_years"], [2023, 2024])

    def test_builder_records_missing_bucket_and_crops_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw" / "2024" / "01" / "01"
            raw.mkdir(parents=True)
            config = {
                "source_image": {"width": 6, "height": 4},
                "reflectivity_crop": {"left": 2, "top": 1, "right_exclusive": 6, "bottom_exclusive": 4},
            }
            for minute in (0, 2, 4, 6, 8, 10, 30, 32, 34, 36, 38, 40):
                image = np.zeros((4, 6, 4), dtype=np.uint8)
                image[:, :2, :3] = 255
                image[1:4, 2:6, 1] = minute + 1
                image[:, :, 3] = 255
                Image.fromarray(image, mode="RGBA").save(raw / f"2024_01_01_00_{minute:02d}.png")
            output = root / "out"
            process_year(2024, root / "raw", output, 15, 2, 2, 6, config, "digest", "rgb-max", False)
            year = output / "year=2024"
            timestamps = np.load(year / "radar_timestamps.npy", allow_pickle=False)
            self.assertEqual(timestamps.tolist(), ["2024-01-01T00:00:00", "2024-01-01T00:30:00"])
            metadata = json.loads((year / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["enforce_timestamp_continuity"])
            frames = np.memmap(year / "radar_frames.dat", dtype=np.uint8, mode="r", shape=(2, 2, 2, 3))
            self.assertTrue(np.all(frames[0, :, :, 0] == 0))
            self.assertTrue(np.all(frames[0, :, :, 1] == 11))
            coverage = (year / "window_coverage.csv").read_text(encoding="utf-8")
            self.assertIn("2024-01-01T00:15:00,0,0,0", coverage)

    def test_dataset_rejects_samples_crossing_timestamp_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset" / "year=2024"
            root.mkdir(parents=True)
            shape = (10, 2, 2, 3)
            frames = np.memmap(root / "radar_frames.dat", dtype=np.uint8, mode="w+", shape=shape)
            frames[:] = 0
            frames.flush()
            del frames
            timestamps = [datetime(2024, 1, 1) + timedelta(minutes=15 * index) for index in range(10)]
            timestamps[4:] = [value + timedelta(minutes=15) for value in timestamps[4:]]
            np.save(root / "radar_timestamps.npy", np.asarray([value.isoformat() for value in timestamps], dtype="<U19"))
            (root / "metadata.json").write_text(json.dumps({"shape": list(shape), "dtype": "uint8", "frames_file": "radar_frames.dat", "timestamps_file": "radar_timestamps.npy", "aggregate_minutes": 15, "enforce_timestamp_continuity": True}), encoding="utf-8")
            target_shape = (10, 2, 2, 1)
            for name, dtype in (("Y_alertario.dat", np.float32), ("M_alertario.dat", np.uint8)):
                array = np.memmap(root / name, dtype=dtype, mode="w+", shape=target_shape)
                array[:] = 0
                array.flush()
                del array
            (root / "targets_alertario_metadata.json").write_text(json.dumps({"shape": list(target_shape), "Y_dtype": "float32", "M_dtype": "uint8", "Y_file": "Y_alertario.dat", "M_file": "M_alertario.dat"}), encoding="utf-8")
            dataset = RadarStationMemmapDataset(root.parent, [2024], t_in=2, t_out=2, stride=1)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.get_sample_classes().tolist(), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
