from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from nowcasting.cli.inspect_radar_capture import inspect_capture


class InspectRadarCaptureTests(unittest.TestCase):
    def test_inspect_capture_writes_reproducible_audit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "captures"
            day = root / "2024" / "01" / "01"
            day.mkdir(parents=True)

            for minute, value in ((0, 20), (5, 80), (10, 160)):
                image = np.zeros((6, 8, 4), dtype=np.uint8)
                image[:, :2, :3] = 255  # Simulates a fixed left-side overlay.
                image[3:, 4:, :3] = value
                image[:, :, 3] = 255
                Image.fromarray(image, mode="RGBA").save(
                    day / f"2024_01_01_00_{minute:02d}.png"
                )

            (root / "notes.txt").write_text("not an image", encoding="utf-8")
            output_dir = Path(temporary_directory) / "audit"
            summary = inspect_capture(root, output_dir, sample_count=3, contact_count=2)

            self.assertEqual(summary["total_files"], 4)
            self.assertEqual(summary["png_files"], 3)
            self.assertEqual(summary["valid_pngs_with_timestamp"], 3)
            self.assertEqual(summary["image_sizes"], {"8x6": 3})
            self.assertEqual(summary["timestamp_start"], "2024-01-01T00:00:00")
            self.assertTrue((output_dir / "sample_stats.csv").is_file())
            self.assertTrue((output_dir / "contact_sheet.png").is_file())
            self.assertTrue((output_dir / "layout_persistence.png").is_file())

            saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["persistence"]["sampled_images_same_size"], 3)


if __name__ == "__main__":
    unittest.main()
