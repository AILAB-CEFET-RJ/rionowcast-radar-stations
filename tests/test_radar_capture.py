from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from nowcasting.radar_capture import load_capture_config, load_reflectivity_rgb


class RadarCaptureTests(unittest.TestCase):
    def test_crop_removes_left_overlay_and_preserves_reflectivity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "capture.json"
            config_path.write_text(
                json.dumps(
                    {
                        "source_image": {"width": 8, "height": 6},
                        "reflectivity_crop": {
                            "left": 2,
                            "top": 1,
                            "right_exclusive": 6,
                            "bottom_exclusive": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            rgba = np.zeros((6, 8, 4), dtype=np.uint8)
            rgba[:, :2, :3] = 255
            rgba[1:5, 2:6, 1] = 150
            rgba[:, :, 3] = 255
            image_path = root / "capture.png"
            Image.fromarray(rgba, mode="RGBA").save(image_path)

            result = load_reflectivity_rgb(image_path, load_capture_config(config_path))

            self.assertEqual(result.shape, (4, 4, 3))
            self.assertTrue(np.all(result[:, :, 0] == 0))
            self.assertTrue(np.all(result[:, :, 1] == 150))

    def test_config_rejects_crop_outside_source_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "invalid.json"
            config_path.write_text(
                json.dumps(
                    {
                        "source_image": {"width": 8, "height": 6},
                        "reflectivity_crop": {
                            "left": 2,
                            "top": 1,
                            "right_exclusive": 9,
                            "bottom_exclusive": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_capture_config(config_path)


if __name__ == "__main__":
    unittest.main()
