from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from nowcasting.operational_inference import prepare_radar_input, validate_checkpoint_configuration


class OperationalInferenceTests(unittest.TestCase):
    def test_prepares_channel_first_input_and_applies_saved_spatial_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for index in range(5):
                rgba = np.zeros((6, 8, 4), dtype=np.uint8)
                rgba[1:5, 2:6, 0] = 20 + index
                rgba[:, :, 3] = 255
                path = root / f"input_{index}.png"
                Image.fromarray(rgba, mode="RGBA").save(path)
                paths.append(path.name)
            event = {"input": [{"png_files": [path]} for path in paths]}
            capture_config = {
                "source_image": {"width": 8, "height": 6},
                "reflectivity_crop": {"left": 2, "top": 1, "right_exclusive": 6, "bottom_exclusive": 5},
            }
            configuration = {
                "step": 5,
                "crop": {"enabled": True, "bounds": {"row_start": 1, "row_stop": 3, "column_start": 0, "column_stop": 2}},
            }

            prepared = prepare_radar_input(event, root, capture_config, configuration, 4, 4)

            self.assertEqual(prepared.shape, (1, 3, 5, 2, 2))
            self.assertAlmostEqual(float(prepared[0, 0, 0, 0, 0]), 20 / 255.0)
            self.assertAlmostEqual(float(prepared[0, 0, 4, 0, 0]), 24 / 255.0)

    def test_rejects_legacy_or_incompatible_checkpoint_geometry(self) -> None:
        config = {"name": "capture", "version": 1}
        with self.assertRaisesRegex(ValueError, "legado"):
            validate_checkpoint_configuration({}, config)
        with self.assertRaisesRegex(ValueError, "difere"):
            validate_checkpoint_configuration({"radar_capture_preprocessing": {"version": 2}}, config)
        validate_checkpoint_configuration({"radar_capture_preprocessing": config}, config)


if __name__ == "__main__":
    unittest.main()
