from __future__ import annotations

import unittest

import pandas as pd

from nowcasting.cli.transform_station_mapping import transform_mapping


class TransformStationMappingTests(unittest.TestCase):
    def test_preserves_legacy_pixels_when_capture_crop_matches_legacy_grid(self) -> None:
        mapping = pd.DataFrame({"station_id": [1], "pixel_i": [655], "pixel_j": [653]})
        config = {
            "source_image": {"width": 1024, "height": 768},
            "reflectivity_crop": {"left": 185, "top": 56, "right_exclusive": 839, "bottom_exclusive": 712},
            "station_mapping_transform": {
                "legacy_grid": {"height": 656, "width": 654},
                "row_offset": 0,
                "column_offset": 0,
            },
        }

        transformed = transform_mapping(mapping, config)

        self.assertEqual(transformed.loc[0, "legacy_pixel_i"], 655)
        self.assertEqual(transformed.loc[0, "legacy_pixel_j"], 653)
        self.assertEqual(transformed.loc[0, "pixel_i"], 655)
        self.assertEqual(transformed.loc[0, "pixel_j"], 653)


if __name__ == "__main__":
    unittest.main()
