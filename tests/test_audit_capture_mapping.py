from __future__ import annotations

import unittest

import pandas as pd

from nowcasting.cli.audit_capture_mapping import station_pixels


class AuditCaptureMappingTests(unittest.TestCase):
    def test_transformed_mapping_resamples_from_canonical_704_grid(self) -> None:
        mapping = pd.DataFrame(
            {"station_id": [1, 2], "pixel_i": [0, 655], "pixel_j": [0, 653]}
        )

        rows, columns = station_pixels(
            mapping, 128, 128, source_height=656, source_width=654
        )

        self.assertEqual(rows.tolist(), [0, 127])
        self.assertEqual(columns.tolist(), [0, 127])


if __name__ == "__main__":
    unittest.main()
