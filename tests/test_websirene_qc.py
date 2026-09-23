from __future__ import annotations

import unittest

import pandas as pd

from nowcasting.websirene_qc import (
    DEFAULT_CONFIG,
    audit_observations,
    haversine_km,
    merge_config,
    nearest_reference_stations,
    paired_metrics,
    station_summary,
)
from nowcasting.station_geometry import direct_resampled_pixels, sparse_target_pixels


class WebSireneQualityControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = merge_config(
            DEFAULT_CONFIG,
            {
                "thresholds": {"constant_run_length": 3},
                "station_acceptance": {
                    "min_accepted_observations": 1,
                    "min_years_with_accepted_data": 1,
                    "max_rejected_fraction": 1.0,
                    "max_suspect_fraction": 1.0,
                },
            },
        )

    def test_audit_assigns_rejected_suspect_and_informational_flags(self) -> None:
        frame = pd.DataFrame(
            {
                "nome": ["A"] * 7,
                "observation_datetime": [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:05:00Z",
                    "2024-01-01T00:10:00Z",
                    "2024-01-01T01:00:00Z",
                    "bad-date",
                    "2024-01-01T01:05:00Z",
                ],
                "m15": [1.0, 2.0, 60.0, -1.0, 2.0, 1.0, 180.0],
            }
        )
        audited = audit_observations(frame, station_id=1, year=2024, config=self.config)

        self.assertEqual(audited.loc[0, "qc_status"], "suspect")
        self.assertIn("duplicate_timestamp_conflict", audited.loc[0, "qc_flags"])
        self.assertEqual(audited.loc[2, "qc_status"], "suspect")
        self.assertIn("m15_above_suspect_threshold", audited.loc[2, "qc_flags"])
        self.assertEqual(audited.loc[3, "qc_status"], "rejected")
        self.assertEqual(audited.loc[5, "qc_status"], "rejected")
        self.assertEqual(audited.loc[6, "qc_status"], "rejected")
        self.assertIn("gap_before", audited.loc[4, "qc_flags"])

        summary = station_summary(audited, self.config)
        self.assertEqual(summary.loc[0, "station_status"], "approved")

    def test_nearest_reference_pair_and_metrics(self) -> None:
        web = pd.DataFrame(
            {"station_id": [1, 2], "latitude": [-22.90, -23.50], "longitude": [-43.20, -43.20]}
        )
        reference = pd.DataFrame(
            {"station_id": [101], "latitude": [-22.91], "longitude": [-43.20]}
        )
        pairs = nearest_reference_stations(web, reference, max_distance_km=5.0)
        self.assertEqual(pairs["websirene_station_id"].tolist(), [1])
        self.assertLess(pairs.loc[0, "distance_km"], 2.0)
        self.assertAlmostEqual(haversine_km(-22.9, -43.2, -22.9, -43.2), 0.0)

        metrics = paired_metrics(
            pd.DataFrame({"websirene_m15": [1.0, 3.0], "reference_m15": [2.0, 2.0]})
        )
        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["bias"], 0.0)

    def test_sparse_target_transform_can_differ_from_direct_transform(self) -> None:
        rows = pd.Series([303.0, 342.0]).to_numpy()
        columns = pd.Series([324.0, 335.0]).to_numpy()
        direct = direct_resampled_pixels(
            rows, columns, height_orig=656, width_orig=654, height=128, width=128
        )
        pipeline = sparse_target_pixels(
            rows, columns, height_orig=656, width_orig=654,
            source_height=256, source_width=256, height=128, width=128,
        )
        self.assertEqual(direct[0].shape, (2,))
        self.assertTrue(((direct[0] != pipeline[0]) | (direct[1] != pipeline[1])).any())


if __name__ == "__main__":
    unittest.main()
