from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.cli.audit_historical_station_mapping import all_candidate_pixels, candidate_pixels, local_signal
from nowcasting.radar_georeferencing import geographic_station_pixels, orientation_candidates
from nowcasting.cli.audit_historical_radar_events import frame_statistics


class HistoricalStationMappingTests(unittest.TestCase):
    def test_candidate_transpose_matches_swapped_source_geometry(self) -> None:
        mapping = pd.DataFrame({"station_id": [1], "pixel_i": [100], "pixel_j": [200]})
        candidates = candidate_pixels(mapping, height=654, width=656)
        direct_row, direct_column = candidates["direct"]
        transposed_row, transposed_column = candidates["transpose"]
        self.assertEqual((int(direct_row[0]), int(direct_column[0])), (100, 201))
        self.assertEqual((int(transposed_row[0]), int(transposed_column[0])), (200, 100))

    def test_local_signal_uses_neighborhood_maximum(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[3, 3, 1] = 200
        signals = local_signal(frame, np.array([1, 6]), np.array([1, 6]), radius=2)
        self.assertAlmostEqual(float(signals[0]), 200 / 255)
        self.assertEqual(float(signals[1]), 0.0)

    def test_geographic_mapping_places_radar_center_at_bitmap_center(self) -> None:
        mapping = pd.DataFrame({"latitude": [-22.955139], "longitude": [-43.248278]})
        config = {"radar_center_wgs84": {"latitude": -22.955139, "longitude": -43.248278}, "display_range_km": 138.9}
        rows, columns = geographic_station_pixels(mapping, source_height=654, source_width=656, georeferencing=config)
        self.assertEqual((int(rows[0]), int(columns[0])), (326, 328))

    def test_audit_adds_geographic_candidate_when_configured(self) -> None:
        mapping = pd.DataFrame({"station_id": [1], "pixel_i": [100], "pixel_j": [200], "latitude": [-22.955139], "longitude": [-43.248278]})
        config = {"historical_georeferencing": {"radar_center_wgs84": {"latitude": -22.955139, "longitude": -43.248278}, "display_range_km": 138.9}}
        candidates = all_candidate_pixels(mapping, height=654, width=656, config=config)
        self.assertIn("geographic_aeqd_north_up", candidates)

    def test_geographic_orientation_candidates_cover_all_axis_conventions(self) -> None:
        candidates = orientation_candidates(np.array([100]), np.array([200]), height=654, width=656)
        self.assertEqual(len(candidates), 8)
        flip_vh = candidates["geographic_aeqd_flip_vh"]
        transposed = candidates["geographic_aeqd_transpose_north_up"]
        self.assertEqual((int(flip_vh[0][0]), int(flip_vh[1][0])), (553, 455))
        self.assertEqual((int(transposed[0][0]), int(transposed[1][0])), (199, 100))

    def test_frame_statistics_reports_visible_echo_extent(self) -> None:
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        frame[1, 2] = [0, 100, 0]
        frame[3, 4] = [20, 0, 0]
        stats = frame_statistics(frame)
        self.assertEqual(stats["active_pixels"], 2)
        self.assertAlmostEqual(float(stats["active_fraction"]), 0.1)
        self.assertEqual((stats["echo_row_min"], stats["echo_row_max"]), (1, 3))
        self.assertEqual((stats["echo_column_min"], stats["echo_column_max"]), (2, 4))


if __name__ == "__main__":
    unittest.main()
