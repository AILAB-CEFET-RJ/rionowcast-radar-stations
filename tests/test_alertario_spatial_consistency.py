from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nowcasting.spatial_qc import (
    build_neighbor_graph,
    classify_spatial_context,
    compute_neighbor_context,
    radar_patch_metrics,
    summarize_spatial_episodes,
)


class AlertarioSpatialConsistencyTests(unittest.TestCase):
    def test_neighbor_graph_has_local_and_extended_bands(self) -> None:
        mapping = pd.DataFrame({
            "station_id": [1, 2, 3],
            "latitude": [0.0, 0.0, 0.0],
            "longitude": [0.0, 0.03, 0.08],
        })

        graph = build_neighbor_graph(mapping, local_radius_km=5, extended_radius_km=10)
        station_one = graph.loc[graph["source_station_id"] == 1]

        self.assertEqual(station_one["neighbor_station_id"].tolist(), [2, 3])
        self.assertEqual(station_one["distance_band"].tolist(), ["local", "extended"])
        self.assertFalse((graph["source_station_id"] == graph["neighbor_station_id"]).any())

    def test_neighbor_context_distinguishes_exact_and_tolerant_support(self) -> None:
        timestamp = pd.Timestamp("2026-01-01 00:15", tz="UTC")
        candidates = pd.DataFrame({
            "candidate_id": [0], "station_id": [1], "timestamp": [timestamp],
        })
        observations = pd.DataFrame({
            "station_id": [2, 3],
            "timestamp": [timestamp + pd.Timedelta(minutes=5), timestamp],
            "m15": [2.0, 0.0],
        })
        graph = pd.DataFrame({
            "source_station_id": [1, 1],
            "neighbor_station_id": [2, 3],
            "distance_km": [2.0, 8.0],
            "distance_band": ["local", "extended"],
        })

        context = compute_neighbor_context(
            candidates, observations, graph, prefix="alertario",
            wet_threshold_mm15=1.25, temporal_tolerance_minutes=15,
        ).iloc[0]

        self.assertEqual(context["alertario_exact_available_count"], 1)
        self.assertEqual(context["alertario_exact_wet_count"], 0)
        self.assertEqual(context["alertario_local_window_wet_count"], 1)
        self.assertEqual(context["alertario_extended_window_available_count"], 2)
        self.assertEqual(context["alertario_extended_window_nearest_wet_distance_km"], 2.0)

    def test_radar_patch_metrics_handle_edges(self) -> None:
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, 1] = [10, 20, 30]

        metrics = radar_patch_metrics(frame, 0, 0, patch_radii=(1, 2))

        self.assertEqual(metrics["radar_patch_3x3_pixels"], 4)
        self.assertEqual(metrics["radar_patch_5x5_pixels"], 9)
        self.assertEqual(metrics["radar_patch_5x5_active_pixels"], 1)
        self.assertTrue(metrics["radar_echo_present"])

    def test_classification_keeps_websirene_as_asymmetric_evidence(self) -> None:
        base = {
            "m15": 10.0,
            "temporal_inconsistent": False,
            "alertario_local_window_wet_count": 0,
            "alertario_extended_window_wet_count": 0,
            "alertario_extended_window_available_count": 2,
            "radar_context_available": True,
            "radar_echo_present": False,
            "websirene_extended_window_wet_count": 0,
        }
        rows = []
        rows.append({**base, "alertario_local_window_wet_count": 1,
                     "alertario_extended_window_wet_count": 1})
        rows.append({**base, "radar_echo_present": True})
        rows.append({**base, "websirene_extended_window_wet_count": 2})
        rows.append({**base, "temporal_inconsistent": True})
        rows.append({**base, "alertario_extended_window_available_count": 1})
        rows.append({**base, "m15": 1.0})

        classified = classify_spatial_context(pd.DataFrame(rows))

        self.assertEqual(classified["spatial_status"].tolist(), [
            "spatially_supported_local",
            "gauge_isolated_radar_supported",
            "websirene_only_support",
            "multi_source_unsupported",
            "insufficient_spatial_context",
            "below_spatial_screening_threshold",
        ])
        self.assertTrue(classified.loc[3, "high_priority_review"])
        self.assertFalse(classified.loc[2, "multi_source_unsupported"])
        self.assertEqual(classified.loc[2, "review_priority"], "medium")

    def test_episode_summary_uses_highest_review_priority(self) -> None:
        timestamps = pd.to_datetime([
            "2026-01-01 00:00Z", "2026-01-01 00:05Z", "2026-01-01 00:15Z",
        ])
        observations = pd.DataFrame({
            "station_id": [1, 1, 1],
            "station_name": ["Teste"] * 3,
            "timestamp": timestamps,
            "m05": [0.0, 60.0, 0.0],
            "m15": [0.0, 60.0, 0.0],
            "max_consistency_abs_error_mm": [0.0, 60.0, 0.0],
            "spatially_screened": [False, True, False],
            "temporal_inconsistent": [False, False, True],
            "multi_source_unsupported": [False, True, False],
            "radar_echo_present": [False, False, False],
            "alertario_extended_support": [False, False, False],
            "websirene_consensus_support": [False, False, False],
            "review_priority": ["low", "high", "medium"],
            "spatial_status": [
                "below_spatial_screening_threshold",
                "multi_source_unsupported",
                "below_spatial_screening_threshold",
            ],
        })

        episodes = summarize_spatial_episodes(observations, maximum_gap_minutes=5)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes["highest_review_priority"].tolist(), ["high", "medium"])
        self.assertEqual(episodes["candidate_timestamps"].tolist(), [2, 1])
        self.assertEqual(episodes["high_priority_episode"].tolist(), [False, False])

    def test_episode_combines_temporal_and_spatial_evidence_across_rows(self) -> None:
        observations = pd.DataFrame({
            "station_id": [1, 1],
            "station_name": ["Teste", "Teste"],
            "timestamp": pd.to_datetime(["2026-01-01 00:00Z", "2026-01-01 00:30Z"]),
            "m05": [100.0, 0.0],
            "m15": [100.0, 0.0],
            "max_consistency_abs_error_mm": [0.0, 100.0],
            "spatially_screened": [True, False],
            "temporal_inconsistent": [False, True],
            "multi_source_unsupported": [True, False],
            "radar_echo_present": [False, False],
            "alertario_extended_support": [False, False],
            "websirene_consensus_support": [False, False],
            "review_priority": ["medium", "medium"],
            "spatial_status": [
                "multi_source_unsupported",
                "below_spatial_screening_threshold",
            ],
        })

        episodes = summarize_spatial_episodes(
            observations,
            maximum_gap_minutes=60,
        )

        self.assertEqual(len(episodes), 1)
        self.assertTrue(episodes.loc[0, "high_priority_episode"])
        self.assertEqual(episodes.loc[0, "highest_review_priority"], "high")


if __name__ == "__main__":
    unittest.main()
