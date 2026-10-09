from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.forecast_evaluation import (
    align_records, event_metrics, evaluate_records, paired_daily_bootstrap, select_decision_threshold, skill_scores,
)
from nowcasting.cli.export_radar_checkpoint_records import export_years, normalize_configuration, stconvs2s_root


def records(prediction=(0.0, 2.0, 2.0, 0.0)) -> pd.DataFrame:
    return pd.DataFrame({
        "year": [2023] * 4,
        "target_timestamp": ["2023-01-01T00:00:00Z", "2023-01-01T00:15:00Z",
                             "2023-01-02T00:00:00Z", "2023-01-02T00:15:00Z"],
        "horizon": [1, 2, 1, 2], "station_id": [1] * 4,
        "predicted_mm_15min": prediction, "observed_mm_15min": [0.0, 2.0, 0.0, 2.0],
        "is_observed": [True] * 4,
    })


class ForecastEvaluationTests(unittest.TestCase):
    def test_checkpoint_export_defaults_absent_legacy_goes_flag_to_false(self):
        configuration = {
            "dataset_root": "dataset", "test_years": [2024], "step": 5, "stride": 5,
            "model": "stconvs2s-c", "num_layers": 1, "hidden_dim": 8, "kernel_size": 5,
            "stconvs2s_root": "stconvs2s", "target_source": "alertario", "station_mapping": "mapping.csv",
            "mapping_height_orig": 654, "mapping_width_orig": 656, "crop_stations": True,
            "crop_margin_pixels": 20, "input_stations": False,
        }
        self.assertFalse(normalize_configuration(configuration)["input_goes"])

    def test_checkpoint_export_can_override_test_years_for_validation_records(self):
        configuration = {"test_years": [2023, 2024]}
        self.assertEqual(export_years(configuration, None), [2023, 2024])
        self.assertEqual(export_years(configuration, "2022"), [2022])

    def test_checkpoint_export_can_override_legacy_core_checkout_path(self):
        configuration = {"stconvs2s_root": "/lovelace/external/stconvs2s"}
        self.assertEqual(stconvs2s_root(configuration, None), Path("/lovelace/external/stconvs2s"))
        self.assertEqual(stconvs2s_root(configuration, Path("/workstation/stconvs2s")), Path("/workstation/stconvs2s"))

    def test_categorical_metrics_match_known_contingency_table(self):
        metrics = evaluate_records(records(), [1.25])
        categorical = metrics["thresholds"]["1.25"]["global"]
        self.assertEqual((categorical["hits"], categorical["misses"], categorical["false_alarms"], categorical["correct_negatives"]), (1, 1, 1, 1))
        self.assertAlmostEqual(categorical["pod"], 0.5)
        self.assertAlmostEqual(categorical["far"], 0.5)
        self.assertAlmostEqual(categorical["csi"], 1 / 3)
        self.assertAlmostEqual(categorical["frequency_bias"], 1.0)

    def test_alignment_rejects_different_masks(self):
        changed = records()
        changed.loc[0, "is_observed"] = False
        with self.assertRaisesRegex(ValueError, "máscara"):
            align_records({"B1": records(), "C1": changed})

    def test_daily_bootstrap_is_deterministic_and_reports_skill(self):
        baseline = records((0.0, 0.0, 0.0, 0.0))
        candidate = records((0.0, 2.0, 0.0, 2.0))
        first = paired_daily_bootstrap(baseline, candidate, [1.25], replicates=50, seed=7)
        second = paired_daily_bootstrap(baseline, candidate, [1.25], replicates=50, seed=7)
        self.assertEqual(first, second)
        self.assertGreater(first["skill_mae"]["estimate"], 0.0)

    def test_event_metrics_groups_contiguous_exceedances_by_station(self):
        report = event_metrics(records(), 1.25)
        self.assertEqual(report["events"], 2)
        self.assertEqual(report["detected_events"], 1)
        self.assertEqual(report["false_alert_records"], 1)
        self.assertEqual(report["false_alert_events"], 1)
        self.assertEqual(report["false_alert_days"], 1)

    def test_municipal_event_metrics_aggregate_station_maximum(self):
        second_station = records((0.0, 0.0, 0.0, 0.0))
        second_station["station_id"] = 2
        report = event_metrics(pd.concat([records(), second_station], ignore_index=True), 1.25, scope="municipal")
        self.assertEqual(report["scope"], "municipal")
        self.assertEqual(report["events"], 2)
        self.assertEqual(report["detected_events"], 1)

    def test_decision_threshold_is_selected_only_from_candidate_grid(self):
        selected, report = select_decision_threshold(records(), 1.25, [0.5, 1.25, 2.5])
        self.assertEqual(selected, 1.25)
        self.assertAlmostEqual(report["csi"], 1 / 3)

    def test_skill_scores_are_reported_for_global_and_horizons(self):
        baseline = evaluate_records(records((0.0, 0.0, 0.0, 0.0)), [1.25])
        candidate = evaluate_records(records((0.0, 2.0, 0.0, 2.0)), [1.25])
        skill = skill_scores(baseline, candidate)
        self.assertGreater(skill["global"]["mae_skill"], 0.0)
        self.assertEqual(set(skill["horizons"]), {"1", "2"})

    def test_station_metrics_and_intensity_bootstrap_are_available(self):
        baseline = records((0.0, 0.0, 0.0, 0.0))
        candidate = records((0.0, 2.0, 0.0, 2.0))
        report = evaluate_records(candidate, [1.25])
        self.assertEqual(set(report["stations"]), {"1"})
        self.assertEqual(report["stations"]["1"]["global"]["n"], 4)
        bootstrap = paired_daily_bootstrap(
            baseline, candidate, [1.25], replicates=20, seed=7, intensity=(1.25, 6.25),
        )
        self.assertGreater(bootstrap["skill_mae"]["estimate"], 0.0)


if __name__ == "__main__":
    unittest.main()
