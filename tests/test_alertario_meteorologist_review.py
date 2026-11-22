from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from nowcasting.cli.export_alertario_meteorologist_review import export_workbook


class AlertarioMeteorologistReviewTests(unittest.TestCase):
    def test_exports_readable_workbook_and_questions(self) -> None:
        episodes = pd.DataFrame({
            "station_id": [17],
            "station_name": ["Barra/Barrinha"],
            "episode_number": [1],
            "start_timestamp": ["2026-05-07 01:55:00+00:00"],
            "end_timestamp": ["2026-05-07 02:55:00+00:00"],
            "candidate_timestamps": [2],
            "max_m05": [220.0],
            "max_m15": [417.2],
            "max_consistency_abs_error_mm": [417.2],
            "spatially_screened_count": [1],
            "temporal_inconsistent_count": [1],
            "multi_source_unsupported_count": [1],
            "radar_supported_count": [0],
            "alertario_supported_count": [0],
            "websirene_consensus_count": [0],
            "high_priority_episode": [True],
            "highest_review_priority": ["high"],
            "spatial_statuses": ["multi_source_unsupported"],
        })
        observations = pd.DataFrame({
            "station_id": [17],
            "station_name": ["Barra/Barrinha"],
            "timestamp": pd.to_datetime(["2026-05-07 02:00:00Z"]),
            "m05": [220.0],
            "m10": [417.2],
            "m15": [417.2],
            "h01": [417.2],
            "previous_m05": [0.0],
            "next_m05": [0.0],
            "previous_m15": [0.0],
            "next_m15": [0.0],
            "expected_m10_from_m05": [220.0],
            "expected_m15_from_m05": [220.0],
            "expected_h01_from_m05": [220.0],
            "m10_consistency_error_mm": [197.2],
            "m15_consistency_error_mm": [197.2],
            "h01_consistency_error_mm": [197.2],
            "temporal_inconsistent": [True],
            "spatially_screened": [True],
            "alertario_extended_window_wet_count": [0],
            "alertario_extended_window_max_m15": [0.0],
            "alertario_extended_window_nearest_wet_distance_km": [None],
            "radar_context_available": [True],
            "radar_echo_present": [False],
            "websirene_context_available": [False],
            "websirene_consensus_support": [False],
            "spatial_status": ["multi_source_unsupported"],
        })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_path = root / "episodes.csv"
            observations_path = root / "observations.parquet"
            workbook_path = root / "review.xlsx"
            questions_path = root / "questions.md"
            episodes.to_csv(episodes_path, index=False)
            observations.to_parquet(observations_path, index=False)

            export_workbook(
                episodes_path,
                observations_path,
                workbook_path,
                questions_path,
            )

            workbook = load_workbook(workbook_path, data_only=True)
            self.assertEqual(workbook.sheetnames, [
                "LEIA-ME",
                "PRIORITÁRIOS",
                "TODOS OS EPISÓDIOS",
                "OBSERVAÇÕES",
                "LEGENDA",
            ])
            self.assertEqual(workbook["PRIORITÁRIOS"].max_row, 2)
            self.assertEqual(workbook["OBSERVAÇÕES"]["A2"].value, "Alta")
            self.assertIn("06/05/2026 22:55", workbook["PRIORITÁRIOS"]["E2"].value)
            self.assertEqual(workbook["LEGENDA"]["A1"].value, "Termo")
            self.assertIn("Os campos m05", questions_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
