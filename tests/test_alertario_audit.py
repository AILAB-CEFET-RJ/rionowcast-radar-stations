from __future__ import annotations

import unittest

import pandas as pd

from nowcasting.cli.audit_alertario import summarize_file


class AlertarioAuditTests(unittest.TestCase):
    def test_summary_separates_sentinel_missing_and_negative_values(self) -> None:
        frame = pd.DataFrame(
            {
                "estacao_id": [1, 1, 1, 1, 1],
                "dia_utc": [
                    "2024-01-01T00:00:00Z", "2024-01-01T00:15:00Z",
                    "2024-01-01T00:30:00Z", "2024-01-01T00:45:00Z", "bad-date",
                ],
                "m15": [1.0, -99.99, -1.0, None, 2.0],
            }
        )
        summary = summarize_file(frame, sentinel=-99.99, suspect=50.0, reject=175.0)
        year_2024 = summary.loc[summary["year"] == 2024].iloc[0]
        unknown_year = summary.loc[summary["year"] == -1].iloc[0]
        self.assertEqual(year_2024["valid_m15"], 1)
        self.assertEqual(year_2024["sentinel_m15"], 1)
        self.assertEqual(year_2024["negative_m15"], 1)
        self.assertEqual(year_2024["missing_m15"], 1)
        self.assertEqual(unknown_year["invalid_timestamp"], 1)

    def test_summary_distinguishes_exact_duplicates_from_15_minute_aggregation(self) -> None:
        frame = pd.DataFrame(
            {
                "estacao_id": [1, 1, 1],
                "dia_utc": ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z", "2024-01-01T00:05:00Z"],
                "m15": [1.0, 2.0, 2.0],
            }
        )
        summary = summarize_file(frame, sentinel=-99.99, suspect=50.0, reject=175.0).iloc[0]
        self.assertEqual(summary["multiple_rows_15min"], 3)
        self.assertEqual(summary["duplicate_exact_timestamp"], 2)


if __name__ == "__main__":
    unittest.main()
