from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nowcasting.cli.audit_alertario_temporal_consistency import (
    audit_temporal_consistency,
    summarize_flagged_episodes,
    summarize_station_year,
)


def rolling(values: list[float], window: int) -> list[float]:
    return pd.Series(values).rolling(window, min_periods=1).sum().tolist()


def make_frame(m05: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "estacao_id": [1] * len(m05),
        "estacao": ["Teste"] * len(m05),
        "dia_utc": pd.date_range("2026-01-01", periods=len(m05), freq="5min", tz="UTC"),
        "m05": m05,
        "m10": rolling(m05, 2),
        "m15": rolling(m05, 3),
        "h01": rolling(m05, 12),
    })


class AlertarioTemporalConsistencyTests(unittest.TestCase):
    def test_consistent_isolated_increment_is_flagged_as_high_but_not_inconsistent(self) -> None:
        m05 = [0.0] * 15
        m05[5] = 60.0
        audited = audit_temporal_consistency(make_frame(m05))

        spike = audited.iloc[5]
        self.assertTrue(spike["high_m05"])
        self.assertTrue(spike["isolated_m05_spike"])
        self.assertFalse(audited["m10_inconsistent"].any())
        self.assertFalse(audited["m15_inconsistent"].any())
        self.assertFalse(audited["h01_inconsistent"].any())

    def test_reset_after_spike_breaks_rolling_accumulations(self) -> None:
        frame = make_frame([0.0] * 15)
        frame.loc[5, ["m05", "m10", "m15", "h01"]] = 60.0
        audited = audit_temporal_consistency(frame)

        self.assertTrue(audited.loc[5, "isolated_m05_spike"])
        self.assertTrue(audited.loc[5, "isolated_m15_spike"])
        self.assertTrue(audited.loc[6, "m10_inconsistent"])
        self.assertTrue(audited.loc[6, "m15_inconsistent"])
        self.assertTrue(audited.loc[11, "h01_inconsistent"])
        summary = summarize_station_year(audited).iloc[0]
        self.assertGreater(summary["any_temporal_flag"], 1)

    def test_gap_is_not_classified_as_accumulation_inconsistency(self) -> None:
        frame = make_frame([0.0, 0.0, 10.0, 0.0])
        frame.loc[2:, "dia_utc"] += pd.Timedelta(minutes=5)
        audited = audit_temporal_consistency(frame)

        self.assertFalse(audited.loc[2, "m10_comparable"])
        self.assertFalse(audited.loc[2, "m10_inconsistent"])

    def test_exact_duplicate_is_collapsed_and_reported(self) -> None:
        frame = make_frame([0.0, 0.0, 0.0])
        frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
        audited = audit_temporal_consistency(frame)

        self.assertEqual(len(audited), 3)
        self.assertTrue(audited.loc[1, "duplicate_exact_timestamp"])
        self.assertFalse(audited.loc[1, "conflicting_duplicate_timestamp"])

    def test_flagged_timestamps_are_grouped_into_contiguous_episodes(self) -> None:
        audited = audit_temporal_consistency(make_frame([0.0] * 6))
        audited.loc[[1, 2, 4], "high_m05"] = True
        audited.loc[[1, 2, 4], "any_temporal_flag"] = True

        episodes = summarize_flagged_episodes(audited)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes["flagged_timestamps"].tolist(), [2, 1])
        self.assertEqual(episodes["span_minutes"].tolist(), [5, 0])
        self.assertEqual(episodes["high_m05_count"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
