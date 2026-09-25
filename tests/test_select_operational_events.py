from __future__ import annotations

import unittest

import pandas as pd

from nowcasting.cli.select_operational_events import select_events


class SelectOperationalEventsTests(unittest.TestCase):
    def test_selects_separated_events_with_complete_input_and_target_sequences(self) -> None:
        first = pd.Timestamp("2024-01-01T00:00:00Z")
        buckets = {
            first + pd.Timedelta(minutes=15 * index): []
            for index in range(40)
        }
        events = pd.DataFrame(
            {
                "timestamp": [
                    first + pd.Timedelta(minutes=75),
                    first + pd.Timedelta(minutes=90),
                    first + pd.Timedelta(minutes=300),
                ],
                "station_id": [1, 2, 3],
                "station_name": ["A", "B", "C"],
                "m15": [30.0, 20.0, 15.0],
            }
        )

        selected = select_events(
            events,
            buckets,
            aggregate_minutes=15,
            t_in=5,
            t_out=5,
            min_separation_minutes=180,
            limit=12,
        )

        self.assertEqual([event["station_id"] for event in selected], [1, 3])
        self.assertEqual(len(selected[0]["input_timestamps"]), 5)
        self.assertEqual(len(selected[0]["target_timestamps"]), 5)
        self.assertEqual(selected[0]["target_timestamps"][0], events.iloc[0]["timestamp"])

    def test_excludes_event_when_a_required_capture_bucket_is_absent(self) -> None:
        first = pd.Timestamp("2024-01-01T00:00:00Z")
        buckets = {
            first + pd.Timedelta(minutes=15 * index): []
            for index in range(10)
            if index != 5
        }
        events = pd.DataFrame(
            {
                "timestamp": [first + pd.Timedelta(minutes=75)],
                "station_id": [1],
                "station_name": ["A"],
                "m15": [30.0],
            }
        )

        selected = select_events(
            events,
            buckets,
            aggregate_minutes=15,
            t_in=5,
            t_out=5,
            min_separation_minutes=180,
            limit=1,
        )

        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
