from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.cli.build_websirene_targets import load_mapping, process_year


class WebSireneSparseTargetTests(unittest.TestCase):
    def test_builder_aggregates_collocated_stations_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qc_root = root / "qc"
            radar_root = root / "radar"
            (radar_root / "year=2024").mkdir(parents=True)
            mapping_path = root / "mapping.csv"
            mapping_path.write_text("station_id,pixel_i,pixel_j\n1,328,327\n2,328,327\n", encoding="utf-8")
            for station_id, value in ((1, 2.0), (2, 6.0)):
                path = qc_root / f"station_id={station_id}" / "year=2024"
                path.mkdir(parents=True)
                pd.DataFrame({
                    "station_id": [station_id], "timestamp_15min": ["2024-01-01T00:00:00Z"],
                    "m15": [value], "qc_status": ["accepted"],
                }).to_parquet(path / "data.parquet", index=False)
            np.save(radar_root / "year=2024" / "radar_timestamps.npy", np.array(["2024-01-01T00:00:00+00:00"]))
            whitelist = root / "whitelist.csv"
            whitelist.write_text("station_id\n1\n2\n", encoding="utf-8")
            args = Namespace(qc_root=qc_root, mapping=mapping_path, radar_root=radar_root,
                             station_whitelist=whitelist, height=128, width=128,
                             height_orig=656, width_orig=654, observation_to_radar_offset_minutes=0,
                             cell_aggregation="median", overwrite=False)
            process_year(args, 2024, load_mapping(args), {1, 2})
            with np.load(radar_root / "year=2024" / "targets_websirene_sparse.npz") as sparse:
                self.assertEqual(sparse["frame"].tolist(), [0])
                self.assertEqual(sparse["station_id"].tolist(), [-1])
                self.assertEqual(sparse["station_count"].tolist(), [2])
                self.assertAlmostEqual(float(sparse["value"][0]), np.log1p(4.0))
            metadata = json.loads((radar_root / "year=2024" / "targets_websirene_metadata.json").read_text())
            self.assertEqual(metadata["format"], "sparse")
            self.assertEqual(metadata["cell_aggregation"], "median")


if __name__ == "__main__":
    unittest.main()
