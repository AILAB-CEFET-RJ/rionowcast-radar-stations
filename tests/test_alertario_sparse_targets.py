from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.cli.build_alertario_sparse_targets import load_mapping, process_year


class AlertarioSparseTargetTests(unittest.TestCase):
    def test_builder_preserves_station_id_and_excludes_sentinel_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            radar_root = root / "radar"
            output_root = root / "output"
            raw_root.mkdir()
            (radar_root / "year=2024").mkdir(parents=True)
            (output_root / "year=2024").mkdir(parents=True)
            mapping_path = root / "mapping.csv"
            mapping_path.write_text(
                "station_id,pixel_i,pixel_j\n1,328,327\n2,100,100\n",
                encoding="utf-8",
            )
            raw_path = raw_root / "source.parquet"
            pd.DataFrame(
                {
                    "estacao_id": [1, 1, 1, 2],
                    "dia_utc": [
                        "2024-01-01T00:00:00Z",
                        "2024-01-01T00:15:00Z",
                        "2024-01-01T00:30:00Z",
                        "2024-01-01T00:00:00Z",
                    ],
                    "m15": [2.0, -99.99, -1.0, 3.0],
                }
            ).to_parquet(raw_path, index=False)
            np.save(
                radar_root / "year=2024" / "radar_timestamps.npy",
                np.array([
                    "2024-01-01T00:00:00+00:00",
                    "2024-01-01T00:15:00+00:00",
                    "2024-01-01T00:30:00+00:00",
                ]),
            )
            args = Namespace(
                mapping=mapping_path, height=128, width=128, height_orig=656,
                width_orig=654, radar_root=radar_root, output_root=output_root,
                overwrite=False,
            )
            process_year(args, 2024, load_mapping(args), [raw_path])

            with np.load(output_root / "year=2024" / "targets_alertario_sparse.npz") as sparse:
                self.assertEqual(sparse["station_id"].tolist(), [1, 2])
                self.assertEqual(sparse["frame"].tolist(), [0, 0])
                self.assertAlmostEqual(float(sparse["value"][0]), np.log1p(2.0))
            metadata = json.loads(
                (output_root / "year=2024" / "targets_alertario_metadata.json").read_text()
            )
            self.assertTrue(metadata["station_provenance"])
            self.assertEqual(metadata["quality_control_counts"]["sentinel_m15"], 1)
            self.assertEqual(metadata["quality_control_counts"]["negative_m15"], 2)


if __name__ == "__main__":
    unittest.main()
