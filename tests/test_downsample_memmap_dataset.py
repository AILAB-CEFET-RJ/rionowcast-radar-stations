import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "downsample_memmap_dataset.py"
SPEC = importlib.util.spec_from_file_location("downsample_memmap_dataset", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownsampleMemmapDatasetTests(unittest.TestCase):
    def test_downsamples_radar_and_preserves_sparse_alertario_target(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_dir = root / "source" / "year=2024"
            output_dir = root / "output" / "year=2024"
            source_dir.mkdir(parents=True)

            radar = np.memmap(
                source_dir / "radar_frames.dat",
                dtype=np.uint8,
                mode="w+",
                shape=(2, 4, 4, 3),
            )
            radar[:] = np.arange(96, dtype=np.uint8).reshape(2, 4, 4, 3)
            radar.flush()
            del radar
            np.save(source_dir / "radar_timestamps.npy", np.array(["a", "b"]))
            with (source_dir / "metadata.json").open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "shape": [2, 4, 4, 3],
                        "dtype": "uint8",
                        "frames_file": "radar_frames.dat",
                        "timestamps_file": "radar_timestamps.npy",
                    },
                    file,
                )

            y = np.memmap(
                source_dir / "Y_alertario.dat",
                dtype=np.float32,
                mode="w+",
                shape=(2, 4, 4, 1),
            )
            m = np.memmap(
                source_dir / "M_alertario.dat",
                dtype=np.uint8,
                mode="w+",
                shape=(2, 4, 4, 1),
            )
            y[:] = 0
            m[:] = 0
            y[0, 0, 0, 0] = 1.5
            y[0, 1, 1, 0] = 2.5
            m[0, 0, 0, 0] = 1
            m[0, 1, 1, 0] = 1
            y.flush()
            m.flush()
            del y
            del m
            with (source_dir / "targets_alertario_metadata.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(
                    {
                        "shape": [2, 4, 4, 1],
                        "Y_dtype": "float32",
                        "M_dtype": "uint8",
                        "Y_file": "Y_alertario.dat",
                        "M_file": "M_alertario.dat",
                    },
                    file,
                )

            output_dir.mkdir(parents=True)
            _, source_shape = MODULE.downsample_radar(source_dir, output_dir, 2, 2, 1)
            MODULE.downsample_targets(
                source_dir, output_dir, "alertario", source_shape, 2, 2, 1
            )

            result_radar = np.memmap(
                output_dir / "radar_frames.dat",
                dtype=np.uint8,
                mode="r",
                shape=(2, 2, 2, 3),
            )
            np.testing.assert_array_equal(
                result_radar[0],
                np.array(
                    [[[15, 16, 17], [21, 22, 23]],
                     [[39, 40, 41], [45, 46, 47]]],
                    dtype=np.uint8,
                ),
            )

            result_y = np.memmap(
                output_dir / "Y_alertario.dat",
                dtype=np.float32,
                mode="r",
                shape=(2, 2, 2, 1),
            )
            result_m = np.memmap(
                output_dir / "M_alertario.dat",
                dtype=np.uint8,
                mode="r",
                shape=(2, 2, 2, 1),
            )
            self.assertEqual(result_m[0, 0, 0, 0], 1)
            self.assertEqual(result_y[0, 0, 0, 0], 2.5)


if __name__ == "__main__":
    unittest.main()
