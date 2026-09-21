import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.dataset import RadarStationMemmapDataset, parse_years
from nowcasting.losses import MaskedMAELoss, WeightedMaskedMAELoss


def create_year(root: Path, year: int) -> None:
    year_dir = root / f"year={year}"
    year_dir.mkdir(parents=True)
    shape_radar = (12, 2, 2, 3)
    shape_target = (12, 2, 2, 1)
    radar = np.memmap(year_dir / "radar_frames.dat", dtype=np.uint8, mode="w+", shape=shape_radar)
    target = np.memmap(year_dir / "Y_alertario.dat", dtype=np.float32, mode="w+", shape=shape_target)
    mask = np.memmap(year_dir / "M_alertario.dat", dtype=np.uint8, mode="w+", shape=shape_target)
    radar[:] = 0
    target[:] = 0
    mask[:] = 0
    target[5, 0, 0, 0] = np.log1p(2.0)
    mask[5, 0, 0, 0] = 1
    radar.flush()
    target.flush()
    mask.flush()
    with (year_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"shape": list(shape_radar), "dtype": "uint8"}, file)
    with (year_dir / "targets_alertario_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"shape": list(shape_target), "Y_dtype": "float32", "M_dtype": "uint8",
             "Y_file": "Y_alertario.dat", "M_file": "M_alertario.dat"}, file,
        )


class NowcastingDatasetTests(unittest.TestCase):
    def test_parse_years_supports_ranges_and_lists(self):
        self.assertEqual(parse_years("2021,2019-2020,2021"), [2019, 2020, 2021])

    def test_dataset_contains_only_the_requested_years_without_internal_split(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_year(root, 2020)
            create_year(root, 2021)
            dataset = RadarStationMemmapDataset(root, [2020, 2021], stride=5, split_name="train")
            self.assertEqual(len(dataset), 4)
            self.assertEqual({year for year, _ in dataset.samples}, {2020, 2021})
            x, y, m = dataset[0]
            self.assertEqual(tuple(x.shape), (3, 5, 2, 2))
            self.assertEqual(tuple(y.shape), (1, 5, 2, 2))
            self.assertEqual(tuple(m.shape), (1, 5, 2, 2))

    def test_weighted_loss_gives_more_weight_to_extreme_target(self):
        prediction = torch.zeros((1, 1, 1, 1, 2))
        target = torch.tensor([[[[[0.0, np.log1p(20.0)]]]]])
        mask = torch.ones_like(target)
        mae = MaskedMAELoss()(prediction, target, mask)
        weighted = WeightedMaskedMAELoss((1, 1, 1, 20))(prediction, target, mask)
        self.assertGreater(weighted.item(), mae.item())


if __name__ == "__main__":
    unittest.main()
