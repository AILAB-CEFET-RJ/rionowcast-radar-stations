import json
import hashlib
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
from nowcasting.station_dataset import StationSequenceDataset
from nowcasting.station_model import StationMLP


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
    target[5, 1, 1, 0] = np.log1p(20.0)
    mask[5, 1, 1, 0] = 1
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


def create_sparse_year(root: Path, year: int) -> None:
    year_dir = root / f"year={year}"
    year_dir.mkdir(parents=True)
    shape_radar = (12, 2, 2, 3)
    shape_target = (12, 2, 2, 1)
    radar = np.memmap(year_dir / "radar_frames.dat", dtype=np.uint8, mode="w+", shape=shape_radar)
    radar[:] = 0
    radar.flush()
    np.savez(
        year_dir / "targets_alertario_sparse.npz",
        frame=np.array([1, 5, 5], dtype=np.int32),
        row=np.array([0, 0, 1], dtype=np.uint16),
        column=np.array([0, 0, 1], dtype=np.uint16),
        value=np.array([np.log1p(1.0), np.log1p(2.0), np.log1p(20.0)], dtype=np.float32),
    )
    with (year_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"shape": list(shape_radar), "dtype": "uint8"}, file)
    with (year_dir / "targets_alertario_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"shape": list(shape_target), "format": "sparse", "sparse_file": "targets_alertario_sparse.npz"},
            file,
        )


def create_goes_year(root: Path, year: int, available: list[bool] | None = None) -> None:
    year_dir = root / f"year={year}"
    radar_timestamps = np.array(
        [f"{year}-01-01T{index // 4:02d}:{(index % 4) * 15:02d}:00" for index in range(12)], dtype="<U19"
    )
    np.save(year_dir / "radar_timestamps.npy", radar_timestamps)
    metadata_path = year_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"timestamps_file": "radar_timestamps.npy", "aggregate_minutes": 15, "enforce_timestamp_continuity": True})
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    goes_dir = year_dir / "goes16"
    goes_dir.mkdir()
    shape = (12, 2, 2, 2)
    goes = np.memmap(goes_dir / "goes_frames.dat", dtype=np.float32, mode="w+", shape=shape)
    goes[:] = 200.0
    goes.flush()
    del goes
    availability = np.asarray(available or [True] * 12, dtype=np.uint8)
    np.save(goes_dir / "goes_available.npy", availability)
    np.save(goes_dir / "goes_source_timestamps.npy", np.full((12, 2), f"{year}-01-01T00:00:00+00:00", dtype="<U32"))
    digest = hashlib.sha256(np.asarray(radar_timestamps, dtype="<U32").tobytes()).hexdigest()
    (goes_dir / "metadata.json").write_text(json.dumps({
        "shape": list(shape), "dtype": "float32", "frames_file": "goes_frames.dat",
        "availability_file": "goes_available.npy", "channels": ["C13", "C14"],
        "normalization": {"kind": "divide", "divisor": 400.0},
        "radar_timestamps_sha256": digest,
        "goes_config_sha256": "goes-test", "target_grid": {"height": 2, "width": 2},
    }), encoding="utf-8")


class NowcastingDatasetTests(unittest.TestCase):
    def test_parse_years_supports_ranges_and_lists(self):
        self.assertEqual(parse_years("2021,2019-2020,2021"), [2019, 2020, 2021])

    def test_dataset_contains_only_the_requested_years_without_internal_split(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_year(root, 2020)
            create_year(root, 2021)
            dataset = RadarStationMemmapDataset(root, [2020, 2021], stride=5, split_name="train")
            self.assertEqual(len(dataset), 2)
            self.assertEqual({year for year, _ in dataset.samples}, {2020, 2021})
            x, y, m = dataset[0]
            self.assertEqual(tuple(x.shape), (3, 5, 2, 2))
            self.assertEqual(tuple(y.shape), (1, 5, 2, 2))
            self.assertEqual(tuple(m.shape), (1, 5, 2, 2))

    def test_station_crop_preserves_observations_inside_the_roi(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_year(root, 2020)
            mapping = root / "stations.csv"
            mapping.write_text("station_id,pixel_i,pixel_j\n1,0,0\n", encoding="utf-8")

            dataset = RadarStationMemmapDataset(
                root, [2020], stride=5, split_name="crop", crop_stations=True,
                crop_margin_pixels=0, station_mapping=mapping,
                mapping_height_orig=2, mapping_width_orig=2,
            )
            x, y, mask = dataset[0]

            self.assertEqual(tuple(x.shape), (3, 5, 1, 1))
            self.assertEqual(tuple(y.shape), (1, 5, 1, 1))
            self.assertEqual(int(mask.sum()), 1)
            self.assertAlmostEqual(y[0, 0, 0, 0].item(), np.log1p(2.0))
            self.assertEqual(dataset.crop_metadata["shape"], [1, 1])

    def test_station_crop_filters_and_reindexes_sparse_targets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_sparse_year(root, 2020)
            mapping = root / "stations.csv"
            mapping.write_text("station_id,pixel_i,pixel_j\n1,0,0\n", encoding="utf-8")

            dataset = RadarStationMemmapDataset(
                root, [2020], stride=5, split_name="sparse-crop", crop_stations=True,
                crop_margin_pixels=0, station_mapping=mapping,
                mapping_height_orig=2, mapping_width_orig=2,
            )
            _, y, mask = dataset[0]

            self.assertEqual(tuple(y.shape), (1, 5, 1, 1))
            self.assertEqual(int(mask.sum()), 1)
            self.assertAlmostEqual(y[0, 0, 0, 0].item(), np.log1p(2.0))

    def test_station_history_adds_value_and_mask_input_channels(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_sparse_year(root, 2020)
            dataset = RadarStationMemmapDataset(
                root, [2020], stride=5, split_name="fusion", input_stations=True,
            )
            x, y, mask = dataset[0]

            self.assertEqual(tuple(x.shape), (5, 5, 2, 2))
            self.assertAlmostEqual(x[3, 1, 0, 0].item(), np.log1p(1.0))
            self.assertEqual(x[4, 1, 0, 0].item(), 1.0)
            self.assertEqual(x[4, 0, 0, 0].item(), 0.0)
            self.assertAlmostEqual(y[0, 0, 0, 0].item(), np.log1p(2.0))
            self.assertEqual(int(mask.sum()), 2)

    def test_goes_adds_channels_and_filters_incomplete_input_history(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_sparse_year(root, 2020)
            create_goes_year(root, 2020, [True, True, False, True, True, True, True, True, True, True, True, True])
            dataset = RadarStationMemmapDataset(root, [2020], t_in=2, t_out=2, stride=1, input_goes=True)
            self.assertEqual(dataset.samples, [(2020, 0), (2020, 3), (2020, 4), (2020, 5), (2020, 6), (2020, 7), (2020, 8)])
            x, _, _ = dataset[0]
            self.assertEqual(tuple(x.shape), (5, 2, 2, 2))
            self.assertAlmostEqual(x[3, 0, 0, 0].item(), 0.5)
            self.assertAlmostEqual(x[4, 1, 1, 1].item(), 0.5)

    def test_station_sequence_dataset_and_model_use_values_and_masks(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_sparse_year(root, 2020)
            mapping = root / "stations.csv"
            mapping.write_text("station_id,pixel_i,pixel_j\n1,0,0\n2,1,1\n", encoding="utf-8")
            dataset = StationSequenceDataset(
                root, [2020], mapping=mapping, stride=5,
                mapping_height_orig=2, mapping_width_orig=2,
            )
            inputs, target, mask = dataset[0]
            output = StationMLP(station_count=2)(inputs.unsqueeze(0))

            self.assertEqual(tuple(inputs.shape), (5, 2, 2))
            self.assertEqual(tuple(target.shape), (5, 2))
            self.assertEqual(int(mask.sum()), 2)
            self.assertEqual(tuple(output.shape), (1, 5, 2))

    def test_station_sequence_dataset_rejects_samples_crossing_radar_gaps(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_sparse_year(root, 2020)
            year_dir = root / "year=2020"
            timestamps = np.array([
                "2020-01-01T00:00:00", "2020-01-01T00:15:00", "2020-01-01T00:30:00",
                "2020-01-01T00:45:00", "2020-01-01T01:15:00", "2020-01-01T01:30:00",
                "2020-01-01T01:45:00", "2020-01-01T02:00:00", "2020-01-01T02:15:00",
                "2020-01-01T02:30:00", "2020-01-01T02:45:00", "2020-01-01T03:00:00",
            ], dtype="<U19")
            np.save(year_dir / "radar_timestamps.npy", timestamps)
            metadata_path = year_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({
                "timestamps_file": "radar_timestamps.npy",
                "aggregate_minutes": 15,
                "enforce_timestamp_continuity": True,
            })
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            mapping = root / "stations.csv"
            mapping.write_text("station_id,pixel_i,pixel_j\n1,0,0\n2,1,1\n", encoding="utf-8")

            dataset = StationSequenceDataset(
                root, [2020], mapping=mapping, t_in=2, t_out=2, stride=1,
                mapping_height_orig=2, mapping_width_orig=2,
            )

            self.assertEqual(len(dataset), 6)

    def test_weighted_loss_gives_more_weight_to_extreme_target(self):
        prediction = torch.zeros((1, 1, 1, 1, 2))
        target = torch.tensor([[[[[0.0, np.log1p(20.0)]]]]])
        mask = torch.ones_like(target)
        mae = MaskedMAELoss()(prediction, target, mask)
        weighted = WeightedMaskedMAELoss((1, 1, 1, 20))(prediction, target, mask)
        self.assertGreater(weighted.item(), mae.item())


if __name__ == "__main__":
    unittest.main()
