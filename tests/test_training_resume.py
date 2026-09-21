import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_nowcasting.py"
SPEC = importlib.util.spec_from_file_location("train_nowcasting", SCRIPT_PATH)
train_nowcasting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_nowcasting)


def configuration() -> dict:
    return {
        "dataset_root": "/data/dataset",
        "model": "stconvs2s-c",
        "num_layers": 3,
        "hidden_dim": 32,
        "kernel_size": 5,
        "step": 5,
        "stride": 5,
        "target_source": "alertario",
        "loss": "masked-huber",
        "huber_delta": 0.1,
        "loss_weights": "1,5,10,20",
        "sampler_thresholds": "1.25,6.25,12.5",
        "balanced_sampler": False,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.001,
        "seed": 1000,
        "distributed": False,
        "world_size": 1,
        "train_years": [2012, 2013],
        "val_years": [2014],
        "test_years": [2015],
        "stconvs2s_commit": "abc123",
    }


class TrainingResumeTests(unittest.TestCase):
    def test_rng_state_restores_python_numpy_and_torch_sequences(self):
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        state = train_nowcasting.capture_rng_state()

        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        train_nowcasting.restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(1).item())

        self.assertEqual(expected, actual)

    def test_resume_checkpoint_rejects_changed_training_configuration(self):
        saved_configuration = configuration()
        state = {
            "checkpoint_version": train_nowcasting.CHECKPOINT_VERSION,
            "configuration": saved_configuration,
            "rng_states": [{}],
        }
        train_nowcasting.validate_resume_checkpoint(state, saved_configuration, world_size=1)

        changed = configuration()
        changed["batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "batch_size"):
            train_nowcasting.validate_resume_checkpoint(state, changed, world_size=1)

        with self.assertRaisesRegex(ValueError, "1 rank"):
            train_nowcasting.validate_resume_checkpoint(state, saved_configuration, world_size=2)

    def test_atomic_checkpoint_write_replaces_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iteration_1_last.pt"
            train_nowcasting.atomic_torch_save({"epoch": 1}, path)
            train_nowcasting.atomic_torch_save({"epoch": 2}, path)

            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 2)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
