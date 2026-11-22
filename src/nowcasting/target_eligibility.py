"""Shared sample-eligibility rules for sparse observational targets."""

from __future__ import annotations

import numpy as np


REQUIRE_EACH_OUTPUT_FRAME = "require_observation_in_each_output_frame"


def requires_observation_in_each_output_frame(metadata: dict) -> bool:
    """Return the opt-in target-coverage policy recorded in metadata."""
    policy = metadata.get("sample_eligibility")
    if policy is None:
        return False
    if not isinstance(policy, dict):
        raise ValueError("sample_eligibility deve ser um objeto JSON.")
    required = policy.get(REQUIRE_EACH_OUTPUT_FRAME, False)
    if not isinstance(required, bool):
        raise ValueError(f"sample_eligibility.{REQUIRE_EACH_OUTPUT_FRAME} deve ser booleano.")
    return required


def observed_frame_mask(frame_indices: np.ndarray, frame_count: int) -> np.ndarray:
    """Build a boolean mask indicating frames with at least one observation."""
    indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_count < 0:
        raise ValueError("frame_count não pode ser negativo.")
    if len(indices) and (indices.min() < 0 or indices.max() >= frame_count):
        raise ValueError("Índice de target fora do eixo temporal do radar.")
    observed = np.zeros(frame_count, dtype=bool)
    observed[indices] = True
    return observed


def filter_starts_with_complete_output_coverage(
    candidate_starts: list[int], observed_frames: np.ndarray, *, t_in: int, t_out: int,
) -> list[int]:
    """Keep windows for which every forecast horizon has an observation."""
    observed = np.asarray(observed_frames, dtype=bool)
    missing_prefix = np.concatenate(
        ([0], np.cumsum(~observed, dtype=np.int64))
    )
    return [
        start
        for start in candidate_starts
        if missing_prefix[start + t_in + t_out] == missing_prefix[start + t_in]
    ]
