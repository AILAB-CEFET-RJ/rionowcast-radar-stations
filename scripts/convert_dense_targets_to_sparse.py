#!/usr/bin/env python3
"""Convert annual dense station targets into compact sparse observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte Y/M memmaps densos em targets esparsos.")
    parser.add_argument("--source-year-dir", type=Path, required=True)
    parser.add_argument("--output-year-dir", type=Path, required=True)
    parser.add_argument("--source-metadata", default="targets_alertario_metadata.json")
    parser.add_argument("--output-height", type=int, required=True)
    parser.add_argument("--output-width", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    with (args.source_year_dir / args.source_metadata).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    shape = tuple(metadata["shape"])
    if len(shape) != 4 or shape[-1] != 1:
        raise ValueError(f"Shape de target invalido: {shape}")
    n_frames, source_height, source_width, _ = shape
    y = np.memmap(args.source_year_dir / metadata["Y_file"], dtype=np.dtype(metadata["Y_dtype"]), mode="r", shape=shape)
    m = np.memmap(args.source_year_dir / metadata["M_file"], dtype=np.dtype(metadata["M_dtype"]), mode="r", shape=shape)
    frames, rows, columns, values = [], [], [], []
    for start in range(0, n_frames, args.chunk_size):
        stop = min(start + args.chunk_size, n_frames)
        local_t, source_i, source_j, _ = np.nonzero(m[start:stop])
        if len(local_t):
            frames.append((start + local_t).astype(np.int32))
            rows.append(np.rint(source_i * (args.output_height - 1) / max(source_height - 1, 1)).astype(np.uint16))
            columns.append(np.rint(source_j * (args.output_width - 1) / max(source_width - 1, 1)).astype(np.uint16))
            values.append(np.asarray(y[start:stop][local_t, source_i, source_j, 0], dtype=np.float32))
        print(f"frames {start}:{stop}/{n_frames}", flush=True)
    frame = np.concatenate(frames) if frames else np.empty(0, dtype=np.int32)
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.uint16)
    column = np.concatenate(columns) if columns else np.empty(0, dtype=np.uint16)
    value = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    # Collisions created by resampling retain the largest observed precipitation.
    keys = np.lexsort((column, row, frame))
    frame, row, column, value = frame[keys], row[keys], column[keys], value[keys]
    starts = np.r_[True, (frame[1:] != frame[:-1]) | (row[1:] != row[:-1]) | (column[1:] != column[:-1])]
    groups = np.flatnonzero(starts)
    value = np.maximum.reduceat(value, groups)
    frame, row, column = frame[groups], row[groups], column[groups]
    args.output_year_dir.mkdir(parents=True, exist_ok=True)
    sparse_file = "targets_alertario_sparse.npz"
    np.savez_compressed(args.output_year_dir / sparse_file, frame=frame, row=row, column=column, value=value)
    output = dict(metadata)
    output.update({"format": "sparse", "shape": [n_frames, args.output_height, args.output_width, 1], "sparse_file": sparse_file, "sparse_fields": {"frame": "int32", "row": "uint16", "column": "uint16", "value": "float32"}, "observation_count": int(len(value)), "spatial_resampling": "sparse-nearest", "source_spatial_shape": [source_height, source_width]})
    output.pop("Y_file", None); output.pop("M_file", None)
    with (args.output_year_dir / args.source_metadata).open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
