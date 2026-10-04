"""Write model-agnostic station forecast records for paired evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def export_flat_station_forecasts(
    dataset, station_pixels: np.ndarray, station_ids: list[int], prediction_log: np.ndarray, output: Path,
    *, experiment_id: str,
) -> dict:
    """Export flattened ``sample,horizon,station`` predictions as Parquet.

    Dataset sample order is part of the identifier, but the emitted keys use
    target timestamp, horizon and station, so records remain comparable across
    model implementations and future dataset years.
    """
    expected = len(dataset) * dataset.t_out * len(station_ids)
    prediction = np.asarray(prediction_log, dtype=np.float32)
    if len(prediction) != expected:
        raise ValueError(f"Previsões: esperado {expected}, recebido {len(prediction)}.")
    timestamps: dict[int, np.ndarray] = {}
    rows: list[dict] = []
    offset = 0
    for sample_index, (year, start) in enumerate(dataset.samples):
        if year not in timestamps:
            metadata = json.loads((dataset.radar_root / f"year={year}" / "metadata.json").read_text(encoding="utf-8"))
            timestamps[year] = np.load(dataset.radar_root / f"year={year}" / metadata.get("timestamps_file", "radar_timestamps.npy"), allow_pickle=False)
        _, target, mask = dataset[sample_index]
        target_values = np.expm1(target.numpy()[0][:, station_pixels[:, 0], station_pixels[:, 1]])
        observed = mask.numpy()[0][:, station_pixels[:, 0], station_pixels[:, 1]] > 0
        current = np.maximum(np.expm1(prediction[offset:offset + dataset.t_out * len(station_ids)]), 0.0)
        current = current.reshape(dataset.t_out, len(station_ids))
        offset += dataset.t_out * len(station_ids)
        for horizon in range(dataset.t_out):
            timestamp = timestamps[year][start + dataset.t_in + horizon]
            for station_index, station_id in enumerate(station_ids):
                rows.append({"experiment_id": experiment_id, "year": year, "target_timestamp": str(timestamp),
                             "horizon": horizon + 1, "station_id": station_id,
                             "predicted_mm_15min": float(current[horizon, station_index]),
                             "observed_mm_15min": float(target_values[horizon, station_index]),
                             "is_observed": bool(observed[horizon, station_index])})
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output, index=False)
    metadata = {"format": "station_forecast_records_v1", "experiment_id": experiment_id,
                "rows": len(frame), "years": sorted({int(year) for year, _ in dataset.samples}),
                "t_in": dataset.t_in, "t_out": dataset.t_out, "stride": dataset.stride,
                "station_count": len(station_ids), "prediction_unit": "mm/15min"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata
