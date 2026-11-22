#!/usr/bin/env python3
"""Audit spatial support for AlertaRio observations without excluding targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nowcasting.spatial_qc import (
    build_neighbor_graph,
    classify_spatial_context,
    compute_neighbor_context,
    radar_patch_metrics,
    summarize_spatial_episodes,
)
from nowcasting.station_geometry import direct_resampled_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audita suporte espacial de candidatos AlertaRio usando vizinhos, radar RGB e, "
            "opcionalmente, observações WebSirene já aprovadas pelo QC."
        )
    )
    parser.add_argument("--candidate-observations", type=Path, required=True)
    parser.add_argument("--alertario-root", type=Path, required=True)
    parser.add_argument("--alertario-mapping", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--websirene-root", type=Path)
    parser.add_argument("--websirene-mapping", type=Path)
    parser.add_argument("--websirene-whitelist", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato não suportado: {path}")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "wet_threshold_mm15",
        "spatial_screening_threshold_mm15",
        "temporal_tolerance_minutes",
        "candidate_cadence_minutes",
        "review_episode_gap_minutes",
        "alertario_neighbors",
        "radar",
        "websirene",
        "automatic_exclusion",
    }
    if missing := required - set(config):
        raise ValueError(f"Configuração espacial incompleta: {sorted(missing)}")
    if config["automatic_exclusion"] is not False:
        raise ValueError("A versão diagnóstica não permite exclusão automática.")
    return config


def normalize_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path).rename(columns={"nome": "station_name"})
    required = {"station_id", "latitude", "longitude"}
    if missing := required - set(mapping):
        raise ValueError(f"{path}: colunas ausentes: {sorted(missing)}")
    mapping = mapping.dropna(subset=list(required)).copy()
    mapping["station_id"] = pd.to_numeric(mapping["station_id"], errors="raise").astype(int)
    if mapping["station_id"].duplicated().any():
        raise ValueError(f"{path}: station_id duplicado.")
    if "station_name" not in mapping:
        mapping["station_name"] = mapping["station_id"].astype(str)
    return mapping


def load_candidates(path: Path, mapping: pd.DataFrame) -> pd.DataFrame:
    candidates = _read_table(path).rename(
        columns={"estacao_id": "station_id", "dia_utc": "timestamp", "estacao": "station_name"}
    )
    required = {"station_id", "timestamp", "m15"}
    if missing := required - set(candidates):
        raise ValueError(f"Candidatos: colunas ausentes: {sorted(missing)}")
    candidates["station_id"] = pd.to_numeric(candidates["station_id"], errors="raise").astype(int)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="raise")
    candidates["m15"] = pd.to_numeric(candidates["m15"], errors="coerce")
    if candidates.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("Os candidatos contêm station_id/timestamp duplicado.")
    unknown_station_ids = set(candidates["station_id"]) - set(mapping["station_id"])
    if unknown_station_ids:
        raise ValueError(f"Candidatos sem mapeamento: {sorted(unknown_station_ids)}")
    names = mapping.set_index("station_id")["station_name"]
    if "station_name" not in candidates:
        candidates["station_name"] = candidates["station_id"].map(names)
    else:
        candidates["station_name"] = candidates["station_name"].fillna(
            candidates["station_id"].map(names)
        )
    for column in ("m05", "max_consistency_abs_error_mm"):
        if column not in candidates:
            candidates[column] = np.nan
    temporal_flags = [
        column for column in (
            "conflicting_duplicate_timestamp",
            "accumulation_hierarchy_violation",
            "m10_inconsistent",
            "m15_inconsistent",
            "h01_inconsistent",
        ) if column in candidates
    ]
    candidates["temporal_inconsistent"] = (
        candidates[temporal_flags].fillna(False).astype(bool).any(axis=1)
        if temporal_flags else False
    )
    candidates = candidates.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=np.int64))
    return candidates


def required_timestamps(
    candidates: pd.DataFrame,
    *,
    tolerance_minutes: int,
    cadence_minutes: int,
) -> pd.DatetimeIndex:
    offsets = range(-tolerance_minutes, tolerance_minutes + 1, cadence_minutes)
    values = {
        timestamp + pd.Timedelta(minutes=offset)
        for timestamp in candidates["timestamp"]
        for offset in offsets
    }
    return pd.DatetimeIndex(sorted(values))


def load_alertario_context(
    root: Path,
    candidates: pd.DataFrame,
    *,
    tolerance_minutes: int,
    cadence_minutes: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum Parquet AlertaRio encontrado em {root}")
    requested = required_timestamps(
        candidates,
        tolerance_minutes=tolerance_minutes,
        cadence_minutes=cadence_minutes,
    )
    station_ids = set(candidates["station_id"].astype(int))
    frames = []
    rows_scanned = 0
    rows_selected = 0
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=["estacao_id", "dia_utc", "m15"])
        rows_scanned += len(frame)
        frame = frame.rename(columns={"estacao_id": "station_id", "dia_utc": "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        # Keep all stations: candidate station IDs only define times, not possible neighbors.
        frame = frame.loc[frame["timestamp"].isin(requested)].copy()
        if not frame.empty:
            frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce")
            frame["m15"] = pd.to_numeric(frame["m15"], errors="coerce")
            frame = frame.loc[
                frame["station_id"].notna() & frame["m15"].notna() & frame["m15"].ge(0)
            ]
            frame["station_id"] = frame["station_id"].astype(int)
            rows_selected += len(frame)
            frames.append(frame)
        if index == 1 or index % 50 == 0 or index == len(files):
            print(f"AlertaRio files {index}/{len(files)}", flush=True)
    observations = (
        pd.concat(frames, ignore_index=True).groupby(
            ["station_id", "timestamp"], as_index=False,
        )["m15"].max()
        if frames else pd.DataFrame(columns=["station_id", "timestamp", "m15"])
    )
    return observations, {
        "files": len(files),
        "rows_scanned": rows_scanned,
        "valid_rows_selected": rows_selected,
        "candidate_station_ids": len(station_ids),
        "requested_timestamps": len(requested),
    }


def load_websirene_context(
    root: Path,
    whitelist_path: Path,
    candidates: pd.DataFrame,
    *,
    tolerance_minutes: int,
    cadence_minutes: int,
) -> tuple[pd.DataFrame, set[int], dict[str, int]]:
    whitelist = pd.read_csv(whitelist_path)
    approved = set(pd.to_numeric(whitelist["station_id"], errors="coerce").dropna().astype(int))
    requested = required_timestamps(
        candidates,
        tolerance_minutes=tolerance_minutes,
        cadence_minutes=cadence_minutes,
    )
    years = sorted(candidates["timestamp"].dt.year.unique())
    frames = []
    files_found = 0
    for station_id in sorted(approved):
        for year in years:
            path = root / f"station_id={station_id}" / f"year={year}" / "data.parquet"
            if not path.is_file():
                continue
            files_found += 1
            frame = pd.read_parquet(path, columns=["timestamp_15min", "m15", "qc_status"])
            frame = frame.loc[frame["qc_status"].eq("accepted")].copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp_15min"], utc=True, errors="coerce")
            frame = frame.loc[frame["timestamp"].isin(requested), ["timestamp", "m15"]]
            if frame.empty:
                continue
            frame["station_id"] = station_id
            frames.append(frame)
    observations = (
        pd.concat(frames, ignore_index=True).groupby(
            ["station_id", "timestamp"], as_index=False,
        )["m15"].max()
        if frames else pd.DataFrame(columns=["station_id", "timestamp", "m15"])
    )
    return observations, approved, {
        "approved_stations": len(approved),
        "files_found_for_candidate_years": files_found,
        "valid_rows_selected": int(len(observations)),
    }


def extract_radar_context(
    candidates: pd.DataFrame,
    mapping: pd.DataFrame,
    dataset_root: Path,
    radar_config: dict,
) -> pd.DataFrame:
    floor_minutes = int(radar_config["observation_timestamp_floor_minutes"])
    offset = pd.Timedelta(minutes=int(radar_config["observation_to_radar_offset_minutes"]))
    radii = tuple(int(value) for value in radar_config["patch_radii_pixels"])
    empty_metrics = {
        key: None
        for radius in radii
        for key in (
            f"radar_patch_{2 * radius + 1}x{2 * radius + 1}_pixels",
            f"radar_patch_{2 * radius + 1}x{2 * radius + 1}_active_pixels",
            f"radar_patch_{2 * radius + 1}x{2 * radius + 1}_active_fraction",
            f"radar_patch_{2 * radius + 1}x{2 * radius + 1}_max_channel",
        )
    }
    cache: dict[int, tuple[np.memmap, dict[pd.Timestamp, int], dict, dict[int, tuple[int, int]]]] = {}
    rows = []
    for candidate in candidates.itertuples(index=False):
        radar_timestamp = candidate.timestamp.floor(f"{floor_minutes}min") + offset
        year = int(radar_timestamp.year)
        result = {
            "candidate_id": int(candidate.candidate_id),
            "radar_timestamp": radar_timestamp,
            "radar_context_available": False,
            "radar_echo_present": False,
            **empty_metrics,
        }
        if year not in cache:
            year_dir = dataset_root / f"year={year}"
            metadata_path = year_dir / "metadata.json"
            timestamps_path = year_dir / "radar_timestamps.npy"
            if not metadata_path.is_file() or not timestamps_path.is_file():
                cache[year] = (None, {}, {}, {})  # type: ignore[arg-type]
            else:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                shape = tuple(int(value) for value in metadata["shape"])
                frames = np.memmap(
                    year_dir / metadata["frames_file"],
                    dtype=np.dtype(metadata["dtype"]),
                    mode="r",
                    shape=shape,
                )
                timestamps = pd.to_datetime(
                    np.load(timestamps_path, allow_pickle=False), utc=True,
                )
                timestamp_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
                mapped_rows, mapped_columns = direct_resampled_pixels(
                    mapping["pixel_i"].to_numpy(float),
                    mapping["pixel_j"].to_numpy(float),
                    height_orig=int(radar_config["source_height"]),
                    width_orig=int(radar_config["source_width"]),
                    height=shape[1],
                    width=shape[2],
                )
                pixels = {
                    int(station_id): (int(row), int(column))
                    for station_id, row, column in zip(
                        mapping["station_id"], mapped_rows, mapped_columns,
                    )
                }
                cache[year] = (frames, timestamp_index, metadata, pixels)
        frames, timestamp_index, _, pixels = cache[year]
        frame_index = timestamp_index.get(radar_timestamp)
        pixel = pixels.get(int(candidate.station_id))
        if frames is not None and frame_index is not None and pixel is not None:
            metrics = radar_patch_metrics(
                np.asarray(frames[frame_index]),
                pixel[0],
                pixel[1],
                patch_radii=radii,
                active_channel_threshold=int(radar_config["active_channel_threshold"]),
                minimum_active_pixels=int(radar_config["minimum_active_pixels"]),
            )
            result.update(metrics)
            result.update({
                "radar_context_available": True,
                "radar_frame_index": int(frame_index),
                "radar_row": pixel[0],
                "radar_column": pixel[1],
            })
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    optional_websirene = (
        args.websirene_root, args.websirene_mapping, args.websirene_whitelist,
    )
    if any(value is not None for value in optional_websirene) and not all(
        value is not None for value in optional_websirene
    ):
        raise ValueError("Informe os três argumentos WebSirene ou nenhum deles.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} não está vazio; use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    alert_mapping = normalize_mapping(args.alertario_mapping)
    required_mapping_columns = {"pixel_i", "pixel_j"}
    if missing := required_mapping_columns - set(alert_mapping):
        raise ValueError(f"Mapeamento AlertaRio sem coordenadas de pixel: {sorted(missing)}")
    candidates = load_candidates(args.candidate_observations, alert_mapping)
    tolerance = int(config["temporal_tolerance_minutes"])
    cadence = int(config["candidate_cadence_minutes"])
    wet_threshold = float(config["wet_threshold_mm15"])

    alert_cfg = config["alertario_neighbors"]
    alert_graph = build_neighbor_graph(
        alert_mapping,
        local_radius_km=float(alert_cfg["local_radius_km"]),
        extended_radius_km=float(alert_cfg["extended_radius_km"]),
        same_network=True,
    )
    alert_observations, alert_stats = load_alertario_context(
        args.alertario_root,
        candidates,
        tolerance_minutes=tolerance,
        cadence_minutes=cadence,
    )
    alert_context = compute_neighbor_context(
        candidates,
        alert_observations,
        alert_graph,
        prefix="alertario",
        wet_threshold_mm15=wet_threshold,
        temporal_tolerance_minutes=tolerance,
    )
    radar_context = extract_radar_context(
        candidates, alert_mapping, args.dataset_root, config["radar"],
    )

    web_stats = {
        "configured": False,
        "approved_stations": 0,
        "files_found_for_candidate_years": 0,
        "valid_rows_selected": 0,
    }
    empty_web_graph = pd.DataFrame(columns=alert_graph.columns)
    web_graph = empty_web_graph
    empty_web_observations = pd.DataFrame(columns=["station_id", "timestamp", "m15"])
    web_observations = empty_web_observations
    if all(value is not None for value in optional_websirene):
        web_mapping = normalize_mapping(args.websirene_mapping)
        web_observations, approved, loaded_stats = load_websirene_context(
            args.websirene_root,
            args.websirene_whitelist,
            candidates,
            tolerance_minutes=tolerance,
            cadence_minutes=cadence,
        )
        web_mapping = web_mapping.loc[web_mapping["station_id"].isin(approved)]
        web_cfg = config["websirene"]
        web_graph = build_neighbor_graph(
            alert_mapping,
            web_mapping,
            local_radius_km=float(web_cfg["local_radius_km"]),
            extended_radius_km=float(web_cfg["extended_radius_km"]),
            same_network=False,
        )
        web_stats = {"configured": True, **loaded_stats}
    web_context = compute_neighbor_context(
        candidates,
        web_observations,
        web_graph,
        prefix="websirene",
        wet_threshold_mm15=wet_threshold,
        temporal_tolerance_minutes=tolerance,
    )

    context = candidates.merge(alert_context, on="candidate_id", validate="one_to_one")
    context = context.merge(radar_context, on="candidate_id", validate="one_to_one")
    context = context.merge(web_context, on="candidate_id", validate="one_to_one")
    context["websirene_source_configured"] = bool(web_stats["configured"])
    context["websirene_context_available"] = context[
        "websirene_extended_window_available_count"
    ].gt(0)
    context = classify_spatial_context(
        context,
        screening_threshold_mm15=float(config["spatial_screening_threshold_mm15"]),
        min_alertario_neighbors=int(alert_cfg["minimum_available_neighbors"]),
        min_websirene_consensus=int(
            config["websirene"]["minimum_wet_stations_for_consensus"]
        ),
    )
    episodes = summarize_spatial_episodes(
        context,
        maximum_gap_minutes=int(config["review_episode_gap_minutes"]),
    )

    context.to_parquet(args.output_dir / "spatial_observations.parquet", index=False)
    priority = pd.Categorical(context["review_priority"], ["high", "medium", "low"], ordered=True)
    review = context.assign(_priority=priority).sort_values(
        ["_priority", "max_consistency_abs_error_mm", "m15"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns="_priority")
    review.to_csv(args.output_dir / "review_candidates.csv", index=False)
    episodes.to_csv(args.output_dir / "spatial_episode_summary.csv", index=False)
    alert_graph.to_csv(args.output_dir / "station_neighbor_graph.csv", index=False)
    if web_stats["configured"]:
        web_graph.to_csv(args.output_dir / "websirene_neighbor_graph.csv", index=False)

    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "automatic_exclusions": 0,
        "candidate_observations": str(args.candidate_observations),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "candidates": int(len(context)),
        "stations": int(context["station_id"].nunique()),
        "episodes": int(len(episodes)),
        "high_priority_episodes": int(episodes["high_priority_episode"].sum()),
        "spatial_status_counts": {
            str(key): int(value)
            for key, value in context["spatial_status"].value_counts().items()
        },
        "review_priority_counts": {
            str(key): int(value)
            for key, value in context["review_priority"].value_counts().items()
        },
        "high_priority_review": int(context["high_priority_review"].sum()),
        "radar_context_available": int(context["radar_context_available"].sum()),
        "alertario": alert_stats,
        "websirene": {
            **web_stats,
            "policy": (
                "accepted observations and approved stations only; positive consensus may add "
                "support; absence is never negative evidence; WebSirene cannot determine rejection"
            ),
        },
        "outputs": {
            "spatial_observations": "spatial_observations.parquet",
            "review_candidates": "review_candidates.csv",
            "spatial_episode_summary": "spatial_episode_summary.csv",
            "station_neighbor_graph": "station_neighbor_graph.csv",
            "websirene_neighbor_graph": (
                "websirene_neighbor_graph.csv" if web_stats["configured"] else None
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(
        f"Auditoria espacial concluída | candidatos={len(context)} | "
        f"prioridade_alta={summary['high_priority_review']} | saída={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
