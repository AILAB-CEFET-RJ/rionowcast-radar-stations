"""Spatial quality-control diagnostics for sparse rain-gauge observations."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


SPATIAL_STATUSES = (
    "below_spatial_screening_threshold",
    "spatially_supported_local",
    "spatially_supported_extended",
    "gauge_isolated_radar_supported",
    "websirene_only_support",
    "multi_source_unsupported",
    "insufficient_spatial_context",
)


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return the great-circle distance between two WGS84 coordinates."""
    radius_km = 6371.0088
    lat_a, lon_a, lat_b, lon_b = map(
        math.radians, (latitude_a, longitude_a, latitude_b, longitude_b),
    )
    return 2 * radius_km * math.asin(
        math.sqrt(
            math.sin((lat_b - lat_a) / 2) ** 2
            + math.cos(lat_a)
            * math.cos(lat_b)
            * math.sin((lon_b - lon_a) / 2) ** 2
        )
    )


def build_neighbor_graph(
    source_mapping: pd.DataFrame,
    neighbor_mapping: pd.DataFrame | None = None,
    *,
    local_radius_km: float = 5.0,
    extended_radius_km: float = 10.0,
    same_network: bool = True,
) -> pd.DataFrame:
    """Build a directed station-neighbor graph with explicit distance bands."""
    if local_radius_km <= 0 or extended_radius_km < local_radius_km:
        raise ValueError("Os raios devem ser positivos e o raio ampliado não pode ser menor.")
    neighbor_mapping = source_mapping if neighbor_mapping is None else neighbor_mapping
    required = {"station_id", "latitude", "longitude"}
    for label, mapping in (("origem", source_mapping), ("vizinhos", neighbor_mapping)):
        missing = required - set(mapping.columns)
        if missing:
            raise ValueError(f"Mapeamento de {label}: colunas ausentes: {sorted(missing)}")

    sources = source_mapping.dropna(subset=list(required)).copy()
    neighbors = neighbor_mapping.dropna(subset=list(required)).copy()
    rows: list[dict[str, object]] = []
    for source in sources.itertuples(index=False):
        for neighbor in neighbors.itertuples(index=False):
            if same_network and int(source.station_id) == int(neighbor.station_id):
                continue
            distance = haversine_km(
                float(source.latitude),
                float(source.longitude),
                float(neighbor.latitude),
                float(neighbor.longitude),
            )
            if distance > extended_radius_km:
                continue
            rows.append({
                "source_station_id": int(source.station_id),
                "neighbor_station_id": int(neighbor.station_id),
                "distance_km": distance,
                "distance_band": "local" if distance <= local_radius_km else "extended",
            })
    return pd.DataFrame(
        rows,
        columns=[
            "source_station_id", "neighbor_station_id", "distance_km", "distance_band",
        ],
    ).sort_values(
        ["source_station_id", "distance_km", "neighbor_station_id"], ignore_index=True,
    )


def _neighbor_statistics(
    values: pd.DataFrame,
    *,
    wet_threshold_mm15: float,
) -> dict[str, float | int | None]:
    if values.empty:
        return {
            "available_count": 0,
            "wet_count": 0,
            "wet_fraction": None,
            "median_m15": None,
            "max_m15": None,
            "nearest_wet_distance_km": None,
        }
    wet = values["m15"].ge(wet_threshold_mm15)
    wet_distances = values.loc[wet, "distance_km"]
    return {
        "available_count": int(len(values)),
        "wet_count": int(wet.sum()),
        "wet_fraction": float(wet.mean()),
        "median_m15": float(values["m15"].median()),
        "max_m15": float(values["m15"].max()),
        "nearest_wet_distance_km": (
            float(wet_distances.min()) if not wet_distances.empty else None
        ),
    }


def compute_neighbor_context(
    candidates: pd.DataFrame,
    observations: pd.DataFrame,
    neighbor_graph: pd.DataFrame,
    *,
    prefix: str,
    wet_threshold_mm15: float = 1.25,
    temporal_tolerance_minutes: int = 15,
) -> pd.DataFrame:
    """Compute exact and time-tolerant neighbor support for each candidate."""
    required_candidates = {"candidate_id", "station_id", "timestamp"}
    required_observations = {"station_id", "timestamp", "m15"}
    if missing := required_candidates - set(candidates.columns):
        raise ValueError(f"Candidatos: colunas ausentes: {sorted(missing)}")
    if missing := required_observations - set(observations.columns):
        raise ValueError(f"Observações: colunas ausentes: {sorted(missing)}")
    if wet_threshold_mm15 < 0 or temporal_tolerance_minutes < 0:
        raise ValueError("Limiar de chuva e tolerância temporal não podem ser negativos.")

    candidate_data = candidates[list(required_candidates)].copy()
    candidate_data["timestamp"] = pd.to_datetime(candidate_data["timestamp"], utc=True)
    observation_data = observations[list(required_observations)].copy()
    observation_data["timestamp"] = pd.to_datetime(observation_data["timestamp"], utc=True)
    observation_data["m15"] = pd.to_numeric(observation_data["m15"], errors="coerce")
    observation_data = observation_data.loc[
        observation_data["m15"].notna() & observation_data["m15"].ge(0)
    ]
    graph_groups = {
        int(station_id): group.copy()
        for station_id, group in neighbor_graph.groupby("source_station_id", sort=False)
    }
    tolerance = pd.Timedelta(minutes=temporal_tolerance_minutes)
    output_rows: list[dict[str, object]] = []
    for candidate in candidate_data.itertuples(index=False):
        graph = graph_groups.get(int(candidate.station_id))
        result: dict[str, object] = {"candidate_id": int(candidate.candidate_id)}
        for scope in ("exact", "local_window", "extended_window"):
            for metric, value in _neighbor_statistics(
                pd.DataFrame(columns=["m15", "distance_km"]),
                wet_threshold_mm15=wet_threshold_mm15,
            ).items():
                result[f"{prefix}_{scope}_{metric}"] = value
        if graph is None or graph.empty:
            output_rows.append(result)
            continue

        neighbor_ids = set(graph["neighbor_station_id"].astype(int))
        nearby = observation_data.loc[
            observation_data["station_id"].astype(int).isin(neighbor_ids)
            & observation_data["timestamp"].between(
                candidate.timestamp - tolerance, candidate.timestamp + tolerance,
            )
        ]
        if nearby.empty:
            output_rows.append(result)
            continue
        nearby = nearby.groupby("station_id", as_index=False)["m15"].max().merge(
            graph[["neighbor_station_id", "distance_km", "distance_band"]],
            left_on="station_id",
            right_on="neighbor_station_id",
            how="inner",
            validate="one_to_one",
        )
        exact = observation_data.loc[
            observation_data["station_id"].astype(int).isin(neighbor_ids)
            & observation_data["timestamp"].eq(candidate.timestamp)
        ].groupby("station_id", as_index=False)["m15"].max().merge(
            graph[["neighbor_station_id", "distance_km"]],
            left_on="station_id",
            right_on="neighbor_station_id",
            how="inner",
            validate="one_to_one",
        )
        scopes = {
            "exact": exact,
            "local_window": nearby.loc[nearby["distance_band"] == "local"],
            "extended_window": nearby,
        }
        for scope, values in scopes.items():
            for metric, value in _neighbor_statistics(
                values, wet_threshold_mm15=wet_threshold_mm15,
            ).items():
                result[f"{prefix}_{scope}_{metric}"] = value
        output_rows.append(result)
    return pd.DataFrame(output_rows)


def radar_patch_metrics(
    frame: np.ndarray,
    row: int,
    column: int,
    *,
    patch_radii: tuple[int, ...] = (1, 2),
    active_channel_threshold: int = 0,
    minimum_active_pixels: int = 1,
) -> dict[str, float | int | bool]:
    """Measure visible RGB echo around one station without physical calibration."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("O frame de radar deve ter shape (altura, largura, 3).")
    if not patch_radii or min(patch_radii) < 0 or minimum_active_pixels <= 0:
        raise ValueError("Raios e número mínimo de pixels ativos devem ser positivos.")
    if not (0 <= row < frame.shape[0] and 0 <= column < frame.shape[1]):
        raise ValueError("Coordenada da estação fora do frame de radar.")

    result: dict[str, float | int | bool] = {}
    largest_active_pixels = 0
    for radius in sorted(set(patch_radii)):
        top, bottom = max(0, row - radius), min(frame.shape[0], row + radius + 1)
        left, right = max(0, column - radius), min(frame.shape[1], column + radius + 1)
        patch = frame[top:bottom, left:right]
        active = patch.max(axis=2) > active_channel_threshold
        size = radius * 2 + 1
        active_pixels = int(active.sum())
        result[f"radar_patch_{size}x{size}_pixels"] = int(active.size)
        result[f"radar_patch_{size}x{size}_active_pixels"] = active_pixels
        result[f"radar_patch_{size}x{size}_active_fraction"] = float(active.mean())
        result[f"radar_patch_{size}x{size}_max_channel"] = int(patch.max())
        largest_active_pixels = active_pixels
    result["radar_echo_present"] = largest_active_pixels >= minimum_active_pixels
    return result


def classify_spatial_context(
    context: pd.DataFrame,
    *,
    screening_threshold_mm15: float = 6.25,
    min_alertario_neighbors: int = 2,
    min_websirene_consensus: int = 2,
) -> pd.DataFrame:
    """Assign diagnostic statuses without converting flags into exclusions."""
    required = {
        "m15",
        "temporal_inconsistent",
        "alertario_local_window_wet_count",
        "alertario_extended_window_wet_count",
        "alertario_extended_window_available_count",
        "radar_context_available",
        "radar_echo_present",
        "websirene_extended_window_wet_count",
    }
    if missing := required - set(context.columns):
        raise ValueError(f"Contexto espacial: colunas ausentes: {sorted(missing)}")
    if screening_threshold_mm15 < 0 or min_alertario_neighbors <= 0 or min_websirene_consensus <= 0:
        raise ValueError("Limiares espaciais inválidos.")

    result = context.copy()
    result["spatially_screened"] = pd.to_numeric(result["m15"], errors="coerce").ge(
        screening_threshold_mm15
    )
    result["alertario_context_adequate"] = result[
        "alertario_extended_window_available_count"
    ].ge(min_alertario_neighbors)
    result["alertario_local_support"] = result[
        "alertario_local_window_wet_count"
    ].ge(1)
    result["alertario_extended_support"] = result[
        "alertario_extended_window_wet_count"
    ].ge(1)
    result["websirene_consensus_support"] = result[
        "websirene_extended_window_wet_count"
    ].ge(min_websirene_consensus)

    statuses = []
    for row in result.itertuples(index=False):
        if not row.spatially_screened:
            status = "below_spatial_screening_threshold"
        elif row.alertario_local_support:
            status = "spatially_supported_local"
        elif row.alertario_extended_support:
            status = "spatially_supported_extended"
        elif bool(row.radar_context_available) and bool(row.radar_echo_present):
            status = "gauge_isolated_radar_supported"
        elif row.websirene_consensus_support:
            status = "websirene_only_support"
        elif not row.alertario_context_adequate or not bool(row.radar_context_available):
            status = "insufficient_spatial_context"
        else:
            status = "multi_source_unsupported"
        statuses.append(status)
    result["spatial_status"] = statuses
    result["multi_source_unsupported"] = result["spatial_status"].eq(
        "multi_source_unsupported"
    )
    result["high_priority_review"] = (
        result["multi_source_unsupported"] & result["temporal_inconsistent"].astype(bool)
    )
    result["review_priority"] = np.select(
        [
            result["high_priority_review"],
            result["multi_source_unsupported"]
            | result["temporal_inconsistent"].astype(bool)
            | result["spatial_status"].eq("websirene_only_support"),
        ],
        ["high", "medium"],
        default="low",
    )
    return result


def summarize_spatial_episodes(
    observations: pd.DataFrame,
    *,
    maximum_gap_minutes: int = 5,
) -> pd.DataFrame:
    """Group nearby diagnostic candidates without declaring physical rain events."""
    if maximum_gap_minutes <= 0:
        raise ValueError("maximum_gap_minutes deve ser positivo.")
    if observations.empty:
        return pd.DataFrame()
    data = observations.sort_values(["station_id", "timestamp"]).reset_index(drop=True).copy()
    difference = data.groupby("station_id")["timestamp"].diff()
    starts = difference.isna() | difference.gt(pd.Timedelta(minutes=maximum_gap_minutes))
    data["episode_number"] = starts.groupby(data["station_id"]).cumsum().astype(int)
    rows = []
    priority_order = {"low": 0, "medium": 1, "high": 2}
    for keys, group in data.groupby(["station_id", "station_name", "episode_number"], sort=True):
        station_id, station_name, episode_number = keys
        temporal_inconsistent_count = int(group["temporal_inconsistent"].sum())
        multi_source_unsupported_count = int(group["multi_source_unsupported"].sum())
        high_priority_episode = bool(
            temporal_inconsistent_count > 0 and multi_source_unsupported_count > 0
        )
        row_priority = max(group["review_priority"], key=priority_order.get)
        rows.append({
            "station_id": int(station_id),
            "station_name": station_name,
            "episode_number": int(episode_number),
            "start_timestamp": group["timestamp"].min(),
            "end_timestamp": group["timestamp"].max(),
            "candidate_timestamps": int(len(group)),
            "max_m05": float(group["m05"].max()),
            "max_m15": float(group["m15"].max()),
            "max_consistency_abs_error_mm": float(group["max_consistency_abs_error_mm"].max()),
            "spatially_screened_count": int(group["spatially_screened"].sum()),
            "temporal_inconsistent_count": temporal_inconsistent_count,
            "multi_source_unsupported_count": multi_source_unsupported_count,
            "radar_supported_count": int(group["radar_echo_present"].sum()),
            "alertario_supported_count": int(group["alertario_extended_support"].sum()),
            "websirene_consensus_count": int(group["websirene_consensus_support"].sum()),
            "high_priority_episode": high_priority_episode,
            "highest_review_priority": "high" if high_priority_episode else row_priority,
            "spatial_statuses": "|".join(sorted(set(group["spatial_status"]))),
        })
    return pd.DataFrame(rows)
