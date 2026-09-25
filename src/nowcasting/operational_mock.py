from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap

from nowcasting.operational_inference import aggregate_capture_bucket
from nowcasting.radar_capture import crop_bounds
from nowcasting.station_geometry import direct_resampled_pixels


def load_event_observations(
    alertario_root: Path,
    station_ids: set[int],
    timestamps: list[pd.Timestamp],
    max_m15: float,
) -> pd.DataFrame:
    """Load only valid AlertaRio observations needed for one retrospective event."""
    wanted = set(timestamps)
    frames = []
    for path in sorted(alertario_root.rglob("*.parquet")):
        frame = pd.read_parquet(path, columns=["estacao_id", "estacao", "dia_utc", "m15"])
        frame = frame.loc[frame["estacao_id"].isin(station_ids)].copy()
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["dia_utc"], utc=True, errors="coerce").dt.floor("15min")
        frame["m15"] = pd.to_numeric(frame["m15"], errors="coerce")
        frame = frame.loc[
            frame["timestamp"].isin(wanted) & frame["m15"].ge(0) & frame["m15"].le(max_m15),
            ["timestamp", "estacao_id", "estacao", "m15"],
        ]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "station_id", "station_name", "m15"])
    observations = pd.concat(frames, ignore_index=True)
    observations = observations.groupby(["timestamp", "estacao_id", "estacao"], as_index=False)["m15"].max()
    return observations.rename(columns={"estacao_id": "station_id", "estacao": "station_name"})


def build_persistence_forecast(
    observations: pd.DataFrame,
    input_timestamps: list[pd.Timestamp],
    target_timestamps: list[pd.Timestamp],
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """Repeat the latest available station observation for each forecast horizon."""
    history = observations.loc[observations["timestamp"].isin(input_timestamps)].copy()
    if history.empty:
        return pd.DataFrame(
            columns=["station_id", "station_name", "latitude", "longitude", "row", "column", "source_timestamp", "target_timestamp", "mock_prediction_mm_15min"]
        )
    latest = history.sort_values("timestamp").groupby("station_id", as_index=False).tail(1)
    latest = latest.merge(stations, on="station_id", how="inner", validate="one_to_one")
    records = []
    for item in latest.itertuples(index=False):
        for target_timestamp in target_timestamps:
            records.append(
                {
                    "station_id": int(item.station_id),
                    "station_name": item.station_name,
                    "latitude": item.latitude,
                    "longitude": item.longitude,
                    "row": int(item.row),
                    "column": int(item.column),
                    "source_timestamp": item.timestamp.isoformat(),
                    "target_timestamp": target_timestamp.isoformat(),
                    "mock_prediction_mm_15min": float(item.m15),
                }
            )
    return pd.DataFrame(records)


def station_positions(mapping: pd.DataFrame, capture_config: dict[str, Any], height: int, width: int) -> pd.DataFrame:
    left, top, right, bottom = crop_bounds(capture_config)
    result = mapping.copy()
    rows, columns = direct_resampled_pixels(
        result["pixel_i"].to_numpy(float),
        result["pixel_j"].to_numpy(float),
        height_orig=bottom - top,
        width_orig=right - left,
        height=height,
        width=width,
    )
    result["row"] = rows
    result["column"] = columns
    return result


def crop_station_region(
    frames: np.ndarray,
    stations: pd.DataFrame,
    margin_pixels: int,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int, int, int]]:
    """Crop radar frames to the station envelope and reindex station pixels."""
    if margin_pixels < 0:
        raise ValueError("crop margin não pode ser negativa.")
    if stations.empty:
        raise ValueError("Não há estações para definir a ROI cropada.")
    height, width = frames.shape[1:3]
    top = max(0, int(stations["row"].min()) - margin_pixels)
    bottom = min(height, int(stations["row"].max()) + margin_pixels + 1)
    left = max(0, int(stations["column"].min()) - margin_pixels)
    right = min(width, int(stations["column"].max()) + margin_pixels + 1)
    cropped_stations = stations.copy()
    cropped_stations["row"] -= top
    cropped_stations["column"] -= left
    return frames[:, top:bottom, left:right, :], cropped_stations, (top, bottom, left, right)


def aggregate_event_history(
    event: dict[str, Any], capture_root: Path, capture_config: dict[str, Any], width: int, height: int
) -> np.ndarray:
    frames = []
    for bucket in event["input"]:
        paths = [capture_root / relative for relative in bucket["png_files"]]
        frames.append(aggregate_capture_bucket(paths, capture_config, width, height))
    return np.stack(frames)


def save_mock_product(
    frames: np.ndarray,
    event: dict[str, Any],
    forecast: pd.DataFrame,
    output_path: Path,
    crop_bounds_128: tuple[int, int, int, int] | None,
) -> None:
    """Render history plus station-persistence forecasts; no dense rainfall field."""
    figure, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for index, axis in enumerate(axes[0]):
        axis.imshow(frames[index])
        axis.set_title(f"Radar {event['input'][index]['timestamp_utc'][11:16]} UTC")
        axis.axis("off")

    vmax = max(12.5, float(forecast["mock_prediction_mm_15min"].max())) if not forecast.empty else 12.5
    for index, axis in enumerate(axes[1]):
        axis.imshow(frames[-1])
        timestamp = event["target"][index]["timestamp_utc"]
        horizon = forecast.loc[forecast["target_timestamp"] == timestamp]
        scatter = axis.scatter(
            horizon["column"], horizon["row"], c=horizon["mock_prediction_mm_15min"],
            cmap="YlOrRd", vmin=0, vmax=vmax, s=55, edgecolors="white", linewidths=0.6,
        )
        axis.set_title(f"MOCK persistencia T+{15 * (index + 1)} min")
        axis.axis("off")
    if not forecast.empty:
        figure.colorbar(scatter, ax=axes[1].tolist(), label="mm/15 min", shrink=0.8)
    roi = "ROI cropada das estacoes" if crop_bounds_128 is not None else "grade completa do radar"
    figure.suptitle(
        f"Demonstracao de produto ({roi}): radar historico e previsao por persistencia nas estacoes\n"
        "Mock deterministico; nao representa previsao de modelo treinado nem campo continuo de chuva."
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def grid_boundary(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the perimeter of a latitude/longitude grid in display order."""
    rows, columns = latitude.shape
    if rows < 2 or columns < 2:
        raise ValueError("Grade geografica deve ter pelo menos 2 x 2 pontos.")
    top = np.arange(columns)
    right = np.arange(1, rows)
    bottom = np.arange(columns - 2, -1, -1)
    left = np.arange(rows - 2, 0, -1)
    row = np.concatenate((np.zeros(columns, dtype=int), right, np.full(columns - 1, rows - 1), left))
    column = np.concatenate((top, np.full(rows - 1, columns - 1), bottom, np.zeros(rows - 2, dtype=int)))
    return latitude[row, column], longitude[row, column]


def render_station_forecast_map(
    forecast: pd.DataFrame,
    stations: pd.DataFrame,
    event: dict[str, Any],
    geographic_grid_path: Path,
    full_height: int,
    full_width: int,
    crop_bounds_128: tuple[int, int, int, int] | None,
    output_path: Path,
    show_station_names: bool,
    map_tiles: str,
    municipal_basemap_path: Path | None,
) -> None:
    """Write a Folium product map with one layer for each forecast horizon."""
    if not geographic_grid_path.is_file():
        raise FileNotFoundError(f"Grade geografica não encontrada: {geographic_grid_path}")
    with np.load(geographic_grid_path) as grid:
        latitude = grid["lat"]
        longitude = grid["lon"]
    row_indices = np.clip(
        ((np.arange(full_height) + 0.5) * latitude.shape[0] / full_height).astype(int),
        0,
        latitude.shape[0] - 1,
    )
    column_indices = np.clip(
        ((np.arange(full_width) + 0.5) * longitude.shape[1] / full_width).astype(int),
        0,
        longitude.shape[1] - 1,
    )
    latitude_128 = latitude[row_indices][:, column_indices]
    longitude_128 = longitude[row_indices][:, column_indices]
    radar_latitude, radar_longitude = grid_boundary(latitude_128, longitude_128)
    station_map = folium.Map(
        location=[float(stations["latitude"].mean()), float(stations["longitude"].mean())],
        zoom_start=10,
        tiles=None,
        control_scale=True,
    )
    if map_tiles == "openstreetmap":
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(station_map)
    elif map_tiles != "none":
        raise ValueError(f"map_tiles inválido: {map_tiles!r}")
    if municipal_basemap_path is not None:
        if not municipal_basemap_path.is_file():
            raise FileNotFoundError(f"GeoJSON municipal não encontrado: {municipal_basemap_path}")
        municipal_data = json.loads(municipal_basemap_path.read_text(encoding="utf-8"))
        folium.GeoJson(
            municipal_data,
            name="Municipios do IBGE (2024)",
            style_function=lambda _: {
                "color": "#5c677d", "weight": 1, "fillColor": "#d9e2ec", "fillOpacity": 0.25,
            },
            tooltip=folium.GeoJsonTooltip(fields=["NM_MUN"], aliases=["Município"]),
        ).add_to(station_map)
    folium.GeoJson(
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [list(zip(radar_longitude, radar_latitude))]}},
        name="Area da grade do Radar Sumare",
        style_function=lambda _: {"color": "#2171b5", "weight": 2, "fillColor": "#9ecae1", "fillOpacity": 0.08},
    ).add_to(station_map)
    if crop_bounds_128 is not None:
        top, bottom, left, right = crop_bounds_128
        crop_latitude, crop_longitude = grid_boundary(
            latitude_128[top:bottom, left:right], longitude_128[top:bottom, left:right]
        )
        folium.GeoJson(
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [list(zip(crop_longitude, crop_latitude))]}},
            name=f"ROI cropada ({bottom - top} x {right - left})",
            style_function=lambda _: {"color": "#f16913", "weight": 2, "dashArray": "6 4", "fillColor": "#fdae6b", "fillOpacity": 0.12},
        ).add_to(station_map)

    vmax = max(12.5, float(forecast["mock_prediction_mm_15min"].max())) if not forecast.empty else 12.5
    colormap = LinearColormap(colors=["#ffffcc", "#fd8d3c", "#bd0026"], vmin=0, vmax=vmax, caption="Mock de persistencia (mm/15 min)")
    for index, bucket in enumerate(event["target"]):
        timestamp = bucket["timestamp_utc"]
        horizon = forecast.loc[forecast["target_timestamp"] == timestamp]
        layer = folium.FeatureGroup(name=f"Mock T+{15 * (index + 1)} min", show=index == 0).add_to(station_map)
        for station in horizon.itertuples(index=False):
            value = float(station.mock_prediction_mm_15min)
            popup = (
                f"<b>{escape(str(station.station_name))}</b><br>"
                f"ID: {station.station_id}<br>"
                f"Horizonte: T+{15 * (index + 1)} min<br>"
                f"Mock de persistencia: {value:.2f} mm/15 min<br>"
                f"Observacao de origem: {escape(str(station.source_timestamp))}"
            )
            folium.CircleMarker(
                [station.latitude, station.longitude], radius=6, color="#7f0000", weight=1,
                fill=True, fill_color=colormap(value), fill_opacity=0.9,
                tooltip=f"{station.station_name}: {value:.2f} mm/15 min",
                popup=folium.Popup(popup, max_width=320),
            ).add_to(layer)
    names_layer = folium.FeatureGroup(name="Nomes das estacoes", show=show_station_names).add_to(station_map)
    for station in stations.sort_values("station_id").itertuples(index=False):
        folium.Marker(
            [station.latitude, station.longitude],
            icon=folium.DivIcon(html=(
                f'<div style="font-size: 10px; color: #7f0000; white-space: nowrap;">{escape(str(station.nome))}</div>'
            )),
        ).add_to(names_layer)
    colormap.add_to(station_map)
    folium.LayerControl(collapsed=False).add_to(station_map)
    station_map.fit_bounds([[float(stations["latitude"].min()), float(stations["longitude"].min())], [float(stations["latitude"].max()), float(stations["longitude"].max())]])
    station_map.save(output_path)


def run_mock(
    event: dict[str, Any],
    capture_root: Path,
    capture_config: dict[str, Any],
    alertario_root: Path,
    mapping: pd.DataFrame,
    output_dir: Path,
    width: int,
    height: int,
    max_m15: float,
    crop_stations: bool = False,
    crop_margin_pixels: int = 20,
    geographic_grid_path: Path | None = None,
    show_station_names: bool = False,
    map_tiles: str = "none",
    municipal_basemap_path: Path | None = None,
) -> dict[str, Any]:
    input_timestamps = [pd.Timestamp(bucket["timestamp_utc"]) for bucket in event["input"]]
    target_timestamps = [pd.Timestamp(bucket["timestamp_utc"]) for bucket in event["target"]]
    stations = station_positions(mapping, capture_config, height, width)
    frames = aggregate_event_history(event, capture_root, capture_config, width, height)
    crop = None
    if crop_stations:
        frames, stations, crop = crop_station_region(frames, stations, crop_margin_pixels)
    observations = load_event_observations(
        alertario_root, set(stations["station_id"].astype(int)), input_timestamps + target_timestamps, max_m15
    )
    forecast = build_persistence_forecast(observations, input_timestamps, target_timestamps, stations)

    output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output_dir / "retrospective_observations.csv", index=False)
    forecast.to_csv(output_dir / "mock_station_forecast.csv", index=False)
    save_mock_product(frames, event, forecast, output_dir / "mock_product.png", crop)
    if geographic_grid_path is not None:
        render_station_forecast_map(
            forecast, stations, event, geographic_grid_path, height, width, crop,
            output_dir / "mock_station_forecast_map.html", show_station_names, map_tiles, municipal_basemap_path,
        )
    summary = {
        "mode": "mock_station_persistence",
        "disclaimer": "Mock deterministico para demonstracao. Nao usa checkpoint e nao representa previsao de modelo treinado.",
        "dense_rainfall_field": "not_generated: the current model is supervised only at station pixels",
        "event_timestamp_utc": event["event_timestamp_utc"],
        "event_station": event["station_name"],
        "event_m15_mm_15min": event["m15_mm_15min"],
        "input_frames": len(event["input"]),
        "target_horizons": len(event["target"]),
        "stations_with_persistence_forecast": int(forecast["station_id"].nunique()) if not forecast.empty else 0,
        "retrospective_observations": int(len(observations)),
        "crop": {
            "enabled": crop is not None,
            "margin_pixels": crop_margin_pixels if crop is not None else None,
            "bounds_128": list(crop) if crop is not None else None,
            "shape": list(frames.shape[1:3]),
        },
        "map": "mock_station_forecast_map.html" if geographic_grid_path is not None else None,
        "map_tiles": map_tiles if geographic_grid_path is not None else None,
        "municipal_basemap": str(municipal_basemap_path) if municipal_basemap_path is not None else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
