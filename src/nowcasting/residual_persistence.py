"""Causal persistence skip connection for station-supervised radar forecasts."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def persistence_from_station_channels(
    values: torch.Tensor, masks: torch.Tensor, horizons: int, output_channels: int = 1,
) -> torch.Tensor:
    """Repeat each pixel's last causally observed station value over horizons.

    ``values`` and ``masks`` have shape ``[batch, time, height, width]``.
    Pixels with no valid history receive zero, matching the B1 convention. The
    values stay in the dataset target space, currently log1p(mm/15 min).
    """
    if values.ndim != 4 or masks.shape != values.shape:
        raise ValueError("Valores e máscaras de estações devem ter shape [batch, tempo, altura, largura].")
    if horizons <= 0 or output_channels <= 0:
        raise ValueError("Horizontes e canais de saída devem ser positivos.")
    baseline = torch.zeros_like(values[:, 0])
    for time_index in range(values.shape[1]):
        baseline = torch.where(masks[:, time_index].bool(), values[:, time_index], baseline)
    return baseline[:, None, None].expand(-1, output_channels, horizons, -1, -1)


class ResidualPersistenceForecaster(nn.Module):
    """Forecast as B1 station persistence plus an STConvS2S radar correction."""

    def __init__(self, core: nn.Module, nonstation_channels: int, horizons: int, output_channels: int = 1):
        super().__init__()
        if nonstation_channels <= 0 or horizons <= 0 or output_channels <= 0:
            raise ValueError("Configuração residual inválida.")
        self.core = core
        self.nonstation_channels = nonstation_channels
        self.horizons = horizons
        self.output_channels = output_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.nonstation_channels + 2:
            raise ValueError("Entrada residual deve conter canais não-estação, valores e máscara de estações.")
        residual = self.core(inputs[:, :self.nonstation_channels])
        baseline = persistence_from_station_channels(
            inputs[:, -2], inputs[:, -1], self.horizons, self.output_channels,
        )
        if residual.shape != baseline.shape:
            raise ValueError(
                f"Correção STConvS2S {tuple(residual.shape)} incompatível com persistência {tuple(baseline.shape)}."
            )
        return baseline + residual


def build_forecaster(
    constructor: Callable, sample_x: torch.Tensor, sample_y: torch.Tensor, *, num_layers: int,
    hidden_dim: int, kernel_size: int, device: torch.device, step: int,
    forecast_formulation: str, input_stations: bool,
) -> nn.Module:
    """Construct direct or residual-persistence forecasters from a dataset sample."""
    if forecast_formulation == "direct":
        return constructor(
            (1, *sample_x.shape), num_layers, hidden_dim, kernel_size, device, 0.0, step,
            output_channels=sample_y.shape[0],
        )
    if forecast_formulation != "residual-persistence":
        raise ValueError(f"Formulação de previsão desconhecida: {forecast_formulation!r}.")
    if not input_stations:
        raise ValueError("--forecast-formulation residual-persistence requer --input-stations.")
    nonstation_channels = sample_x.shape[0] - 2
    core = constructor(
        (1, nonstation_channels, *sample_x.shape[1:]), num_layers, hidden_dim, kernel_size, device, 0.0, step,
        output_channels=sample_y.shape[0],
    )
    return ResidualPersistenceForecaster(
        core, nonstation_channels=nonstation_channels, horizons=sample_y.shape[1], output_channels=sample_y.shape[0],
    )
