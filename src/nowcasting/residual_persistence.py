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


class ConditionalLateFusionForecaster(nn.Module):
    """Fuse independent radar and GOES corrections over causal persistence.

    The C13 branch never receives RGB channels and the radar branch never
    receives C13.  Their corrections are combined only at the forecast layer.
    A learned gate is largest where the latest RGB frame has no visual echo,
    which expresses the experimental hypothesis that infrared cloud-top
    temperature is most useful when local radar evidence is weak.
    """

    def __init__(
        self, radar_core: nn.Module, goes_core: nn.Module, *, radar_channels: int,
        goes_channels: int, horizons: int, output_channels: int = 1, goes_branch_control: bool = False,
    ):
        super().__init__()
        if radar_channels <= 0 or goes_channels <= 0 or horizons <= 0 or output_channels <= 0:
            raise ValueError("Configuração de fusão condicional inválida.")
        self.radar_core = radar_core
        self.goes_core = goes_core
        self.radar_channels = radar_channels
        self.goes_channels = goes_channels
        self.horizons = horizons
        self.output_channels = output_channels
        self.goes_branch_control = goes_branch_control
        # softplus(raw_slope) keeps the gate monotonically smaller when echo is present.
        self.gate_intercept = nn.Parameter(torch.tensor(0.0))
        self.gate_raw_slope = nn.Parameter(torch.tensor(0.54132485))  # softplus ~= 1

    def goes_gate(self, radar: torch.Tensor) -> torch.Tensor:
        """Return a [batch, 1, 1, height, width] C13 gate from the last RGB frame."""
        if radar.ndim != 5 or radar.shape[1] != self.radar_channels:
            raise ValueError("Entrada RGB incompatível com a porta condicional.")
        latest = radar[:, :, -1]
        echo_present = latest.amax(dim=1, keepdim=True).gt(0).to(radar.dtype)
        slope = torch.nn.functional.softplus(self.gate_raw_slope)
        return torch.sigmoid(self.gate_intercept - slope * echo_present).unsqueeze(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected_channels = self.radar_channels + self.goes_channels + 2
        if inputs.ndim != 5 or inputs.shape[1] != expected_channels:
            raise ValueError("Entrada condicional deve conter RGB, GOES, valores e máscara de estações.")
        radar = inputs[:, :self.radar_channels]
        goes_end = self.radar_channels + self.goes_channels
        goes = inputs[:, self.radar_channels:goes_end]
        if self.goes_branch_control:
            goes = torch.zeros_like(goes)
        radar_residual = self.radar_core(radar)
        goes_residual = self.goes_core(goes)
        baseline = persistence_from_station_channels(
            inputs[:, -2], inputs[:, -1], self.horizons, self.output_channels,
        )
        if radar_residual.shape != baseline.shape or goes_residual.shape != baseline.shape:
            raise ValueError("Correções RGB/C13 incompatíveis com a persistência causal.")
        return baseline + radar_residual + self.goes_gate(radar) * goes_residual


def uses_persistence_baseline(forecast_formulation: str) -> bool:
    return forecast_formulation in {"residual-persistence", "conditional-late-fusion"}


def build_forecaster(
    constructor: Callable, sample_x: torch.Tensor, sample_y: torch.Tensor, *, num_layers: int,
    hidden_dim: int, kernel_size: int, device: torch.device, step: int,
    forecast_formulation: str, input_stations: bool, input_goes: bool = False,
    conditional_goes_control: bool = False,
) -> nn.Module:
    """Construct direct or residual-persistence forecasters from a dataset sample."""
    if forecast_formulation == "direct":
        return constructor(
            (1, *sample_x.shape), num_layers, hidden_dim, kernel_size, device, 0.0, step,
            output_channels=sample_y.shape[0],
        )
    if forecast_formulation == "conditional-late-fusion":
        if not input_stations:
            raise ValueError("--forecast-formulation conditional-late-fusion requer --input-stations.")
        if not input_goes:
            raise ValueError("--forecast-formulation conditional-late-fusion requer --input-goes.")
        radar_channels = 3
        goes_channels = sample_x.shape[0] - radar_channels - 2
        if goes_channels <= 0:
            raise ValueError("Fusão condicional requer ao menos um canal GOES.")
        radar_core = constructor(
            (1, radar_channels, *sample_x.shape[1:]), num_layers, hidden_dim, kernel_size, device, 0.0, step,
            output_channels=sample_y.shape[0],
        )
        goes_core = constructor(
            (1, goes_channels, *sample_x.shape[1:]), num_layers, hidden_dim, kernel_size, device, 0.0, step,
            output_channels=sample_y.shape[0],
        )
        return ConditionalLateFusionForecaster(
            radar_core, goes_core, radar_channels=radar_channels, goes_channels=goes_channels,
            horizons=sample_y.shape[1], output_channels=sample_y.shape[0],
            goes_branch_control=conditional_goes_control,
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
