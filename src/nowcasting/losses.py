from __future__ import annotations

import torch
from torch import nn


PRECIPITATION_THRESHOLDS_MM15 = (1.25, 6.25, 12.5)


def _masked_mean(error: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    return (error * mask).sum() / (mask.sum() + eps)


def _huber(error: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = error.abs()
    return torch.where(
        absolute < delta,
        0.5 * absolute.square(),
        delta * (absolute - 0.5 * delta),
    )


def _precipitation_weights(target_log: torch.Tensor, class_weights: tuple[float, ...]):
    target_mm15 = torch.expm1(target_log)
    weights = torch.full_like(target_mm15, class_weights[0])
    for threshold, weight in zip(PRECIPITATION_THRESHOLDS_MM15, class_weights[1:]):
        weights = torch.where(target_mm15 >= threshold, weight, weights)
    return weights


class MaskedMAELoss(nn.Module):
    def forward(self, prediction, target, mask):
        return _masked_mean((prediction - target).abs(), mask, 1e-6)


class MaskedHuberLoss(nn.Module):
    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta

    def forward(self, prediction, target, mask):
        return _masked_mean(_huber(prediction - target, self.delta), mask, 1e-6)


class WeightedMaskedMAELoss(nn.Module):
    def __init__(self, class_weights=(1.0, 5.0, 10.0, 20.0)):
        super().__init__()
        self.class_weights = _validate_weights(class_weights)

    def forward(self, prediction, target, mask):
        weights = _precipitation_weights(target, self.class_weights) * mask
        return (weights * (prediction - target).abs()).sum() / (weights.sum() + 1e-6)


class WeightedMaskedHuberLoss(WeightedMaskedMAELoss):
    def __init__(self, delta: float = 0.1, class_weights=(1.0, 5.0, 10.0, 20.0)):
        super().__init__(class_weights)
        self.delta = delta

    def forward(self, prediction, target, mask):
        weights = _precipitation_weights(target, self.class_weights) * mask
        return (weights * _huber(prediction - target, self.delta)).sum() / (weights.sum() + 1e-6)


def _validate_weights(values) -> tuple[float, float, float, float]:
    weights = tuple(float(value) for value in values)
    if len(weights) != 4 or any(value <= 0 for value in weights):
        raise ValueError("São necessários quatro pesos positivos: fraca, moderada, forte, extrema.")
    return weights
