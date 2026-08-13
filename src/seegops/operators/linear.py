from __future__ import annotations

from dataclasses import dataclass
import re
import numpy as np

from .base import Operator
from ..core import Signal, ValidationError


@dataclass
class ApplyGain(Operator):
    def apply(self, value: Signal) -> Signal:
        gains = value.attrs.get("gain_per_channel")
        if gains is None:
            raise ValidationError("gain_per_channel metadata is absent")
        if value.attrs.get("gain_applied"):
            raise ValidationError("gain has already been applied")
        axis = value.axis("channel")
        shape = [1] * value.data.ndim
        shape[axis] = len(gains)
        data = value.data.astype(np.float64) * np.asarray(gains).reshape(shape)
        attrs = dict(value.attrs)
        attrs["gain_applied"] = True
        return value.with_step(
            data,
            attrs=attrs,
            unit=value.attrs.get("physical_unit", value.unit),
            operator="apply_gain",
            parameters={},
        )


@dataclass
class BipolarReference(Operator):
    pairs: list[tuple[int, int]] | None = None

    def apply(self, value: Signal) -> Signal:
        axis = value.axis("channel")
        n = value.data.shape[axis]
        pairs = self.pairs or [(i, i + 1) for i in range(n - 1)]
        if any(min(a, b) < 0 or max(a, b) >= n or a == b for a, b in pairs):
            raise ValidationError("invalid bipolar channel pair")
        data = np.stack([np.take(value.data, a, axis=axis) - np.take(value.data, b, axis=axis) for a, b in pairs], axis=axis)
        old_names = value.coordinate("channel")
        names = np.asarray([f"{old_names[a]}-{old_names[b]}" for a, b in pairs])
        coords = dict(value.coords)
        coords["channel"] = names
        attrs = dict(value.attrs)
        matrix = np.zeros((len(pairs), n), dtype=int)
        for row, (a, b) in enumerate(pairs):
            matrix[row, a], matrix[row, b] = 1, -1
        attrs.update({"reference": "bipolar", "reference_pairs": pairs, "reference_matrix": matrix.tolist()})
        return value.with_step(data, coords=coords, attrs=attrs, valid_mask=None, operator="bipolar_reference", parameters={"pairs": pairs})


@dataclass
class CommonAverageReference(Operator):
    channel_indices: list[int] | None = None

    def apply(self, value: Signal) -> Signal:
        axis = value.axis("channel")
        indices = self.channel_indices or list(range(value.data.shape[axis]))
        reference = np.mean(np.take(value.data, indices, axis=axis), axis=axis, keepdims=True)
        data = value.data - reference
        attrs = dict(value.attrs)
        attrs.update({"reference": "common_average", "reference_channels": indices})
        return value.with_step(data, attrs=attrs, valid_mask=None, operator="common_average_reference", parameters={"channel_indices": indices})


@dataclass
class Baseline(Operator):
    interval_s: tuple[float, float]
    mode: str = "subtract"

    def apply(self, value: Signal) -> Signal:
        time = value.coordinate("time")
        selected = (time >= self.interval_s[0]) & (time <= self.interval_s[1])
        if not selected.any():
            raise ValidationError("baseline interval contains no samples")
        axis = value.axis("time")
        baseline = np.mean(np.compress(selected, value.data, axis=axis), axis=axis, keepdims=True)
        if self.mode == "subtract":
            data, unit = value.data - baseline, value.unit
        elif self.mode == "ratio":
            data, unit = value.data / baseline, "ratio"
        elif self.mode == "db":
            if np.any(value.data <= 0) or np.any(baseline <= 0):
                raise ValidationError("dB baseline requires strictly positive power-like data")
            data, unit = 10 * np.log10(value.data / baseline), "dB"
        else:
            raise ValidationError("mode must be subtract, ratio, or db")
        return value.with_step(data, unit=unit, valid_mask=None, operator="baseline", parameters=self.parameters)
