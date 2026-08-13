from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import Operator
from ..core import Feature, ProvenanceStep, Signal, ValidationError


@dataclass
class FFTPowerSpectrum(Operator):
    dim: str = "time"
    detrend: bool = True
    window: str = "hann"

    def apply(self, value: Signal) -> Signal:
        axis = value.axis(self.dim)
        n = value.data.shape[axis]
        if n < 4:
            raise ValidationError("at least four samples are required for a spectrum")
        x = value.data.astype(np.float64)
        if self.detrend:
            x = x - np.mean(x, axis=axis, keepdims=True)
        if self.window == "hann":
            w = np.hanning(n)
        elif self.window == "boxcar":
            w = np.ones(n)
        else:
            raise ValidationError("window must be hann or boxcar")
        shape = [1] * x.ndim
        shape[axis] = n
        tapered = x * w.reshape(shape)
        fft = np.fft.rfft(tapered, axis=axis)
        scale = value.sampling_rate * np.sum(w ** 2)
        power = np.abs(fft) ** 2 / scale
        frequencies = np.fft.rfftfreq(n, 1 / value.sampling_rate)
        dims = list(value.dims)
        dims[axis] = "frequency"
        coords = {k: v for k, v in value.coords.items() if k != self.dim}
        coords["frequency"] = frequencies
        return value.with_step(power, dims=dims, coords=coords, unit=f"{value.unit}^2/Hz", valid_mask=None, operator="fft_power_spectrum", parameters=self.parameters)


@dataclass
class BandPower(Operator):
    fmin: float
    fmax: float
    reduction: str = "mean"
    require_cycles: float | None = None
    source_duration_s: float | None = None

    def apply(self, value: Signal) -> Feature:
        axis = value.axis("frequency")
        freq = value.coordinate("frequency")
        if not (0 <= self.fmin < self.fmax <= value.sampling_rate / 2):
            raise ValidationError("band must lie inside [0, Nyquist]")
        selected = (freq >= self.fmin) & (freq <= self.fmax)
        if selected.sum() < 1:
            raise ValidationError("frequency grid contains no bin in requested band")
        if self.require_cycles is not None:
            duration = self.source_duration_s
            if duration is None:
                raise ValidationError("source_duration_s is required when enforcing minimum cycles")
            if duration * self.fmin < self.require_cycles:
                raise ValidationError(
                    f"window contains {duration * self.fmin:.2f} cycles at {self.fmin:g} Hz; "
                    f"requires at least {self.require_cycles:g}"
                )
        subset = np.compress(selected, value.data, axis=axis)
        if self.reduction == "mean":
            data = np.mean(subset, axis=axis)
        elif self.reduction == "integral":
            data = np.trapz(subset, x=freq[selected], axis=axis)
        else:
            raise ValidationError("reduction must be mean or integral")
        dims = tuple(d for d in value.dims if d != "frequency")
        coords = {d: value.coords[d] for d in dims if d in value.coords}
        step = ProvenanceStep("band_power", "0.1.0", self.parameters)
        return Feature(data, dims, coords, value.unit, "band_power", {"fmin": self.fmin, "fmax": self.fmax}, value.provenance + (step,))

