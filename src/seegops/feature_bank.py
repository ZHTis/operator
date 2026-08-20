from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from .core import Signal, ValidationError
from .pipeline import Pipeline
from .sampling import Sample, SampleTable
from .tables import TrialTable


@dataclass(frozen=True)
class BandFeatureSpec:
    name: str
    fmin: float
    fmax: float
    minimum_cycles: float | None = 4.0

    def __post_init__(self) -> None:
        if not self.name or not (0 <= self.fmin < self.fmax):
            raise ValidationError("invalid band feature specification")


@dataclass
class TrialSignalProvider:
    """Lazily preprocess and cache one trial, then serve all of its samples."""

    signal: Signal
    trials: TrialTable | None = None
    preprocessing: Pipeline | None = None
    _cached_trial_id: str | None = field(default=None, init=False, repr=False)
    _cached_signal: Signal | None = field(default=None, init=False, repr=False)

    def _slice_signal(self, start: int, stop: int) -> Signal:
        if self.signal.axis("time") != self.signal.data.ndim - 1:
            raise ValidationError("TrialSignalProvider requires time as the last dimension")
        data = np.asarray(self.signal.data[..., start:stop])
        coords = dict(self.signal.coords)
        coords["time"] = np.arange(stop - start) / self.signal.sampling_rate
        attrs = dict(self.signal.attrs)
        attrs.update({"source_start_sample": start, "source_stop_sample_exclusive": stop})
        return Signal(
            data=data,
            dims=self.signal.dims,
            coords=coords,
            sampling_rate=self.signal.sampling_rate,
            unit=self.signal.unit,
            attrs=attrs,
            valid_mask=None,
            provenance=self.signal.provenance,
        )

    def trial_signal(self, trial_id: str | None) -> tuple[Signal, int]:
        if trial_id is None or self.trials is None:
            start, stop = 0, self.signal.data.shape[-1]
            cache_id = "__recording__"
        else:
            trial = self.trials.by_id(trial_id)
            start = round(trial.onset_s * self.signal.sampling_rate)
            stop = round(trial.offset_s * self.signal.sampling_rate)
            cache_id = trial_id
        if cache_id != self._cached_trial_id:
            value = self._slice_signal(start, stop)
            if self.preprocessing is not None:
                processed = self.preprocessing.run(value)
                if not isinstance(processed, Signal):
                    raise ValidationError("preprocessing pipeline must return Signal")
                value = processed
            self._cached_trial_id = cache_id
            self._cached_signal = value
        assert self._cached_signal is not None
        return self._cached_signal, start

    def segment(self, sample: Sample) -> Signal:
        trial_signal, global_start = self.trial_signal(sample.trial_id)
        local_start = sample.start_sample - global_start
        local_stop = sample.stop_sample_exclusive - global_start
        if local_start < 0 or local_stop > trial_signal.data.shape[-1]:
            raise ValidationError(f"sample {sample.sample_id} lies outside its trial signal")
        return self._slice_from_trial(trial_signal, local_start, local_stop, sample)

    @staticmethod
    def _slice_from_trial(value: Signal, start: int, stop: int, sample: Sample) -> Signal:
        data = np.asarray(value.data[..., start:stop])
        coords = dict(value.coords)
        coords["time"] = np.arange(stop - start) / value.sampling_rate
        attrs = dict(value.attrs)
        attrs.update({"sample_id": sample.sample_id, "sample_kind": sample.sample_kind})
        return Signal(
            data=data,
            dims=value.dims,
            coords=coords,
            sampling_rate=value.sampling_rate,
            unit=value.unit,
            attrs=attrs,
            provenance=value.provenance,
        )


@dataclass(frozen=True)
class FeatureCollection:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    sample_table: SampleTable
    specifications: Mapping[str, Any]

    def select_samples(self, samples: SampleTable) -> "FeatureCollection":
        allowed = {sample.sample_id for sample in samples.samples}
        frame = self.frame.loc[self.frame.sample_id.isin(allowed)].copy()
        if set(frame.sample_id) != allowed:
            raise ValidationError("selected FeatureCollection is missing samples")
        return FeatureCollection(frame, self.feature_columns, samples, self.specifications)


@dataclass(frozen=True)
class FeatureBank:
    """Branching channel-wise features with a shared Welch PSD per sample."""

    time_features: tuple[str, ...] = ("rms", "standard_deviation", "mad", "line_length")
    bands: tuple[BandFeatureSpec, ...] = ()
    welch_segment_s: float = 1.0
    allow_channel_reduction: bool = False

    def _validate_signal(self, value: Signal) -> None:
        if "channel" not in value.dims:
            raise ValidationError("FeatureBank requires a channel dimension")
        if value.axis("time") != value.data.ndim - 1:
            raise ValidationError("FeatureBank requires time as the last dimension")
        if value.axis("channel") != 0:
            raise ValidationError("FeatureBank currently requires channel as the first dimension")
        unknown = set(self.time_features) - {"rms", "standard_deviation", "mad", "line_length"}
        if unknown:
            raise ValidationError(f"unknown time features: {sorted(unknown)}")

    def transform(self, provider: TrialSignalProvider, samples: SampleTable) -> FeatureCollection:
        if samples.sampling_rate != provider.signal.sampling_rate:
            raise ValidationError("SampleTable and Signal sampling rates differ")
        channel_names = np.asarray(provider.signal.coordinate("channel"), dtype=object)
        rows: list[pd.DataFrame] = []
        feature_columns = list(self.time_features) + [f"log_power_{band.name}" for band in self.bands]
        for sample in samples.samples:
            value = provider.segment(sample)
            self._validate_signal(value)
            x = np.asarray(value.data, dtype=np.float64)
            n_channels, n_time = x.shape
            names = np.asarray(value.coordinate("channel"), dtype=object)
            if len(names) != n_channels:
                raise ValidationError("channel coordinate length mismatch")
            data: dict[str, Any] = {
                "sample_id": np.repeat(sample.sample_id, n_channels),
                "channel_index": np.arange(n_channels, dtype=int),
                "source_channel_index": np.asarray(
                    value.attrs.get("source_channel_indices", np.arange(n_channels)),
                    dtype=int,
                ),
                "channel": names,
            }
            centered = x - np.mean(x, axis=-1, keepdims=True)
            if "rms" in self.time_features:
                data["rms"] = np.sqrt(np.mean(centered ** 2, axis=-1))
            if "standard_deviation" in self.time_features:
                data["standard_deviation"] = np.std(x, axis=-1)
            if "mad" in self.time_features:
                medians = np.median(x, axis=-1, keepdims=True)
                data["mad"] = np.median(np.abs(x - medians), axis=-1)
            if "line_length" in self.time_features:
                data["line_length"] = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)
            if self.bands:
                duration_s = n_time / value.sampling_rate
                nperseg = min(n_time, max(4, round(self.welch_segment_s * value.sampling_rate)))
                frequencies, psd = scipy_signal.welch(
                    x,
                    fs=value.sampling_rate,
                    axis=-1,
                    nperseg=nperseg,
                    noverlap=nperseg // 2,
                    detrend="constant",
                    scaling="density",
                )
                for band in self.bands:
                    column = f"log_power_{band.name}"
                    if band.fmax > value.sampling_rate / 2:
                        raise ValidationError(f"{band.name} exceeds Nyquist")
                    if band.minimum_cycles is not None and duration_s * band.fmin < band.minimum_cycles:
                        data[column] = np.full(n_channels, np.nan)
                        continue
                    selected = (frequencies >= band.fmin) & (frequencies <= band.fmax)
                    if selected.sum() < 2:
                        data[column] = np.full(n_channels, np.nan)
                        continue
                    power = np.trapezoid(psd[:, selected], frequencies[selected], axis=-1)
                    data[column] = np.log10(np.maximum(power, np.finfo(float).tiny))
            frame = pd.DataFrame(data)
            if not self.allow_channel_reduction and len(frame) != len(names):
                raise ValidationError("a feature branch removed the channel dimension")
            rows.append(frame)
        result = pd.concat(rows, ignore_index=True)
        if set(result.sample_id) != {sample.sample_id for sample in samples.samples}:
            raise ValidationError("FeatureBank output is missing samples")
        return FeatureCollection(
            frame=result,
            feature_columns=tuple(feature_columns),
            sample_table=samples,
            specifications={
                "time_features": list(self.time_features),
                "bands": [band.__dict__ for band in self.bands],
                "welch_segment_s": self.welch_segment_s,
                "allow_channel_reduction": self.allow_channel_reduction,
            },
        )


@dataclass(frozen=True)
class TargetCollection:
    frame: pd.DataFrame
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class ForceTargetBank:
    targets: tuple[str, ...] = ("force_mean", "force_slope", "force_abs_slope")

    def transform(self, force: Signal, samples: SampleTable) -> TargetCollection:
        if force.dims != ("time",):
            raise ValidationError("ForceTargetBank requires a one-dimensional time Signal")
        if force.sampling_rate != samples.sampling_rate:
            raise ValidationError("force and SampleTable sampling rates differ")
        rows = []
        for sample in samples.samples:
            values = np.asarray(
                force.data[sample.start_sample:sample.stop_sample_exclusive], dtype=float
            )
            finite = np.isfinite(values)
            row: dict[str, Any] = {"sample_id": sample.sample_id}
            if finite.sum() < 3:
                mean = slope = absolute_slope = np.nan
            else:
                y = values[finite]
                t = np.flatnonzero(finite) / force.sampling_rate
                centered_t = t - t.mean()
                denominator = np.dot(centered_t, centered_t)
                mean = float(y.mean())
                slope = float(np.dot(centered_t, y - mean) / denominator) if denominator else np.nan
                absolute_slope = abs(slope)
            values_by_name = {
                "force_mean": mean,
                "force_slope": slope,
                "force_abs_slope": absolute_slope,
            }
            for target in self.targets:
                if target not in values_by_name:
                    raise ValidationError(f"unknown force target {target!r}")
                row[target] = values_by_name[target]
            rows.append(row)
        return TargetCollection(pd.DataFrame(rows), self.targets)
