from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from .core import Signal
from .feature_bank import FeatureCollection, TrialSignalProvider


def basic_qc(signal: Signal, *, seconds: float = 10.0) -> dict[str, Any]:
    n = min(signal.data.shape[-1], round(seconds * signal.sampling_rate))
    x = np.asarray(signal.data[..., :n], dtype=np.float64)
    flat = x.reshape((-1, n))
    rms = np.sqrt(np.mean(flat ** 2, axis=1))
    ptp = np.ptp(flat, axis=1)
    warnings = []
    if signal.attrs.get("source_adc") == "SignalGeneratorADC":
        warnings.append("BCI2000 header declares SignalGeneratorADC; treat as synthetic until acquisition provenance proves otherwise.")
    if flat.shape[0] >= 8:
        rms_cv = float(np.std(rms) / np.mean(rms)) if np.mean(rms) else float("nan")
        if rms_cv < 0.02:
            warnings.append("Channel RMS values are unusually homogeneous; inspect for generated or duplicated signals.")
    else:
        rms_cv = None
    return {
        "analyzed_seconds": n / signal.sampling_rate,
        "n_channels": int(flat.shape[0]),
        "finite_fraction": float(np.isfinite(flat).mean()),
        "constant_channels": np.flatnonzero(ptp == 0).tolist(),
        "rms_median_raw": float(np.median(rms)),
        "rms_min_raw": float(np.min(rms)),
        "rms_max_raw": float(np.max(rms)),
        "peak_to_peak_median_raw": float(np.median(ptp)),
        "rms_coefficient_of_variation": rms_cv,
        "warnings": warnings,
    }


@dataclass(frozen=True)
class ChannelQCTable:
    frame: pd.DataFrame


@dataclass(frozen=True)
class SampleQCTable:
    frame: pd.DataFrame


def channel_qc_from_trials(
    provider: TrialSignalProvider,
    *,
    trial_ids: Sequence[str],
    recording_id: str,
    seconds_per_trial: float = 10.0,
    line_frequencies_hz: Sequence[float] = (50.0, 100.0, 150.0, 200.0, 250.0),
    minimum_finite_fraction: float = 0.99,
    maximum_line_noise_ratio: float = 20.0,
    rms_robust_z_limit: float = 8.0,
) -> ChannelQCTable:
    """Fit channel-level QC from training trials only."""
    segments = []
    names = None
    source_indices = None
    fs = provider.signal.sampling_rate
    for trial_id in trial_ids:
        trial_signal, _ = provider.trial_signal(trial_id)
        n = min(trial_signal.data.shape[-1], max(4, round(seconds_per_trial * fs)))
        segments.append(np.asarray(trial_signal.data[..., :n], dtype=np.float64))
        names = np.asarray(trial_signal.coordinate("channel"), dtype=object)
        source_indices = np.asarray(
            trial_signal.attrs.get("source_channel_indices", np.arange(len(names))),
            dtype=int,
        )
    if not segments or names is None or source_indices is None:
        raise ValueError("trial_ids must contain at least one trial")
    x = np.concatenate(segments, axis=-1)
    finite_fraction = np.mean(np.isfinite(x), axis=-1)
    safe = np.where(np.isfinite(x), x, np.nan)
    rms = np.sqrt(np.nanmean(safe ** 2, axis=-1))
    peak_to_peak = np.nanmax(safe, axis=-1) - np.nanmin(safe, axis=-1)
    rms_median = np.nanmedian(rms)
    rms_mad = np.nanmedian(np.abs(rms - rms_median))
    rms_scale = 1.4826 * rms_mad
    rms_robust_z = (
        np.abs(rms - rms_median) / rms_scale
        if rms_scale > 0 else np.zeros_like(rms)
    )
    nperseg = min(x.shape[-1], max(4, round(2 * fs)))
    frequencies, psd = scipy_signal.welch(
        np.nan_to_num(x), fs=fs, axis=-1, nperseg=nperseg, noverlap=nperseg // 2
    )
    ratios = []
    for frequency in line_frequencies_hz:
        if frequency >= fs / 2:
            continue
        line = (frequencies >= frequency - 0.5) & (frequencies <= frequency + 0.5)
        neighbors = (
            ((frequencies >= frequency - 4) & (frequencies <= frequency - 2))
            | ((frequencies >= frequency + 2) & (frequencies <= frequency + 4))
        )
        if line.any() and neighbors.any():
            ratios.append(
                np.mean(psd[:, line], axis=-1)
                / np.maximum(np.median(psd[:, neighbors], axis=-1), np.finfo(float).tiny)
            )
    line_noise_ratio = np.max(np.stack(ratios), axis=0) if ratios else np.zeros(len(names))
    valid = (
        (finite_fraction >= minimum_finite_fraction)
        & (peak_to_peak > 0)
        & (rms_robust_z <= rms_robust_z_limit)
        & (line_noise_ratio <= maximum_line_noise_ratio)
    )
    reasons = []
    for finite, ptp, rz, line_ratio in zip(
        finite_fraction, peak_to_peak, rms_robust_z, line_noise_ratio
    ):
        row_reasons = []
        if finite < minimum_finite_fraction:
            row_reasons.append("nonfinite")
        if ptp <= 0:
            row_reasons.append("constant")
        if rz > rms_robust_z_limit:
            row_reasons.append("rms_outlier")
        if line_ratio > maximum_line_noise_ratio:
            row_reasons.append("line_noise")
        reasons.append(";".join(row_reasons))
    return ChannelQCTable(pd.DataFrame({
        "recording": recording_id,
        "channel_index": np.arange(len(names), dtype=int),
        "source_channel_index": source_indices,
        "channel": names,
        "channel_qc_finite_fraction": finite_fraction,
        "channel_qc_rms": rms,
        "channel_qc_peak_to_peak": peak_to_peak,
        "channel_qc_rms_robust_z": rms_robust_z,
        "channel_qc_line_noise_ratio": line_noise_ratio,
        "channel_qc_valid": valid,
        "channel_qc_reason": reasons,
    }))


def sample_qc_from_features(
    features: FeatureCollection,
    *,
    rms_robust_z_limit: float = 8.0,
) -> SampleQCTable:
    """Flag non-finite features and extreme window RMS without dropping rows."""
    frame = features.frame[["sample_id", "channel_index", *features.feature_columns]].copy()
    finite = np.isfinite(frame.loc[:, features.feature_columns].to_numpy(dtype=float)).all(axis=1)
    if "rms" in frame:
        grouped = frame.groupby("channel_index", observed=True)["rms"]
        median = grouped.transform("median")
        mad = grouped.transform(lambda values: np.median(np.abs(values - np.median(values))))
        scale = 1.4826 * mad
        robust_z = np.where(scale > 0, np.abs(frame.rms - median) / scale, 0.0)
    else:
        robust_z = np.zeros(len(frame))
    valid = finite & (robust_z <= rms_robust_z_limit)
    reasons = np.where(
        ~finite,
        "nonfinite_feature",
        np.where(robust_z > rms_robust_z_limit, "window_rms_outlier", ""),
    )
    return SampleQCTable(pd.DataFrame({
        "sample_id": frame.sample_id,
        "channel_index": frame.channel_index,
        "sample_qc_rms_robust_z": robust_z,
        "sample_qc_valid": valid,
        "sample_qc_reason": reasons,
    }))
