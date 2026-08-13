from __future__ import annotations

from typing import Any
import numpy as np

from .core import Signal


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
