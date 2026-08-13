from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from ..core import ProvenanceStep, Signal, ValidationError, file_fingerprint


_DTYPES = {"int16": "<i2", "int32": "<i4", "float32": "<f4"}


@dataclass(frozen=True)
class StateDefinition:
    name: str
    bits: int
    initial: int
    byte_offset: int
    bit_offset: int


@dataclass
class BCI2000Recording:
    path: Path
    header: str
    parameters: dict[str, str]
    states_def: dict[str, StateDefinition]
    signal: Signal
    raw_states: np.ndarray

    def state(self, name: str) -> np.ndarray:
        """Decode a BCI2000 state into one unsigned integer per sample."""
        if name not in self.states_def:
            raise KeyError(f"unknown state {name!r}; available={sorted(self.states_def)}")
        d = self.states_def[name]
        start_bit = d.byte_offset * 8 + d.bit_offset
        out = np.zeros(self.raw_states.shape[0], dtype=np.uint64)
        for k in range(d.bits):
            absolute = start_bit + k
            byte = absolute // 8
            bit = absolute % 8
            out |= ((self.raw_states[:, byte] >> bit) & 1).astype(np.uint64) << k
        return out

    @property
    def storage_time(self) -> datetime | None:
        raw = self.parameters.get("StorageTime")
        try:
            return datetime.fromisoformat(raw) if raw else None
        except ValueError:
            return None

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "shape": list(self.signal.data.shape),
            "sampling_rate_hz": self.signal.sampling_rate,
            "duration_s": self.signal.data.shape[-1] / self.signal.sampling_rate,
            "dtype": str(self.signal.data.dtype),
            "unit": self.signal.unit,
            "storage_time": self.storage_time.isoformat() if self.storage_time else None,
            "states": sorted(self.states_def),
            "fingerprint": self.signal.attrs["source_fingerprint"],
        }


def _read_header(path: Path) -> tuple[str, int]:
    with path.open("rb") as f:
        prefix = f.read(256)
        match = re.search(rb"HeaderLen=\s*(\d+)", prefix)
        if not match:
            raise ValidationError(f"not a supported BCI2000 file: {path}")
        length = int(match.group(1))
        f.seek(0)
        return f.read(length).decode("latin1"), length


def _value(header: str, name: str) -> str | None:
    pattern = rf"\b{name}=\s*([^\r\n]+)"
    m = re.search(pattern, header)
    if not m:
        return None
    return m.group(1).split("//", 1)[0].strip().split()[0]


def _parse_states(header: str) -> dict[str, StateDefinition]:
    section = header.split("[ State Vector Definition ]", 1)[1].split("[ Parameter Definition ]", 1)[0]
    result: dict[str, StateDefinition] = {}
    for line in section.replace("\r", "").splitlines():
        parts = line.split()
        if len(parts) == 5:
            try:
                result[parts[0]] = StateDefinition(parts[0], *map(int, parts[1:]))
            except ValueError:
                pass
    return result


def _parse_parameters(header: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for line in header.replace("\r", "").splitlines():
        body = line.split("//", 1)[0]
        m = re.search(r"\b([A-Za-z_/][A-Za-z0-9_/]*)=\s*([^\s]+)", body)
        if m:
            params[m.group(1).lstrip("/")] = m.group(2).replace("%20", " ")
    return params


def inspect_bci2000(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    header, header_len = _read_header(p)
    source_ch = int(re.search(r"SourceCh=\s*(\d+)", header).group(1))
    state_len = int(re.search(r"StatevectorLen=\s*(\d+)", header).group(1))
    fmt_match = re.search(r"DataFormat=\s*(\w+)", header[:256])
    data_format = fmt_match.group(1) if fmt_match else {"0": "int16", "1": "float32", "2": "int32"}.get(_value(header, "SignalType") or "0", "int16")
    sampling_text = _value(header, "SamplingRate") or "0"
    sampling_rate = float(re.sub(r"[^0-9.eE+-]", "", sampling_text))
    itemsize = np.dtype(_DTYPES[data_format]).itemsize
    frame_bytes = source_ch * itemsize + state_len
    payload = p.stat().st_size - header_len
    if payload % frame_bytes:
        raise ValidationError("payload size is not divisible by BCI2000 sample frame size")
    n_samples = payload // frame_bytes
    return {
        "path": str(p.resolve()),
        "header_len": header_len,
        "source_ch": source_ch,
        "statevector_len": state_len,
        "data_format": data_format,
        "sampling_rate": sampling_rate,
        "n_samples": n_samples,
        "duration_s": n_samples / sampling_rate,
        "header": header,
        "parameters": _parse_parameters(header),
        "states_def": _parse_states(header),
    }


def _gains(header: str, n: int) -> tuple[np.ndarray, str]:
    m = re.search(r"SourceChGain=\s*\d+\s+([^\r\n/]+)", header)
    if not m:
        return np.ones(n), "raw"
    tokens = m.group(1).split()
    values, units = [], []
    for token in tokens[:n]:
        g = re.match(r"([-+0-9.eE]+)(.*)", token)
        values.append(float(g.group(1)))
        units.append(g.group(2) or "raw")
    if len(values) != n:
        return np.ones(n), "raw"
    unit = units[0] if len(set(units)) == 1 else "mixed"
    return np.asarray(values), unit


def read_bci2000(path: str | Path, *, channel_names: Iterable[str] | None = None) -> BCI2000Recording:
    """Memory-map BCI2000 data without copying the recording into RAM."""
    info = inspect_bci2000(path)
    p = Path(path)
    n_ch, n = info["source_ch"], info["n_samples"]
    sample_dtype = np.dtype(_DTYPES[info["data_format"]])
    record_dtype = np.dtype([
        ("signal", sample_dtype, (n_ch,)),
        ("state", "u1", (info["statevector_len"],)),
    ])
    records = np.memmap(p, mode="r", dtype=record_dtype, offset=info["header_len"], shape=(n,))
    raw = records["signal"].T
    gains, unit = _gains(info["header"], n_ch)
    names = list(channel_names or [f"CH{i + 1:03d}" for i in range(n_ch)])
    if len(names) != n_ch:
        raise ValidationError("channel_names length differs from SourceCh")
    attrs = {
        "source_format": "BCI2000",
        "source_path": str(p.resolve()),
        "source_fingerprint": file_fingerprint(p),
        "storage_time": info["parameters"].get("StorageTime"),
        "subject_run": info["parameters"].get("SubjectRun"),
        "source_module": info["parameters"].get("ModuleName"),
        "source_adc": "SignalGeneratorADC" if "SignalGeneratorADC" in info["header"] else "unknown",
        "gain_per_channel": gains.tolist(),
        "physical_unit": unit,
        "gain_applied": False,
        "online_reference": "unknown",
    }
    step = ProvenanceStep("read_bci2000", "0.1.0", {"path": str(p.resolve()), "memory_map": True})
    signal = Signal(
        data=raw,
        dims=("channel", "time"),
        coords={"channel": np.asarray(names), "time": np.arange(n) / info["sampling_rate"]},
        sampling_rate=info["sampling_rate"],
        unit="ADU",
        attrs=attrs,
        provenance=(step,),
    )
    return BCI2000Recording(p, info["header"], info["parameters"], info["states_def"], signal, records["state"])
