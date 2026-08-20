"""Composable, provenance-aware operators for sEEG data."""

from .core import Feature, Signal, ValidationError
from .feature_bank import (
    BandFeatureSpec,
    FeatureBank,
    FeatureCollection,
    ForceTargetBank,
    TargetCollection,
    TrialSignalProvider,
)
from .feature_table import FeatureTable, FeatureTableAssembler
from .pipeline import Pipeline
from .qc import ChannelQCTable, SampleQCTable
from .sampling import ContinuousWindowSampler, EventLockedSampler, Sample, SampleTable
from .tables import Event, EventTable, Trial, TrialTable

__all__ = [
    "Feature", "Signal", "ValidationError", "Pipeline",
    "Event", "EventTable", "Trial", "TrialTable",
    "Sample", "SampleTable", "ContinuousWindowSampler", "EventLockedSampler",
    "BandFeatureSpec", "FeatureBank", "FeatureCollection",
    "ForceTargetBank", "TargetCollection", "TrialSignalProvider",
    "FeatureTable", "FeatureTableAssembler", "ChannelQCTable", "SampleQCTable",
]
__version__ = "0.1.0"
