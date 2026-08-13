"""Composable, provenance-aware operators for sEEG data."""

from .core import Feature, Signal, ValidationError
from .pipeline import Pipeline
from .tables import Event, EventTable, Trial, TrialTable

__all__ = [
    "Feature", "Signal", "ValidationError", "Pipeline",
    "Event", "EventTable", "Trial", "TrialTable",
]
__version__ = "0.1.0"
