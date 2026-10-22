"""qtraffic - quantum-assisted traffic signal optimization with QAOA."""

from .network import EW_GREEN, NS_GREEN, TrafficNetwork
from .policies import FixedTimePolicy, MaxQueuePolicy
from .scenarios import SCENARIOS, make_scenario
from .simulator import SimResult, TrafficSimulator, TrafficState

__all__ = [
    "EW_GREEN",
    "NS_GREEN",
    "SCENARIOS",
    "FixedTimePolicy",
    "MaxQueuePolicy",
    "SimResult",
    "TrafficNetwork",
    "TrafficSimulator",
    "TrafficState",
    "make_scenario",
]

__version__ = "0.1.0"
