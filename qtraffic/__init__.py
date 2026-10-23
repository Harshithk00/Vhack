"""qtraffic - quantum-assisted traffic signal optimization with QAOA."""

from .classical import ExactSolver, SimulatedAnnealingSolver, SolveResult
from .formulation import QuboWeights, SignalProblem, build_signal_problem
from .network import EW_GREEN, NS_GREEN, TrafficNetwork
from .policies import FixedTimePolicy, MaxQueuePolicy, QuboPolicy
from .qaoa import QAOAConfig, QAOASolver
from .qubo import QUBO
from .scenarios import SCENARIOS, make_scenario
from .simulator import SimResult, TrafficSimulator, TrafficState

__all__ = [
    "EW_GREEN",
    "NS_GREEN",
    "QUBO",
    "SCENARIOS",
    "ExactSolver",
    "FixedTimePolicy",
    "MaxQueuePolicy",
    "QAOAConfig",
    "QAOASolver",
    "QuboPolicy",
    "QuboWeights",
    "SignalProblem",
    "SimulatedAnnealingSolver",
    "SolveResult",
    "build_signal_problem",
    "SimResult",
    "TrafficNetwork",
    "TrafficSimulator",
    "TrafficState",
    "make_scenario",
]

__version__ = "0.1.1"
