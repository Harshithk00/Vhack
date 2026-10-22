"""Signal-control policies the simulator can run.

The two classical baselines here need no optimisation at all and mirror what is
deployed in most cities today:

* ``FixedTimePolicy`` - pre-timed plan that alternates the green every N slots;
* ``MaxQueuePolicy``  - actuated control: give the green to the longer queue.
"""

from __future__ import annotations

import numpy as np

from .classical import SolveResult
from .formulation import QuboWeights, build_signal_problem
from .network import TrafficNetwork
from .simulator import TrafficState


class FixedTimePolicy:
    def __init__(self, cycle_slots: int = 1):
        self.cycle_slots = cycle_slots
        self.name = "Fixed-time"
        self._slot = 0

    def decide(self, network: TrafficNetwork, state: TrafficState, slot_steps: int) -> np.ndarray:
        phase = (self._slot // self.cycle_slots) % 2
        self._slot += 1
        return np.full(network.n, phase, dtype=int)


class MaxQueuePolicy:
    """Actuated control with hysteresis: switch only if the red queue is clearly longer."""

    def __init__(self, hysteresis: float = 0.5):
        self.hysteresis = hysteresis
        self.name = "Max-queue (actuated)"

    def decide(self, network: TrafficNetwork, state: TrafficState, slot_steps: int) -> np.ndarray:
        phase = state.phase.copy()
        ew_longer = state.q_ew > state.q_ns + self.hysteresis
        ns_longer = state.q_ns > state.q_ew + self.hysteresis
        phase[ew_longer] = 1
        phase[ns_longer] = 0
        return phase


class QuboPolicy:
    """Model-predictive control: each slot, build the signal QUBO from the live queues,
    solve it with any solver (exact, annealing or QAOA) and apply the first slot of the
    plan (receding horizon)."""

    def __init__(self, solver, horizon: int = 2, weights: QuboWeights | None = None, name: str | None = None):
        self.solver = solver
        self.horizon = horizon
        self.weights = weights or QuboWeights()
        self.name = name or f"QUBO + {solver.name}"
        self.history: list[SolveResult] = []

    def decide(self, network: TrafficNetwork, state: TrafficState, slot_steps: int) -> np.ndarray:
        problem = build_signal_problem(network, state, self.horizon, slot_steps, self.weights)
        kwargs = {}
        if getattr(self.solver, "supports_warm_start", False) and self.history:
            # reuse last slot's QAOA angles: traffic changes slowly, so do the landscapes
            kwargs["initial_params"] = self.history[-1].info.get("params")
        result = self.solver.solve(problem.qubo, **kwargs)
        self.history.append(result)
        return problem.decode(result.x)[0]
