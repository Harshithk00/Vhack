"""Signal-control policies the simulator can run.

The two classical baselines here need no optimisation at all and mirror what is
deployed in most cities today:

* ``FixedTimePolicy`` - pre-timed plan that alternates the green every N slots;
* ``MaxQueuePolicy``  - actuated control: give the green to the longer queue.
"""

from __future__ import annotations

import numpy as np

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
