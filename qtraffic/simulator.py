"""Queue-based traffic simulator used to score signal plans.

A fluid-queue model with stochastic (Poisson) arrivals: each step, the green
approach at every intersection discharges up to ``saturation_flow`` vehicles, a
share of them joins the downstream queue, and new vehicles arrive from outside.
The step in which a signal changes loses ``switch_loss`` of its capacity to the
amber / all-red interval.

Signals are re-planned once per *slot* (``slot_steps`` steps, 30 s by default):
a policy looks at the current queues and returns one phase bit per intersection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .network import EW_GREEN, TrafficNetwork


@dataclass
class TrafficState:
    q_ns: np.ndarray
    q_ew: np.ndarray
    phase: np.ndarray  # current phase bit per intersection
    step: int = 0

    def copy(self) -> "TrafficState":
        return TrafficState(self.q_ns.copy(), self.q_ew.copy(), self.phase.copy(), self.step)

    @property
    def total_queue(self) -> float:
        return float(self.q_ns.sum() + self.q_ew.sum())


class Policy(Protocol):
    name: str

    def decide(self, network: TrafficNetwork, state: TrafficState, slot_steps: int) -> np.ndarray: ...


@dataclass
class SimResult:
    policy: str
    scenario: str
    step_seconds: float
    queues: np.ndarray  # (steps, n, 2): NS and EW queue after each step
    phases: np.ndarray  # (slots, n)
    arrivals: float
    exited: float
    total_delay: float  # vehicle-steps spent queueing
    switches: int
    decision_seconds: list[float] = field(default_factory=list)

    @property
    def avg_wait_s(self) -> float:
        """Average queueing delay per vehicle that entered the network (seconds)."""
        return self.total_delay * self.step_seconds / max(self.arrivals, 1.0)

    @property
    def throughput_vph(self) -> float:
        hours = self.queues.shape[0] * self.step_seconds / 3600
        return self.exited / hours

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "scenario": self.scenario,
            "avg_wait_s": round(self.avg_wait_s, 2),
            "throughput_vph": round(float(self.throughput_vph), 1),
            "mean_queue": round(float(self.queues.sum(axis=(1, 2)).mean()), 2),
            "max_queue": round(float(self.queues.max()), 2),
            "switches": self.switches,
            "decision_ms": round(1000 * float(np.mean(self.decision_seconds or [0.0])), 2),
        }


class TrafficSimulator:
    def __init__(self, network: TrafficNetwork, slot_steps: int = 6, seed: int = 0):
        self.network = network
        self.slot_steps = slot_steps
        self.seed = seed

    def initial_state(self, mean_queue: float = 4.0) -> TrafficState:
        rng = np.random.default_rng(self.seed + 10_000)
        n = self.network.n
        return TrafficState(
            q_ns=rng.poisson(mean_queue, n).astype(float),
            q_ew=rng.poisson(mean_queue, n).astype(float),
            phase=rng.integers(0, 2, n),
        )

    def step(self, state: TrafficState, phase: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
        """Advance one step in place. Returns ``(arrivals, exited)``."""
        net = self.network
        capacity = np.full(net.n, net.saturation_flow)
        capacity[phase != state.phase] *= 1.0 - net.switch_loss
        state.phase = phase.copy()

        ew_green = phase == EW_GREEN
        served_ew = np.where(ew_green, np.minimum(state.q_ew, capacity), 0.0)
        served_ns = np.where(~ew_green, np.minimum(state.q_ns, capacity), 0.0)
        state.q_ew -= served_ew
        state.q_ns -= served_ns

        exited = 0.0
        for i in range(net.n):
            for served, downstream, queue in (
                (served_ew[i], net.downstream_ew(i), state.q_ew),
                (served_ns[i], net.downstream_ns(i), state.q_ns),
            ):
                onward = served * net.through_ratio
                if downstream is None:
                    exited += served
                else:
                    queue[downstream] += onward
                    exited += served - onward

        new_ns = rng.poisson(net.arrival_ns).astype(float)
        new_ew = rng.poisson(net.arrival_ew).astype(float)
        state.q_ns += new_ns
        state.q_ew += new_ew
        state.step += 1
        return float(new_ns.sum() + new_ew.sum()), exited

    def run(self, policy: Policy, n_slots: int = 40, state: TrafficState | None = None) -> SimResult:
        state = (state or self.initial_state()).copy()
        rng = np.random.default_rng(self.seed)
        queues, phases, decisions = [], [], []
        arrivals = exited = delay = 0.0
        switches = 0

        for _ in range(n_slots):
            t0 = time.perf_counter()
            phase = np.asarray(policy.decide(self.network, state.copy(), self.slot_steps), dtype=int)
            decisions.append(time.perf_counter() - t0)
            switches += int((phase != state.phase).sum())
            phases.append(phase)
            for _ in range(self.slot_steps):
                a, e = self.step(state, phase, rng)
                arrivals += a
                exited += e
                delay += state.total_queue
                queues.append(np.stack([state.q_ns, state.q_ew], axis=1).copy())

        return SimResult(
            policy=policy.name,
            scenario=self.network.name,
            step_seconds=self.network.step_seconds,
            queues=np.array(queues),
            phases=np.array(phases),
            arrivals=arrivals,
            exited=exited,
            total_delay=delay,
            switches=switches,
            decision_seconds=decisions,
        )
