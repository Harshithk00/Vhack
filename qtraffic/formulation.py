"""Traffic signal timing as a QUBO.

Decision variables: ``x[i, t] = 1`` if intersection ``i`` shows east-west green in
slot ``t`` of a short look-ahead horizon (0 = north-south green). With ``N``
intersections and horizon ``T`` the problem has ``N * T`` binary variables, i.e.
``N * T`` qubits.

Each queue at the end of slot ``t`` is predicted with a linear fluid model::

    Q_ew[i, t] = Q0_ew[i] + (t+1) L lam_ew[i]            # current queue + arrivals
               - c   * sum_{s<=t} x[i, s]                  # served on our green
               + r c^ * sum_{s<=t} x[up_ew(i), s]          # platoons released upstream

and the mirror image for north-south with ``1 - x``. Here ``L`` is the slot length
in steps, ``c`` the green capacity per slot, ``r`` the through ratio and ``c^`` the
capacity the upstream signal can actually use (capped by its queue).

Objective::

    E(x) = w_q * sum_{i,t} (Q_ns[i,t]^2 + Q_ew[i,t]^2)       # long queues cost more
         + w_s * sum_{i,t} [x[i,t] != x[i,t-1]]               # amber/all-red lost time

Squaring the queues is what makes the problem genuinely quadratic: it couples a
signal to its own future slots and, through the upstream-inflow term, to its
neighbours, which rewards "green waves" without hand-coding them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import TrafficNetwork
from .qubo import QUBO
from .simulator import TrafficState


@dataclass
class QuboWeights:
    queue: float = 1.0
    switch: float = 6.0


@dataclass
class SignalProblem:
    network: TrafficNetwork
    state: TrafficState
    horizon: int
    slot_steps: int
    weights: QuboWeights
    qubo: QUBO = field(init=False)

    def __post_init__(self) -> None:
        self.qubo = _build_qubo(self)

    @property
    def num_qubits(self) -> int:
        return self.network.n * self.horizon

    def var(self, i: int, t: int) -> int:
        return t * self.network.n + i

    def label(self, k: int) -> str:
        t, i = divmod(k, self.network.n)
        return f"{self.network.label(i)}@t{t}"

    def decode(self, x) -> np.ndarray:
        """Bit vector -> plan of shape ``(horizon, n)``."""
        return np.asarray(x, int).reshape(self.horizon, self.network.n)

    def encode(self, plan) -> np.ndarray:
        return np.asarray(plan, int).reshape(-1)

    # --- direct (non-QUBO) evaluation, used for tests and for display -------------

    def service(self) -> tuple[float, np.ndarray, np.ndarray]:
        net, st, L = self.network, self.state, self.slot_steps
        c = net.saturation_flow * L
        c_hat_ns = np.minimum(c, st.q_ns + L * net.arrival_ns)
        c_hat_ew = np.minimum(c, st.q_ew + L * net.arrival_ew)
        return c, c_hat_ns, c_hat_ew

    def predicted_queues(self, plan) -> np.ndarray:
        """Model queues ``(horizon, n, 2)`` [NS, EW] at the end of each slot."""
        net, st, L = self.network, self.state, self.slot_steps
        plan = np.asarray(plan, float).reshape(self.horizon, net.n)
        c, c_hat_ns, c_hat_ew = self.service()
        r = net.through_ratio
        served_ew = np.cumsum(plan, axis=0)  # EW green slots so far
        out = np.zeros((self.horizon, net.n, 2))
        for t in range(self.horizon):
            served_ns = (t + 1) - served_ew[t]
            for i in range(net.n):
                q_ns = st.q_ns[i] + (t + 1) * L * net.arrival_ns[i] - c * served_ns[i]
                q_ew = st.q_ew[i] + (t + 1) * L * net.arrival_ew[i] - c * served_ew[t, i]
                if (u := net.upstream_ns(i)) is not None:
                    q_ns += r * c_hat_ns[u] * served_ns[u]
                if (u := net.upstream_ew(i)) is not None:
                    q_ew += r * c_hat_ew[u] * served_ew[t, u]
                out[t, i] = q_ns, q_ew
        return out

    def plan_cost(self, plan) -> float:
        plan = np.asarray(plan, int).reshape(self.horizon, self.network.n)
        queues = self.predicted_queues(plan)
        prev = np.vstack([self.state.phase[None, :], plan[:-1]])
        switches = (plan != prev).sum()
        return float(self.weights.queue * (queues**2).sum() + self.weights.switch * switches)


def _build_qubo(p: SignalProblem) -> QUBO:
    net, st, L, T = p.network, p.state, p.slot_steps, p.horizon
    n_vars = net.n * T
    linear = np.zeros(n_vars)
    quad = np.zeros((n_vars, n_vars))
    offset = 0.0
    c, c_hat_ns, c_hat_ew = p.service()
    r = net.through_ratio
    w_q, w_s = p.weights.queue, p.weights.switch

    def add_square(const: float, coef: np.ndarray) -> None:
        nonlocal offset
        offset += w_q * const**2
        linear[:] += w_q * (2 * const * coef + coef**2)  # x^2 = x for binaries
        quad[:] += w_q * 2 * np.triu(np.outer(coef, coef), 1)

    for t in range(T):
        for i in range(net.n):
            # east-west queue: affine in x
            const = st.q_ew[i] + (t + 1) * L * net.arrival_ew[i]
            coef = np.zeros(n_vars)
            for s in range(t + 1):
                coef[p.var(i, s)] -= c
                if (u := net.upstream_ew(i)) is not None:
                    coef[p.var(u, s)] += r * c_hat_ew[u]
            add_square(const, coef)

            # north-south queue: affine in (1 - x)
            const = st.q_ns[i] + (t + 1) * L * net.arrival_ns[i]
            coef = np.zeros(n_vars)
            for s in range(t + 1):
                const -= c
                coef[p.var(i, s)] += c
                if (u := net.upstream_ns(i)) is not None:
                    const += r * c_hat_ns[u]
                    coef[p.var(u, s)] -= r * c_hat_ns[u]
            add_square(const, coef)

            # switching penalty: x_a XOR x_b = x_a + x_b - 2 x_a x_b
            k = p.var(i, t)
            if t == 0:
                prev = float(st.phase[i])
                offset += w_s * prev
                linear[k] += w_s * (1 - 2 * prev)
            else:
                j = p.var(i, t - 1)
                linear[k] += w_s
                linear[j] += w_s
                quad[j, k] -= 2 * w_s

    return QUBO(linear, quad, offset)


def build_signal_problem(
    network: TrafficNetwork,
    state: TrafficState,
    horizon: int = 2,
    slot_steps: int = 6,
    weights: QuboWeights | None = None,
) -> SignalProblem:
    return SignalProblem(network, state, horizon, slot_steps, weights or QuboWeights())
