"""Classical QUBO solvers used as references for QAOA.

* ``ExactSolver`` enumerates all ``2^n`` plans (vectorised) - the ground truth.
* ``SimulatedAnnealingSolver`` is the classical heuristic QAOA has to beat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .qubo import QUBO, bit_matrix


@dataclass
class SolveResult:
    solver: str
    x: np.ndarray
    energy: float
    runtime_s: float
    evaluations: int = 0
    info: dict = field(default_factory=dict)


class ExactSolver:
    name = "Exact (brute force)"

    def solve(self, qubo: QUBO) -> SolveResult:
        t0 = time.perf_counter()
        energies = qubo.all_energies()
        best = int(np.argmin(energies))
        x = bit_matrix(qubo.num_vars, best, best + 1)[0].astype(int)
        return SolveResult(
            self.name, x, float(energies[best]), time.perf_counter() - t0, evaluations=energies.size
        )


class SimulatedAnnealingSolver:
    name = "Simulated annealing"

    def __init__(self, sweeps: int = 200, restarts: int = 8, seed: int = 0):
        self.sweeps = sweeps
        self.restarts = restarts
        self.rng = np.random.default_rng(seed)

    def solve(self, qubo: QUBO) -> SolveResult:
        t0 = time.perf_counter()
        n = qubo.num_vars
        h = qubo.linear
        Js = qubo.quadratic + qubo.quadratic.T
        scale = max(np.abs(h).max(initial=0), np.abs(Js).sum(axis=1).max(initial=0), 1e-9)
        temps = np.geomspace(scale, scale * 1e-4, self.sweeps)

        best_x, best_e, evals = None, np.inf, 0
        for _ in range(self.restarts):
            x = self.rng.integers(0, 2, n)
            field_ = h + Js @ x
            for temp in temps:
                for k in self.rng.permutation(n):
                    delta = (1 - 2 * x[k]) * field_[k]
                    evals += 1
                    if delta <= 0 or self.rng.random() < np.exp(-delta / temp):
                        field_ += Js[:, k] * (1 - 2 * x[k])
                        x[k] ^= 1
            e = qubo.energy(x)
            if e < best_e:
                best_x, best_e = x.copy(), e
        return SolveResult(self.name, best_x, float(best_e), time.perf_counter() - t0, evaluations=evals)
