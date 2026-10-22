"""QAOA solver for the signal-timing QUBO, built on Qiskit.

The circuit is written out explicitly so every gate maps to a piece of the traffic
model::

    |+>^n  ->  [ exp(-i gamma_l H_C)  exp(-i beta_l sum X) ]  for l = 1..p  ->  measure

    exp(-i gamma H_C) = prod RZ(2 gamma h_k)        on each signal-slot qubit
                      * prod RZZ(2 gamma J_kl)      on each coupled pair (same signal
                                                    over time, or neighbouring signals)

Three execution backends share the same optimiser loop:

* ``"statevector"`` - exact Qiskit ``Statevector`` simulation (default);
* ``"fast"``        - an equivalent vectorised NumPy simulation (the cost layer is
                      diagonal), ~10-50x faster; used inside closed-loop runs and
                      tested against the Qiskit result;
* ``"aer"``         - shot-based ``AerSimulator`` with an optional depolarising
                      noise model, closest to real hardware.

Angles are trained with COBYLA, optionally layer by layer using the INTERP
initialisation of Zhou et al. (PRX 2020), and the objective can be the plain
energy expectation or CVaR (Barkoutsos et al., Quantum 2020).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from .classical import SolveResult
from .qubo import QUBO, bit_matrix

BASIS_GATES = ["cx", "rz", "sx", "x"]


@dataclass
class QAOAConfig:
    reps: int = 2
    backend: str = "statevector"  # "statevector" | "fast" | "aer"
    maxiter: int = 150
    restarts: int = 2
    layerwise: bool = True
    cvar_alpha: float = 1.0  # 1.0 = plain expectation
    shots: int = 2048  # aer backend only
    noise: float = 0.0  # aer backend only: 2-qubit depolarising probability
    final_samples: int = 1024
    seed: int = 0


def linear_ramp(reps: int, delta: float = 0.75) -> np.ndarray:
    """Annealing-inspired start: gamma ramps up, beta ramps down."""
    frac = (np.arange(reps) + 0.5) / reps
    return np.concatenate([delta * frac, delta * (1 - frac)])


def interp(params: np.ndarray) -> np.ndarray:
    """INTERP: extend optimal depth-p angles to a depth-(p+1) starting point."""
    p = len(params) // 2

    def grow(v):
        padded = np.concatenate([[0.0], v, [0.0]])
        j = np.arange(p + 1)
        return (j / p) * padded[j] + ((p - j) / p) * padded[j + 1]

    return np.concatenate([grow(params[:p]), grow(params[p:])])


def cvar(probs: np.ndarray, energies: np.ndarray, alpha: float) -> float:
    if alpha >= 1.0:
        return float(probs @ energies)
    order = np.argsort(energies)
    p_sorted = probs[order]
    cum = np.cumsum(p_sorted)
    take = np.clip(alpha - (cum - p_sorted), 0.0, p_sorted)
    return float(take @ energies[order] / alpha)


def fast_probabilities(diag: np.ndarray, params: np.ndarray, n: int) -> np.ndarray:
    """Exact QAOA output distribution without building a circuit."""
    p = len(params) // 2
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for gamma, beta in zip(params[:p], params[p:]):
        psi *= np.exp(-1j * gamma * diag)
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi = psi.reshape((2,) * n)
        for ax in range(n):
            a0, a1 = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
            psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
        psi = psi.reshape(-1)
    return np.abs(psi) ** 2


class QAOASolver:
    supports_warm_start = True

    def __init__(self, config: QAOAConfig | None = None, **overrides):
        self.config = replace(config or QAOAConfig(), **overrides)
        self.rng = np.random.default_rng(self.config.seed)
        self._aer = None

    @property
    def name(self) -> str:
        noise = f", noise={self.config.noise:g}" if self.config.backend == "aer" and self.config.noise else ""
        return f"QAOA (p={self.config.reps}{noise})"

    # --- circuits ----------------------------------------------------------------

    @staticmethod
    def build_circuit(qubo: QUBO, reps: int, measure: bool = False):
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        n = qubo.num_vars
        h, J, _ = qubo.to_ising()
        gammas, betas = ParameterVector("γ", reps), ParameterVector("β", reps)
        qc = QuantumCircuit(n, name="QAOA")
        qc.h(range(n))
        for layer in range(reps):
            qc.barrier()
            for k in range(n):
                if h[k] != 0:
                    qc.rz(2 * gammas[layer] * h[k], k)
            for k, l in zip(*np.nonzero(J)):
                qc.rzz(2 * gammas[layer] * J[k, l], int(k), int(l))
            for k in range(n):
                qc.rx(2 * betas[layer], k)
        if measure:
            qc.measure_all()
        return qc

    @staticmethod
    def bind(qc, params: np.ndarray):
        reps = len(params) // 2
        gammas = [p for p in qc.parameters if p.name.startswith("γ")]
        betas = [p for p in qc.parameters if p.name.startswith("β")]
        mapping = dict(zip(sorted(gammas, key=lambda p: p.index), params[:reps]))
        mapping.update(zip(sorted(betas, key=lambda p: p.index), params[reps:]))
        return qc.assign_parameters(mapping)

    def circuit_stats(self, qubo: QUBO, reps: int | None = None) -> dict:
        from qiskit import transpile

        reps = reps or self.config.reps
        qn, _ = qubo.normalized()
        qc = self.bind(self.build_circuit(qn, reps), linear_ramp(reps))
        tc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
        ops = tc.count_ops()
        return {
            "qubits": qubo.num_vars,
            "reps": reps,
            "parameters": 2 * reps,
            "zz_terms": qubo.num_interactions(),
            "depth": tc.depth(),
            "cx": int(ops.get("cx", 0)),
            "gates": int(sum(ops.values())),
        }

    # --- distributions -----------------------------------------------------------

    def _distribution_fn(self, qn: QUBO, diag: np.ndarray, reps: int):
        n, backend = qn.num_vars, self.config.backend
        if backend == "fast":
            return lambda params: fast_probabilities(diag, params, n)
        if backend == "statevector":
            from qiskit.quantum_info import Statevector

            qc = self.build_circuit(qn, reps)
            return lambda params: Statevector(self.bind(qc, params)).probabilities()
        if backend == "aer":
            from qiskit import transpile
            from qiskit_aer import AerSimulator

            sim = AerSimulator(noise_model=self._noise_model())
            tc = transpile(self.build_circuit(qn, reps, measure=True), basis_gates=BASIS_GATES, optimization_level=1)

            def probs(params):
                seed = int(self.rng.integers(2**31))
                counts = sim.run(self.bind(tc, params), shots=self.config.shots, seed_simulator=seed).result().get_counts()
                out = np.zeros(2**n)
                for key, v in counts.items():
                    out[int(key.replace(" ", ""), 2)] = v
                return out / out.sum()

            return probs
        raise ValueError(f"unknown backend {backend!r}")

    def _noise_model(self):
        if not self.config.noise:
            return None
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        model = NoiseModel()
        model.add_all_qubit_quantum_error(depolarizing_error(self.config.noise, 2), ["cx"])
        model.add_all_qubit_quantum_error(depolarizing_error(self.config.noise / 10, 1), ["sx", "x"])
        return model

    # --- optimisation ------------------------------------------------------------

    def solve(self, qubo: QUBO, initial_params: np.ndarray | None = None) -> SolveResult:
        cfg = self.config
        t0 = time.perf_counter()
        qn, scale = qubo.normalized()
        diag = qn.all_energies()
        history: list[float] = []
        evals = 0
        fns = {}

        def optimise(reps: int, x0: np.ndarray, maxiter: int):
            nonlocal evals
            fn = fns.setdefault(reps, self._distribution_fn(qn, diag, reps))

            def objective(params):
                nonlocal evals
                evals += 1
                value = cvar(fn(params), diag, cfg.cvar_alpha)
                history.append(value / scale)
                return value

            res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.25})
            return res.x, res.fun

        if initial_params is not None and len(initial_params) == 2 * cfg.reps:
            best_params, best_val = optimise(cfg.reps, np.asarray(initial_params, float), max(30, cfg.maxiter // 3))
        else:
            best_params, best_val = None, np.inf
            for restart in range(cfg.restarts):
                start = linear_ramp(1 if cfg.layerwise else cfg.reps)
                if restart:
                    start = start + self.rng.normal(scale=0.3, size=start.size)
                params = start
                for reps in range(1 if cfg.layerwise else cfg.reps, cfg.reps + 1):
                    if cfg.layerwise and reps > 1:
                        params = interp(params)
                    params, val = optimise(reps, params, cfg.maxiter)
                if val < best_val:
                    best_params, best_val = params, val

        probs = self._distribution_fn(qn, diag, cfg.reps)(best_params) if cfg.backend == "aer" else fns[cfg.reps](best_params)
        probs = probs / probs.sum()
        samples = self.rng.choice(probs.size, size=cfg.final_samples, p=probs)
        best_index = int(samples[np.argmin(diag[samples])])
        x = bit_matrix(qn.num_vars, best_index, best_index + 1)[0].astype(int)
        ground = np.isclose(diag, diag.min(), atol=1e-9)
        top = np.argsort(probs)[::-1][:16]

        return SolveResult(
            self.name,
            x,
            float(qubo.energy(x)),
            time.perf_counter() - t0,
            evaluations=evals,
            info={
                "params": best_params,
                "reps": cfg.reps,
                "expectation": float(probs @ diag) / scale,
                "p_optimal": float(probs[ground].sum()),
                "p_selected": float(probs[best_index]),
                "most_likely": int(np.argmax(probs)),
                "history": history,
                "top_states": [(int(b), float(probs[b]), float(diag[b] / scale)) for b in top],
            },
        )
