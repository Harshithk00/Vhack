"""A small QUBO container and its mapping to an Ising / Pauli Hamiltonian.

Energy of a bit vector ``x``::

    E(x) = offset + sum_k linear[k] x_k + sum_{k<l} quadratic[k, l] x_k x_l

Bit ``k`` is qubit ``k`` throughout (little-endian, as in Qiskit), so index ``b``
of :meth:`QUBO.all_energies` is the bitstring whose qubit ``k`` equals
``(b >> k) & 1``. With ``x = (1 - z) / 2`` the energy becomes an Ising model in
spins ``z = +-1``, i.e. a diagonal Hamiltonian of Z and ZZ terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ENUMERATION_VARS = 24


def bit_matrix(n: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Rows are the bitstrings ``start .. stop-1``; column ``k`` is qubit ``k``."""
    idx = np.arange(start, 2**n if stop is None else stop, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)


@dataclass
class QUBO:
    linear: np.ndarray
    quadratic: np.ndarray
    offset: float = 0.0

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, float).copy()
        q = np.asarray(self.quadratic, float)
        n = self.linear.shape[0]
        if q.shape != (n, n):
            raise ValueError(f"quadratic must be {n}x{n}, got {q.shape}")
        # fold any diagonal / lower-triangle entries so the matrix is strictly upper
        self.linear += np.diag(q)
        self.quadratic = np.triu(q, 1) + np.tril(q, -1).T
        self.offset = float(self.offset)

    @property
    def num_vars(self) -> int:
        return self.linear.shape[0]

    def energy(self, x) -> float | np.ndarray:
        x = np.asarray(x, float)
        xs = np.atleast_2d(x)
        e = self.offset + xs @ self.linear + np.einsum("bi,ij,bj->b", xs, self.quadratic, xs)
        return float(e[0]) if x.ndim == 1 else e

    def all_energies(self, chunk: int = 1 << 16) -> np.ndarray:
        """Energy of every bitstring, indexed like a Qiskit statevector."""
        n = self.num_vars
        if n > MAX_ENUMERATION_VARS:
            raise ValueError(f"refusing to enumerate 2^{n} states")
        out = np.empty(2**n)
        for start in range(0, 2**n, chunk):
            stop = min(start + chunk, 2**n)
            out[start:stop] = self.energy(bit_matrix(n, start, stop))
        return out

    def scaled(self, factor: float) -> "QUBO":
        return QUBO(self.linear * factor, self.quadratic * factor, self.offset * factor)

    def normalized(self) -> tuple["QUBO", float]:
        """Scale so the largest Ising coefficient is 1 (keeps QAOA angles in a sane range)."""
        h, J, _ = self.to_ising()
        biggest = max(np.abs(h).max(initial=0.0), np.abs(J).max(initial=0.0)) or 1.0
        return self.scaled(1.0 / biggest), 1.0 / biggest

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(h, J, const)`` with ``E = const + sum h_k z_k + sum_{k<l} J_kl z_k z_l``."""
        Q = self.quadratic
        h = -self.linear / 2 - (Q.sum(axis=1) + Q.sum(axis=0)) / 4
        J = Q / 4
        const = self.offset + self.linear.sum() / 2 + Q.sum() / 4
        return h, J, float(const)

    def to_pauli(self):
        """Cost Hamiltonian as a Qiskit ``SparsePauliOp`` plus the constant energy shift."""
        from qiskit.quantum_info import SparsePauliOp

        h, J, const = self.to_ising()
        terms = [("Z", [k], h[k]) for k in range(self.num_vars) if h[k] != 0]
        terms += [("ZZ", [k, l], J[k, l]) for k, l in zip(*np.nonzero(J))]
        if not terms:
            terms = [("I", [0], 0.0)]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=self.num_vars), const

    def num_interactions(self) -> int:
        return int(np.count_nonzero(self.quadratic))
