import numpy as np
import pytest

from qtraffic import (
    QUBO,
    ExactSolver,
    QAOASolver,
    QuboPolicy,
    TrafficSimulator,
    build_signal_problem,
    make_scenario,
)
from qtraffic.qaoa import cvar, fast_probabilities, interp, linear_ramp


def signal_qubo(rows=2, cols=2, horizon=1, scenario="rush_hour", seed=2):
    net = make_scenario(scenario, rows, cols)
    return build_signal_problem(net, TrafficSimulator(net, seed=seed).initial_state(), horizon=horizon).qubo


def test_fast_backend_matches_qiskit_statevector():
    from qiskit.quantum_info import Statevector

    qn, _ = signal_qubo(2, 2, 2).normalized()
    params = np.array([0.4, 0.9, 0.7, 0.2])
    qc = QAOASolver.bind(QAOASolver.build_circuit(qn, reps=2), params)
    expected = Statevector(qc).probabilities()
    got = fast_probabilities(qn.all_energies(), params, qn.num_vars)
    assert np.allclose(got, expected, atol=1e-10)


def test_circuit_structure():
    q = signal_qubo(2, 2, 2)
    qc = QAOASolver.build_circuit(q, reps=3)
    ops = qc.count_ops()
    assert qc.num_qubits == 8
    assert ops["rzz"] == 3 * q.num_interactions()
    assert ops["rx"] == 3 * 8
    assert len(qc.parameters) == 6


def test_interp_and_ramp_shapes():
    assert linear_ramp(3).shape == (6,)
    grown = interp(np.array([0.5, 0.2]))
    assert np.allclose(grown, [0.5, 0.5, 0.2, 0.2])


def test_cvar_reduces_to_expectation():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    energies = np.array([3.0, -1.0, 2.0, 0.0])
    assert np.isclose(cvar(probs, energies, 1.0), probs @ energies)
    assert np.isclose(cvar(probs, energies, 0.2), -1.0)


@pytest.mark.parametrize("backend", ["statevector", "fast"])
def test_qaoa_finds_optimum_on_small_grid(backend):
    q = signal_qubo(2, 2, 1)
    exact = ExactSolver().solve(q)
    res = QAOASolver(reps=2, backend=backend).solve(q)
    assert np.isclose(res.energy, exact.energy)
    assert res.info["p_optimal"] > 1 / 2**q.num_vars


def test_deeper_circuits_concentrate_on_optimum():
    q = signal_qubo(2, 2, 2)
    p1 = QAOASolver(reps=1, backend="fast").solve(q).info["p_optimal"]
    p3 = QAOASolver(reps=3, backend="fast").solve(q).info["p_optimal"]
    assert p3 > p1


def test_noisy_aer_backend_runs():
    q = signal_qubo(1, 2, 1)
    res = QAOASolver(reps=1, backend="aer", shots=512, noise=0.01, maxiter=30, restarts=1).solve(q)
    assert res.x.shape == (2,)
    assert np.isclose(res.energy, q.energy(res.x))


def test_qaoa_policy_warm_starts():
    net = make_scenario("event_surge", 2, 2)
    policy = QuboPolicy(QAOASolver(reps=2, backend="fast"), horizon=1)
    TrafficSimulator(net).run(policy, n_slots=3)
    assert len(policy.history) == 3
    assert policy.history[1].evaluations < policy.history[0].evaluations


def test_trivial_qubo():
    q = QUBO(np.array([1.0, -1.0]), np.zeros((2, 2)))
    assert list(QAOASolver(reps=1, backend="fast").solve(q).x) == [0, 1]
