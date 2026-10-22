import numpy as np
import pytest

from qtraffic import (
    QUBO,
    ExactSolver,
    SimulatedAnnealingSolver,
    TrafficSimulator,
    build_signal_problem,
    make_scenario,
)
from qtraffic.qubo import bit_matrix


def random_qubo(n, seed=0):
    rng = np.random.default_rng(seed)
    return QUBO(rng.normal(size=n), rng.normal(size=(n, n)), rng.normal())


def test_folding_keeps_energy():
    rng = np.random.default_rng(1)
    lin, quad = rng.normal(size=4), rng.normal(size=(4, 4))
    q = QUBO(lin, quad)
    for x in bit_matrix(4):
        assert np.isclose(q.energy(x), lin @ x + x @ quad @ x)


def test_ising_matches_qubo():
    q = random_qubo(6)
    h, J, const = q.to_ising()
    for x in bit_matrix(6):
        z = 1 - 2 * x
        assert np.isclose(q.energy(x), const + h @ z + z @ J @ z)


def test_pauli_operator_matches_qubo():
    from qiskit.quantum_info import Statevector

    q = random_qubo(5, seed=2)
    op, const = q.to_pauli()
    energies = q.all_energies()
    for b in [0, 3, 17, 31]:
        sv = Statevector.from_int(b, 2**5)
        assert np.isclose(sv.expectation_value(op).real + const, energies[b])


@pytest.mark.parametrize("scenario", ["rush_hour", "event_surge"])
@pytest.mark.parametrize("grid,horizon", [((2, 2), 2), ((1, 3), 3)])
def test_signal_qubo_equals_direct_cost(scenario, grid, horizon):
    net = make_scenario(scenario, *grid)
    state = TrafficSimulator(net, seed=4).initial_state()
    problem = build_signal_problem(net, state, horizon=horizon)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, problem.num_qubits)
        assert np.isclose(problem.qubo.energy(x), problem.plan_cost(problem.decode(x)))


def test_signal_qubo_has_spatial_and_temporal_couplings():
    net = make_scenario("rush_hour", 2, 2)
    problem = build_signal_problem(net, TrafficSimulator(net).initial_state(), horizon=2)
    Q = problem.qubo.quadratic
    # same intersection, consecutive slots
    assert Q[problem.var(0, 0), problem.var(0, 1)] != 0
    # upstream/downstream neighbours (A1 -> A2 on the east-west street)
    assert Q[problem.var(0, 0), problem.var(1, 0)] != 0


def test_annealing_finds_exact_optimum():
    net = make_scenario("event_surge", 2, 2)
    problem = build_signal_problem(net, TrafficSimulator(net, seed=1).initial_state(), horizon=3)
    exact = ExactSolver().solve(problem.qubo)
    sa = SimulatedAnnealingSolver(seed=0).solve(problem.qubo)
    assert np.isclose(exact.energy, problem.qubo.all_energies().min())
    assert sa.energy <= exact.energy + 1e-6
