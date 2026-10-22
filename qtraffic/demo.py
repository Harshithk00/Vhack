"""Terminal demo: python -m qtraffic.demo --scenario rush_hour"""

from __future__ import annotations

import argparse

import numpy as np

from . import (
    SCENARIOS,
    ExactSolver,
    FixedTimePolicy,
    MaxQueuePolicy,
    QAOASolver,
    QuboPolicy,
    SimulatedAnnealingSolver,
    TrafficSimulator,
    build_signal_problem,
    make_scenario,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Quantum-assisted traffic signal optimisation with QAOA")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="rush_hour")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=2, help="look-ahead slots of 30 s")
    parser.add_argument("--reps", type=int, default=2, help="QAOA depth p")
    parser.add_argument("--backend", choices=["statevector", "fast", "aer"], default="statevector")
    parser.add_argument("--noise", type=float, default=0.0, help="2-qubit depolarising error (aer only)")
    parser.add_argument("--minutes", type=int, default=15, help="closed-loop simulation length")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args(argv)

    net = make_scenario(args.scenario, args.rows, args.cols, seed=args.seed)
    state = TrafficSimulator(net, seed=args.seed).initial_state()
    problem = build_signal_problem(net, state, horizon=args.horizon)
    print(f"\n🚦 {net.describe()}")
    print(f"   {SCENARIOS[args.scenario]}")
    print(f"   QUBO: {problem.num_qubits} qubits, {problem.qubo.num_interactions()} couplings, {2**problem.num_qubits:,} possible plans\n")

    print("Queues now (NS / EW):  " + "  ".join(f"{net.label(i)} {state.q_ns[i]:.0f}/{state.q_ew[i]:.0f}" for i in range(net.n)))

    qaoa = QAOASolver(reps=args.reps, backend=args.backend, noise=args.noise, seed=args.seed)
    results = [qaoa.solve(problem.qubo), ExactSolver().solve(problem.qubo), SimulatedAnnealingSolver(seed=args.seed).solve(problem.qubo)]
    optimum = results[1].energy
    print(f"\n{'solver':<28}{'cost':>10}{'optimal?':>10}{'time (s)':>10}")
    for r in results:
        print(f"{r.solver:<28}{r.energy:>10.1f}{'yes' if np.isclose(r.energy, optimum) else 'no':>10}{r.runtime_s:>10.3f}")
    info = results[0].info
    print(f"\nQAOA: P(optimal plan) = {info['p_optimal']:.1%} vs {1 / 2**problem.num_qubits:.2%} for a random guess")
    stats = qaoa.circuit_stats(problem.qubo)
    print(f"      circuit depth {stats['depth']}, {stats['cx']} CNOTs, {stats['parameters']} trained angles")

    plan = problem.decode(results[0].x)
    print("\nQAOA timing plan:")
    for t, slot in enumerate(plan):
        cells = "  ".join(f"{net.label(i)}:{'EW' if b else 'NS'}" for i, b in enumerate(slot))
        print(f"  {30 * t:>3}-{30 * (t + 1):<3}s  {cells}")

    n_slots = args.minutes * 2
    print(f"\nClosed loop, {args.minutes} min of identical random traffic:")
    policies = [
        FixedTimePolicy(),
        MaxQueuePolicy(),
        QuboPolicy(QAOASolver(reps=args.reps, backend="fast", seed=args.seed), args.horizon),
    ]
    rows = [TrafficSimulator(net, seed=args.seed).run(p, n_slots=n_slots).summary() for p in policies]
    print(f"{'controller':<28}{'avg wait (s)':>14}{'veh/h':>10}{'switches':>10}")
    for row in rows:
        print(f"{row['policy']:<28}{row['avg_wait_s']:>14.1f}{row['throughput_vph']:>10.0f}{row['switches']:>10}")
    gain = 1 - rows[2]["avg_wait_s"] / rows[0]["avg_wait_s"]
    verb = "cuts" if gain >= 0 else "raises"
    print(f"\n➡  QAOA control {verb} average waiting time by {abs(gain):.0%} vs fixed-time signals.\n")


if __name__ == "__main__":
    main()
