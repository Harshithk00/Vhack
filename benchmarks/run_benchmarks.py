"""Run every experiment in the study and write CSVs + figures to ``results/``.

    python benchmarks/run_benchmarks.py           # full study (~5-10 min on a laptop)
    python benchmarks/run_benchmarks.py --quick   # smoke run (~1 min)

Experiments
1. closed_loop - average wait / throughput of each controller in each scenario
2. quality     - QAOA approximation ratio and P(optimum) vs depth p and problem size
3. scaling     - qubits, circuit depth, CNOTs and solve time as the city grows
4. noise       - QAOA on a noisy simulated device (depolarising errors)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qtraffic import (  # noqa: E402
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
from qtraffic import viz  # noqa: E402
from qtraffic.viz import plt  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"


def instance(scenario, rows, cols, horizon, seed):
    net = make_scenario(scenario, rows, cols, seed=seed)
    state = TrafficSimulator(net, seed=seed).initial_state()
    return build_signal_problem(net, state, horizon=horizon)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- 1. closed loop -------------------------------------------------------------


def closed_loop(seeds, n_slots, grid=(2, 2), horizon=2):
    rows = []
    timelines = {}
    for scenario in SCENARIOS:
        net = make_scenario(scenario, *grid)
        for seed in seeds:
            policies = [
                FixedTimePolicy(),
                MaxQueuePolicy(),
                QuboPolicy(SimulatedAnnealingSolver(seed=seed), horizon),
                QuboPolicy(ExactSolver(), horizon),
                QuboPolicy(QAOASolver(reps=2, backend="fast", seed=seed), horizon),
            ]
            for policy in policies:
                res = TrafficSimulator(net, seed=seed).run(policy, n_slots=n_slots)
                row = res.summary() | {"seed": seed, "qubits": net.n * horizon if isinstance(policy, QuboPolicy) else 0}
                rows.append(row)
                if seed == seeds[0]:
                    timelines[(scenario, res.policy)] = res.queues.sum(axis=(1, 2))
        log(f"closed loop: {scenario} done")
    return pd.DataFrame(rows), timelines


def plot_closed_loop(df, timelines):
    agg = df.groupby(["scenario", "policy"], sort=False)["avg_wait_s"].mean().unstack()
    policies = list(dict.fromkeys(df["policy"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(policies)
    x = np.arange(len(agg.index))
    for k, p in enumerate(policies):
        ax.bar(x + k * width, agg[p], width, label=p, color=viz.policy_color(p))
    ax.set_xticks(x + width * (len(policies) - 1) / 2, [s.replace("_", " ") for s in agg.index])
    ax.set_ylabel("average wait per vehicle (s)")
    ax.set_title("Closed-loop control on a 2x2 grid (lower is better)")
    ax.set_ylim(0, agg.values.max() * 1.3)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    viz.save(fig, FIG / "closed_loop_wait.png")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for (scenario, policy), series in timelines.items():
        if scenario == "rush_hour":
            minutes = np.arange(series.size) * 5 / 60
            ax.plot(minutes, series, label=policy, color=viz.policy_color(policy), lw=1.6)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("vehicles queued (whole grid)")
    ax.set_title("Rush hour: total queue over time")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    viz.save(fig, FIG / "rush_hour_queues.png")


# --- 2. solution quality --------------------------------------------------------


def quality(configs, reps_list, seeds):
    rows = []
    for rows_, cols, horizon in configs:
        for scenario in ["off_peak", "rush_hour", "event_surge"]:
            for seed in seeds:
                problem = instance(scenario, rows_, cols, horizon, seed)
                q = problem.qubo
                energies = q.all_energies()
                e_min, e_max = energies.min(), energies.max()
                exact = ExactSolver().solve(q)
                sa = SimulatedAnnealingSolver(seed=seed).solve(q)
                base = {"grid": f"{rows_}x{cols}", "horizon": horizon, "qubits": q.num_vars, "scenario": scenario, "seed": seed}
                rows.append(base | {"solver": "Exact", "p": 0, "found_optimum": True, "approx_ratio": 1.0, "runtime_s": exact.runtime_s})
                rows.append(
                    base
                    | {"solver": "SA", "p": 0, "found_optimum": bool(np.isclose(sa.energy, e_min)),
                       "approx_ratio": (e_max - sa.energy) / (e_max - e_min), "runtime_s": sa.runtime_s}
                )
                for reps in reps_list:
                    res = QAOASolver(reps=reps, backend="fast", seed=seed).solve(q)
                    rows.append(
                        base
                        | {
                            "solver": "QAOA",
                            "p": reps,
                            "found_optimum": bool(np.isclose(res.energy, e_min)),
                            "approx_ratio": (e_max - res.info["expectation"]) / (e_max - e_min),
                            "p_optimal": res.info["p_optimal"],
                            "uniform_p_optimal": 1 / 2**q.num_vars,
                            "evaluations": res.evaluations,
                            "runtime_s": res.runtime_s,
                        }
                    )
        log(f"quality: {rows_}x{cols} H={horizon} done")
    return pd.DataFrame(rows)


def plot_quality(df):
    qa = df[df.solver == "QAOA"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for qubits, grp in qa.groupby("qubits"):
        m = grp.groupby("p")[["approx_ratio", "p_optimal", "uniform_p_optimal"]].mean()
        axes[0].plot(m.index, m.approx_ratio, marker="o", label=f"{qubits} qubits")
        axes[1].plot(m.index, m.p_optimal / m.uniform_p_optimal, marker="o", label=f"{qubits} qubits")
    axes[0].set(xlabel="QAOA depth p", ylabel="approximation ratio  ⟨E⟩", title="Solution quality vs depth")
    axes[1].set(xlabel="QAOA depth p", ylabel="P(optimum) / random guess", title="Amplification of the best plan", yscale="log")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
        ax.set_xticks(sorted(qa.p.unique()))
        ax.spines[["top", "right"]].set_visible(False)
    viz.save(fig, FIG / "qaoa_quality_vs_depth.png")


# --- 3. scalability -------------------------------------------------------------


def scaling(configs, timing_limit):
    rows = []
    for rows_, cols, horizon in configs:
        problem = instance("rush_hour", rows_, cols, horizon, 0)
        q = problem.qubo
        row = {"grid": f"{rows_}x{cols}", "horizon": horizon, "qubits": q.num_vars, "zz_terms": q.num_interactions()}
        for reps in (1, 2, 3):
            stats = QAOASolver(reps=reps).circuit_stats(q)
            row |= {f"depth_p{reps}": stats["depth"], f"cx_p{reps}": stats["cx"]}
        if q.num_vars <= timing_limit:
            row["exact_s"] = ExactSolver().solve(q).runtime_s
            row["sa_s"] = SimulatedAnnealingSolver().solve(q).runtime_s
            row["qaoa_fast_p2_s"] = QAOASolver(reps=2, backend="fast").solve(q).runtime_s
            if q.num_vars <= 12:
                row["qaoa_qiskit_p2_s"] = QAOASolver(reps=2, backend="statevector").solve(q).runtime_s
        rows.append(row)
        log(f"scaling: {q.num_vars} qubits done")
    return pd.DataFrame(rows).sort_values("qubits")


def plot_scaling(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for reps, style in ((1, "o-"), (2, "s-"), (3, "^-")):
        axes[0].plot(df.qubits, df[f"cx_p{reps}"], style, label=f"CNOTs, p={reps}")
    axes[0].set(xlabel="qubits (signals x horizon)", ylabel="CNOT count after transpiling", title="Circuit cost grows with the city")
    timed = df.dropna(subset=["exact_s"])
    for col, label, color in (
        ("exact_s", "Exact brute force", "#3b82f6"),
        ("sa_s", "Simulated annealing", "#7c66dc"),
        ("qaoa_fast_p2_s", "QAOA p=2 (simulated)", viz.QAOA_COLOR),
        ("qaoa_qiskit_p2_s", "QAOA p=2 (Qiskit statevector)", "#f97316"),
    ):
        if col in timed:
            sub = timed.dropna(subset=[col])
            axes[1].plot(sub.qubits, sub[col], "o-", label=label, color=color)
    axes[1].set(xlabel="qubits", ylabel="wall-clock seconds", yscale="log", title="Solve time on a laptop")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    viz.save(fig, FIG / "scaling.png")


# --- 4. noise -------------------------------------------------------------------


def noise(levels, shots, maxiter):
    rows = []
    for rows_, cols, horizon in ((2, 2, 1), (2, 2, 2)):
        problem = instance("rush_hour", rows_, cols, horizon, 0)
        q = problem.qubo
        energies = q.all_energies()
        e_min, e_max = energies.min(), energies.max()
        for reps in (1, 2):
            for level in levels:
                res = QAOASolver(reps=reps, backend="aer", noise=level, shots=shots, maxiter=maxiter, restarts=1).solve(q)
                rows.append(
                    {
                        "qubits": q.num_vars,
                        "p": reps,
                        "noise": level,
                        "approx_ratio": (e_max - res.info["expectation"]) / (e_max - e_min),
                        "p_optimal": res.info["p_optimal"],
                        "found_optimum": bool(np.isclose(res.energy, e_min)),
                    }
                )
        log(f"noise: {q.num_vars} qubits done")
    return pd.DataFrame(rows)


def plot_noise(df):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (qubits, reps), grp in df.groupby(["qubits", "p"]):
        ax.plot(grp.noise * 100, grp.approx_ratio, "o-", label=f"{qubits} qubits, p={reps}")
    ax.set(xlabel="2-qubit gate error (%)", ylabel="approximation ratio  ⟨E⟩", title="QAOA on a noisy simulated device")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    viz.save(fig, FIG / "noise.png")


def hero_figure():
    net = make_scenario("rush_hour", 3, 3)
    sim = TrafficSimulator(net, seed=1)
    state = sim.initial_state(mean_queue=6)
    problem = build_signal_problem(net, state, horizon=1)
    plan = problem.decode(QAOASolver(reps=2, backend="fast").solve(problem.qubo).x)[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    viz.draw_network(axes[0], net, state.q_ns, state.q_ew, state.phase, "Current signals and queues")
    after = state.copy()
    rng = np.random.default_rng(0)
    for _ in range(sim.slot_steps):
        sim.step(after, plan, rng)
    viz.draw_network(axes[1], net, after.q_ns, after.q_ew, plan, "30 s later, with the QAOA plan")
    viz.save(fig, FIG / "hero.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    if args.quick:
        seeds, n_slots = [0], 20
        quality_cfg, reps_list = [(2, 2, 1), (2, 2, 2)], [1, 2]
        scaling_cfg, timing_limit = [(1, 2, 1), (2, 2, 1), (2, 2, 2), (2, 3, 2)], 12
        noise_levels, shots, maxiter = [0.0, 0.02], 512, 25
    else:
        seeds, n_slots = [0, 1, 2], 60
        quality_cfg, reps_list = [(2, 2, 1), (2, 2, 2), (2, 3, 2), (2, 2, 3), (2, 4, 2)], [1, 2, 3, 4]
        scaling_cfg = [(1, 2, 1), (2, 2, 1), (1, 3, 2), (2, 2, 2), (3, 3, 1), (2, 3, 2), (2, 2, 3), (2, 4, 2), (3, 3, 2), (2, 3, 3), (3, 4, 2), (4, 4, 2)]
        timing_limit = 18
        noise_levels, shots, maxiter = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05], 1024, 60

    hero_figure()
    cl, timelines = closed_loop(seeds, n_slots)
    cl.to_csv(OUT / "closed_loop.csv", index=False)
    plot_closed_loop(cl, timelines)

    qd = quality(quality_cfg, reps_list, seeds)
    qd.to_csv(OUT / "quality.csv", index=False)
    plot_quality(qd)

    sc = scaling(scaling_cfg, timing_limit)
    sc.to_csv(OUT / "scaling.csv", index=False)
    plot_scaling(sc)

    nz = noise(noise_levels, shots, maxiter)
    nz.to_csv(OUT / "noise.csv", index=False)
    plot_noise(nz)

    write_summary(cl, qd, sc, nz)
    log(f"done - results in {OUT.relative_to(ROOT)}/")


def write_summary(cl, qd, sc, nz):
    wait = cl.groupby(["scenario", "policy"], sort=False)["avg_wait_s"].mean().unstack().round(1)
    qa = qd[qd.solver == "QAOA"].groupby(["qubits", "p"])[["approx_ratio", "p_optimal", "found_optimum", "runtime_s"]].mean().round(3)
    sa = qd[qd.solver == "SA"].groupby("qubits")[["found_optimum", "runtime_s"]].mean().round(3)
    lines = [
        "# Benchmark summary",
        "",
        "Generated by `python benchmarks/run_benchmarks.py`. Numbers are means over seeds.",
        "",
        "## Average wait per vehicle (s), 2x2 grid, 30 min of traffic",
        "",
        wait.to_markdown(),
        "",
        "## QAOA solution quality (fast exact simulation)",
        "",
        qa.to_markdown(),
        "",
        "## Simulated annealing on the same instances",
        "",
        sa.to_markdown(),
        "",
        "## Scalability",
        "",
        sc.round(4).to_markdown(index=False),
        "",
        "## Noise",
        "",
        nz.round(3).to_markdown(index=False),
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
