"""Interactive demo: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

faulthandler.enable()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qtraffic import (  # noqa: E402
    SCENARIOS,
    EW_GREEN,
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
from qtraffic.qubo import bit_matrix  # noqa: E402
from qtraffic.viz import GREEN, RED, ROAD, policy_color  # noqa: E402

st.set_page_config(page_title="Quantum Traffic Signals", page_icon="🚦", layout="wide")

# --- sidebar ----------------------------------------------------------------------

with st.sidebar:
    st.header("🚦 City setup")
    scenario = st.selectbox("Traffic scenario", list(SCENARIOS), index=1, format_func=lambda s: s.replace("_", " ").title())
    st.caption(SCENARIOS[scenario])
    c1, c2 = st.columns(2)
    rows = c1.number_input("Rows", 1, 3, 2)
    cols = c2.number_input("Columns", 1, 4, 2)
    horizon = st.slider("Look-ahead slots (30 s each)", 1, 3, 2)
    seed = st.number_input("Random seed", 0, 999, 2)

    st.header("⚛️ QAOA")
    reps = st.slider("Circuit depth p", 1, 4, 2)
    backend = st.radio(
        "Backend",
        ["fast", "statevector", "aer"],
        format_func={"fast": "Exact simulation (NumPy)", "statevector": "Qiskit Statevector", "aer": "Qiskit Aer, noisy shots"}.get,
    )
    noise = st.slider("2-qubit gate error", 0.0, 0.05, 0.01, 0.005, disabled=backend != "aer")
    cvar_alpha = st.slider("CVaR α (1 = plain expectation)", 0.05, 1.0, 1.0, 0.05)

    st.divider()
    st.caption("**Team NULL POINTER**  \nHarshit Kumar (Team Lead)  \nHimansu Rauniyar  \nAditya Pattar  \nCSE, SIT Tumkur")

n_qubits = int(rows * cols * horizon)
limit = {"fast": 18, "statevector": 14, "aer": 12}[backend]

st.title("Quantum-Assisted Traffic Signal Optimization")
st.caption(
    "Team **NULL POINTER** · **Harshit Kumar** (Team Lead), Himansu Rauniyar, Aditya Pattar · "
    "Dept. of Computer Science & Engineering, SIT Tumkur · [GitHub](https://github.com/Harshithk00/Vhack)"
)
st.markdown(
    "Each signal-and-time-slot is **one qubit**: `0` = north–south green, `1` = east–west green. "
    "QAOA searches all plans at once for the one that keeps queues short without needless switching."
)

if n_qubits > limit:
    st.error(f"{n_qubits} qubits is too many for the {backend} backend in a live demo (limit {limit}). Shrink the grid or horizon.")
    st.stop()


# --- cached computations ------------------------------------------------------------


@st.cache_data(show_spinner=False)
def make_problem(scenario, rows, cols, horizon, seed):
    net = make_scenario(scenario, rows, cols, seed=seed)
    state = TrafficSimulator(net, seed=seed).initial_state()
    return build_signal_problem(net, state, horizon=horizon)


@st.cache_data(show_spinner=False)
def solve_all(scenario, rows, cols, horizon, seed, reps, backend, noise, cvar_alpha):
    problem = make_problem(scenario, rows, cols, horizon, seed)
    qaoa = QAOASolver(reps=reps, backend=backend, noise=noise if backend == "aer" else 0.0, cvar_alpha=cvar_alpha, seed=seed)
    return {
        "qaoa": qaoa.solve(problem.qubo),
        "exact": ExactSolver().solve(problem.qubo),
        "sa": SimulatedAnnealingSolver(seed=seed).solve(problem.qubo),
        "stats": qaoa.circuit_stats(problem.qubo),
        "energies": problem.qubo.all_energies(),
    }


@st.cache_data(show_spinner=False)
def circuit_figure(scenario, rows, cols, horizon, seed, reps):
    problem = make_problem(scenario, rows, cols, horizon, seed)
    qn, _ = problem.qubo.normalized()
    qc = QAOASolver.build_circuit(qn, min(reps, 1), measure=True)
    return qc.draw("mpl", fold=40, style="iqp", scale=0.6)


@st.cache_data(show_spinner=False)
def run_city(scenario, rows, cols, horizon, seed, reps, n_slots):
    net = make_scenario(scenario, rows, cols, seed=seed)
    policies = [
        FixedTimePolicy(),
        MaxQueuePolicy(),
        QuboPolicy(SimulatedAnnealingSolver(seed=seed), horizon),
        QuboPolicy(QAOASolver(reps=reps, backend="fast", seed=seed), horizon),
    ]
    if net.n * horizon <= 14:
        policies.insert(3, QuboPolicy(ExactSolver(), horizon))
    return [TrafficSimulator(net, seed=seed).run(p, n_slots=n_slots) for p in policies]


def plan_label(bits, n) -> str:
    slots = np.asarray(bits).reshape(-1, n)
    return " · ".join("".join("E" if b else "N" for b in slot) for slot in slots)


def network_figure(net, q_ns, q_ew, phase, title):
    fig = go.Figure()
    for r in range(net.rows):
        fig.add_trace(go.Scatter(x=[-0.6, net.cols - 0.4], y=[-r, -r], mode="lines", line=dict(color=ROAD, width=10), hoverinfo="skip"))
    for c in range(net.cols):
        fig.add_trace(go.Scatter(x=[c, c], y=[0.6, -net.rows + 0.4], mode="lines", line=dict(color=ROAD, width=10), hoverinfo="skip"))
    scale = 0.38 / max(1.0, float(np.max(q_ns)), float(np.max(q_ew)))
    for i in range(net.n):
        r, c = net.position(i)
        ew = phase[i] == EW_GREEN
        fig.add_trace(go.Scatter(x=[c - 0.1 - q_ew[i] * scale, c - 0.1], y=[-r, -r], mode="lines", line=dict(color=GREEN if ew else RED, width=14), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[c, c], y=[-r + 0.1, -r + 0.1 + q_ns[i] * scale], mode="lines", line=dict(color=RED if ew else GREEN, width=14), hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=[c], y=[-r], mode="markers+text", text=[net.label(i)], textposition="bottom right",
                marker=dict(size=22, color="#1f2937"),
                hovertext=f"{net.label(i)}<br>{'EW' if ew else 'NS'} green<br>NS queue {q_ns[i]:.1f}<br>EW queue {q_ew[i]:.1f}",
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title=title, showlegend=False, height=330, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


problem = make_problem(scenario, rows, cols, horizon, seed)
net, state = problem.network, problem.state

m = st.columns(4)
m[0].metric("Qubits", n_qubits)
m[1].metric("Possible plans", f"{2**n_qubits:,}")
m[2].metric("ZZ couplings", problem.qubo.num_interactions())
m[3].metric("Signals × slots", f"{net.n} × {horizon}")

tab_plan, tab_circuit, tab_city, tab_bench, tab_how = st.tabs(
    ["🚦 Signal plan", "⚛️ Quantum circuit", "🏙️ City simulation", "📊 Benchmarks", "🧮 How it works"]
)

with st.spinner(f"Optimising a {n_qubits}-qubit QAOA circuit…"):
    sol = solve_all(scenario, rows, cols, horizon, seed, reps, backend, noise, cvar_alpha)
qaoa, exact, sa, energies = sol["qaoa"], sol["exact"], sol["sa"], sol["energies"]

# --- tab: plan -------------------------------------------------------------------------

with tab_plan:
    plan = problem.decode(qaoa.x)
    left, right = st.columns(2)
    left.plotly_chart(network_figure(net, state.q_ns, state.q_ew, state.phase, "Now: signals and queues"), width="stretch")
    pred = problem.predicted_queues(plan)[0]
    right.plotly_chart(network_figure(net, pred[:, 0], pred[:, 1], plan[0], "Predicted after 30 s with the QAOA plan"), width="stretch")

    gap = (qaoa.energy - exact.energy) / max(energies.max() - exact.energy, 1e-9)
    if np.isclose(qaoa.energy, exact.energy):
        st.success(f"QAOA found the optimal plan out of {2**n_qubits:,} (P(optimum) after training = {qaoa.info['p_optimal']:.1%}).")
    else:
        st.warning(f"QAOA's plan is {gap:.1%} of the energy range away from optimal - try a deeper circuit.")

    table = pd.DataFrame(
        {
            "Solver": ["QAOA", "Exact (brute force)", "Simulated annealing"],
            "Plan (N/E per signal, · per slot)": [plan_label(r.x, net.n) for r in (qaoa, exact, sa)],
            "Cost": [round(r.energy, 1) for r in (qaoa, exact, sa)],
            "Time (s)": [round(r.runtime_s, 3) for r in (qaoa, exact, sa)],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    slots = pd.DataFrame(
        [[("EW 🟢" if plan[t, i] else "NS 🟢") for i in range(net.n)] for t in range(horizon)],
        columns=[net.label(i) for i in range(net.n)],
        index=[f"{30 * t}-{30 * (t + 1)} s" for t in range(horizon)],
    )
    st.markdown("**QAOA timing plan** (only the first slot is applied, then the plan is re-optimised)")
    st.dataframe(slots, width="stretch")

# --- tab: circuit -----------------------------------------------------------------------

with tab_circuit:
    s = sol["stats"]
    c = st.columns(4)
    c[0].metric("Depth p", reps)
    c[1].metric("Trainable angles", s["parameters"])
    c[2].metric("CNOTs (transpiled)", s["cx"])
    c[3].metric("Circuit depth", s["depth"])

    left, right = st.columns(2)
    hist = qaoa.info["history"]
    fig = go.Figure(go.Scatter(y=hist, mode="lines", line=dict(color="#e5484d")))
    fig.add_hline(y=exact.energy, line_dash="dot", annotation_text="optimal cost")
    fig.update_layout(title="Training: COBYLA on ⟨cost⟩", xaxis_title="circuit evaluation", yaxis_title="objective", height=340, margin=dict(t=40))
    left.plotly_chart(fig, width="stretch")

    top = qaoa.info["top_states"][:12]
    opt = energies.min()
    labels = [plan_label(bit_matrix(n_qubits, b, b + 1)[0], net.n) for b, _, _ in top]
    colors = ["#2bb673" if np.isclose(energies[b], opt) else "#7c66dc" for b, _, _ in top]
    fig = go.Figure(go.Bar(x=[p for _, p, _ in top], y=labels, orientation="h", marker_color=colors))
    fig.add_vline(x=1 / 2**n_qubits, line_dash="dot", annotation_text="random guess")
    fig.update_layout(title="Most likely measured plans (green = optimal)", xaxis_title="probability", height=340, margin=dict(t=40), yaxis=dict(autorange="reversed"))
    right.plotly_chart(fig, width="stretch")

    with st.expander("Show the circuit (one QAOA layer)", expanded=n_qubits <= 8):
        st.pyplot(circuit_figure(scenario, rows, cols, horizon, seed, reps))
    st.caption("RZ gates encode each signal's own queue pressure, RZZ gates couple a signal to its next slot and to its neighbours, RX is the mixer.")

# --- tab: city simulation ---------------------------------------------------------------

with tab_city:
    n_slots = st.slider("Simulated minutes", 5, 30, 15) * 2
    if n_qubits > 12:
        st.info("Closed-loop QAOA re-optimises every 30 s; keep it to ≤ 12 qubits for a snappy demo.")
    if st.button("▶ Run the city", type="primary") or "city" in st.session_state:
        with st.spinner("Simulating every controller on identical traffic…"):
            results = run_city(scenario, rows, cols, horizon, seed, reps, n_slots)
        st.session_state["city"] = True
        fig = go.Figure()
        for res in results:
            minutes = np.arange(res.queues.shape[0]) * net.step_seconds / 60
            fig.add_trace(go.Scatter(x=minutes, y=res.queues.sum(axis=(1, 2)), name=res.policy, line=dict(color=policy_color(res.policy))))
        fig.update_layout(title="Vehicles waiting across the grid", xaxis_title="minutes", yaxis_title="queued vehicles", height=380, margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
        summary = pd.DataFrame([r.summary() for r in results]).drop(columns="scenario")
        best = summary.loc[summary.avg_wait_s.idxmin(), "policy"]
        fixed = summary.avg_wait_s.iloc[0]
        qaoa_wait = summary[summary.policy.str.contains("QAOA")].avg_wait_s.iloc[0]
        st.metric("QAOA controller vs fixed-time signals", f"{qaoa_wait:.1f} s wait", f"{(qaoa_wait - fixed) / fixed:+.0%}", delta_color="inverse")
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption(f"Best this run: **{best}**. Same random arrivals for every controller.")

# --- tab: benchmarks --------------------------------------------------------------------

with tab_bench:
    figs = ROOT / "results" / "figures"
    shown = False
    for name, caption in [
        ("closed_loop_wait.png", "Average wait per vehicle by controller and scenario"),
        ("qaoa_quality_vs_depth.png", "QAOA quality improves with depth"),
        ("scaling.png", "Scalability: circuit cost and solve time"),
        ("noise.png", "Effect of hardware noise"),
    ]:
        if (figs / name).exists():
            st.image(str(figs / name), caption=caption)
            shown = True
    if not shown:
        st.info("Run `python benchmarks/run_benchmarks.py` to generate the full study.")

# --- tab: how it works ------------------------------------------------------------------

with tab_how:
    st.markdown("#### Decision variables")
    st.latex(r"x_{i,t} = \begin{cases}1 & \text{east–west green at signal } i \text{ in slot } t\\ 0 & \text{north–south green}\end{cases}")
    st.markdown("#### Predicted queues (linear in x)")
    st.latex(r"Q^{EW}_{i,t} = Q^{EW}_{i,0} + (t{+}1)L\lambda^{EW}_i - c\sum_{s\le t} x_{i,s} + r\,\hat c\sum_{s\le t} x_{\text{up}(i),s}")
    st.markdown("#### Cost (quadratic in x → QUBO → Ising Hamiltonian)")
    st.latex(r"E(x) = w_q \sum_{i,t}\left[(Q^{NS}_{i,t})^2 + (Q^{EW}_{i,t})^2\right] + w_s\sum_{i,t}\left[x_{i,t}\ne x_{i,t-1}\right]")
    st.latex(r"x = \tfrac{1-Z}{2} \;\Rightarrow\; H_C = \sum_k h_k Z_k + \sum_{k<l} J_{kl} Z_k Z_l")
    st.markdown("#### QAOA")
    st.latex(r"|\gamma,\beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l \sum_k X_k}\, e^{-i\gamma_l H_C}\, |+\rangle^{\otimes n}")
    st.markdown(
        "Angles are trained with COBYLA (layer by layer, INTERP initialisation), the circuit is sampled, and the "
        "lowest-cost sampled plan is applied for 30 s before re-planning with fresh queue data."
    )
