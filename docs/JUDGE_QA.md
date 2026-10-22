# Judge Q&A cheat sheet

Short, true answers. Every number here comes from `results/summary.md`.

## The quantum part

**Is there a quantum advantage?**
Not at this size, and we say so. Brute force solves 18 qubits in about 0.1 s. What we show is
feasibility: QAOA finds optimal plans, its quality improves with depth, it survives realistic
noise, and the formulation scales to hardware-sized problems.

**Why QAOA and not quantum annealing?**
QAOA runs on gate-based IBM Quantum hardware through Qiskit, and its depth p trades circuit cost
for quality, which we measured (0.85 → 0.93 from p = 1 to 4). Annealing work on traffic
(Neukart et al., 2017) targets routing, not signal timing.

**What does each gate do?**
H puts every plan in superposition. RZ(2γh) encodes each signal's own queue pressure. RZZ(2γJ)
encodes couplings: the same signal across time slots, and neighbouring signals along a street.
RX(2β) is the mixer. We measure 1024 shots and keep the cheapest plan.

**How are the angles trained?**
COBYLA minimises the expected cost. Layer-by-layer INTERP initialisation (Zhou et al., PRX 2020)
starts depth p+1 from the best depth-p angles. In closed loop, the previous slot's angles
warm-start the next solve, which needs far fewer circuit evaluations than a cold start.

**How often does QAOA return the optimum?**
Every instance up to 12 qubits at p = 2, and 78% at 16 qubits (100% at p = 4). The probability of
measuring the optimum stays small (3–5% for 8 qubits), but that is 8–12× a random guess, and
sampling 1024 shots finds it.

**What about noise?**
On Qiskit Aer with depolarising noise, QAOA still found the optimum at 1% two-qubit error. The
approximation ratio drops as noise rises above about 1–2%, which is why hardware runs need error
mitigation.

**Did you run on real hardware?**
Not yet. Everything is simulated (Qiskit Statevector, and Aer with noise). The same circuit is
hardware-ready through Qiskit Runtime; that is our first next step.

## The traffic model

**Why square the queues?**
It makes the cost quadratic, which is what a QUBO needs, and it penalises long queues more than
short ones. The cross terms of the square couple neighbouring signals, so green waves emerge
without being programmed.

**How realistic is the simulator?**
A stochastic queue model: Poisson arrivals, saturation flow of about 1440 veh/h per green
approach, 70% of vehicles continue downstream, and capacity is lost on each switch (amber/all-red).
It is simplified (one-way streets, two phases, no turns), and the README says so.

**Is the comparison fair?**
Yes. Every controller sees identical random arrivals, over 30 simulated minutes and 3 seeds.
Fixed-time even wins in light balanced traffic, which we report.

**What are the scenarios?**
Off-peak, rush hour (heavy eastbound arterial), event surge (a stadium letting out at one
junction) and random demand.

## Scale and impact

**How many qubits would a real city need?**
Qubits = signals × time slots. A 4×4 grid planned 1 minute ahead is 32 qubits, about 450 CNOTs at
p = 2. A full city would be split into corridors or sub-grids that each fit on one device, solved
in parallel.

**Why does this matter if classical is faster today?**
Signal coordination is a real combinatorial problem that grows exponentially, and it is a natural
QUBO. We built the full pipeline (formulation, circuit, closed-loop control, benchmarks), so it
can move to better hardware as it arrives.

**What would you do with more time?**
Run on IBM Quantum hardware with error mitigation, add corridor decomposition, and validate
on SUMO with real detector data.

## About the work

**What is original here?**
Signal timing (not routing) as a QUBO with squared predicted queues, so coordination emerges from
the physics. QAOA tested as a live receding-horizon controller in closed loop. Honest benchmarks
against four classical baselines, including noise and scaling.

**Can we see the code?**
Yes: github.com/Harshithk00/Vhack (MIT licensed), with 26 automated tests. The live dashboard is
at qtraffic.harshithk.tech.
