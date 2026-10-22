# Problem formulation

## 1. Traffic model

A `rows x cols` grid of signalised intersections on one-way streets: east-west
streets flow eastbound, north-south streets flow southbound. Every intersection
has two approaches (NS and EW) and shows exactly one green at a time.

| Symbol | Meaning | Default |
|---|---|---|
| step | simulation time step | 5 s |
| `L` | slot length (steps between re-planning) | 6 steps = 30 s |
| `s` | saturation flow of a green approach | 2 veh/step (~1440 veh/h) |
| `c = sL` | vehicles a green can discharge per slot | 12 |
| `r` | share of discharged vehicles continuing to the next signal | 0.7 |
| `λ` | external arrival rate per approach (Poisson) | per scenario |
| switch loss | capacity lost in the step a signal changes (amber / all-red) | 50 % |

The simulator (`qtraffic/simulator.py`) is a fluid-queue model with stochastic
arrivals. It scores controllers on average queueing delay per vehicle, throughput,
queue lengths and number of switches. It is deliberately *separate* from the
optimisation model, so the controllers are judged on traffic outcomes, not on
their own objective.

## 2. Decision variables

```
x[i, t] = 1  ->  intersection i shows east-west green in slot t
x[i, t] = 0  ->  north-south green
```

`N` signals and a look-ahead of `T` slots give `n = N·T` binary variables, which
become `n` qubits. Qubit `k = t·N + i`.

## 3. Predicted queues

For every approach, the queue at the end of slot `t` is linear in `x`:

```
Q_ew[i,t] = Q0_ew[i] + (t+1)·L·λ_ew[i]  -  c · Σ_{s≤t} x[i,s]  +  r·ĉ_ew[u] · Σ_{s≤t} x[u,s]
Q_ns[i,t] = Q0_ns[i] + (t+1)·L·λ_ns[i]  -  c · Σ_{s≤t} (1-x[i,s])  +  r·ĉ_ns[v] · Σ_{s≤t} (1-x[v,s])
```

where `u` / `v` are the upstream intersections on the EW / NS street, and
`ĉ = min(c, Q0 + L·λ)` caps the platoon an upstream green can release by what is
actually queued there.

## 4. Objective (QUBO)

```
E(x) = w_q · Σ_{i,t} ( Q_ns[i,t]² + Q_ew[i,t]² )  +  w_s · Σ_{i,t} [ x[i,t] ≠ x[i,t-1] ]
```

* **Squared queues** penalise long queues more than short ones (fairness, and
  spill-back risk). Squaring a linear expression gives the quadratic terms:
  * the same signal over consecutive slots (`x[i,s]·x[i,s']`): serving a
    direction twice has diminishing returns;
  * neighbouring signals (`x[u,s]·x[i,t]`): a negative coupling when both are
    green along the same street, so **green waves emerge** from the model rather
    than being hand-coded.
* **Switching penalty**: `x_a ≠ x_b` is `x_a + x_b − 2·x_a·x_b`; for the first slot
  the previous phase is a known constant, so the term is linear.

Because `x² = x` for binaries, every term folds into

```
E(x) = offset + Σ_k a_k x_k + Σ_{k<l} b_kl x_k x_l
```

`tests/test_qubo.py` checks that this QUBO energy equals a direct evaluation of the
queue model for random plans.

## 5. Ising Hamiltonian

Substituting `x = (1 − Z)/2`:

```
H_C = Σ_k h_k Z_k + Σ_{k<l} J_kl Z_k Z_l + const
h_k = −a_k/2 − Σ_l (b_kl + b_lk)/4,     J_kl = b_kl / 4
```

The Hamiltonian is rescaled so `max |h|, |J| = 1`; this keeps good QAOA angles in
the same range for every instance.

## 6. QAOA

```
|γ, β⟩ = Π_{l=1..p}  e^{−iβ_l Σ X_k} · e^{−iγ_l H_C}  · H^{⊗n} |0⟩
```

* cost layer: `RZ(2γ h_k)` on each qubit and `RZZ(2γ J_kl)` on each coupled pair;
* mixer: `RX(2β)` on each qubit;
* angles trained by COBYLA on `⟨H_C⟩` (or CVaR_α), layer by layer with the INTERP
  initialisation, starting from a linear-ramp schedule;
* the trained circuit is sampled (1024 shots) and the lowest-cost sampled plan is
  returned;
* in closed loop the previous slot's angles warm-start the next optimisation.

## 7. Control loop

Every 30 s: read queues → build QUBO → solve → apply the first slot of the plan →
repeat (model-predictive / receding-horizon control).

## Known simplifications

* One-way streets, two phases per signal, no turning movements or pedestrians.
* The prediction model is linear (a queue can be predicted negative); the squared
  objective penalises over-serving an empty approach, which keeps this harmless.
* Arrival rates are assumed known; in deployment they would come from detectors.
