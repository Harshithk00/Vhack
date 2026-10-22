# Team NULL POINTER: pitch scripts

Harshit Kumar (Team Lead) · Himansu Rauniyar · Aditya Pattar · Dept. of CSE, SIT Tumkur

Links to have open: poster QR code → [vhack.harshithk.tech](https://vhack.harshithk.tech) (video),
[qtraffic.harshithk.tech](https://qtraffic.harshithk.tech) (live dashboard),
[github.com/Harshithk00/Vhack](https://github.com/Harshithk00/Vhack) (code, MIT).

## The one sentence everyone must be able to say

> We turned traffic-signal timing into a QUBO where every signal in every 30-second slot is a
> qubit, solved it with QAOA in Qiskit, and cut rush-hour waiting by 64% in simulation.

## 60-second pitch (when a judge stops at the poster)

| Who | Time | Say | Point at |
|---|---|---|---|
| **Harshit** | 0–15 s | "Hi, we're team NULL POINTER from CSE, SIT Tumkur. Most traffic lights still run on fixed timers, so at rush hour green time goes to empty roads while the busy road backs up. Timing many signals together is combinatorial: just 4 signals planned a minute ahead have 256 possible plans." | Introduction |
| **Himansu** | 15–35 s | "Each signal in each 30-second slot is one qubit: 0 is north–south green, 1 is east–west. The cost is the square of every predicted queue plus a switching penalty. Squaring couples neighbouring signals, so green waves appear on their own. We map that to an Ising Hamiltonian, and this QAOA circuit, built in Qiskit, finds the best plan. We re-plan every 30 seconds." | Theory card → circuit |
| **Aditya** | 35–55 s | "On identical simulated traffic, our controller cut rush-hour waiting by 64% versus fixed timers and beat queue-actuated signals too. QAOA found the true optimum on every instance up to 12 qubits, and it still did with 1% gate noise." | Results figure, −64% |
| **Harshit** | 55–60 s | "Honestly, classical solvers are still faster at this size. Next is IBM Quantum hardware and splitting cities into corridors. Scan the QR code to try it live." | QR code |

## 3-minute version (if the judge has time)

1. **Harshit, problem (30 s):** fixed timers; 83.5 s average wait at rush hour in our simulation;
   plans grow as 2^(signals × slots), so a 4×4 grid has 2^32.
2. **Himansu, method (60 s):** qubit encoding → predicted queues (linear in the bits) → QUBO →
   Ising H = Σ hZ + Σ JZZ → QAOA (H layer, RZ/RZZ cost, RX mixer, p layers) → COBYLA trains γ, β
   with INTERP initialisation → 1024 shots → cheapest plan runs for 30 s → re-plan with live
   queues and warm-started angles.
3. **Aditya, live demo (60 s):** open qtraffic.harshithk.tech → Rush Hour, 2×2, 2 slots
   (8 qubits) → "QAOA found the optimal plan out of 256" → Quantum circuit tab (training curve,
   optimum ~8× likelier than random) → City simulation → Run → switch backend to *Qiskit Aer,
   noisy shots*.
4. **Harshit, honest results and next steps (30 s):** approximation ratio 0.85 → 0.93 as p goes
   1 → 4; still optimal at 1% noise; classical solvers faster today; next: IBM hardware, corridor
   decomposition, SUMO and live detector data.

## Rules for the day

* Everyone speaks; some rubrics score it directly.
* Lead with the number (−64%) and the live demo; judges remember both.
* Say the limitation before you are asked. Honest comparison earns points.
* If the live site is slow, fall back to the video on vhack.harshithk.tech.
* Bring a phone with both links open, and a charger.
