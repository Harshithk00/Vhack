# Pitch and demo script (about 3 minutes)

## 1. Problem (20 s)

Most city signals still run fixed timings. At rush hour they waste green on
empty roads while queues on the busy road grow. Coordinating every signal in a
city is a combinatorial problem: 4 signals planned 1 minute ahead already have
256 possible plans, and a 4x4 grid has 2^32.

## 2. Idea (30 s)

Turn signal timing into a QUBO, the native input of quantum optimizers:

* one qubit = one signal in one 30-second slot (0 = NS green, 1 = EW green);
* cost = squared predicted queues + a penalty for switching;
* map it to an Ising Hamiltonian and solve it with **QAOA in Qiskit**;
* re-plan every 30 s with fresh queue data (model-predictive control).

Green waves are not hand-coded: they fall out of the coupling terms between
neighbouring signals.

## 3. Live demo (90 s): `streamlit run app/streamlit_app.py`

1. **Signal plan tab**: Rush Hour, 2x2 grid, 2 slots → 8 qubits. Show the queues,
   then QAOA's plan, with "QAOA found the optimal plan out of 256".
2. **Quantum circuit tab**: the training curve, the probability bars (the optimum
   is ~8x more likely than a random guess), and the circuit itself.
3. **City simulation tab**: Run the city. Same random traffic for every controller;
   QAOA control cuts the average wait from ~84 s (fixed-time) to ~30 s at rush
   hour.
4. Switch the backend to **Qiskit Aer, noisy shots** to show it still works with
   1 % gate errors.

## 4. Honest findings (30 s)

* In simulation QAOA reaches the exact optimum on every instance up to 12 qubits,
  and on most at 16.
* Deeper circuits help: the approximation ratio goes from 0.85 to 0.93 as p goes
  from 1 to 4.
* At this size, classical brute force and simulated annealing are faster.
  Quantum advantage would need hundreds of qubits with low noise, which is why
  we report CNOT counts and noise sensitivity.
* The pipeline is hardware-ready: the same circuit runs on IBM Quantum through
  Qiskit Runtime.

## 5. What's next (10 s)

Run on IBM Quantum hardware; split cities into corridors so each subproblem
fits on a device; connect to SUMO and real detector feeds.
