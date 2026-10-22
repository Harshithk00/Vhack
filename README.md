# Quantum-Assisted Traffic Signal Optimization Using QAOA

> Hackathon project: can a quantum optimizer decide when our traffic lights turn green?

Urban congestion wastes time, fuel and clean air. This project models a small city
grid of traffic signals as a combinatorial optimization problem, solves it with the
**Quantum Approximate Optimization Algorithm (QAOA)** in **Qiskit**, and compares it
with classical optimizers under different traffic conditions.

## Status

- [x] Phase 1: traffic network model, scenarios and queue simulator
- [x] Phase 2: QUBO formulation and classical solvers
- [x] Phase 3: QAOA solver
- [ ] Phase 4: benchmarks
- [ ] Phase 5: interactive dashboard
- [ ] Phase 6: docs and polish

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```
