"""Ready-made traffic conditions used by the benchmarks and the dashboard.

Arrival rates are in vehicles per 5-second step. Main entries are the west column
(eastbound traffic) and the north row (southbound traffic); interior intersections
only get light side-street demand, the rest comes from upstream signals.
"""

from __future__ import annotations

import numpy as np

from .network import TrafficNetwork

SIDE_STREET_SHARE = 0.25

SCENARIOS: dict[str, str] = {
    "off_peak": "Light, balanced traffic in both directions",
    "rush_hour": "Heavy eastbound commuter flow on the arterial, normal cross traffic",
    "event_surge": "Balanced traffic plus a stadium letting out onto one intersection",
    "random": "Randomly drawn demand per approach (seeded)",
}


def _entry_rates(rows: int, cols: int, main_ns: float, main_ew: float) -> tuple[np.ndarray, np.ndarray]:
    ns = np.full((rows, cols), main_ns * SIDE_STREET_SHARE)
    ew = np.full((rows, cols), main_ew * SIDE_STREET_SHARE)
    ns[0, :] = main_ns
    ew[:, 0] = main_ew
    return ns.ravel(), ew.ravel()


def make_scenario(name: str, rows: int = 2, cols: int = 2, seed: int = 0) -> TrafficNetwork:
    if name == "off_peak":
        ns, ew = _entry_rates(rows, cols, 0.30, 0.30)
    elif name == "rush_hour":
        ns, ew = _entry_rates(rows, cols, 0.30, 1.00)
    elif name == "event_surge":
        ns, ew = _entry_rates(rows, cols, 0.45, 0.45)
        hot = (rows * cols) // 2
        ns[hot] += 0.70
    elif name == "random":
        rng = np.random.default_rng(seed)
        ns, ew = _entry_rates(rows, cols, 1.0, 1.0)
        ns = ns * rng.uniform(0.2, 0.7, rows * cols)
        ew = ew * rng.uniform(0.2, 0.7, rows * cols)
    else:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return TrafficNetwork(rows, cols, arrival_ns=ns, arrival_ew=ew, name=name)
