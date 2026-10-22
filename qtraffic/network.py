"""Road network model: a grid of signalised intersections on one-way streets.

Every intersection has two approaches:

* NS - traffic on the north-south street, flowing southbound;
* EW - traffic on the east-west street, flowing eastbound.

A signal shows exactly one green at a time, so each intersection's decision for a
time slot is a single bit::

    x = 0  ->  north-south green
    x = 1  ->  east-west green

Vehicles discharged on a green either continue to the next intersection downstream
(``through_ratio``) or turn off the grid. Time is measured in simulation *steps*
(5 seconds by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NS_GREEN = 0
EW_GREEN = 1


@dataclass
class TrafficNetwork:
    rows: int
    cols: int
    arrival_ns: np.ndarray
    arrival_ew: np.ndarray
    saturation_flow: float = 2.0  # vehicles a green approach discharges per step (~1440 veh/h)
    through_ratio: float = 0.7  # share of discharged vehicles continuing downstream
    switch_loss: float = 0.5  # share of one slot's first-step capacity lost to amber/all-red
    step_seconds: float = 5.0
    name: str = "custom"

    def __post_init__(self) -> None:
        n = self.rows * self.cols
        self.arrival_ns = np.broadcast_to(np.asarray(self.arrival_ns, float), (n,)).copy()
        self.arrival_ew = np.broadcast_to(np.asarray(self.arrival_ew, float), (n,)).copy()
        if self.rows < 1 or self.cols < 1:
            raise ValueError("network needs at least one intersection")

    @property
    def n(self) -> int:
        return self.rows * self.cols

    def index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def position(self, i: int) -> tuple[int, int]:
        return divmod(i, self.cols)

    def label(self, i: int) -> str:
        r, c = self.position(i)
        return f"{chr(ord('A') + r)}{c + 1}"

    def downstream_ew(self, i: int) -> int | None:
        r, c = self.position(i)
        return self.index(r, c + 1) if c + 1 < self.cols else None

    def downstream_ns(self, i: int) -> int | None:
        r, c = self.position(i)
        return self.index(r + 1, c) if r + 1 < self.rows else None

    def upstream_ew(self, i: int) -> int | None:
        r, c = self.position(i)
        return self.index(r, c - 1) if c > 0 else None

    def upstream_ns(self, i: int) -> int | None:
        r, c = self.position(i)
        return self.index(r - 1, c) if r > 0 else None

    def links(self) -> list[tuple[int, int, str]]:
        """Directed road links ``(from, to, direction)`` between intersections."""
        out = []
        for i in range(self.n):
            if (j := self.downstream_ew(i)) is not None:
                out.append((i, j, "EW"))
            if (j := self.downstream_ns(i)) is not None:
                out.append((i, j, "NS"))
        return out

    def describe(self) -> str:
        demand = self.arrival_ns.sum() + self.arrival_ew.sum()
        veh_per_hour = demand * 3600 / self.step_seconds
        return (
            f"{self.name}: {self.rows}x{self.cols} grid, {self.n} signals, "
            f"external demand {veh_per_hour:,.0f} veh/h"
        )
