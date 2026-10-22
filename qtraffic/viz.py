"""Matplotlib figures shared by the benchmarks and the README."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .network import EW_GREEN, TrafficNetwork  # noqa: E402

GREEN, RED, ROAD = "#2bb673", "#e5484d", "#9aa4b2"
POLICY_COLORS = {
    "Fixed-time": "#9aa4b2",
    "Max-queue (actuated)": "#f5a524",
    "QUBO + Simulated annealing": "#7c66dc",
    "QUBO + Exact (brute force)": "#3b82f6",
}
QAOA_COLOR = "#e5484d"


def policy_color(name: str) -> str:
    return POLICY_COLORS.get(name, QAOA_COLOR if name.startswith("QUBO + QAOA") else "#555")


def draw_network(ax, network: TrafficNetwork, q_ns, q_ew, phase, title: str = "") -> None:
    """Grid of signals: bars show queue lengths, the bar colour shows green/red."""
    ax.set_aspect("equal")
    for r in range(network.rows):
        ax.plot([-0.6, network.cols - 0.4], [-r, -r], color=ROAD, lw=6, zorder=0, solid_capstyle="round")
    for c in range(network.cols):
        ax.plot([c, c], [0.6, -network.rows + 0.4], color=ROAD, lw=6, zorder=0, solid_capstyle="round")
    scale = 0.35 / max(1.0, float(np.max(q_ns)), float(np.max(q_ew)))
    for i in range(network.n):
        r, c = network.position(i)
        y = -r
        ew = phase[i] == EW_GREEN
        ax.barh(y, -q_ew[i] * scale, left=c - 0.08, height=0.12, color=GREEN if ew else RED, zorder=2)
        ax.bar(c, q_ns[i] * scale, bottom=y + 0.08, width=0.12, color=RED if ew else GREEN, zorder=2)
        ax.scatter([c], [y], s=180, color="#1f2937", zorder=3)
        ax.text(c + 0.1, y - 0.18, network.label(i), fontsize=8, color="#374151")
    ax.set_xlim(-0.7, network.cols - 0.3)
    ax.set_ylim(-network.rows + 0.3, 0.7)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
