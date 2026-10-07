"""Small plotting helpers: every figure is saved to reports/figures/ and closed.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  

from project_paths import FIGURES  

plt.rcParams.update({"figure.figsize": (12, 4), "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})

STATE_ORDER = ["SHUTDOWN", "MAINTENANCE", "RAMP_UP", "NORMAL", "HIGH_DEMAND"]
STATE_COLORS = {"SHUTDOWN": "tab:gray", "MAINTENANCE": "tab:purple", "RAMP_UP": "tab:orange",
                "NORMAL": "tab:blue", "HIGH_DEMAND": "tab:red"}


def save(fig: plt.Figure, name: str) -> None:
    """Save `fig` as reports/figures/<name>.png and close it."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=110)
    plt.close(fig)
    print(f"   [figure saved] reports/figures/{name}.png")
