"""Result figures for the README.

Three panels, each answering one question: did calibration work, where does
the model actually have an edge, and what would that edge have earned.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"


def _style(ax) -> None:
    """Recede the frame so the marks carry the chart."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def plot_reliability(ax, curve: pd.DataFrame) -> None:
    """Predicted probability against observed frequency."""
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    # Marker area tracks bin population, so sparsely populated bins cannot
    # masquerade as evidence. The top bin holds seven forecasts.
    counts = curve["n"].to_numpy()
    sizes = 18 + 190 * np.sqrt(counts / counts.max())

    ax.plot(
        curve["mean_predicted"], curve["observed_rate"],
        color=BLUE, linewidth=2.0, zorder=2,
    )
    ax.scatter(
        curve["mean_predicted"], curve["observed_rate"],
        s=sizes, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3,
    )

    sparse = curve.loc[curve["n"] < 50]
    for _, row in sparse.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["mean_predicted"], row["observed_rate"]),
            textcoords="offset points", xytext=(-4, -16),
            ha="right", color=MUTED, fontsize=8,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability", color=MUTED, fontsize=9)
    ax.set_ylabel("Observed frequency", color=MUTED, fontsize=9)
    ax.set_title(
        "Calibration holds across the range",
        color=INK, fontsize=11, fontweight="bold", loc="left", pad=10,
    )
    ax.text(
        0.30, 0.20, "perfect calibration",
        transform=ax.transAxes, color=MUTED, fontsize=8, rotation=37,
    )
    ax.text(
        0.04, 0.93, "marker size = forecasts in bin",
        transform=ax.transAxes, color=MUTED, fontsize=8,
    )
    _style(ax)


def plot_edge_profile(ax, profile: pd.DataFrame) -> None:
    """Realised return by how far the model departs from the market."""
    labels = ["<0%", "0-5%", "5-10%", "10-20%", "20-40%", ">40%"]
    values = profile["roi"].to_numpy() * 100
    colors = [BLUE if v > 0 else RED for v in values]

    bars = ax.bar(labels, values, color=colors, width=0.62, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)

    for bar, value in zip(bars, values):
        offset = 0.7 if value > 0 else -1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.1f}%",
            ha="center", color=INK, fontsize=8.5, fontweight="bold",
        )

    ax.set_ylim(min(values) - 4, max(values) + 4)
    ax.set_xlabel("Model probability vs market (relative edge)", color=MUTED, fontsize=9)
    ax.set_ylabel("Return per unit staked", color=MUTED, fontsize=9)
    ax.set_title(
        "Profit lives in a band, not the tail",
        color=INK, fontsize=11, fontweight="bold", loc="left", pad=10,
    )
    _style(ax)


def plot_equity(ax, equity: pd.DataFrame) -> None:
    """Cumulative profit across the walk-forward evaluation period."""
    seasons = equity["Season"].tolist()
    cumulative = equity["cumulative_profit"].to_numpy()
    positions = np.arange(len(seasons))

    ax.plot(positions, cumulative, color=BLUE, linewidth=2.2, zorder=3)
    ax.fill_between(positions, 0, cumulative, color=BLUE, alpha=0.12, zorder=2)
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)

    ax.scatter(
        positions[-1], cumulative[-1], s=70, color=BLUE,
        edgecolor=SURFACE, linewidth=2, zorder=4,
    )
    ax.annotate(
        f"£{cumulative[-1]:,.0f}",
        (positions[-1], cumulative[-1]), textcoords="offset points",
        xytext=(-6, 12), ha="right", color=INK, fontsize=10, fontweight="bold",
    )

    ax.set_xticks(positions[::2])
    ax.set_xticklabels([seasons[i] for i in range(0, len(seasons), 2)], rotation=0)
    ax.set_ylabel("Cumulative profit (£1,000 start)", color=MUTED, fontsize=9)
    ax.set_title(
        "Fractional-Kelly bankroll, 12 held-out seasons",
        color=INK, fontsize=11, fontweight="bold", loc="left", pad=10,
    )
    _style(ax)


def main() -> str:
    """Render the three-panel results figure."""
    results = json.loads((config.REPORT_DIR / "results.json").read_text())
    curve = pd.DataFrame(results["reliability_curve"])
    profile = pd.DataFrame(results["edge_profile"])
    equity = pd.DataFrame(results["equity_by_season"])

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
    plot_reliability(axes[0], curve)
    plot_edge_profile(axes[1], profile)
    plot_equity(axes[2], equity)

    figure.suptitle(
        "Premier League Decision Engine  ·  walk-forward results, 2013/14-2024/25",
        color=INK, fontsize=13, fontweight="bold", x=0.008, ha="left", y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    output = config.REPORT_DIR / "results.png"
    figure.savefig(output, dpi=170, facecolor=SURFACE)
    plt.close(figure)
    return str(output)


if __name__ == "__main__":  # pragma: no cover
    print(main())
