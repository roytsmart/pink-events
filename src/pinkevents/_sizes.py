"""
How big the events are, now that they are patches instead of points.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from ._overview import path_figures
from ._averages import _classes

__all__ = [
    "sizes",
]


def sizes(
    figsize: tuple[float, float] = (16, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The size distribution of the census events.

    Three panels: the distribution of footprint areas on logarithmic axes,
    where a power law would be a straight line; the extent along the slit
    against the extent across it, which are not the same kind of quantity,
    since the slit sweeps a third of an arcsecond every thirty two seconds
    and the crossing extent therefore folds lifetime into size; and the
    area against the significance, which says whether bigger means
    stronger.

    Parameters
    ----------
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    from ._catalog import catalog

    table = catalog()

    area = table["area"]
    extent_x = table["extent_x"]
    extent_y = table["extent_y"]
    sig = table["sig"]

    fig, axs = plt.subplots(ncols=3, figsize=figsize, constrained_layout=True)

    ax = axs[0]
    bins = np.geomspace(area.min() * 0.9, area.max() * 1.1, 12)
    for name, sel, color in _classes(table["direction"]):
        histogram, edges = np.histogram(area[sel], bins=bins)
        centers = np.sqrt(edges[:~0] * edges[1:])
        keep = histogram > 0
        ax.plot(
            centers[keep],
            histogram[keep],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1,
            label=f"{name} ({sel.sum()})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("footprint area (arcsec$^2$)")
    ax.set_ylabel("events per bin")
    ax.legend(fontsize=8)
    ax.set_title("the size distribution", fontsize=10)

    ax = axs[1]
    colors = {0: "tab:purple", -1: "tab:blue", 1: "tab:red"}
    names = {0: "bidirectional", -1: "blue jets", 1: "red jets"}
    for direction, color in colors.items():
        sel = table["direction"] == direction
        ax.scatter(
            extent_x[sel],
            extent_y[sel],
            s=18,
            color=color,
            label=names[direction],
        )
    limit = max(extent_x.max(), extent_y.max()) * 1.1
    ax.plot([0, limit], [0, limit], color="gray", linewidth=0.7, linestyle="dashed")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("extent across the slit (arcsec, folds in lifetime)")
    ax.set_ylabel("extent along the slit (arcsec)")
    ax.legend(fontsize=8)
    ax.set_title("shape, or lifetime in disguise", fontsize=10)

    ax = axs[2]
    for direction, color in colors.items():
        sel = table["direction"] == direction
        ax.scatter(area[sel], sig[sel], s=18, color=color, label=names[direction])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("footprint area (arcsec$^2$)")
    ax.set_ylabel("significance ($\\sigma$)")
    ax.legend(fontsize=8)
    ax.set_title("bigger against stronger", fontsize=10)

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "sizes.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
