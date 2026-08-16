"""
What the events cost: the radiative energy of each footprint.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants
from ._observations import raster
from ._overview import path_figures, _velocity_centers
from ._search import score_maps, components
from ._averages import _classes

__all__ = [
    "energies",
]

#: How long the slit spends on one step of this raster.
timedelta_step = 32.19 * u.s

#: The velocity range whose excess counts as the line.
velocity_line = 150 * u.km / u.s


def _energy_per_event(
    significance_min: float = 7,
    significance_low: float = 4,
) -> dict[str, np.ndarray]:
    """
    The observed Si IV radiative energy of every census patch.

    The excess radiance over the median profile, integrated over the line,
    the footprint, and the time the slit spent there, into all directions.
    A lower bound on the event's total output: the event shines before and
    after the slit visits, and the Si IV line is a small fraction of the
    transition region's radiative losses.

    Parameters
    ----------
    significance_min
        The seed floor of the census.
    significance_low
        The extension threshold of the patches.
    """
    from ._catalog import catalog
    from ._deconvolve import deconvolved

    table = catalog(
        significance_min=significance_min,
        significance_low=significance_low,
    )

    obs = raster()
    sharp = deconvolved()

    axis_time = obs.axis_time
    axis_wavelength = obs.axis_wavelength
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y
    index_time = {axis_time: 0}

    # the patches, re-derived exactly as the catalog derived them
    maps = score_maps(sharp, sharpened=True)
    significance = u.Quantity(maps.significance_any[index_time].ndarray).value

    velocity = _velocity_centers(obs).ndarray.to_value(u.km / u.s)
    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))
    excess = u.Quantity((obs.outputs[index_time] - median).ndarray)

    where_line = np.abs(velocity) < velocity_line.to_value(u.km / u.s)
    where_continuum = velocity > 250

    pos = obs.outputs[index_time].axes.index(axis_wavelength)

    # one wavelength cell, from the velocity pitch
    step = np.abs(np.median(np.diff(velocity)))
    dl = (1393.755 * u.AA * step / 299792.458).to(u.nm)

    # the per-pixel continuum offset, removed so that only the line counts
    offset = (
        np.nanmedian(
            np.where(where_continuum, u.Quantity(excess).value, np.nan), axis=pos
        )
        << excess.unit
    )

    line = (
        np.nansum(np.where(where_line, u.Quantity(excess).value, 0), axis=pos)
        << excess.unit
    )
    num_line = int(where_line.sum())
    intensity = (line - num_line * offset) * dl

    # the area of one pixel on the Sun, foreshortening included
    position = obs.inputs.position[index_time].cell_centers((axis_x, axis_y))
    px = position.x.ndarray.to_value(u.arcsec)
    py = position.y.ndarray.to_value(u.arcsec)
    scale_x = np.abs(np.nanmean(np.diff(px, axis=0))) * u.arcsec
    scale_y = np.abs(np.nanmean(np.diff(py, axis=1))) * u.arcsec
    radius = np.hypot(np.nanmedian(px), np.nanmedian(py))
    mu = float(np.sqrt(1 - (radius / 959.6) ** 2))
    per_arcsec = (1 * u.arcsec).to_value(u.rad) * astropy.constants.au
    area = (scale_x.value * per_arcsec) * (scale_y.value * per_arcsec) / mu

    energy = np.full(len(table["x"]), np.nan)
    for patch in components(significance, significance_low):
        rows, cols = patch[:, 0], patch[:, 1]
        values = significance[rows, cols]
        best = int(np.argmax(values))
        if float(values[best]) < significance_min:
            continue
        x = px[rows[best], cols[best]]
        y = py[rows[best], cols[best]]
        i = int(np.argmin(np.hypot(table["x"] - x, table["y"] - y)))
        total = intensity[rows, cols].sum()
        result = (4 * np.pi * u.sr) * total * area * timedelta_step
        energy[i] = result.to_value(u.erg)

    return {"energy": energy}


def energies(
    figsize: tuple[float, float] = (16, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The energy distribution of the census events.

    Three panels: the distribution of observed Si IV radiative energies on
    logarithmic axes; the survival function, whose upper-end slope is what
    the heating-budget argument turns on; and energy against footprint
    area.

    Parameters
    ----------
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    from ._catalog import catalog

    table = catalog()
    energy = _energy_per_event()["energy"]

    good = np.isfinite(energy) & (energy > 0)

    fig, axs = plt.subplots(ncols=3, figsize=figsize, constrained_layout=True)

    ax = axs[0]
    bins = np.geomspace(energy[good].min() * 0.9, energy[good].max() * 1.1, 12)
    for name, sel, color in _classes(table["direction"]):
        keep = sel & good
        histogram, edges = np.histogram(energy[keep], bins=bins)
        centers = np.sqrt(edges[:~0] * edges[1:])
        nonzero = histogram > 0
        ax.plot(
            centers[nonzero],
            histogram[nonzero],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1,
            label=f"{name} ({keep.sum()})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("observed Si IV radiative energy (erg)")
    ax.set_ylabel("events per bin")
    ax.legend(fontsize=8)
    ax.set_title("the energy distribution", fontsize=10)

    ax = axs[1]
    ordered = np.sort(energy[good])
    survival = np.arange(len(ordered), 0, -1)
    ax.plot(ordered, survival, color="black", drawstyle="steps-post")
    # the crude upper-decade slope, for orientation rather than for print
    upper = ordered > ordered[-1] / 10
    if upper.sum() > 5:
        fit = np.polyfit(np.log10(ordered[upper]), np.log10(survival[upper]), 1)
        grid = np.geomspace(ordered[-1] / 10, ordered[-1], 10)
        ax.plot(
            grid,
            10 ** np.polyval(fit, np.log10(grid)),
            color="tab:orange",
            linestyle="dashed",
            label=f"upper-decade slope {fit[0]:.2f}",
        )
        ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("observed Si IV radiative energy (erg)")
    ax.set_ylabel("events above")
    ax.set_title("the survival function", fontsize=10)

    ax = axs[2]
    colors = {0: "tab:purple", -1: "tab:blue", 1: "tab:red"}
    names = {0: "bidirectional", -1: "blue jets", 1: "red jets"}
    for direction, color in colors.items():
        sel = (table["direction"] == direction) & good
        ax.scatter(
            table["area"][sel], energy[sel], s=18, color=color, label=names[direction]
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("footprint area (arcsec$^2$)")
    ax.set_ylabel("observed Si IV radiative energy (erg)")
    ax.legend(fontsize=8)
    ax.set_title("energy against size", fontsize=10)

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "energy.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
