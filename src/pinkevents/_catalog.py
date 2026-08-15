"""
Every explosive event in the raster, and the magnetic ground it stands on.
"""

import pathlib
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import astropy.visualization
import named_arrays as na
from ._caching import memory
from ._observations import raster
from ._rgb import rgb
from ._network import network, distance_to
from ._magnetic import flux_density
from ._search import score_maps
from ._overview import path_figures, _plot_image

__all__ = [
    "catalog",
    "statistics",
    "profiles",
]

#: How finely the magnetogram times are sampled. The field barely evolves
#: over this, and it keeps the census from downloading a magnetogram for
#: every one of the raster's four hundred steps.
cadence_magnetogram = 10 * u.min


def _time_rounded(time_jd: float) -> astropy.time.Time:
    """The moment, rounded to the magnetogram sampling."""
    step = cadence_magnetogram.to_value(u.day)
    return astropy.time.Time(np.round(time_jd / step) * step, format="jd")


#: Bumped whenever the scoring upstream of the catalog changes: the cache
#: hashes only the function below, and cannot see that `score_maps` moved
#: under it, which once served a stale census as though nothing had
#: happened.
version_scoring = 3


@memory.cache
def _catalog(
    significance_min: float,
    separation: float,
    version: int,
) -> dict[str, npt.NDArray]:
    """
    The census, as plain arrays: one row per event.

    Cached, since building it samples a magnetogram under every event and
    every control point.

    Parameters
    ----------
    significance_min
        The smallest score, in standard deviations of the continuum noise,
        counted as an event.
    separation
        The de-duplication radius in arcseconds: a detection this close to
        a stronger one is the same event.
    """
    obs = raster()

    axis_time = obs.axis_time
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    index_time = {axis_time: 0}

    maps = score_maps(obs)
    significance = maps.significance_any[index_time]
    significance_blue = maps.significance_blue[index_time]
    significance_red = maps.significance_red[index_time]

    position = obs.inputs.position[index_time].cell_centers((axis_x, axis_y))

    net = network()

    def census(sig_map: np.ndarray) -> list[dict]:
        """The greedy champion walk, on whatever significance map."""
        order = np.argsort(-np.nan_to_num(sig_map), axis=None)
        kept = []
        for flat in order:
            index_nd = np.unravel_index(flat, sig_map.shape)
            index = {ax: int(i) for ax, i in zip(significance.axes, index_nd)}
            sig = float(sig_map[index_nd])
            if not np.isfinite(sig) or sig < significance_min:
                break
            here = position[index]
            x = float(here.x.ndarray.to_value(u.arcsec))
            y = float(here.y.ndarray.to_value(u.arcsec))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if any(np.hypot(x - e["x"], y - e["y"]) < separation for e in kept):
                continue
            kept.append({"x": x, "y": y, "sig": sig, "index": index})
        return kept

    sig_blue_map = u.Quantity(significance_blue.ndarray).value
    sig_red_map = u.Quantity(significance_red.ndarray).value

    # The false discovery rate, measured rather than assumed: the same
    # detector walked over the wing deficits, which no real event produces,
    # counts what noise and systematics alone put over the floor.
    num_deficit = len(census(np.maximum(-sig_blue_map, -sig_red_map)))

    events = []
    for found in census(u.Quantity(significance.ndarray).value):
        index = found["index"]
        x = found["x"]
        y = found["y"]
        sig = found["sig"]
        moment = obs.inputs.time[{axis_time: 0, axis_x: index[axis_x]}].ndarray
        jd = float(astropy.time.Time(moment).jd)

        # Which way the event flows: both wings clearing the floor is
        # bidirectional, otherwise the event belongs to its stronger wing.
        sig_blue = float(u.Quantity(significance_blue[index].ndarray).value)
        sig_red = float(u.Quantity(significance_red[index].ndarray).value)
        if min(sig_blue, sig_red) >= significance_min:
            direction = 0
        elif sig_blue > sig_red:
            direction = -1
        else:
            direction = +1

        events.append(
            {
                "x": x,
                "y": y,
                "sig": sig,
                "sig_blue": sig_blue,
                "sig_red": sig_red,
                "direction": direction,
                "jd": jd,
                "index_x": index[axis_x],
            }
        )

    def sample(x: float, y: float, jd: float) -> tuple[float, float]:
        here = na.Cartesian2dVectorArray(x=x * u.arcsec, y=y * u.arcsec)
        field = flux_density(here, _time_rounded(jd)).to_value(u.G)
        d = distance_to(*net, here).to_value(u.arcsec)
        return field, d

    for e in events:
        e["field"], e["d_network"] = sample(e["x"], e["y"], e["jd"])

    # The chance expectation: the same measurement at positions that were
    # not chosen for anything, so that a preference of the events can be
    # told from a property of the raster. Drawn from the pixels the
    # detector was allowed to search, since a control allowed to stand
    # where no event could be found is not the events' chance expectation.
    rng = np.random.default_rng(42)
    searchable = maps.valid[index_time].broadcast_to(position.x.shape).ndarray
    finite = np.isfinite(position.x.ndarray) & np.isfinite(position.y.ndarray)
    flat_valid = np.flatnonzero((finite & searchable).ravel())
    chosen = rng.choice(flat_valid, size=500, replace=False)

    control = []
    shape = position.x.ndarray.shape
    axes = position.x.axes
    for flat in chosen:
        index_nd = np.unravel_index(flat, shape)
        index = {ax: int(i) for ax, i in zip(axes, index_nd)}
        here = position[index]
        x = float(here.x.ndarray.to_value(u.arcsec))
        y = float(here.y.ndarray.to_value(u.arcsec))
        moment = obs.inputs.time[{axis_time: 0, axis_x: index[axis_x]}].ndarray
        jd = float(astropy.time.Time(moment).jd)
        field, d = sample(x, y, jd)
        control.append({"x": x, "y": y, "field": field, "d_network": d})

    return {
        "num_deficit": np.array(num_deficit),
        "x": np.array([e["x"] for e in events]),
        "y": np.array([e["y"] for e in events]),
        "sig": np.array([e["sig"] for e in events]),
        "sig_blue": np.array([e["sig_blue"] for e in events]),
        "sig_red": np.array([e["sig_red"] for e in events]),
        "direction": np.array([e["direction"] for e in events]),
        "jd": np.array([e["jd"] for e in events]),
        "field": np.array([e["field"] for e in events]),
        "d_network": np.array([e["d_network"] for e in events]),
        "control_field": np.array([c["field"] for c in control]),
        "control_d_network": np.array([c["d_network"] for c in control]),
    }


def catalog(
    significance_min: float = 7,
    separation: u.Quantity = 5 * u.arcsec,
) -> dict[str, npt.NDArray]:
    """
    Every explosive event in the raster, one row per event.

    Each event is a local champion of the bidirectional score: detections
    within the de-duplication radius of a stronger one are the same event
    seen in adjacent pixels, since the slit crosses an event of a few
    arcseconds over several steps. The rows carry position, time,
    significance, the line-of-sight flux density under the event from a
    near-co-temporal magnetogram, and the distance to the Mg II network,
    together with the same measurements at five hundred random positions,
    the chance expectation the events are judged against.

    Parameters
    ----------
    significance_min
        The smallest score, in standard deviations of the continuum noise,
        counted as an event.
    separation
        The de-duplication radius.
    """
    return _catalog(
        significance_min=significance_min,
        separation=separation.to_value(u.arcsec),
        version=version_scoring,
    )


def statistics(
    significance_min: float = 7,
    separation: u.Quantity = 5 * u.arcsec,
    figsize: tuple[float, float] = (16, 8),
    dpi: float = 600,
) -> pathlib.Path:
    """
    The census of explosive events and their magnetic context.

    Four panels: the raster with every event marked, the distribution of
    the flux density under the events against the same measurement at
    random positions, the distribution of distance to the Mg II network
    against the same control, and the event significance against the flux
    density under it.

    Parameters
    ----------
    significance_min
        The smallest score, in standard deviations of the continuum noise,
        counted as an event.
    separation
        The de-duplication radius.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    table = catalog(
        significance_min=significance_min,
        separation=separation,
    )

    obs = raster()
    image, colorbar = rgb(obs)
    net = network()

    num = len(table["sig"])

    def cdf(a: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        a = np.sort(a[np.isfinite(a)])
        return a, np.arange(1, len(a) + 1) / len(a)

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(ncols=3, width_ratios=[1.1, 0.08, 1])
        ax_image = fig.add_subplot(grid[0])
        cax = fig.add_subplot(grid[1])
        grid_panels = grid[2].subgridspec(nrows=2, ncols=2)
        axs = grid_panels.subplots()

        _plot_image(ax_image, cax, image, colorbar, obs, network=net)
        colors = {0: "white", -1: "deepskyblue", +1: "orangered"}
        names = {0: "bidirectional", -1: "blue jet", +1: "red jet"}
        for direction, color in colors.items():
            where = table["direction"] == direction
            ax_image.scatter(
                table["x"][where],
                table["y"][where],
                s=30,
                facecolors="none",
                edgecolors=color,
                linewidths=0.6,
                label=f"{names[direction]} ({where.sum()})",
            )
        ax_image.legend(fontsize=7, loc="lower left")
        fdr = float(table["num_deficit"]) / max(num, 1)
        ax_image.set_title(
            f"{num} events at {significance_min}$\\sigma$ or better, "
            f"measured FDR $\\approx$ {fdr * 100:.0f}%",
            fontsize=10,
        )

        ax = axs[0, 0]
        ax.plot(*cdf(table["field"]), color="tab:red", label="events")
        ax.plot(*cdf(table["control_field"]), color="gray", label="random positions")
        ax.set_xscale("log")
        ax.set_xlabel("|flux density| (G)")
        ax.set_ylabel("cumulative fraction")
        ax.legend(fontsize=8)

        ax = axs[0, 1]
        ax.plot(*cdf(table["d_network"]), color="tab:red", label="events")
        ax.plot(
            *cdf(table["control_d_network"]), color="gray", label="random positions"
        )
        ax.set_xlabel("distance to Mg II network (arcsec)")
        ax.set_ylabel("cumulative fraction")

        ax = axs[1, 0]
        ax.scatter(table["field"], table["sig"], s=12, color="tab:red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|flux density| (G)")
        ax.set_ylabel("significance ($\\sigma$)")

        ax = axs[1, 1]
        ax.scatter(table["d_network"], table["sig"], s=12, color="tab:red")
        ax.set_yscale("log")
        ax.set_xlabel("distance to Mg II network (arcsec)")
        ax.set_ylabel("significance ($\\sigma$)")

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "statistics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path


def profiles(
    significance_min: float = 7,
    separation: u.Quantity = 5 * u.arcsec,
    velocity_limit: u.Quantity = 300 * u.km / u.s,
    halfwidth_x: int = 1,
    halfwidth_y: int = 2,
    num_column: int = 6,
    num_row: int = 8,
    dpi: float = 200,
) -> list[pathlib.Path]:
    """
    The Si IV profile of every event in the census, a gallery in pages.

    One small panel per event, in descending order of significance, each
    against the median profile of the raster and stamped with the event's
    number, significance, distance to the Mg II network, and the flux
    density under it.

    Parameters
    ----------
    significance_min
        The smallest score, in standard deviations of the continuum noise,
        counted as an event.
    separation
        The de-duplication radius.
    velocity_limit
        The Doppler velocity range of the panels.
    halfwidth_x
        How many raster steps on either side to average each profile over.
    halfwidth_y
        How many pixels along the slit on either side to average over.
    num_column
        The number of panels across a page.
    num_row
        The number of panels down a page.
    dpi
        The resolution of the saved pages.
    """
    from ._overview import _velocity_centers

    table = catalog(
        significance_min=significance_min,
        separation=separation,
    )

    obs = raster()

    axis_time = obs.axis_time
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    index_time = {axis_time: 0}

    velocity = _velocity_centers(obs)
    velocity = velocity.ndarray.to_value(u.km / u.s)
    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))
    median = u.Quantity(median.ndarray).value

    position = obs.inputs.position[index_time].cell_centers((axis_x, axis_y))
    px = position.x.ndarray.to_value(u.arcsec)
    py = position.y.ndarray.to_value(u.arcsec)
    axes_position = position.x.axes

    num = len(table["sig"])
    per_page = num_column * num_row
    num_page = int(np.ceil(num / per_page))

    path_figures.mkdir(parents=True, exist_ok=True)
    paths = []

    for page in range(num_page):
        fig, axs = plt.subplots(
            nrows=num_row,
            ncols=num_column,
            figsize=(2.6 * num_column, 1.9 * num_row),
            sharex=True,
            constrained_layout=True,
        )
        axs = np.array(axs).ravel()

        for slot, ax in enumerate(axs):
            i = page * per_page + slot
            if i >= num:
                ax.set_visible(False)
                continue

            # The pixel nearest the cataloged position, since the catalog
            # keeps positions rather than indices.
            distance = np.hypot(px - table["x"][i], py - table["y"][i])
            index_nd = np.unravel_index(np.nanargmin(distance), distance.shape)
            index = {ax_: int(j) for ax_, j in zip(axes_position, index_nd)}

            neighborhood = {
                axis_x: slice(
                    max(index[axis_x] - halfwidth_x, 0),
                    index[axis_x] + halfwidth_x + 1,
                ),
                axis_y: slice(
                    max(index[axis_y] - halfwidth_y, 0),
                    index[axis_y] + halfwidth_y + 1,
                ),
            }
            profile = np.nanmean(
                obs.outputs[index_time | neighborhood],
                axis=(axis_x, axis_y),
            )
            profile = u.Quantity(profile.ndarray).value

            ax.plot(velocity, median, color="gray", linewidth=0.8)
            ax.plot(velocity, profile, color="tab:red", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.4, linestyle="dashed")
            ax.set_xlim(-velocity_limit.value, +velocity_limit.value)
            ax.tick_params(labelsize=6)
            arrow = {0: "$\\leftrightarrow$", -1: "$\\leftarrow$", 1: "$\\rightarrow$"}
            ax.text(
                0.03,
                0.83,
                f"{i + 1}{arrow[int(table['direction'][i])]}: "
                f"{table['sig'][i]:.0f}$\\sigma$",
                transform=ax.transAxes,
                fontsize=7,
            )
            ax.text(
                0.97,
                0.83,
                f"{table['d_network'][i]:.0f}'', {table['field'][i]:.1f} G",
                transform=ax.transAxes,
                fontsize=6,
                horizontalalignment="right",
            )

        for ax in axs[-num_column:]:
            if ax.get_visible():
                ax.set_xlabel("LOS velocity (km/s)", fontsize=7)

        path = path_figures / f"profiles_{page + 1}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    return paths
