"""
Hunting the dim bidirectional events directly, in the spectra.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
from ._observations import raster
from ._rgb import rgb
from ._candidates import Candidate, _background
from ._network import network, distance_to
from ._overview import (
    path_figures,
    band_wing,
    band_continuum,
    _velocity_centers,
    _plot_image,
    _plot_profile,
)

__all__ = [
    "dim_events",
]

#: The velocity band treated as the line core.
band_core = 30 * u.km / u.s


def dim_events(
    num: int = 8,
    significance_min: float = 5,
    separation: u.Quantity = 10 * u.arcsec,
    halfwidth_background: u.Quantity = 4 * u.arcsec,
    velocity_limit: u.Quantity = 400 * u.km / u.s,
    halfwidth_x: int = 1,
    halfwidth_y: int = 2,
    figsize: tuple[float, float] = (16, 8),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The strongest dim bidirectional profiles in the raster, found spectrally.

    The pink search found events by their color, and most of what it found
    was continuum. This searches for what the interesting minority actually
    is: a pixel whose spectrum stands above the median in *both* wings,
    beyond whatever its continuum is doing. The score is the lesser of the
    two wing excesses less the continuum excess, so a continuum brightening
    scores nothing, a one-sided noise excursion scores nothing, and the
    blends, which live only on the blue side, cannot lift a pixel on their
    own. Pixels whose neighborhood is bright in the line core, the network,
    are excluded: bright network explosive events are a well known
    population, and dim ones in the cell interiors are the quarry.

    Parameters
    ----------
    num
        The number of events to show, at most.
    significance_min
        The smallest score, in standard deviations of the continuum noise,
        still counted as an event. Without a floor the last panels are
        filled with whatever comes next in line, however weak.
    separation
        The minimum distance between the events kept.
    halfwidth_background
        The half-width of the neighborhood used to decide what is network.
    velocity_limit
        The Doppler velocity range of the profile panels.
    halfwidth_x
        How many raster steps on either side to average each profile over.
    halfwidth_y
        How many pixels along the slit on either side to average over.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    obs = raster()

    axis_time = obs.axis_time
    axis_wavelength = obs.axis_wavelength
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    index_time = {axis_time: 0}

    velocity = _velocity_centers(obs)
    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))
    excess = obs.outputs - median

    speed = np.abs(velocity)

    # Medians rather than means, so that a single bad sample in a band, and
    # the despiked data still holds deep negative ones, cannot drag the
    # band. A negative artifact in the continuum band of a plain pixel
    # pulls its continuum excess down and hands it a score it did not earn.
    def band_mean(a: na.AbstractScalar, where: na.AbstractScalar) -> na.AbstractScalar:
        return np.nanmedian(
            np.where(where, a, np.nan),
            axis=axis_wavelength,
        )

    where_blue = (band_wing[0] < speed) & (speed < band_wing[1]) & (velocity < 0)
    where_red = (band_wing[0] < speed) & (speed < band_wing[1]) & (velocity > 0)
    where_continuum = (band_continuum[0] < velocity) & (velocity < band_continuum[1])

    wing_blue = band_mean(excess, where_blue)
    wing_red = band_mean(excess, where_red)
    continuum = band_mean(excess, where_continuum)

    score = np.minimum(wing_blue, wing_red) - continuum

    # How big the score has to be before it means anything, from the scatter
    # of the one band that should hold nothing.
    noise = np.nanstd(
        np.where(where_continuum, excess - continuum, np.nan),
        axis=axis_wavelength,
    )
    num_wing = int(np.sum(where_blue).ndarray)
    significance = score / (noise / np.sqrt(num_wing))

    # What is network and what is cell interior, decided by the line core of
    # the neighborhood rather than by this pixel, so that a compact event
    # does not disqualify itself by being bright.
    core = band_mean(obs.outputs, speed < band_core)

    position = obs.inputs.position[index_time].cell_centers((axis_x, axis_y))
    scale_x = np.nanmean(np.abs(np.diff(position.x, axis=axis_x))).ndarray
    scale_y = np.nanmean(np.abs(np.diff(position.y, axis=axis_y))).ndarray
    hx = max(
        int((halfwidth_background / scale_x).to_value(u.dimensionless_unscaled)), 1
    )
    hy = max(
        int((halfwidth_background / scale_y).to_value(u.dimensionless_unscaled)), 1
    )

    a = core[index_time].ndarray
    source = (core[index_time].axes.index(axis_x), core[index_time].axes.index(axis_y))
    a = np.moveaxis(a, source=source, destination=(~1, ~0))
    background, coverage = _background(a[np.newaxis], halfwidth=(hx, hy))
    background = background[0]
    coverage = coverage[0]

    keep = (background < np.nanmedian(background)) & (coverage > 0.99)
    keep[: 2 * hx + 1, :] = False
    keep[-(2 * hx + 1) :, :] = False
    keep[:, : 2 * hy + 1] = False
    keep[:, -(2 * hy + 1) :] = False
    keep = na.ScalarArray(keep, axes=(axis_x, axis_y))

    score = np.where(keep, score[index_time], 0)

    order = np.argsort(-score.ndarray, axis=None)

    found = []
    for flat in order:
        if len(found) >= num:
            break
        index_nd = np.unravel_index(flat, score.ndarray.shape)
        index = {ax: int(i) for ax, i in zip(score.axes, index_nd)}
        if score[index].ndarray <= 0:
            break
        if significance[index_time | index].ndarray < significance_min:
            break
        here = position[index]
        if any((here - c.position).length < separation for c in found):
            continue
        found.append(
            Candidate(
                index=index,
                position=here,
                score=float(significance[index_time | index].ndarray),
            )
        )

    image, colorbar = rgb(obs)

    net = network()

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(ncols=3, width_ratios=[1.1, 0.08, 1])
        ax_image = fig.add_subplot(grid[0])
        cax = fig.add_subplot(grid[1])
        grid_profiles = grid[2].subgridspec(
            nrows=(num + 1) // 2,
            ncols=2,
        )
        axs = grid_profiles.subplots(sharex=True, sharey=False)
        axs = np.array(axs).ravel()

        _plot_image(ax_image, cax, image, colorbar, obs, network=net)

        for i, candidate in enumerate(found):

            x = candidate.position.x.ndarray.to_value(u.arcsec)
            y = candidate.position.y.ndarray.to_value(u.arcsec)
            ax_image.annotate(
                str(i + 1),
                xy=(x, y),
                xytext=(8, 8),
                textcoords="offset points",
                color="white",
                fontsize=12,
            )
            ax_image.scatter(
                x,
                y,
                s=120,
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
            )

            index = dict(candidate.index)
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
            profile = obs.outputs[index_time | neighborhood]
            profile = profile.mean((axis_x, axis_y))

            _plot_profile(
                ax=axs[i],
                velocity=velocity,
                profile=profile,
                median=median,
                velocity_limit=velocity_limit,
                label=(
                    f"{i + 1} ({candidate.score:.0f}$" + chr(92) + "sigma$, "
                    f"{distance_to(*net, candidate.position).value:.0f}'')"
                ),
            )

        for ax in axs[len(found) :]:
            ax.set_visible(False)

        axs[0].legend(fontsize=7, loc="center right")
        for ax in axs.reshape(grid_profiles.get_geometry())[~0]:
            ax.set_xlabel(f"LOS velocity ({velocity_limit.unit:latex_inline})")

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "dim_ees.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
