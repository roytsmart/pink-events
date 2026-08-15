"""
Hunting the dim bidirectional events directly, in the spectra.
"""

import dataclasses
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import astropy.time
import named_arrays as na
import iris
from ._observations import raster
from ._rgb import rgb
from ._candidates import Candidate, _background
from ._network import network, distance_to
from ._magnetic import flux_density
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


@dataclasses.dataclass
class ScoreMaps:
    """How much every pixel's spectrum stands above the median, per wing."""

    wing_blue: na.AbstractScalar
    """The excess in the blue wing band, less the continuum excess."""

    wing_red: na.AbstractScalar
    """The excess in the red wing band, less the continuum excess."""

    noise: na.AbstractScalar
    """The uncertainty of a band excess, from the continuum band scatter."""

    core: na.AbstractScalar
    """The line-core brightness, for deciding what is network."""

    @property
    def significance_blue(self) -> na.AbstractScalar:
        """The blue wing excess in standard deviations of the noise."""
        return self.wing_blue / self.noise

    @property
    def significance_red(self) -> na.AbstractScalar:
        """The red wing excess in standard deviations of the noise."""
        return self.wing_red / self.noise

    @property
    def score(self) -> na.AbstractScalar:
        """
        The bidirectional score: the lesser of the two wing excesses.

        A continuum brightening scores nothing, a one-sided excursion scores
        its quiet wing, and the blends, which live only on the blue side,
        cannot lift a pixel on their own.
        """
        return np.minimum(self.wing_blue, self.wing_red)

    @property
    def significance(self) -> na.AbstractScalar:
        """The bidirectional score in standard deviations of the noise."""
        return self.score / self.noise

    @property
    def significance_any(self) -> na.AbstractScalar:
        """
        The greater wing's excess in standard deviations of the noise.

        The detection statistic when one-sided flows count too: still
        immune to continuum brightenings, which raise both wings and the
        continuum alike, and still resistant to the blends, since a band
        median shrugs off a narrow line.
        """
        return np.maximum(self.significance_blue, self.significance_red)


def score_maps(
    obs: iris.sg.SpectrographObservation,
) -> ScoreMaps:
    """
    Score every pixel's spectrum against the median, one wing at a time.

    Each wing band holds only certainly supersonic emission, the sound
    speed plus twice the thermal speed of the ion and beyond, so that
    nothing in it can be explained as thermal broadening of a stationary
    line.

    Parameters
    ----------
    obs
        The observation to score, as returned by :func:`pinkevents.raster`.
    """
    axis_time = obs.axis_time
    axis_wavelength = obs.axis_wavelength
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

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

    continuum = band_mean(excess, where_continuum)
    wing_blue = band_mean(excess, where_blue) - continuum
    wing_red = band_mean(excess, where_red) - continuum

    # How big an excess has to be before it means anything, from the scatter
    # of the one band that should hold nothing.
    noise = np.nanstd(
        np.where(where_continuum, excess - continuum, np.nan),
        axis=axis_wavelength,
    )
    num_wing = int(np.sum(where_blue).ndarray)
    noise = noise / np.sqrt(num_wing)

    core = band_mean(obs.outputs, speed < band_core)

    return ScoreMaps(
        wing_blue=wing_blue,
        wing_red=wing_red,
        noise=noise,
        core=core,
    )


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
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    index_time = {axis_time: 0}

    maps = score_maps(obs)
    score = maps.score
    significance = maps.significance
    core = maps.core

    velocity = _velocity_centers(obs)
    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))

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

    def time_of(candidate: Candidate) -> astropy.time.Time:
        """The moment the slit crossed this event."""
        index_step = {axis_time: 0, axis_x: candidate.index[axis_x]}
        return astropy.time.Time(obs.inputs.time[index_step].ndarray, format="jd")

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
                    f"{distance_to(*net, candidate.position).value:.0f}'', "
                    f"{flux_density(candidate.position, time_of(candidate)).value:.1f} G)"
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
