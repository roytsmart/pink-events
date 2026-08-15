"""
The one plot the first question needs.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import astropy.units as u
import astropy.visualization
import named_arrays as na
import iris
from ._observations import raster
from ._rgb import rgb
from ._candidates import candidates, score

__all__ = [
    "overview",
    "event",
]

#: Where the figures land. Not committed; the code that makes them is.
path_figures = pathlib.Path(__file__).parent.parent.parent / "figures"

#: The velocity band treated as the line wings.
band_wing = (40, 150) * (u.km / u.s)

#: The velocity band treated as pure continuum, beyond the wings and the
#: blends near -200 and -100 km/s. Only the red side is unblended on both
#: sides of the band, so the blue side stays out of it.
band_continuum = (250, 400) * (u.km / u.s)


def _velocity_centers(obs: iris.sg.SpectrographObservation) -> na.AbstractScalar:
    """The Doppler velocity at the center of each wavelength cell."""
    axis = obs.axis_wavelength
    velocity = obs.inputs.velocity
    velocity = velocity[{ax: 0 for ax in velocity.shape if ax != axis}]
    lower = {axis: slice(None, ~0)}
    upper = {axis: slice(+1, None)}
    return (velocity[lower] + velocity[upper]) / 2


def _excess_ratio(
    velocity: na.AbstractScalar,
    profile: na.AbstractScalar,
    median: na.AbstractScalar,
) -> tuple[u.Quantity, u.Quantity]:
    """
    How much of a profile's excess over the median is line, how much is not.

    Returns the mean excess in the wing band and in the continuum band. A
    continuum brightening raises both by the same amount; a line event
    raises the wings far more than the continuum.

    Parameters
    ----------
    velocity
        The Doppler velocity at each sample of the profiles.
    profile
        The profile being asked about.
    median
        The profile of the typical pixel, whatever excess is measured from.
    """
    excess = profile - median
    speed = np.abs(velocity)

    where_wing = (band_wing[0] < speed) & (speed < band_wing[1])
    where_continuum = (band_continuum[0] < velocity) & (velocity < band_continuum[1])

    wing = excess[where_wing].mean().ndarray
    continuum = excess[where_continuum].mean().ndarray

    return wing, continuum


def _plot_profile(
    ax: matplotlib.axes.Axes,
    velocity: na.AbstractScalar,
    profile: na.AbstractScalar,
    median: na.AbstractScalar,
    velocity_limit: u.Quantity,
    label: str,
):
    """
    One profile against the median, with the wing and continuum bands shaded.

    The shading is the continuum test made visible: if the red curve stands
    as far above the gray in the outer bands as in the inner ones, the
    excess is continuum; if it stands above only in the inner bands, it is
    the line.

    Parameters
    ----------
    ax
        The axes to draw on.
    velocity
        The Doppler velocity at each sample of the profiles.
    profile
        The profile being asked about.
    median
        The profile of the typical pixel.
    velocity_limit
        The velocity range to display.
    label
        The name of this profile, drawn in the corner of the panel.
    """
    wing, continuum = _excess_ratio(velocity, profile, median)

    for sign in (-1, +1):
        ax.axvspan(
            sign * band_wing[0].value,
            sign * band_wing[1].value,
            color="tab:blue" if sign < 0 else "tab:red",
            alpha=0.06,
        )
    ax.axvspan(
        band_continuum[0].value,
        band_continuum[1].value,
        color="gray",
        alpha=0.12,
    )

    na.plt.plot(velocity, median, ax=ax, color="gray", label="raster median")
    na.plt.plot(velocity, profile, ax=ax, color="tab:red", label="this pixel")

    # The continuum excess, carried across the whole panel: a purely
    # continuum event's profile rides this line everywhere outside the core.
    na.plt.plot(
        velocity,
        median + continuum,
        ax=ax,
        color="black",
        linewidth=0.8,
        linestyle="dotted",
        label="median + continuum excess",
    )

    ax.axvline(0, color="black", linewidth=0.5, linestyle="dashed")
    ax.set_xlim(-velocity_limit.value, +velocity_limit.value)
    ax.text(0.03, 0.9, label, transform=ax.transAxes, fontsize=12)
    ratio = wing / continuum if continuum != 0 else np.inf
    ax.text(
        0.97,
        0.9,
        f"wing/continuum = {ratio:.1f}",
        transform=ax.transAxes,
        fontsize=9,
        horizontalalignment="right",
    )


def _plot_image(
    ax: matplotlib.axes.Axes,
    cax: matplotlib.axes.Axes,
    image: na.FunctionArray,
    colorbar: na.FunctionArray,
    obs: iris.sg.SpectrographObservation,
):
    """
    The rendered raster and the key that says what its colors mean.

    Parameters
    ----------
    ax
        The axes for the raster.
    cax
        The axes for the color key.
    image
        The rendered raster, as returned by :func:`pinkevents.rgb`.
    colorbar
        The color key, as returned by :func:`pinkevents.rgb`.
    obs
        The observation the rendering was made from, for its axis names and
        rest wavelength.
    """
    axis_wavelength = obs.axis_wavelength
    index_time = {obs.axis_time: 0}

    na.plt.pcolormesh(
        image.inputs[index_time],
        C=image.outputs[index_time],
        axis_rgb=axis_wavelength,
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("helioprojective x (arcsec)")
    ax.set_ylabel("helioprojective y (arcsec)")

    wavelength_rest = na.as_named_array(obs.inputs.wavelength_rest).ndarray
    equivalency = u.doppler_optical(wavelength_rest)

    na.plt.pcolormesh(
        colorbar.inputs.x,
        colorbar.inputs.y.to(u.km / u.s, equivalencies=equivalency),
        C=colorbar.outputs,
        axis_rgb=axis_wavelength,
        ax=cax,
    )
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(f"LOS velocity ({u.km / u.s:latex_inline})")
    cax.set_xlabel(
        f"radiance\n({na.unit(obs.outputs):latex_inline})",
        fontsize=8,
    )
    cax.tick_params(labelsize=8)


def overview(
    velocity_limit: u.Quantity = 400 * u.km / u.s,
    num: int = 6,
    halfwidth_x: int = 1,
    halfwidth_y: int = 2,
    figsize: tuple[float, float] = (16, 8),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The raster, the pink pixels found in it, and their spectra.

    The left panel is the RGB rendering the events were noticed in, with the
    candidates numbered and the color key beside it. The right panels are
    the Si IV profiles at each candidate against the median profile of the
    whole raster, with the continuum test drawn in: the shaded outer band is
    pure continuum, the shaded inner bands are the line wings, and the
    dotted line carries each profile's continuum excess across the panel.
    A profile riding the dotted line is a continuum brightening; a profile
    standing above it in the wings is a line event.

    Parameters
    ----------
    velocity_limit
        The Doppler velocity range of the profile panels. Wide enough by
        default to show the blended lines near -200 and -100 km/s, which
        are present in every profile including the median and are not
        Doppler shifts.
    num
        The number of candidates to show.
    halfwidth_x
        How many raster steps on either side of a candidate to average over,
        to knock the noise down without smearing a small event away.
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

    image, colorbar = rgb(obs)

    found = candidates(
        image=image,
        axis_rgb=axis_wavelength,
        axis_x=axis_x,
        axis_y=axis_y,
        num=num,
    )

    velocity = _velocity_centers(obs)

    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))

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

        _plot_image(ax_image, cax, image, colorbar, obs)

        index_time = {axis_time: 0}

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
                label=str(i + 1),
            )

        for ax in axs[len(found) :]:
            ax.set_visible(False)

        # Out of the top right corner, which every panel's ratio is in.
        axs[0].legend(fontsize=7, loc="center right")
        for ax in axs.reshape(grid_profiles.get_geometry())[~0]:
            ax.set_xlabel(f"LOS velocity ({velocity_limit.unit:latex_inline})")

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "overview.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path


def event(
    position: na.Cartesian2dVectorArray = None,
    snap: u.Quantity = 5 * u.arcsec,
    halfwidth_image: u.Quantity = 12 * u.arcsec,
    velocity_limit: u.Quantity = 400 * u.km / u.s,
    halfwidth_x: int = 1,
    halfwidth_y: int = 2,
    figsize: tuple[float, float] = (12, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    One event, up close: its neighborhood in the rendering and its spectrum.

    Parameters
    ----------
    position
        The helioprojective position of the event. The default is the small
        pink event near (275, -140) arcsec.
    snap
        How far from the given position the event may actually be: the
        pinkest pixel within this distance is the one shown, since a
        position read off an image by eye lands beside the event as easily
        as on it, and the exact pixel named can be a dark hole while the
        event glows an arcsecond away. Zero keeps the given position.
    halfwidth_image
        How far the image panel reaches on every side of the event.
    velocity_limit
        The Doppler velocity range of the profile panel.
    halfwidth_x
        How many raster steps on either side to average the profile over.
    halfwidth_y
        How many pixels along the slit on either side to average over.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    if position is None:
        position = na.Cartesian2dVectorArray(
            x=275 * u.arcsec,
            y=-140 * u.arcsec,
        )

    obs = raster()

    axis_time = obs.axis_time
    axis_wavelength = obs.axis_wavelength
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    image, colorbar = rgb(obs)

    index_time = {axis_time: 0}

    pinkness, centers = score(
        image=image,
        axis_rgb=axis_wavelength,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    pinkness = pinkness[index_time]
    centers = centers[index_time] if axis_time in centers.shape else centers

    distance = (centers - position).length

    if snap > 0:
        # The pinkest pixel near the position named, rather than the pixel
        # at it.
        best = np.argmax(np.where(distance < snap, pinkness, 0))
    else:
        best = np.argmin(distance)

    index = {
        axis_x: int(best[axis_x].ndarray),
        axis_y: int(best[axis_y].ndarray),
    }
    position = centers[index]

    velocity = _velocity_centers(obs)
    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))

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

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(ncols=3, width_ratios=[0.8, 0.08, 1])
        ax_image = fig.add_subplot(grid[0])
        cax = fig.add_subplot(grid[1])
        ax_profile = fig.add_subplot(grid[2])

        _plot_image(ax_image, cax, image, colorbar, obs)

        x = position.x.ndarray.to_value(u.arcsec)
        y = position.y.ndarray.to_value(u.arcsec)
        ax_image.set_xlim(x - halfwidth_image.value, x + halfwidth_image.value)
        ax_image.set_ylim(y - halfwidth_image.value, y + halfwidth_image.value)
        ax_image.scatter(
            x,
            y,
            s=200,
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
        )

        _plot_profile(
            ax=ax_profile,
            velocity=velocity,
            profile=profile,
            median=median,
            velocity_limit=velocity_limit,
            label=f"({x:.0f}, {y:.0f})",
        )
        ax_profile.legend(fontsize=8)
        ax_profile.set_xlabel(f"LOS velocity ({velocity_limit.unit:latex_inline})")

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "event.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
