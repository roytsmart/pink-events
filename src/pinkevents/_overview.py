"""
The one plot the first question needs.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
from ._observations import raster
from ._rgb import rgb
from ._candidates import candidates

__all__ = [
    "overview",
]

#: Where the figures land. Not committed; the code that makes them is.
path_figures = pathlib.Path(__file__).parent.parent.parent / "figures"


def overview(
    velocity_limit: u.Quantity = 250 * u.km / u.s,
    num: int = 6,
    halfwidth_x: int = 1,
    halfwidth_y: int = 2,
    figsize: tuple[float, float] = (16, 8),
    dpi: float = 200,
) -> pathlib.Path:
    """
    The raster, the pink pixels found in it, and their spectra.

    The left panel is the RGB rendering the events were noticed in, with the
    candidates numbered. The right panels are the Si IV profiles at each
    candidate, against the median profile of the whole raster. One displaced
    component is a fast upflow; two components astride the rest wavelength
    are a bidirectional event. This is the plot that tells them apart.

    Parameters
    ----------
    velocity_limit
        The Doppler velocity range of the profile panels.
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

    image, _ = rgb(obs)

    found = candidates(
        image=image,
        axis_rgb=axis_wavelength,
        axis_x=axis_x,
        axis_y=axis_y,
        num=num,
    )

    velocity = obs.inputs.velocity
    velocity = velocity[{ax: 0 for ax in velocity.shape if ax != axis_wavelength}]
    lower = {axis_wavelength: slice(None, ~0)}
    upper = {axis_wavelength: slice(+1, None)}
    velocity = (velocity[lower] + velocity[upper]) / 2

    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(ncols=2, width_ratios=[1.1, 1])
        ax_image = fig.add_subplot(grid[0])
        grid_profiles = grid[1].subgridspec(
            nrows=(num + 1) // 2,
            ncols=2,
        )
        axs = grid_profiles.subplots(sharex=True, sharey=False)
        axs = np.array(axs).ravel()

        index_time = {axis_time: 0}
        na.plt.pcolormesh(
            image.inputs[index_time],
            C=image.outputs[index_time],
            axis_rgb=axis_wavelength,
            ax=ax_image,
        )
        ax_image.set_aspect("equal")
        ax_image.set_xlabel("helioprojective x (arcsec)")
        ax_image.set_ylabel("helioprojective y (arcsec)")

        for i, candidate in enumerate(found):

            ax_image.annotate(
                str(i + 1),
                xy=(
                    candidate.position.x.ndarray.to_value(u.arcsec),
                    candidate.position.y.ndarray.to_value(u.arcsec),
                ),
                xytext=(8, 8),
                textcoords="offset points",
                color="white",
                fontsize=12,
            )
            ax_image.scatter(
                candidate.position.x.ndarray.to_value(u.arcsec),
                candidate.position.y.ndarray.to_value(u.arcsec),
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

            ax = axs[i]
            na.plt.plot(velocity, median, ax=ax, color="gray", label="raster median")
            na.plt.plot(
                velocity, profile, ax=ax, color="tab:red", label=f"candidate {i + 1}"
            )
            ax.axvline(0, color="black", linewidth=0.5, linestyle="dashed")
            ax.set_xlim(-velocity_limit.value, +velocity_limit.value)
            ax.text(
                0.05,
                0.9,
                str(i + 1),
                transform=ax.transAxes,
                fontsize=12,
            )

        for ax in axs[len(found) :]:
            ax.set_visible(False)

        axs[0].legend(fontsize=8)
        for ax in axs.reshape(grid_profiles.get_geometry())[~0]:
            ax.set_xlabel(f"LOS velocity ({velocity_limit.unit:latex_inline})")

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "overview.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
