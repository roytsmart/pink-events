"""
The average profiles: of the raster, and of the events found in it.
"""

import pathlib
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import astropy.units as u
from ._observations import raster, wavelength_partner, time_default, window_default
from ._overview import _velocity_centers, path_figures

__all__ = [
    "average_profile",
    "average_ee",
]

#: The blends within reach of the 1394 line, by name and velocity.
blends_1394 = {
    "Ni II 1393.33": -91.5,
    "Fe II 1392.82": -201.5,
    "S I 1392.59": -250.7,
}


def _event_profiles(
    obs,
    table: dict[str, npt.NDArray],
    halfwidth_x: int = 1,
    halfwidth_y: int = 2,
) -> npt.NDArray:
    """
    The spectrum at every cataloged event, one row per event.

    Parameters
    ----------
    obs
        The observation to read the spectra from. Any observation on the
        raster's grid will do: the Si IV window, its restoration, or the
        Mg II window, which shares the slit.
    table
        The census, as returned by :func:`pinkevents.catalog`.
    halfwidth_x
        How many raster steps on either side to average over.
    halfwidth_y
        How many pixels along the slit on either side to average over.
    """
    axis_time = obs.axis_time
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    position = obs.inputs.position[{axis_time: 0}].cell_centers((axis_x, axis_y))
    px = position.x.ndarray.to_value(u.arcsec)
    py = position.y.ndarray.to_value(u.arcsec)

    profiles = []
    for i in range(len(table["x"])):
        distance = np.hypot(px - table["x"][i], py - table["y"][i])
        ix, iy = np.unravel_index(np.nanargmin(distance), distance.shape)
        neighborhood = {
            axis_time: 0,
            axis_x: slice(max(ix - halfwidth_x, 0), ix + halfwidth_x + 1),
            axis_y: slice(max(iy - halfwidth_y, 0), iy + halfwidth_y + 1),
        }
        piece = u.Quantity(obs.outputs[neighborhood].ndarray).value
        profiles.append(np.nanmean(piece, axis=(0, 1)))

    return np.array(profiles)


#: The classes of the census, by name, selector, and color.
def _classes(direction: npt.NDArray) -> tuple:
    return (
        ("all events", np.ones_like(direction, dtype=bool), "black"),
        ("bidirectional", direction == 0, "tab:purple"),
        ("blue jets", direction == -1, "tab:blue"),
        ("red jets", direction == +1, "tab:red"),
    )


def average_profile(
    figsize: tuple[float, float] = (14, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The average Si IV profile: what every excess is measured against.

    Two panels: the median against the mean and the scaled doublet kernel,
    and the log view with the blends named and the restored median beside
    the raw one.

    Parameters
    ----------
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    from ._deconvolve import deconvolved, _kernel

    obs = raster()
    sharp = deconvolved()

    axes = (obs.axis_time, obs.axis_detector_x, obs.axis_detector_y)
    velocity = _velocity_centers(obs).ndarray.to_value(u.km / u.s)
    median = u.Quantity(np.nanmedian(obs.outputs, axis=axes).ndarray).value
    mean = u.Quantity(np.nanmean(obs.outputs, axis=axes).ndarray).value
    median_sharp = u.Quantity(np.nanmedian(sharp.outputs, axis=axes).ndarray).value

    info = _kernel(time=time_default, window=window_default)
    rest = wavelength_partner.to_value(u.AA)
    velocity_kernel = (info["wavelength"] - rest) / rest * 299792.458
    response = info["kernel"]

    far = np.abs(velocity) > 250
    continuum = np.median(median[far])
    line = median - continuum
    scale = np.trapezoid(line, velocity) / np.trapezoid(response, velocity_kernel)

    fig, axs = plt.subplots(ncols=2, figsize=figsize, constrained_layout=True)

    ax = axs[0]
    ax.plot(velocity, median, color="black", label="median profile")
    ax.plot(velocity, mean, color="tab:orange", linewidth=1, label="mean profile")
    ax.plot(
        velocity_kernel,
        response * scale + continuum,
        color="tab:blue",
        linewidth=1,
        linestyle="dashed",
        label="1403 kernel, scaled",
    )
    ax.set_xlim(-300, 300)
    ax.set_xlabel("LOS velocity (km/s)")
    ax.set_ylabel("radiance (erg nm$^{-1}$ s$^{-1}$ sr$^{-1}$ cm$^{-2}$)")
    ax.legend(fontsize=8)
    ax.set_title("linear", fontsize=10)

    ax = axs[1]
    ax.plot(velocity, median, color="black", label="median profile")
    ax.plot(velocity, mean, color="tab:orange", linewidth=1, label="mean profile")
    ax.plot(
        velocity,
        median_sharp,
        color="tab:red",
        linewidth=1,
        label="median, RL restored",
    )
    ax.set_yscale("log")
    ax.set_xlim(-400, 400)
    ax.set_xlabel("LOS velocity (km/s)")
    for name, v in blends_1394.items():
        ax.axvline(v, color="gray", linewidth=0.5, linestyle="dotted")
        ax.annotate(
            name,
            xy=(v, 2e4),
            fontsize=7,
            rotation=90,
            horizontalalignment="right",
            verticalalignment="top",
        )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("log, with the blends named", fontsize=10)

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "average_profile.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path


def average_ee(
    figsize: tuple[float, float] = (14, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The average explosive-event profile, by class.

    Two panels: the mean radiance of the census events against the quiet
    median, and the mean shape after each event is continuum-subtracted
    and normalized to its own peak.

    Parameters
    ----------
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    from ._catalog import catalog

    table = catalog()
    obs = raster()

    velocity = _velocity_centers(obs).ndarray.to_value(u.km / u.s)
    axes = (obs.axis_time, obs.axis_detector_x, obs.axis_detector_y)
    quiet = u.Quantity(np.nanmedian(obs.outputs, axis=axes).ndarray).value

    profiles = _event_profiles(obs, table)
    far = np.abs(velocity) > 250

    def shape(p: npt.NDArray) -> npt.NDArray:
        q = p - np.nanmedian(p[far])
        return q / np.nanmax(q)

    fig, axs = plt.subplots(ncols=2, figsize=figsize, constrained_layout=True)

    ax = axs[0]
    for name, sel, color in _classes(table["direction"]):
        ax.plot(
            velocity,
            np.nanmean(profiles[sel], axis=0),
            color=color,
            label=f"{name} ({sel.sum()})",
        )
    ax.plot(
        velocity,
        quiet,
        color="gray",
        linewidth=1,
        linestyle="dashed",
        label="quiet median",
    )
    ax.set_yscale("log")
    ax.set_xlim(-300, 300)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="dotted")
    ax.set_xlabel("LOS velocity (km/s)")
    ax.set_ylabel("radiance (erg nm$^{-1}$ s$^{-1}$ sr$^{-1}$ cm$^{-2}$)")
    ax.legend(fontsize=8)
    ax.set_title("mean radiance", fontsize=10)

    ax = axs[1]
    for name, sel, color in _classes(table["direction"]):
        normalized = np.array([shape(p) for p in profiles[sel]])
        ax.plot(velocity, np.nanmean(normalized, axis=0), color=color, label=name)
    ax.plot(
        velocity,
        shape(quiet),
        color="gray",
        linewidth=1,
        linestyle="dashed",
        label="quiet median",
    )
    ax.set_xlim(-300, 300)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="dotted")
    ax.set_xlabel("LOS velocity (km/s)")
    ax.set_ylabel("normalized")
    ax.legend(fontsize=8)
    ax.set_title("mean shape, each event normalized first", fontsize=10)

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "average_ee.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
