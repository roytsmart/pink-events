"""
Whether the chromosphere under the events answers: the Mg II h and k lines.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import iris
from ._observations import time_default
from ._network import window_mgii
from ._overview import path_figures
from ._averages import _event_profiles, _classes

__all__ = [
    "response_mgii",
]

#: The rest wavelengths of the Mg II doublet.
wavelength_k = 2796.352 * u.AA
wavelength_h = 2803.531 * u.AA


def _raster_mgii(
    time: str = time_default,
    wavelength_min: u.Quantity = 2792 * u.AA,
    wavelength_max: u.Quantity = 2808 * u.AA,
) -> iris.sg.SpectrographObservation:
    """
    The Mg II window of the same raster, cropped to the h and k lines.

    Not despiked: the medians and small averages everything here is built
    from shrug off the occasional spike, and despiking the NUV cube costs
    more than it buys.

    Parameters
    ----------
    time
        The time of the observation to download.
    wavelength_min
        The blue edge of the crop.
    wavelength_max
        The red edge of the crop.
    """
    obs = iris.sg.open(
        time=time,
        window=window_mgii,
    )

    axis = obs.axis_wavelength

    wavelength = obs.inputs.wavelength
    wavelength = wavelength[{ax: 0 for ax in wavelength.shape if ax != axis}]
    wavelength = wavelength.ndarray

    where = (wavelength > wavelength_min) & (wavelength < wavelength_max)
    index = np.flatnonzero(where)
    lower = int(index.min())
    upper = int(index.max())

    return obs[{axis: slice(lower, upper)}]


def response_mgii(
    figsize: tuple[float, float] = (16, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The Mg II spectrum under the census events, against the quiet Sun.

    Three panels: the mean Mg II spectrum of each event class over the h
    and k lines against the quiet median; the ratio of each class to the
    quiet median, which shows where in the line the chromosphere answers;
    and the k line alone on a Doppler axis, where a response in the very
    core, the k3 depression, means the disturbance reaches the upper
    chromosphere.

    Parameters
    ----------
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    from ._catalog import catalog

    table = catalog()
    obs = _raster_mgii()

    axis = obs.axis_wavelength
    axes = tuple(ax for ax in obs.outputs.axes if ax != axis)

    wavelength = obs.inputs.wavelength
    wavelength = wavelength[{ax: 0 for ax in wavelength.shape if ax != axis}]
    wavelength = wavelength.ndarray.to_value(u.AA)
    wavelength = (wavelength[:~0] + wavelength[1:]) / 2

    quiet = u.Quantity(np.nanmedian(obs.outputs, axis=axes).ndarray).value

    profiles = _event_profiles(obs, table)

    rest_k = wavelength_k.to_value(u.AA)
    velocity_k = (wavelength - rest_k) / rest_k * 299792.458

    fig, axs = plt.subplots(ncols=3, figsize=figsize, constrained_layout=True)

    ax = axs[0]
    for name, sel, color in _classes(table["direction"]):
        ax.plot(
            wavelength,
            np.nanmean(profiles[sel], axis=0),
            color=color,
            linewidth=1,
            label=f"{name} ({sel.sum()})",
        )
    ax.plot(
        wavelength,
        quiet,
        color="gray",
        linewidth=1,
        linestyle="dashed",
        label="quiet median",
    )
    for rest, name in (
        (wavelength_k.to_value(u.AA), "k"),
        (wavelength_h.to_value(u.AA), "h"),
    ):
        ax.axvline(rest, color="gray", linewidth=0.5, linestyle="dotted")
        ax.annotate(name, xy=(rest, ax.get_ylim()[1]), fontsize=9)
    ax.set_xlabel("wavelength ($\\AA$)")
    ax.set_ylabel("signal (DN)")
    ax.legend(fontsize=7)
    ax.set_title("mean Mg II spectrum", fontsize=10)

    ax = axs[1]
    for name, sel, color in _classes(table["direction"]):
        ax.plot(
            wavelength,
            np.nanmean(profiles[sel], axis=0) / quiet,
            color=color,
            linewidth=1,
            label=name,
        )
    ax.axhline(1, color="gray", linewidth=0.5)
    for rest in (wavelength_k.to_value(u.AA), wavelength_h.to_value(u.AA)):
        ax.axvline(rest, color="gray", linewidth=0.5, linestyle="dotted")
    ax.set_xlabel("wavelength ($\\AA$)")
    ax.set_ylabel("events / quiet")
    ax.set_title("the response spectrum", fontsize=10)

    ax = axs[2]
    for name, sel, color in _classes(table["direction"]):
        ax.plot(
            velocity_k,
            np.nanmean(profiles[sel], axis=0) / quiet,
            color=color,
            linewidth=1,
            label=name,
        )
    ax.axhline(1, color="gray", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="dashed")
    ax.set_xlim(-150, 150)
    ax.set_xlabel("LOS velocity from k ($\\mathrm{km/s}$)")
    ax.set_ylabel("events / quiet")
    ax.set_title("the k core, close up", fontsize=10)

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "mgii.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
