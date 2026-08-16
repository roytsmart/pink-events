"""
The missing rung of the ladder: the C II lines between Mg II and Si IV.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import iris
from ._observations import time_default
from ._overview import path_figures
from ._averages import _event_profiles, _classes

__all__ = [
    "response_cii",
]

#: The C II window of the raster.
window_cii = "C II 1336"

#: The rest wavelengths of the C II doublet. The red member is itself a
#: close pair, 1335.663 and 1335.708, dominated by the latter.
wavelength_cii = (1334.532, 1335.708) * u.AA


def _raster_cii(
    time: str = time_default,
    wavelength_min: u.Quantity = 1333 * u.AA,
    wavelength_max: u.Quantity = 1337.5 * u.AA,
) -> iris.sg.SpectrographObservation:
    """
    The C II window of the same raster, cropped to the doublet.

    Not despiked, for the same reason as the Mg II window: medians and
    small averages shrug off the occasional spike.

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
        window=window_cii,
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


def response_cii(
    figsize: tuple[float, float] = (16, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The C II spectrum under the census events, against the quiet Sun.

    The C II doublet forms between the Mg II cores and the Si IV line, so
    its response fills the rung of the height ladder the other two leave
    open. Three panels, matching the Mg II figure: the mean spectrum of
    each event class, the ratio of each class to the quiet median, and the
    stronger doublet member close up on a Doppler axis.

    Parameters
    ----------
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    from ._catalog import catalog

    table = catalog()
    obs = _raster_cii()

    axis = obs.axis_wavelength
    axes = tuple(ax for ax in obs.outputs.axes if ax != axis)

    wavelength = obs.inputs.wavelength
    wavelength = wavelength[{ax: 0 for ax in wavelength.shape if ax != axis}]
    wavelength = wavelength.ndarray.to_value(u.AA)
    wavelength = (wavelength[:~0] + wavelength[1:]) / 2

    quiet = u.Quantity(np.nanmedian(obs.outputs, axis=axes).ndarray).value

    profiles = _event_profiles(obs, table)

    rest = wavelength_cii[1].to_value(u.AA)
    velocity = (wavelength - rest) / rest * 299792.458

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
    for rest_i in wavelength_cii.to_value(u.AA):
        ax.axvline(rest_i, color="gray", linewidth=0.5, linestyle="dotted")
    ax.set_xlabel("wavelength ($\\AA$)")
    ax.set_ylabel("signal (DN)")
    ax.legend(fontsize=7)
    ax.set_title("mean C II spectrum", fontsize=10)

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
    for rest_i in wavelength_cii.to_value(u.AA):
        ax.axvline(rest_i, color="gray", linewidth=0.5, linestyle="dotted")
    ax.set_xlabel("wavelength ($\\AA$)")
    ax.set_ylabel("events / quiet")
    ax.set_title("the response spectrum", fontsize=10)

    ax = axs[2]
    for name, sel, color in _classes(table["direction"]):
        ax.plot(
            velocity,
            np.nanmean(profiles[sel], axis=0) / quiet,
            color=color,
            linewidth=1,
            label=name,
        )
    ax.axhline(1, color="gray", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="dashed")
    ax.set_xlim(-150, 150)
    ax.set_xlabel("LOS velocity from C II 1335.71 (km/s)")
    ax.set_ylabel("events / quiet")
    ax.set_title("the 1335.71 line, close up", fontsize=10)

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "cii.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
