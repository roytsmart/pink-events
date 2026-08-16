"""
The response of any line to the events, one function for all of them.
"""

import dataclasses
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import iris
from ._observations import _outputs_despiked, time_default
from ._overview import path_figures
from ._averages import _event_profiles, _classes

__all__ = [
    "line_list",
    "response",
]

#: Every line this investigation has had reason to name, by label.
line_list = {
    "C II 1334.53": 1334.532,
    "Ni II 1335.20": 1335.203,
    "C II 1335.71": 1335.708,
    "Cl I 1351.66": 1351.657,
    "Fe II 1353.02": 1353.02,
    "Si II 1353.72": 1353.72,
    "Fe XXI 1354.08": 1354.08,
    "C I 1354.29": 1354.288,
    "O I 1355.60": 1355.598,
    "C I 1355.84": 1355.844,
    "Fe II 1348.11": 1348.11,
    "C I 1348.42": 1348.42,
    "Fe XII 1349.40": 1349.40,
    "S I 1392.59": 1392.59,
    "Fe II 1392.82": 1392.82,
    "Ni II 1393.33": 1393.33,
    "Si IV 1393.76": 1393.755,
    "O IV] 1399.78": 1399.78,
    "O IV] 1401.16": 1401.16,
    "S I 1401.51": 1401.51,
    "Fe II 1401.77": 1401.77,
    "Si IV 1402.77": 1402.770,
    "O IV+S IV 1404.8": 1404.81,
    "S IV 1406.02": 1406.02,
    "Mg II 2791.60": 2791.600,
    "Mg II k 2796.35": 2796.352,
    "Mg II 2798.82": 2798.823,
    "Mg II h 2803.53": 2803.531,
}

#: The windows of the raster, by name and span.
windows = {
    "C II 1336": (1331.7, 1358.4),
    "Si IV 1394": (1380.7, 1406.8),
    "Mg II k 2796": (2783.2, 2835.0),
}


def _observation(
    center: u.Quantity,
    halfwidth: u.Quantity,
    time: str,
) -> iris.sg.SpectrographObservation:
    """
    The raster cropped about a wavelength, from whichever window holds it.

    The Si IV window comes despiked from the shared cache; the other two
    come raw, which the median stacking everything here uses shrugs off.

    Parameters
    ----------
    center
        The wavelength to crop about.
    halfwidth
        How far to keep on either side.
    time
        The time of the observation to download.
    """
    value = center.to_value(u.AA)
    for name, (lo, hi) in windows.items():
        if lo < value < hi:
            window = name
            break
    else:
        raise ValueError(f"no window of this raster holds {center}")

    obs = iris.sg.open(time=time, window=window)

    if window == "Si IV 1394":
        outputs = _outputs_despiked(time=time, window=window)
        obs = dataclasses.replace(
            obs,
            outputs=na.ScalarArray(
                ndarray=outputs << na.unit(obs.outputs),
                axes=obs.outputs.axes,
            ),
        )

    axis = obs.axis_wavelength
    wavelength = obs.inputs.wavelength
    wavelength = wavelength[{ax: 0 for ax in wavelength.shape if ax != axis}]
    wavelength = wavelength.ndarray

    where = (wavelength > center - halfwidth) & (wavelength < center + halfwidth)
    index = np.flatnonzero(where)

    return obs[{axis: slice(int(index.min()), int(index.max()))}]


def response(
    center: u.Quantity,
    halfwidth: u.Quantity = 3 * u.AA,
    velocity_limit: u.Quantity = 150 * u.km / u.s,
    logarithmic: bool = False,
    time: str = time_default,
    figsize: tuple[float, float] = (16, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    How the spectrum about a wavelength answers the census events.

    Three panels: the median-stacked spectrum of each event class against
    the quiet median, with every line the investigation knows labeled; the
    excess of each class over quiet; and the excess close up on a Doppler
    axis about the chosen wavelength. Median stacks throughout, so that a
    single cosmic ray in a single event cannot masquerade as a class.

    Parameters
    ----------
    center
        The wavelength to look about, and the rest wavelength of the
        Doppler axis.
    halfwidth
        How far to keep on either side.
    velocity_limit
        The Doppler range of the close-up panel.
    logarithmic
        Whether the spectrum panel is logarithmic.
    time
        The time of the observation to download.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    from ._catalog import catalog

    table = catalog()
    obs = _observation(center=center, halfwidth=halfwidth, time=time)

    axis = obs.axis_wavelength
    axes = tuple(ax for ax in obs.outputs.axes if ax != axis)

    wavelength = obs.inputs.wavelength
    wavelength = wavelength[{ax: 0 for ax in wavelength.shape if ax != axis}]
    wavelength = wavelength.ndarray.to_value(u.AA)
    wavelength = (wavelength[:~0] + wavelength[1:]) / 2

    quiet = u.Quantity(np.nanmedian(obs.outputs, axis=axes).ndarray).value
    profiles = _event_profiles(obs, table)

    rest = center.to_value(u.AA)
    velocity = (wavelength - rest) / rest * 299792.458

    named = {
        label: w
        for label, w in line_list.items()
        if wavelength.min() < w < wavelength.max()
    }

    fig, axs = plt.subplots(ncols=3, figsize=figsize, constrained_layout=True)

    ax = axs[0]
    for name, sel, color in _classes(table["direction"]):
        ax.plot(
            wavelength,
            np.nanmedian(profiles[sel], axis=0),
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
    if logarithmic:
        ax.set_yscale("log")
    for label, w in named.items():
        emphasis = abs(w - rest) < 0.05
        ax.axvline(
            w,
            color="tab:green" if emphasis else "gray",
            linewidth=1 if emphasis else 0.4,
            linestyle="dashed" if emphasis else "dotted",
        )
        ax.annotate(
            label,
            xy=(w, ax.get_ylim()[1]),
            fontsize=6,
            rotation=90,
            horizontalalignment="right",
            verticalalignment="top",
        )
    ax.set_xlabel("wavelength ($\\AA$)")
    ax.set_ylabel("signal (DN)")
    ax.legend(fontsize=7)
    ax.set_title("median-stacked spectra", fontsize=10)

    ax = axs[1]
    for name, sel, color in _classes(table["direction"]):
        ax.plot(
            wavelength,
            np.nanmedian(profiles[sel], axis=0) - quiet,
            color=color,
            linewidth=1,
            label=name,
        )
    ax.axhline(0, color="gray", linewidth=0.5)
    for label, w in named.items():
        ax.axvline(w, color="gray", linewidth=0.4, linestyle="dotted")
    ax.axvline(rest, color="tab:green", linewidth=1, linestyle="dashed")
    ax.set_xlabel("wavelength ($\\AA$)")
    ax.set_ylabel("events $-$ quiet (DN)")
    ax.set_title("the excess spectrum", fontsize=10)

    ax = axs[2]
    for name, sel, color in _classes(table["direction"]):
        ax.plot(
            velocity,
            np.nanmedian(profiles[sel], axis=0) - quiet,
            color=color,
            linewidth=1,
            label=name,
        )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="dashed")
    ax.set_xlim(-velocity_limit.value, +velocity_limit.value)
    ax.set_xlabel(f"LOS velocity from {rest:.2f} $\\AA$ (km/s)")
    ax.set_ylabel("events $-$ quiet (DN)")
    ax.legend(fontsize=7)
    ax.set_title("close up", fontsize=10)

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / f"response_{rest:.2f}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
