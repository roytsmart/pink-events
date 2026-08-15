"""
Where the network is, from the Mg II k core of the same raster.
"""

import numpy as np
import numpy.typing as npt
import astropy.units as u
import named_arrays as na
import iris
from ._caching import memory
from ._observations import time_default, slice_velocity
from ._candidates import _background

__all__ = [
    "network",
    "distance_to",
]

#: The spectral window holding the Mg II k line.
window_mgii = "Mg II k 2796"

#: The velocity band about the k core averaged into the network map.
band_core_mgii = 30 * u.km / u.s


@memory.cache
def _core_mgii(
    time: str,
    band: float,
) -> dict[str, npt.NDArray]:
    """
    The Mg II k core intensity on the raster grid, without units.

    Loading the NUV window is slow and two thousand wavelengths wide, so the
    two-dimensional map is cached, and only as plain arrays, which are the
    one thing which can be both stored and memory-mapped reliably.

    Parameters
    ----------
    time
        The time of the observation to download.
    band
        The half-width of the velocity band about the core, in km/s.
    """
    obs = iris.sg.open(
        time=time,
        window=window_mgii,
    )

    obs = slice_velocity(obs, band * (u.km / u.s))

    axis_time = obs.axis_time
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    core = np.nanmean(obs.outputs, axis=obs.axis_wavelength)
    core = core[{axis_time: 0}]

    position = obs.inputs.position[{axis_time: 0}].cell_centers((axis_x, axis_y))

    axes = (axis_x, axis_y)
    return {
        "x": position.x.transpose(axes).ndarray.to_value(u.arcsec),
        "y": position.y.transpose(axes).ndarray.to_value(u.arcsec),
        "core": core.transpose(axes).ndarray.value,
    }


def _otsu(a: npt.NDArray, num_bin: int = 256) -> float:
    """
    The threshold separating a bimodal distribution, by Otsu's method.

    The value maximizing the variance between the two classes it makes,
    which for the network against the cell interiors lands in the valley
    between the two modes without anyone choosing it.

    Parameters
    ----------
    a
        The values to split. Not-finite entries are ignored.
    num_bin
        The number of histogram bins.
    """
    a = a[np.isfinite(a)]
    count, edges = np.histogram(a, bins=num_bin)
    center = (edges[:~0] + edges[1:]) / 2

    weight_below = np.cumsum(count)
    weight_above = np.cumsum(count[::-1])[::-1]

    mean_below = np.cumsum(count * center) / np.maximum(weight_below, 1)
    mean_above = np.cumsum((count * center)[::-1])[::-1] / np.maximum(weight_above, 1)

    variance = (
        weight_below[:~0]
        * weight_above[1:]
        * np.square(mean_below[:~0] - mean_above[1:])
    )

    return float(center[np.argmax(variance)])


def network(
    time: str = time_default,
    halfwidth_smooth: u.Quantity = 2 * u.arcsec,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    A map of the chromospheric network, and where each of its pixels is.

    The Mg II k core of the same raster is the canonical network tracer,
    and it comes from the same slit at the same times as the Si IV window,
    so there is no coalignment between instruments to get wrong. The core
    intensity is smoothed a little, its logarithm split by Otsu's method,
    which puts the threshold in the valley of a bimodal distribution
    rather than at a percentile someone chose, and everything above the
    threshold is network.

    Returns the mask and the x and y of each of its pixels, in arcseconds,
    as plain arrays with the raster axes in the order (step, slit).

    Parameters
    ----------
    time
        The time of the observation to download.
    halfwidth_smooth
        The half-width of the boxcar taken over the core intensity before
        thresholding, to keep single noisy pixels out of the mask.
    """
    a = _core_mgii(
        time=time,
        band=band_core_mgii.to_value(u.km / u.s),
    )
    x = a["x"]
    y = a["y"]
    core = a["core"]

    scale_x = np.abs(np.nanmean(np.diff(x, axis=0)))
    scale_y = np.abs(np.nanmean(np.diff(y, axis=1)))
    hx = max(int(halfwidth_smooth.to_value(u.arcsec) / scale_x), 1)
    hy = max(int(halfwidth_smooth.to_value(u.arcsec) / scale_y), 1)

    smooth, coverage = _background(core[np.newaxis], halfwidth=(hx, hy))
    smooth = smooth[0]
    coverage = coverage[0]

    with np.errstate(invalid="ignore", divide="ignore"):
        logarithm = np.log10(np.where(smooth > 0, smooth, np.nan))

    threshold = _otsu(logarithm[coverage > 0.99])

    mask = logarithm > threshold

    return mask, x, y


def distance_to(
    mask: npt.NDArray,
    x: npt.NDArray,
    y: npt.NDArray,
    position: na.Cartesian2dVectorArray,
) -> u.Quantity:
    """
    How far a position is from the nearest network pixel.

    Zero for a position inside the network.

    Parameters
    ----------
    mask
        The network mask, as returned by :func:`network`.
    x
        The x coordinate of each pixel of the mask, in arcseconds.
    y
        The y coordinate of each pixel of the mask, in arcseconds.
    position
        The position being asked about.
    """
    x0 = float(na.as_named_array(position.x).ndarray.to_value(u.arcsec))
    y0 = float(na.as_named_array(position.y).ndarray.to_value(u.arcsec))
    dx = x[mask] - x0
    dy = y[mask] - y0
    return np.sqrt(np.min(np.square(dx) + np.square(dy))) * u.arcsec
