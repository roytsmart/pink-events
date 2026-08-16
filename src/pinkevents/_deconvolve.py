"""
Sharpening the 1394 spectra with the doublet partner as the kernel.
"""

import pathlib
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
import iris
from ._caching import memory
from ._observations import raster, raster_partner, time_default, window_default

__all__ = [
    "kernel",
    "deconvolved",
    "deconvolution",
    "where_bands_sharpened",
]

#: The regularization of the Wiener filter, as a fraction of the kernel's
#: peak spectral power. Chosen by the delta test in :func:`deconvolution`:
#: small enough that the median profile collapses by an order of magnitude,
#: large enough that the quiet continuum does not ring.
regularization = 0.03


def where_bands_sharpened(
    velocity: npt.NDArray,
    median_sharp: npt.NDArray,
    speed_sound: float,
    band_continuum: tuple[float, float],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    The bands for deconvolved spectra, with their edges measured.

    Deconvolution rings: the quiet profile's core throws negative lobes a
    few tens of km/s to either side, scaled by the core's brightness, so
    the wing bands must start beyond where the deconvolved median has
    settled, and no earlier than the sound speed. The blends sharpen, so
    their exclusions shrink to where the deconvolved median actually
    stands above the continuum. Both are read off the deconvolved median
    rather than chosen.

    Parameters
    ----------
    velocity
        The Doppler velocity of each sample, in km/s.
    median_sharp
        The deconvolved median profile, in any units.
    speed_sound
        The slowest speed that counts as supersonic, in km/s.
    band_continuum
        The continuum band, in km/s.
    """
    where_continuum = (band_continuum[0] < velocity) & (velocity < band_continuum[1])

    level = np.nanmedian(median_sharp[where_continuum])
    peak = np.nanmax(median_sharp) - level
    settled = np.abs(median_sharp - level) < 0.02 * peak

    # The inner edge: the first speed past the sound speed where the
    # deconvolved median has settled and stays settled.
    speed = np.abs(velocity)
    edge = speed_sound
    order = np.argsort(speed)
    run = 0
    for i in order:
        if speed[i] < speed_sound:
            continue
        run = run + 1 if settled[i] else 0
        if run >= 5:
            edge = max(speed_sound, speed[i])
            break

    # The blends: where the deconvolved median stands off the continuum on
    # the far blue side, padded by one sample.
    blended = ~settled & (velocity < -edge)
    blended = blended | np.roll(blended, 1) | np.roll(blended, -1)

    hi = 150.0
    where_blue = (edge < speed) & (speed < hi) & (velocity < 0) & ~blended
    where_red = (edge < speed) & (speed < hi) & (velocity > 0)

    return where_blue, where_red, where_continuum


@memory.cache
def _kernel(
    time: str,
    window: str,
) -> dict[str, npt.NDArray]:
    """
    The deconvolution kernel, without units: the median 1403 profile.

    The doublet partner forms at the same temperature from the same ion
    moving at the same velocities, through the same spectrograph, so its
    median profile carries the thermal, nonthermal, and instrumental
    broadening of the 1394 line with half the opacity and no blends within
    its window.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    """
    obs = raster_partner(time=time, window=window)

    axis = obs.axis_wavelength
    axes = tuple(ax for ax in obs.outputs.axes if ax != axis)
    profile = np.nanmedian(obs.outputs, axis=axes)
    profile = u.Quantity(profile.ndarray).value

    # The continuum under the line does not belong in the kernel: it is not
    # part of the emission profile of a single parcel of plasma.
    num_edge = 10
    profile = profile - (profile[:num_edge].mean() + profile[-num_edge:].mean()) / 2
    profile = np.clip(profile, 0, None)
    profile = profile / profile.sum()

    wavelength = obs.inputs.wavelength
    wavelength = wavelength[{ax: 0 for ax in wavelength.shape if ax != axis}]
    wavelength = wavelength.ndarray.to_value(u.AA)
    wavelength = (wavelength[:~0] + wavelength[1:]) / 2

    return {"kernel": profile, "wavelength": wavelength}


def kernel(
    time: str = time_default,
    window: str = window_default,
) -> npt.NDArray:
    """
    The deconvolution kernel: the median 1403 profile, normalized to one.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    """
    return _kernel(time=time, window=window)["kernel"]


def _wiener(
    spectra: npt.NDArray,
    response: npt.NDArray,
    axis: int,
) -> npt.NDArray:
    """
    Deconvolve spectra by a kernel, linearly and all with the same filter.

    Linear so that the noise stays tractable and the operation commutes
    with everything downstream, and one filter for every pixel so that the
    band statistics mean the same thing everywhere, which is what lets the
    deficit control calibrate them.

    Parameters
    ----------
    spectra
        The spectra to sharpen.
    response
        The kernel, normalized to unit sum.
    axis
        The logical position of the spectral axis of `spectra`.
    """
    n = spectra.shape[axis]
    length = 1 << int(np.ceil(np.log2(n + len(response))))

    # The kernel centered on its own peak, so that deconvolution does not
    # translate the spectra.
    center = int(np.argmax(response))
    padded = np.zeros(length)
    padded[: len(response)] = response
    padded = np.roll(padded, -center)
    transfer = np.fft.rfft(padded)

    # The spectra padded with their own edge values, so that the filter
    # does not wrap the line into the continuum.
    spectra = np.moveaxis(spectra, axis, -1)
    edge_lo = spectra[..., :1]
    edge_hi = spectra[..., ~0:]
    pad_lo = np.broadcast_to(edge_lo, spectra.shape[:-1] + ((length - n) // 2,))
    pad_hi = np.broadcast_to(
        edge_hi, spectra.shape[:-1] + (length - n - pad_lo.shape[-1],)
    )
    padded = np.concatenate([pad_lo, spectra, pad_hi], axis=-1)

    power = np.abs(transfer) ** 2
    factor = np.conj(transfer) / (power + regularization * power.max())

    sharpened = np.fft.irfft(np.fft.rfft(padded, axis=-1) * factor, n=length, axis=-1)
    sharpened = sharpened[..., pad_lo.shape[-1] : pad_lo.shape[-1] + n]

    return np.moveaxis(sharpened, -1, axis)


@memory.cache
def _outputs_deconvolved(
    time: str,
    window: str,
    beta: float,
) -> npt.NDArray:
    """
    The despiked 1394 signal, deconvolved by the doublet kernel.

    Cached without units as a single-precision array, which is the one
    thing which can be both stored and memory-mapped reliably.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    beta
        The regularization, keyed into the cache so that tuning it does not
        serve stale spectra.
    """
    obs = raster(time=time, window=window)
    response = kernel(time=time, window=window)

    axis = obs.outputs.axes.index(obs.axis_wavelength)
    values = u.Quantity(obs.outputs.ndarray).value

    return _wiener(values, response, axis=axis).astype(np.float32)


def deconvolved(
    time: str = time_default,
    window: str = window_default,
) -> iris.sg.SpectrographObservation:
    """
    The raster with every spectrum deconvolved by the doublet kernel.

    In the sharpened spectra, emission at a velocity means plasma at that
    velocity relative to the typical profile: the width of the line has
    been removed rather than padded around, so the wing bands can start at
    the sound speed itself.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    """
    import dataclasses

    obs = raster(time=time, window=window)

    values = _outputs_deconvolved(
        time=time,
        window=window,
        beta=regularization,
    )

    return dataclasses.replace(
        obs,
        outputs=na.ScalarArray(
            ndarray=values.astype(float) << na.unit(obs.outputs),
            axes=obs.outputs.axes,
        ),
    )


def deconvolution(
    figsize: tuple[float, float] = (16, 5),
    dpi: float = 300,
) -> pathlib.Path:
    """
    The delta test: what deconvolution does to the profiles it must nail.

    Three panels: the kernel against the 1394 median profile, both
    normalized; the deconvolved median profile, which must collapse toward
    a delta at rest, with the blends standing out sharpened; and the
    deconvolved spectrum of the strongest event against its original, to
    show what the census will integrate.

    Parameters
    ----------
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved figure.
    """
    from ._overview import path_figures, _velocity_centers

    obs = raster()
    sharp = deconvolved()

    axis_time = obs.axis_time
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    velocity = _velocity_centers(obs).ndarray.to_value(u.km / u.s)

    axes = (axis_time, axis_x, axis_y)
    median = u.Quantity(np.nanmedian(obs.outputs, axis=axes).ndarray).value
    median_sharp = u.Quantity(np.nanmedian(sharp.outputs, axis=axes).ndarray).value

    response = kernel()
    info = _kernel(time=time_default, window=window_default)
    wavelength_kernel = info["wavelength"]
    rest = 1402.770
    velocity_kernel = (wavelength_kernel - rest) / rest * 299792.458

    # the strongest census event, for the before-and-after
    from ._catalog import catalog

    table = catalog()
    i = int(np.argmax(table["sig"]))
    position = obs.inputs.position[{axis_time: 0}].cell_centers((axis_x, axis_y))
    px = position.x.ndarray.to_value(u.arcsec)
    py = position.y.ndarray.to_value(u.arcsec)
    distance = np.hypot(px - table["x"][i], py - table["y"][i])
    ix, iy = np.unravel_index(np.nanargmin(distance), distance.shape)
    index = {axis_time: 0, axis_x: int(ix), axis_y: int(iy)}
    example = u.Quantity(obs.outputs[index].ndarray).value
    example_sharp = u.Quantity(sharp.outputs[index].ndarray).value

    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(ncols=3, figsize=figsize, constrained_layout=True)

        ax = axs[0]
        ax.plot(velocity, median / median.max(), color="gray", label="median 1394")
        ax.plot(
            velocity_kernel,
            response / response.max(),
            color="tab:blue",
            label="kernel (median 1403)",
        )
        ax.set_xlim(-300, 300)
        ax.set_xlabel("LOS velocity (km/s)")
        ax.set_ylabel("normalized")
        ax.legend(fontsize=8)

        ax = axs[1]
        ax.plot(velocity, median, color="gray", label="median 1394")
        ax.plot(velocity, median_sharp, color="tab:red", label="deconvolved")
        ax.set_yscale("symlog", linthresh=100)
        ax.set_xlim(-300, 300)
        ax.axvline(0, color="black", linewidth=0.5, linestyle="dashed")
        ax.set_xlabel("LOS velocity (km/s)")
        ax.legend(fontsize=8)
        ax.set_title("the delta test", fontsize=10)

        ax = axs[2]
        ax.plot(velocity, example, color="gray", label="event, original")
        ax.plot(velocity, example_sharp, color="tab:red", label="event, deconvolved")
        ax.set_xlim(-300, 300)
        ax.axvline(0, color="black", linewidth=0.5, linestyle="dashed")
        ax.set_xlabel("LOS velocity (km/s)")
        ax.legend(fontsize=8)

    path_figures.mkdir(parents=True, exist_ok=True)
    path = path_figures / "deconvolution.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path
