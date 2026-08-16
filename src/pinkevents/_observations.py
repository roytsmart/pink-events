"""
The one raster this investigation is about.
"""

import dataclasses
import numpy.typing as npt
import astropy.units as u
import named_arrays as na
import iris
from ._caching import memory

__all__ = [
    "raster",
    "raster_partner",
    "slice_velocity",
]

#: The time of the IRIS observation containing the pink events.
time_default = "2013-10-22 11:30"

#: The IRIS spectral window in which the pink events were noticed.
window_default = "Si IV 1394"

#: The rest wavelength of the other member of the Si IV doublet, which the
#: window is wide enough to contain.
wavelength_partner = 1402.770 * u.AA


@memory.cache
def _outputs_despiked(
    time: str,
    window: str,
) -> npt.NDArray:
    """
    The despiked signal of the raster, without units.

    Despiking the whole spectral window takes several minutes, so the result
    is cached. Only the signal is cached, and only its values, since the
    coordinates are cheap to load and a plain array is the one thing which
    can be both stored and memory-mapped reliably.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    """
    result = iris.sg.open(
        time=time,
        window=window,
    )

    result = na.despike(
        array=result,
        axis=(result.axis_wavelength, result.axis_detector_y),
    )

    return result.outputs.ndarray.value


def raster(
    time: str = time_default,
    window: str = window_default,
    velocity_limit: u.Quantity = 400 * u.km / u.s,
) -> iris.sg.SpectrographObservation:
    """
    The raster in physical units, cleaned of spikes, cropped to the line.

    The cosmic ray spikes have been removed from the signal, and the signal
    has been converted from instrument units into a spectral radiance. The
    sky coordinates are left exactly where IRIS put them, since where the
    events sit relative to the network is one of the questions.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    velocity_limit
        How far into the wings to keep, measured as a Doppler shift from the
        rest wavelength. The window is much wider than the line, and the
        continuum beyond the wings only adds to the cost of everything
        downstream.
    """
    result = iris.sg.open(
        time=time,
        window=window,
    )

    outputs = _outputs_despiked(
        time=time,
        window=window,
    )

    result = dataclasses.replace(
        result,
        outputs=na.ScalarArray(
            ndarray=outputs << na.unit(result.outputs),
            axes=result.outputs.axes,
        ),
    )

    result = slice_velocity(result, velocity_limit)

    return result.radiance


def raster_partner(
    time: str = time_default,
    window: str = window_default,
    velocity_limit: u.Quantity = 200 * u.km / u.s,
) -> iris.sg.SpectrographObservation:
    """
    The same raster, cropped about the other member of the Si IV doublet.

    The doublet partner at 1402.77 forms at the same temperature from the
    same ion moving at the same velocities, through the same spectrograph,
    so its profile is the profile of the 1394 line with half the opacity
    and none of the 1394 line's blends nearby.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    velocity_limit
        How far from the partner's rest wavelength to keep. The default
        stays inside the Fe II blend at -214 km/s.
    """
    result = iris.sg.open(
        time=time,
        window=window,
    )

    outputs = _outputs_despiked(
        time=time,
        window=window,
    )

    result = dataclasses.replace(
        result,
        outputs=na.ScalarArray(
            ndarray=outputs << na.unit(result.outputs),
            axes=result.outputs.axes,
        ),
    )

    result = slice_velocity(
        result,
        velocity_limit,
        wavelength_rest=wavelength_partner,
    )

    return result.radiance


def slice_velocity(
    obs: iris.sg.SpectrographObservation,
    velocity_limit: u.Quantity,
    wavelength_rest: "None | u.Quantity" = None,
) -> iris.sg.SpectrographObservation:
    """
    Crop the spectral axis to a Doppler velocity range about a line.

    Parameters
    ----------
    obs
        The observation to crop.
    velocity_limit
        How far from the rest wavelength to keep, on both sides.
    wavelength_rest
        The rest wavelength the velocities are measured from. If
        :obj:`None`, the window's own line is used.
    """
    axis = obs.axis_wavelength

    if wavelength_rest is not None:
        wavelength = obs.inputs.wavelength
        wavelength = wavelength[{ax: 0 for ax in wavelength.shape if ax != axis}]
        equivalency = u.doppler_optical(wavelength_rest)
        velocity = na.ScalarArray(
            ndarray=wavelength.ndarray.to(u.km / u.s, equivalencies=equivalency),
            axes=wavelength.axes,
        )
    else:
        velocity = obs.inputs.velocity
        velocity = velocity[{ax: 0 for ax in velocity.shape if ax != axis}]

    where = (-velocity_limit < velocity) & (velocity < +velocity_limit)
    indices = na.indices(where.shape)[axis][where]
    lower = int(indices.min().ndarray)
    upper = int(indices.max().ndarray)

    # The upper vertex belongs to the cell before it, so the cell slice stops
    # one short of the vertex slice.
    return obs[{axis: slice(lower, upper)}]
