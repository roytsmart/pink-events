"""
The false-color rendering the pink events were noticed in.
"""

import numpy as np
import astropy.units as u
import named_arrays as na
import iris
from ._observations import slice_velocity

__all__ = [
    "rgb",
]

#: The Doppler velocity mapped to each end of the visible spectrum.
velocity_color_default = 150 * u.km / u.s

#: The percentile of the signal placed at the top of the brightness scale.
percentile_default = 99.5


def rgb(
    obs: iris.sg.SpectrographObservation,
    velocity_color: u.Quantity = velocity_color_default,
    percentile: float = percentile_default,
    num_chunk: int = 25,
) -> tuple[na.FunctionArray, na.FunctionArray]:
    """
    Render the raster as an RGB image, the way the presentation did.

    Returns the RGB image and the colorbar, as
    :func:`named_arrays.colorsynth.rgb_and_colorbar` would, but computed in
    chunks along the raster axis: :func:`colorsynth.rgb` transiently
    allocates about seventeen times the size of its input
    (sun-data/colorsynth#10), which for this raster is more memory than a
    laptop has.

    Parameters
    ----------
    obs
        The raster to render.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum.
    percentile
        The percentile of the signal placed at the top of the brightness
        scale, computed over the whole raster so that every chunk is scaled
        alike.
    num_chunk
        The number of raster steps to render at a time.
    """
    # Only the wavelengths that are mapped to a color. The window reaches
    # far into the continuum on both sides, and while the color matching
    # functions give those wavelengths no hue, the presentation rendered only
    # the colored range, and the continuum left in would integrate to a
    # bright magenta veil over the whole raster.
    obs = slice_velocity(obs, velocity_color)

    axis = obs.axis_wavelength
    axis_x = obs.axis_detector_x

    wavelength_rest = na.as_named_array(obs.inputs.wavelength_rest)
    equivalency = u.doppler_optical(wavelength_rest.ndarray)
    wavelength_min = (-velocity_color).to(u.AA, equivalencies=equivalency)
    wavelength_max = (+velocity_color).to(u.AA, equivalencies=equivalency)

    # The centers of the wavelength cells, since the signal lives on the
    # cells and the colors should be evaluated where the signal is.
    lower = {axis: slice(None, ~0)}
    upper = {axis: slice(+1, None)}
    wavelength = obs.inputs.wavelength
    wavelength = wavelength[{ax: 0 for ax in wavelength.shape if ax != axis}]
    wavelength = (wavelength[lower] + wavelength[upper]) / 2

    axis_orthogonal = tuple(ax for ax in obs.outputs.shape if ax != axis)

    spd_max = np.nanpercentile(
        obs.outputs,
        percentile,
        axis=axis_orthogonal,
    )

    # Zero, the way `SpectrographObservation.show` pins it, and passed
    # explicitly so that every chunk and the colorbar are normalized alike:
    # left to :func:`colorsynth.rgb`, the floor would be the minimum of
    # whatever it is handed, which paints a seam at every chunk boundary,
    # and the despiked radiance reaches a hundred thousand below zero in the
    # noise, which put the zero level at four fifths of the brightness scale
    # and washed the whole raster pale.
    spd_min = 0 * na.unit(obs.outputs)

    num_x = obs.outputs.shape[axis_x]
    chunks = []
    for start in range(0, num_x, num_chunk):
        chunk = obs.outputs[{axis_x: slice(start, start + num_chunk)}]
        chunk = na.colorsynth.rgb(
            spd=chunk,
            wavelength=wavelength,
            axis=axis,
            spd_min=spd_min,
            spd_max=spd_max,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        chunks.append(chunk)

    image = na.concatenate(chunks, axis=axis_x)

    colorbar = na.colorsynth.colorbar(
        spd=obs.outputs,
        wavelength=wavelength,
        axis=axis,
        spd_min=spd_min,
        spd_max=spd_max,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )

    image = na.FunctionArray(
        inputs=obs.inputs.position,
        outputs=image,
    )

    return image, colorbar
