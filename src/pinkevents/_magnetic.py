"""
The magnetic field under each event, from co-temporal HMI magnetograms.
"""

import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import sdo

__all__ = [
    "flux_density",
]


def flux_density(
    position: na.Cartesian2dVectorArray,
    time: astropy.time.Time,
    halfwidth: u.Quantity = 3 * u.arcsec,
) -> u.Quantity:
    """
    The line-of-sight flux density under a position, from the nearest
    HMI magnetogram in time.

    The signed field is averaged over a box and the absolute value taken
    afterwards, rather than averaging the absolute field: the magnetogram
    noise is several gauss per pixel and does not care about sign, so it
    averages away in the signed mean but would fold into a false floor of
    its own size in the unsigned one.

    The magnetogram is taken at the moment the raster's slit crossed the
    position, so no correction for solar rotation is needed; what remains
    is the absolute pointing of the two instruments, an arcsecond or two,
    small against a network element.

    Parameters
    ----------
    position
        The helioprojective position to sample.
    time
        The time the position was observed.
    halfwidth
        The half-width of the box averaged over.
    """
    mag = sdo.hmi.open(time.isot)

    axis_time = mag.axis_time
    axis_x = mag.axis_detector_x
    axis_y = mag.axis_detector_y

    index_time = {axis_time: 0}
    inputs = mag.inputs

    def keyword(a, unit: "None | u.UnitBase" = None) -> float:
        a = na.as_named_array(a)
        if axis_time in a.shape:
            a = a[index_time]
        value = a.ndarray
        if unit is not None:
            value = value.to_value(unit)
        return float(value)

    crval_x = keyword(inputs.crval.position.x, u.arcsec)
    crval_y = keyword(inputs.crval.position.y, u.arcsec)
    crpix_x = keyword(inputs.crpix.components[axis_x])
    crpix_y = keyword(inputs.crpix.components[axis_y])
    cdelt_x = keyword(inputs.cdelt.position.x, u.arcsec)
    cdelt_y = keyword(inputs.cdelt.position.y, u.arcsec)

    m_xx = keyword(inputs.pc.position.x.components[axis_x])
    m_xy = keyword(inputs.pc.position.x.components[axis_y])
    m_yx = keyword(inputs.pc.position.y.components[axis_x])
    m_yy = keyword(inputs.pc.position.y.components[axis_y])

    # The transformation run backward: the forward one takes a pixel to the
    # sky through the rotation and the plate scale, so the sky goes to a
    # pixel through their inverses.
    q_x = (keyword(position.x, u.arcsec) - crval_x) / cdelt_x
    q_y = (keyword(position.y, u.arcsec) - crval_y) / cdelt_y

    determinant = m_xx * m_yy - m_xy * m_yx
    p_x = (m_yy * q_x - m_xy * q_y) / determinant + crpix_x
    p_y = (m_xx * q_y - m_yx * q_x) / determinant + crpix_y

    # Half a pixel more than the position, since `AbstractWcsVector` places
    # its grid on the cell edges and the signal lives on the cells.
    half = int(abs(halfwidth.to_value(u.arcsec) / cdelt_x)) + 1

    box = {
        axis_x: slice(int(p_x) - half, int(p_x) + half + 1),
        axis_y: slice(int(p_y) - half, int(p_y) + half + 1),
    }

    field = mag.outputs[index_time | box]

    return np.abs(np.nanmean(field.ndarray))
