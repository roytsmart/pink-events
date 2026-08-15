"""
Finding the small, dim, pink things in the rendered image.
"""

import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "Candidate",
    "candidates",
]


@dataclasses.dataclass
class Candidate:
    """A pink pixel worth pulling a spectrum at."""

    index: dict[str, int]
    """The array index of the pixel, one entry per spatial axis."""

    position: na.Cartesian2dVectorArray
    """The helioprojective position of the pixel."""

    score: float
    """How much pinker than its surroundings the pixel is."""


def _background(
    a: np.ndarray,
    halfwidth: tuple[int, int],
) -> np.ndarray:
    """
    A boxcar average over the two trailing axes, via summed-area tables.

    Returns the average and the fraction of each box which held a finite
    value, since a box hanging off the edge of the detector or over a data
    gap averages whatever is left and says nothing about the neighborhood.

    Parameters
    ----------
    a
        The array to smooth. The two trailing axes are the spatial ones.
    halfwidth
        The half-width of the box along each of the two trailing axes.
    """
    hx, hy = halfwidth

    finite = np.isfinite(a)
    total = np.where(finite, a, 0)

    def box(b: np.ndarray) -> np.ndarray:
        b = np.pad(b, [(0, 0)] * (b.ndim - 2) + [(hx + 1, hx), (hy + 1, hy)])
        s = b.cumsum(~1).cumsum(~0)
        return (
            s[..., 2 * hx + 1 :, 2 * hy + 1 :]
            - s[..., : -(2 * hx + 1), 2 * hy + 1 :]
            - s[..., 2 * hx + 1 :, : -(2 * hy + 1)]
            + s[..., : -(2 * hx + 1), : -(2 * hy + 1)]
        )

    coverage = box(finite.astype(float)) / ((2 * hx + 1) * (2 * hy + 1))

    return box(total) / np.maximum(box(finite.astype(float)), 1), coverage


def candidates(
    image: na.FunctionArray,
    axis_rgb: str,
    axis_x: str,
    axis_y: str,
    halfwidth_background: u.Quantity = 4 * u.arcsec,
    num: int = 6,
    separation: u.Quantity = 10 * u.arcsec,
) -> list[Candidate]:
    """
    The pixels most pinker than their own neighborhoods, in the dark lanes.

    The rendered raster is pink nearly everywhere, since the continuum under
    the line integrates to a magenta veil, so pink in an absolute sense
    selects nothing. What the eye picked out were small patches pinker than
    their immediate surroundings, inside the dark cell interiors. Each pixel
    is therefore compared to a local boxcar background: its score is how
    much its red and blue channels rise above that background together,
    beyond what its green channel does, and pixels whose background is
    brighter than typical, the network, are excluded.

    Parameters
    ----------
    image
        The rendered raster, as returned by :func:`pinkevents.rgb`.
    axis_rgb
        The logical axis of length three carrying the color channels.
    axis_x
        The logical axis of the raster steps.
    axis_y
        The logical axis along the slit.
    halfwidth_background
        The half-width of the neighborhood a pixel is compared against.
    num
        The number of candidates to keep.
    separation
        The minimum distance between the candidates kept, so that one event
        does not use up the whole list.
    """
    rgb = image.outputs

    # Centers only along the axes which are actually vertex grids: asking
    # for cells along the size-one time axis leaves nothing at all.
    position = image.inputs.cell_centers((axis_x, axis_y))

    # The pixel scale along each spatial axis, so that the neighborhood is
    # the same patch of sky in both directions however anisotropic the
    # raster is.
    scale_x = np.nanmean(np.abs(np.diff(position.x, axis=axis_x))).ndarray
    scale_y = np.nanmean(np.abs(np.diff(position.y, axis=axis_y))).ndarray
    hx = max(
        int((halfwidth_background / scale_x).to_value(u.dimensionless_unscaled)), 1
    )
    hy = max(
        int((halfwidth_background / scale_y).to_value(u.dimensionless_unscaled)), 1
    )

    a = rgb.ndarray
    source = (rgb.axes.index(axis_rgb), rgb.axes.index(axis_x), rgb.axes.index(axis_y))
    a = np.moveaxis(a, source=source, destination=(0, ~1, ~0))

    background, coverage = _background(a, halfwidth=(hx, hy))
    excess = np.clip(a - background, 0, None)

    r, g, b = excess
    # Pink is red and blue rising together beyond green: the lesser of the
    # two wings, less the core, so that a plain brightening scores nothing.
    score = np.minimum(r, b) - g

    # Only the dark lanes: a pixel whose surroundings are brighter than the
    # typical neighborhood is in or beside the network, which is not where
    # these events were noticed.
    luminance = background.mean(0)
    score = np.where(luminance < np.median(luminance), score, 0)

    # Only where the whole neighborhood is really there. The detector edges
    # and the data-gap columns leave the background computed from a sliver,
    # and a pixel beats a sliver too easily to mean anything.
    score = np.where(coverage.min(0) > 0.99, score, 0)
    score[..., : 2 * hx + 1, :] = 0
    score[..., -(2 * hx + 1) :, :] = 0
    score[..., :, : 2 * hy + 1] = 0
    score[..., :, -(2 * hy + 1) :] = 0

    axes_spatial = tuple(ax for ax in rgb.axes if ax != axis_rgb)
    axes_moved = tuple(ax for ax in axes_spatial if ax not in (axis_x, axis_y))
    axes_moved = axes_moved + (axis_x, axis_y)
    score = na.ScalarArray(score, axes=axes_moved)

    order = np.argsort(-score.ndarray, axis=None)

    result = []
    for flat in order:
        if len(result) >= num:
            break
        index_nd = np.unravel_index(flat, score.ndarray.shape)
        index = {ax: int(i) for ax, i in zip(score.axes, index_nd)}
        if score[index].ndarray <= 0:
            break
        here = position[{ax: index[ax] for ax in position.shape}]
        if any((here - c.position).length < separation for c in result):
            continue
        result.append(
            Candidate(
                index=index,
                position=here,
                score=float(score[index].ndarray),
            )
        )

    return result
