"""
An investigation of the small, dim, pink events in the supergranule
cell interiors of an IRIS raster.
"""

from ._observations import raster
from ._rgb import rgb
from ._candidates import Candidate, candidates
from ._overview import overview, event

__all__ = [
    "raster",
    "rgb",
    "Candidate",
    "candidates",
    "overview",
    "event",
]
