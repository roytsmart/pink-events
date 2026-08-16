"""
An investigation of the small, dim, pink events in the supergranule
cell interiors of an IRIS raster.
"""

from ._observations import raster
from ._rgb import rgb
from ._candidates import Candidate, candidates
from ._overview import overview, event
from ._search import dim_events
from ._network import network, distance_to
from ._magnetic import flux_density
from ._catalog import catalog, statistics, profiles
from ._deconvolve import kernel, deconvolved, deconvolution
from ._averages import average_profile, average_ee
from ._mgii import response_mgii
from ._cii import response_cii
from ._response import response, line_list
from ._sizes import sizes

__all__ = [
    "raster",
    "rgb",
    "Candidate",
    "candidates",
    "overview",
    "event",
    "dim_events",
    "network",
    "distance_to",
    "flux_density",
    "catalog",
    "statistics",
    "profiles",
    "kernel",
    "deconvolved",
    "deconvolution",
    "average_profile",
    "average_ee",
    "response_mgii",
    "response_cii",
    "response",
    "line_list",
    "sizes",
]
