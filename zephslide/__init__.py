"""
Zephslide -- a from-scratch 2D limit-equilibrium slope-stability engine
(Bishop's Simplified Method, circular slip-surface search), built as a
standalone companion to Zephmatic (the kinematic/stereonet screening tool).

See README.md for scope, methodology, citations, and validation approach.
"""
from .geometry import Material, Bench, Slope, simple_slope
from .lem_core import bishop_fos, fellenius_fos, SliceResult, FoSResult
from .search import search_critical_circle, SearchResult
from .model import SlopeStabilityModel

__all__ = [
    "Material", "Bench", "Slope", "simple_slope",
    "bishop_fos", "fellenius_fos", "SliceResult", "FoSResult",
    "search_critical_circle", "SearchResult",
    "SlopeStabilityModel",
]
