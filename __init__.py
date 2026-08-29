"""
Zephrock -- a from-scratch generalized Hoek-Brown rock-mass strength
parameter generator (GSI/UCS/mi/D -> equivalent Mohr-Coulomb c'/phi',
rock-mass strength, tensile strength, sigma3max for slopes or tunnels),
built as a standalone companion to Zephmatic/Zephslide/Zephflac.

See hoek_brown.py's module docstring for scope, methodology, and citations.
"""
from .hoek_brown import HoekBrownMaterial, MohrCoulombEquivalent, self_consistency_check
from .power_curve import PowerCurveCriterion
from .barton_bandis import BartonBandisJoint

__all__ = ["HoekBrownMaterial", "MohrCoulombEquivalent", "self_consistency_check",
           "PowerCurveCriterion", "BartonBandisJoint"]
