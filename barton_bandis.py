# -*- coding: utf-8 -*-
"""
Barton-Bandis empirical joint shear-strength criterion (Barton & Choubey
1977; Barton & Bandis 1990) -- a THIRD, physically distinct criterion added
to Zephrock alongside the GSI-based generalized Hoek-Brown (hoek_brown.py,
rock-mass strength) and the fitted power curve (power_curve.py, lab-test
fitting). This one is not a rock-mass strength model at all: it is the
peak dilatant SHEAR STRENGTH OF A SINGLE ROCK JOINT/DISCONTINUITY, the
standard criterion used for joint shear-strength input to Zephmatic-style
kinematic/wedge stability checks and to jointed-rock numerical models --
a different physical quantity than sigci/GSI-based rock-mass strength.

CRITERION (standard form, unchanged from the original publications):

    tau = sigma_n * tan( JRC * log10(JCS / sigma_n) + phi_r )

    JRC    : Joint Roughness Coefficient (0-20 scale, from comb-profile
             comparison charts -- Barton & Choubey 1977 -- or a field
             tilt-test back-calculation; NOT computed by this module.
    JCS    : Joint wall Compressive Strength (MPa) -- equals the intact
             rock's UCS for an unweathered joint, or a lower Schmidt-hammer-
             estimated value for a weathered joint wall.
    phi_r  : residual friction angle (deg) -- typically a few degrees below
             the basic friction angle phi_b for a weathered joint, or
             approx equal to phi_b for an unweathered one.
    sigma_n: effective normal stress across the joint (MPa).

VALIDITY: this criterion is only valid over sigma_n < JCS (beyond that the
roughness/dilation contribution the log term represents no longer applies
physically -- e.g. the joint asperities would be sheared through, not
dilating over them); this module warns rather than silently extrapolating.

Instantaneous cohesion/friction (the tangent Mohr-Coulomb line at a given
sigma_n -- how a nonlinear joint criterion is normally reported alongside
a linear-criterion-based limit-equilibrium/numerical tool) are computed via
the exact analytic derivative, not a numerical approximation -- see
instantaneous_params() docstring for the derivation.
"""
import math
import warnings
from dataclasses import dataclass


@dataclass
class BartonBandisJoint:
    """A single rock joint/discontinuity's peak shear-strength model.

    JRC    : Joint Roughness Coefficient (0 = smooth planar, ~20 = very rough).
    JCS    : Joint wall Compressive Strength (MPa).
    phi_r  : residual friction angle (deg).
    """
    JRC: float
    JCS: float
    phi_r: float

    def _check_range(self, sigma_n: float):
        if sigma_n <= 0:
            raise ValueError("sigma_n must be > 0.")
        if sigma_n >= self.JCS:
            warnings.warn(
                f"sigma_n ({sigma_n:.3f} MPa) >= JCS ({self.JCS:.3f} MPa): the "
                f"Barton-Bandis criterion is not physically valid in this range "
                f"(the log10(JCS/sigma_n) term goes to zero or negative, "
                f"implying no/negative roughness contribution) -- result is "
                f"returned but should not be trusted.", stacklevel=3)

    def shear_strength(self, sigma_n: float) -> float:
        """Peak shear strength tau (MPa) at effective normal stress sigma_n (MPa)."""
        self._check_range(sigma_n)
        dilation_angle_deg = self.JRC * math.log10(self.JCS / sigma_n)
        return sigma_n * math.tan(math.radians(dilation_angle_deg + self.phi_r))

    def secant_friction_angle_deg(self, sigma_n: float) -> float:
        """phi_sec = atan(tau/sigma_n) -- the angle of a line from the origin
        through this one point (NOT the tangent/instantaneous angle)."""
        tau = self.shear_strength(sigma_n)
        return math.degrees(math.atan(tau / sigma_n))

    def instantaneous_params(self, sigma_n: float):
        """Exact tangent-line (instantaneous cohesion c_i, friction angle
        phi_i) at a single normal stress -- the standard way a nonlinear
        joint criterion is converted to local Mohr-Coulomb-equivalent
        parameters for a linear solver.

        Derivation: let theta(sigma_n) = JRC*log10(JCS/sigma_n) + phi_r (deg).
        tau = sigma_n * tan(theta_rad).
        dtau/dsigma_n = tan(theta_rad) + sigma_n * sec^2(theta_rad) * dtheta_rad/dsigma_n
        dtheta_deg/dsigma_n = -JRC / (sigma_n * ln(10))
        dtheta_rad/dsigma_n = (pi/180) * dtheta_deg/dsigma_n
        phi_i = atan(dtau/dsigma_n); c_i = tau - sigma_n*dtau/dsigma_n.
        """
        self._check_range(sigma_n)
        theta_deg = self.JRC * math.log10(self.JCS / sigma_n) + self.phi_r
        theta_rad = math.radians(theta_deg)
        tau = sigma_n * math.tan(theta_rad)

        dtheta_deg_dsn = -self.JRC / (sigma_n * math.log(10))
        dtheta_rad_dsn = math.radians(dtheta_deg_dsn)
        dtau_dsn = math.tan(theta_rad) + sigma_n * (1.0 / math.cos(theta_rad)) ** 2 * dtheta_rad_dsn

        phi_i = math.degrees(math.atan(dtau_dsn))
        c_i = tau - sigma_n * dtau_dsn
        return max(c_i, 0.0), phi_i
