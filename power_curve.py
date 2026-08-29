# -*- coding: utf-8 -*-
"""
Generalized power-curve shear-strength criterion -- a second, independent
failure criterion for Zephrock, alongside the GSI-based generalized
Hoek-Brown -> Mohr-Coulomb path in hoek_brown.py.

WHAT THIS IS: unlike hoek_brown.py (which derives strength from GSI/sigci/
mi/D field-mapping inputs), this criterion is fit DIRECTLY to shear-strength
test data you supply -- pairs of (normal stress, shear stress) from direct
shear tests, or (sigma3, sigma1) triaxial pairs converted to (sigma_n, tau)
via the same Balmer (1952) relations used elsewhere in this module. This
matches the "fit a criterion to lab data" workflow RSData/RocData offers
as an alternative to the GSI-based rock-mass approach.

FORM: a two-parameter nonlinear power-law shear envelope,

    tau = sigci * A * (sigma_n / sigci) ** B

(A, B fit by least squares in log-log space). This is a standard, widely
used nonlinear Mohr envelope form for rock and rockfill shear strength
(distinct from, and simpler than, RSData's own internal power-curve
formulation, which was not independently verified against a reliable
source before implementing this -- see the module-level honesty note
below). Treat this as a legitimate, useful nonlinear-envelope fitting tool
in its own right, not a claimed bit-for-bit replication of RSData's
specific power-curve algorithm.

Instantaneous cohesion/friction at any normal stress (the tangent to the
curve at that point, the standard way a nonlinear envelope's local
Mohr-Coulomb-equivalent parameters are reported) are exact analytic
derivatives of the fitted power law, not a numerical approximation:

    tau = sigci*A*(sigma_n/sigci)**B
    dtau/dsigma_n = A*B*(sigma_n/sigci)**(B-1)          (exact)
    phi_i = arctan(dtau/dsigma_n)
    c_i   = tau - sigma_n * dtau/dsigma_n                (tangent-line intercept)
"""
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PowerCurveCriterion:
    """A fitted nonlinear shear-strength envelope: tau = sigci*A*(sigma_n/sigci)**B.

    sigci : a reference/normalizing stress (MPa) -- conventionally the
            intact rock UCS, but any consistent reference stress works.
    A, B  : fitted dimensionless power-curve parameters.
    """
    sigci: float
    A: float
    B: float

    @classmethod
    def fit_from_shear_data(cls, sigma_n: Sequence[float], tau: Sequence[float],
                             sigci: float) -> "PowerCurveCriterion":
        """Least-squares fit in log-log space: ln(tau/sigci) = ln(A) + B*ln(sigma_n/sigci).
        Requires all sigma_n, tau > 0 (points with sigma_n<=0 or tau<=0 are
        dropped, not silently zeroed -- a power law is undefined there)."""
        sigma_n = np.asarray(sigma_n, dtype=float)
        tau = np.asarray(tau, dtype=float)
        mask = (sigma_n > 0) & (tau > 0)
        dropped = int((~mask).sum())
        if dropped:
            print(f"PowerCurveCriterion.fit_from_shear_data: dropped {dropped} "
                  f"point(s) with sigma_n<=0 or tau<=0 (undefined in log-log space).")
        sn, t = sigma_n[mask], tau[mask]
        if len(sn) < 2:
            raise ValueError("Need at least 2 valid (sigma_n>0, tau>0) points to fit.")
        x = np.log(sn / sigci)
        y = np.log(t / sigci)
        B, lnA = np.polyfit(x, y, 1)
        return cls(sigci=sigci, A=math.exp(lnA), B=B)

    def shear_strength(self, sigma_n) -> float:
        sigma_n = np.asarray(sigma_n, dtype=float)
        return self.sigci * self.A * (sigma_n / self.sigci) ** self.B

    def instantaneous_params(self, sigma_n: float):
        """Exact tangent-line (instantaneous cohesion, friction angle) at a
        single normal stress. Returns (c_i_MPa, phi_i_deg)."""
        if sigma_n <= 0:
            raise ValueError("sigma_n must be > 0 for this power-law form.")
        tau = self.shear_strength(sigma_n)
        dtau_dsn = self.A * self.B * (sigma_n / self.sigci) ** (self.B - 1)
        phi_i = math.degrees(math.atan(dtau_dsn))
        c_i = tau - sigma_n * dtau_dsn
        return max(c_i, 0.0), phi_i

    def r_squared(self, sigma_n: Sequence[float], tau: Sequence[float]) -> float:
        """Goodness-of-fit (log-log space) for the data used to fit -- report
        this alongside any fitted A/B so the fit quality is visible, not
        just asserted."""
        sigma_n = np.asarray(sigma_n, dtype=float)
        tau = np.asarray(tau, dtype=float)
        mask = (sigma_n > 0) & (tau > 0)
        x = np.log(sigma_n[mask] / self.sigci)
        y = np.log(tau[mask] / self.sigci)
        y_pred = math.log(self.A) + self.B * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
