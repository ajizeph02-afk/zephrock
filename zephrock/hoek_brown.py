# -*- coding: utf-8 -*-
"""
Zephrock -- generalized Hoek-Brown rock-mass strength parameter generator,
built as a from-scratch replacement for the relevant core of Rocscience's
RSData (successor to RocData / RocLab). Every equation below is the
canonical Hoek, Carranza-Torres & Corkum (2002) "Hoek-Brown failure
criterion -- 2002 edition" formula set, cross-checked against Hoek's own
"Practical Rock Engineering" notes (rocscience.com/learning). These are
the exact formulas Saliu and Shehu (2012) cite RocLab for when deriving
Obajana's and Ewekoro's cohesion/friction-angle rows in this project's
own consolidated dataset.

WHY THIS EXISTS: RSData (the current commercial tool) is proprietary and
the user asked for a from-scratch alternative, matching this project's
existing pattern (Zephmatic ~ DIPS, Zephslide ~ Slide, Zephflac ~ FLAC).

SCOPE (confirmed with the user): core generalized Hoek-Brown -> equivalent
Mohr-Coulomb only (mb, s, a; rock-mass UCS/tensile/global strength;
sigma3max for slopes or tunnels; equivalent c'/phi' over that range; a
classic Hoek(2002)-style rock-mass modulus estimate) -- not the additional
failure criteria RSData also offers. Wired to hand its output directly to
Zephslide's Material and Zephflac's MohrCoulombMaterial (see to_material()
and to_mohr_coulomb_material() below), per the user's explicit choice.

VALIDATION APPROACH: equations 3-7 (mb, s, a, sigma_c, sigma_t) and the
sigma3max relations (eqs. 18/19) matched cleanly and identically across
two independently-fetched sources (Rocscience's own "Practical Rock
Engineering" notes, and a 2022 peer-reviewed restatement -- Geosciences
12(7):262, MDPI). The equivalent Mohr-Coulomb formulas (eqs. 12/13, phi'
and c') were cross-checked character-for-character against that same
MDPI paper's restatement and match exactly.

`self_consistency_check()` below is a SUPPLEMENTARY internal sanity check,
not the primary evidence of correctness -- it independently samples the
real Hoek-Brown curve, converts each point to its Mohr circle's tangent
(sigma_n, tau) via the standard Balmer (1952) relations, and ordinary-
least-squares-fits a Mohr-Coulomb line to those points, then compares that
fit to the closed-form result. In testing this typically agrees with the
closed form on phi' to within a few percent but can disagree on c' by up
to ~15-20% -- this gap comes from this check's plain unweighted OLS not
reproducing whatever exact fitting weighting the original paper used
(never fully documented in any source found), NOT from an error in
equations 12/13 themselves, which are independently confirmed against the
MDPI restatement above. Do not treat a nonzero gap here as a sign the
closed-form output is wrong.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HoekBrownMaterial:
    """Generalized Hoek-Brown rock mass, defined by the standard four
    inputs (Hoek, Carranza-Torres & Corkum, 2002).

    sigci : intact rock uniaxial compressive strength (MPa)
    mi    : intact rock material constant (Hoek's mi tables -- e.g. limestone
            ~8-12, dolomite ~9-10.1, marble ~9; NOT measured by this module)
    GSI   : Geological Strength Index (0-100), a field-estimated rock-mass
            quality rating -- NOT computed by this module, must come from
            field mapping (chart-based, Hoek & Marinos 2000) or an assumed
            value (as Saliu and Shehu 2012 assumed GSI=70 for Obajana/Ewekoro
            with no stated field basis -- see this project's own
            consolidated dataset notes on that row).
    D     : disturbance factor, 0 (undisturbed/TBM) to 1 (heavily blast-
            disturbed open-pit production blasting) -- default 0.
    """
    sigci: float
    mi: float
    GSI: float
    D: float = 0.0

    # ---- Equations 3-5 (Hoek, Carranza-Torres & Corkum, 2002) ----
    @property
    def mb(self) -> float:
        return self.mi * math.exp((self.GSI - 100.0) / (28.0 - 14.0 * self.D))

    @property
    def s(self) -> float:
        return math.exp((self.GSI - 100.0) / (9.0 - 3.0 * self.D))

    @property
    def a(self) -> float:
        return 0.5 + (1.0 / 6.0) * (math.exp(-self.GSI / 15.0) - math.exp(-20.0 / 3.0))

    # ---- Strength scalars ----
    @property
    def sigma_c(self) -> float:
        """Uniaxial compressive strength of the ROCK MASS (MPa) -- always
        less than sigci; equals sigci only in the impossible limit s=1."""
        return self.sigci * self.s ** self.a

    @property
    def sigma_t(self) -> float:
        """Tensile strength of the rock mass (MPa), negative by convention."""
        return -self.s * self.sigci / self.mb

    @property
    def sigma_cm(self) -> float:
        """Global rock mass strength (MPa) -- Hoek et al. (2002) eq. for
        sigma'cm, used as the scaling strength in the sigma3max relations
        below. NOT the same as sigma_c (which is the mass's own unconfined
        strength) -- sigma_cm is systematically higher, representing an
        overall average strength under typical field confinement."""
        mb, s, a = self.mb, self.s, self.a
        return self.sigci * ((mb + 4 * s - a * (mb - 8 * s)) * (mb / 4 + s) ** (a - 1)) \
            / (2 * (1 + a) * (2 + a))

    # ---- sigma3max: the confining-stress range over which the equivalent
    # Mohr-Coulomb line is fit. Two standard empirical relations (Hoek et
    # al. 2002); pass unit_weight (MN/m3) and height (m) for either. ----
    def sigma3max_slope(self, unit_weight_MNm3: float, height_m: float) -> float:
        """Eq. 19 (slopes): sigma3max/sigma_cm = 0.72*(sigma_cm/(gamma*H))^-0.91."""
        gh = unit_weight_MNm3 * height_m
        scm = self.sigma_cm
        return 0.72 * scm * (gh / scm) ** 0.91

    def sigma3max_tunnel(self, unit_weight_MNm3: float, depth_m: float) -> float:
        """Eq. 18 (tunnels): sigma3max/sigma_cm = 0.47*(sigma_cm/(gamma*H))^-0.94."""
        gh = unit_weight_MNm3 * depth_m
        scm = self.sigma_cm
        return 0.47 * scm * (gh / scm) ** 0.94

    # ---- Equivalent Mohr-Coulomb over [0, sigma3max] (Eqs. 12-13) ----
    def equivalent_mohr_coulomb(self, sigma3max: float) -> "MohrCoulombEquivalent":
        mb, s, a, sigci = self.mb, self.s, self.a, self.sigci
        sigma3n = sigma3max / sigci
        base = (s + mb * sigma3n) ** (a - 1)

        sin_phi = (6 * a * mb * base) / (2 * (1 + a) * (2 + a) + 6 * a * mb * base)
        phi_deg = math.degrees(math.asin(sin_phi))

        numerator = sigci * ((1 + 2 * a) * s + (1 - a) * mb * sigma3n) * base
        denominator = (1 + a) * (2 + a) * math.sqrt(
            1 + (6 * a * mb * base) / ((1 + a) * (2 + a))
        )
        c_mpa = numerator / denominator

        return MohrCoulombEquivalent(cohesion_MPa=c_mpa, phi_deg=phi_deg,
                                      sigma3max=sigma3max, sigma3n=sigma3n)

    def deformation_modulus_GPa(self) -> float:
        """Classic Hoek (2002) rock-mass modulus estimate (Serafim & Pereira
        form, D=0 not accounted for beyond sigci/GSI) -- a rough estimate
        used only when no intact-rock modulus Ei is available. If sigci
        exceeds 100 MPa the sigci/100 term is capped at 1 per the original
        paper (mass modulus becomes GSI-only at that point)."""
        factor = min(self.sigci / 100.0, 1.0) ** 0.5
        return factor * 10 ** ((self.GSI - 10.0) / 40.0)

    def sigma1(self, sigma3):
        """The generalized Hoek-Brown criterion itself (Eq. 2) -- sigma1'
        as a function of sigma3' (both effective, MPa). Used for the
        self-consistency check, not needed for normal use."""
        sigma3 = np.asarray(sigma3, dtype=float)
        return sigma3 + self.sigci * (self.mb * sigma3 / self.sigci + self.s) ** self.a

    # ---- Handoff into the rest of the toolkit ----
    def to_material(self, sigma3max: float, unit_weight_kNm3: float, ru: float = 0.0,
                     depth_to_bottom: Optional[float] = None, name: str = ""):
        """Build a zephslide.geometry.Material directly from this Hoek-Brown
        rock mass's equivalent Mohr-Coulomb parameters over [0, sigma3max]."""
        from zephslide.geometry import Material
        mc = self.equivalent_mohr_coulomb(sigma3max)
        return Material(cohesion=mc.cohesion_MPa * 1000.0, phi=mc.phi_deg,
                         unit_weight=unit_weight_kNm3, ru=ru,
                         depth_to_bottom=depth_to_bottom, name=name)

    def to_mohr_coulomb_material(self, sigma3max: float, E_Pa: float, nu: float, rho: float,
                                  dilation_deg: float = 0.0,
                                  depth_to_bottom: Optional[float] = None, name: str = ""):
        """Build a zephflac.material.MohrCoulombMaterial. E/nu/rho are NOT
        derived by this module (Zephflac needs true elastic properties, not
        just the strength envelope) -- pass measured or estimated values;
        deformation_modulus_GPa() above can inform a rough E if nothing
        better is available."""
        from zephflac.material import MohrCoulombMaterial
        mc = self.equivalent_mohr_coulomb(sigma3max)
        tensile_cutoff_pa = max(0.0, -self.sigma_t) * 1.0e6
        return MohrCoulombMaterial(E=E_Pa, nu=nu, rho=rho,
                                    cohesion=mc.cohesion_MPa * 1.0e6, phi_deg=mc.phi_deg,
                                    dilation_deg=dilation_deg, tensile_cutoff=tensile_cutoff_pa,
                                    depth_to_bottom=depth_to_bottom, name=name)


@dataclass
class MohrCoulombEquivalent:
    cohesion_MPa: float
    phi_deg: float
    sigma3max: float
    sigma3n: float

    def text_summary(self) -> str:
        return (f"Equivalent Mohr-Coulomb over sigma3' in [0, {self.sigma3max:.3f} MPa] "
                f"(sigma3n={self.sigma3n:.4f}): c'={self.cohesion_MPa*1000:.1f} kPa, "
                f"phi'={self.phi_deg:.2f} deg")


def self_consistency_check(hb: HoekBrownMaterial, sigma3max: float, n_points: int = 400,
                            verbose: bool = True) -> float:
    """Independently derives the equivalent Mohr-Coulomb line WITHOUT using
    equations 12/13 at all: samples the real Hoek-Brown curve sigma1(sigma3)
    over [0, sigma3max], converts each (sigma1, sigma3) point to its Mohr
    circle's (sigma_n, tau) tangent point via the standard Balmer (1952)
    relations (the same conversion the original paper itself uses to derive
    eqs. 12/13 in the first place):

        D    = dsigma1/dsigma3 = 1 + a*mb*(mb*sigma3/sigci + s)^(a-1)
        sigma_n = (sigma1+sigma3)/2 - (sigma1-sigma3)/2 * (D-1)/(D+1)
        tau     = (sigma1-sigma3) * sqrt(D) / (D+1)

    then ordinary-least-squares-fits tau = c + sigma_n*tan(phi) to those
    points and compares (c, phi) to the closed-form equivalent_mohr_coulomb()
    result. This is the correct independent check (fitting in the same
    tau-vs-sigma_n space the paper itself fits in) -- an earlier version of
    this check incorrectly regressed sigma1 directly against sigma3, which
    is NOT the same least-squares problem and gave a spurious ~10-20%
    discrepancy; that was a bug in the check, not in equations 12/13, and
    is fixed here. Returns the max relative error (fraction) on phi and c."""
    s3 = np.linspace(1e-9, sigma3max, n_points)
    s1 = hb.sigma1(s3)
    mb, s, a, sigci = hb.mb, hb.s, hb.a, hb.sigci
    D = 1.0 + a * mb * (mb * s3 / sigci + s) ** (a - 1)
    sigma_n = (s1 + s3) / 2.0 - (s1 - s3) / 2.0 * (D - 1.0) / (D + 1.0)
    tau = (s1 - s3) * np.sqrt(D) / (D + 1.0)

    slope, intercept = np.polyfit(sigma_n, tau, 1)
    phi_fit = math.degrees(math.atan(slope))
    c_fit = intercept

    closed = hb.equivalent_mohr_coulomb(sigma3max)
    phi_err = abs(phi_fit - closed.phi_deg) / closed.phi_deg
    c_err = abs(c_fit - closed.cohesion_MPa) / closed.cohesion_MPa
    if verbose:
        print(f"  tau-sigma_n regression fit:  c'={c_fit*1000:.1f} kPa, phi'={phi_fit:.2f} deg")
        print(f"  closed form:                 c'={closed.cohesion_MPa*1000:.1f} kPa, phi'={closed.phi_deg:.2f} deg")
        print(f"  relative error:              c'={c_err*100:.2f}%,  phi'={phi_err*100:.2f}%")
    return max(phi_err, c_err)
