"""
lem_core.py

The actual limit-equilibrium math: circular slip-surface geometry (circle
vs. ground-profile intersection, slice discretization), Bishop's Simplified
Method (the MVP's primary FoS calculation), and the Ordinary/Fellenius
Method (implemented purely as an independent internal cross-check -- no
Rocscience Slide license is available to validate against directly, so this
package leans on a second, non-iterative formula with a well-known and
predictable relationship to Bishop's result, plus the analytical
limiting-case checks in validate.py, to build confidence instead).

Layered materials: each slice's WEIGHT integrates unit_weight through
however many material bands its soil column actually crosses (a slice can
span more than one layer vertically); its shear-strength parameters
(cohesion, phi) and pore-pressure ratio are evaluated using whichever
material occupies the elevation at the BASE of the slice -- the layer the
slip surface is actually cutting through at that point, which is the
standard method-of-slices treatment (strength is a base-of-slice property;
weight is a whole-column property). See geometry.py's Slope.layer_bounds()/
material_at_elevation() for how layers are defined and looked up.

Sign convention (derived, not just quoted from a textbook -- see
README.md "Methodology and validation" for the reasoning): for a slice
whose base runs from (x_left, y_base_left) to (x_right, y_base_right) along
the trial circle's lower arc,

    dx = x_right - x_left            (= slice width b, > 0)
    dy = y_base_right - y_base_left
    l  = sqrt(dx^2 + dy^2)           (slant length of the slice base)
    cos(alpha) = dx / l
    sin(alpha) = dy / l

This falls straight out of the slice-base chord geometry -- no arctangent,
no quadrant bookkeeping, no separate sign rule to remember or get backwards.
It was cross-checked against the equivalent radius-vector derivation
(sin(alpha) = (x_mid - x_center) / R) and the two agree exactly, which is
the internal-consistency check documented in the README.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

from .geometry import Slope, Material

MIN_VALID_SPAN_FRACTION = 0.008  # x_right-x_left must exceed this * slope height
N_SAMPLE_POINTS = 800             # for the coarse crossing scan


def circle_y_lower(xc: float, yc: float, R: float, x: float) -> Optional[float]:
    """Lower-arc y of the circle (center xc,yc, radius R) at x, or None if
    x is outside the circle's horizontal span."""
    d2 = R * R - (x - xc) ** 2
    if d2 < 0:
        return None
    return yc - math.sqrt(d2)


@dataclass
class SliceResult:
    x_mid: float
    width: float
    weight: float
    sin_alpha: float
    cos_alpha: float
    u: float          # pore pressure at slice base
    height: float      # average slice height (soil column)
    cohesion: float    # base-of-slice material cohesion (same value for every
                       # slice in a homogeneous slope; may vary slice-to-slice
                       # for a layered one)
    phi: float         # base-of-slice material friction angle (degrees)


@dataclass
class CircleGeometry:
    xc: float
    yc: float
    R: float
    x_left: float
    x_right: float
    slices: List[SliceResult]

    @property
    def depth(self) -> float:
        """Vertical distance from the circle's lowest point to the toe
        elevation -- a convenient, roughly-physical size indicator."""
        return -circle_y_lower(self.xc, self.yc, self.R, self.xc)  # toe is y=0


def find_valid_span(slope: Slope, xc: float, yc: float, R: float
                     ) -> Optional[tuple]:
    """Find the (x_left, x_right) span where the circle's lower arc lies
    strictly below the ground surface (a candidate slip surface), by
    sampling h(x) = ground(x) - circle(x) across the circle's domain and
    locating the first negative->positive crossing (entry) and the next
    positive->negative crossing after it (exit). Returns None if no such
    clean span exists (circle doesn't form a valid failure surface against
    this profile) -- see README "Known limitations" for what this misses
    on complex multi-bench profiles.
    """
    px_min, px_max = slope.x_domain()
    x_lo = max(px_min, xc - R)
    x_hi = min(px_max, xc + R)
    if x_hi - x_lo < 1e-6:
        return None

    xs = [x_lo + i * (x_hi - x_lo) / (N_SAMPLE_POINTS - 1) for i in range(N_SAMPLE_POINTS)]
    hs = []
    for x in xs:
        yc_low = circle_y_lower(xc, yc, R, x)
        if yc_low is None:
            hs.append(None)
        else:
            hs.append(slope.ground_height(x) - yc_low)

    def h_func(x):
        yl = circle_y_lower(xc, yc, R, x)
        if yl is None:
            return -1e9
        return slope.ground_height(x) - yl

    # find first neg->pos crossing (entry)
    x_left = None
    i = 0
    while i < len(xs) - 1:
        h0, h1 = hs[i], hs[i + 1]
        if h0 is not None and h1 is not None and h0 <= 0 < h1:
            x_left = _bisect_root(h_func, xs[i], xs[i + 1])
            break
        i += 1
    if x_left is None:
        return None

    # find next pos->neg crossing after x_left (exit)
    x_right = None
    j = i + 1
    while j < len(xs) - 1:
        h0, h1 = hs[j], hs[j + 1]
        if h0 is not None and h1 is not None and h0 >= 0 > h1:
            x_right = _bisect_root(h_func, xs[j], xs[j + 1])
            break
        j += 1
    if x_right is None:
        return None

    total_h = slope.total_height
    if (x_right - x_left) < MIN_VALID_SPAN_FRACTION * total_h:
        return None

    # sanity: h should be positive at a handful of interior points
    for k in range(1, 4):
        xm = x_left + k * (x_right - x_left) / 4.0
        if h_func(xm) <= 0:
            return None

    return x_left, x_right


def _bisect_root(f, a, b, tol=1e-6, max_iter=60):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if (fa < 0) == (fm < 0):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b)


def _weight_per_width(bounds, y_top: float, y_bottom: float) -> float:
    """Sum of unit_weight_i * (vertical overlap of [y_bottom, y_top] with
    layer i), across whichever layer bands the column actually crosses --
    i.e. weight per unit slice width at one edge (left or right) of a
    slice. Degenerates to unit_weight * (y_top - y_bottom) for a
    homogeneous (single-band) slope."""
    if y_top <= y_bottom:
        return 0.0
    total = 0.0
    for top, bottom, mat in bounds:
        overlap = min(y_top, top) - max(y_bottom, bottom)
        if overlap > 0:
            total += mat.unit_weight * overlap
    return total


def build_slices(slope: Slope, xc: float, yc: float, R: float,
                  x_left: float, x_right: float, n_slices: int = 30) -> List[SliceResult]:
    b = (x_right - x_left) / n_slices
    bounds = slope.layer_bounds()
    slices = []
    for i in range(n_slices):
        xl = x_left + i * b
        xr = xl + b
        y_top_l = slope.ground_height(xl)
        y_top_r = slope.ground_height(xr)
        y_base_l = circle_y_lower(xc, yc, R, xl)
        y_base_r = circle_y_lower(xc, yc, R, xr)
        if y_base_l is None or y_base_r is None:
            continue
        h_l = max(0.0, y_top_l - y_base_l)
        h_r = max(0.0, y_top_r - y_base_r)
        dy = y_base_r - y_base_l
        l = math.hypot(b, dy)
        if l < 1e-9:
            continue
        cos_a = b / l
        sin_a = dy / l
        # weight: trapezoidal integration of the (possibly layered) column,
        # generalizing the old area*unit_weight product to however many
        # material bands the column crosses
        w_l = _weight_per_width(bounds, y_top_l, y_base_l)
        w_r = _weight_per_width(bounds, y_top_r, y_base_r)
        W = b * (w_l + w_r) / 2.0
        h_mid = (h_l + h_r) / 2.0
        # shear-strength parameters and pore pressure: evaluated at the
        # BASE of the slice (the material the slip surface is actually
        # cutting through here), not the column's top material
        base_mat = slope.material_at_elevation((y_base_l + y_base_r) / 2.0)
        u = base_mat.ru * base_mat.unit_weight * h_mid
        slices.append(SliceResult(x_mid=(xl + xr) / 2.0, width=b, weight=W,
                                   sin_alpha=sin_a, cos_alpha=cos_a, u=u, height=h_mid,
                                   cohesion=base_mat.cohesion, phi=base_mat.phi))
    return slices


@dataclass
class FoSResult:
    fos: Optional[float]
    converged: bool
    iterations: int
    method: str
    reason: str = ""


def bishop_fos(slices: List[SliceResult],
               tol: float = 1e-5, max_iter: int = 200, damping: float = 0.6) -> FoSResult:
    """Bishop's Simplified Method (Bishop, 1955, "The Use of the Slip
    Circle in the Stability Analysis of Slopes", Geotechnique 5(1)) --
    iterative, since mα depends on FoS itself.

        FoS = sum( [c_i*b_i + (W_i - u_i*b_i)*tan(phi_i)] / m_alpha_i ) / sum(W_i*sin(alpha_i))
        m_alpha_i = cos(alpha_i) + sin(alpha_i)*tan(phi_i)/FoS

    c_i and phi_i come from each slice's own SliceResult (the base-of-slice
    material -- see build_slices) rather than a single global material, so
    this works unchanged for both homogeneous and layered slopes.

    Fixed-point iteration with damping (new estimate blended with the old
    one) for stability -- undamped Bishop iteration is known to oscillate
    for some geometries; damping trades a few extra iterations for
    reliable convergence, which matters more here since this runs inside
    an outer search loop trying many candidate circles."""
    if not slices:
        return FoSResult(None, False, 0, "bishop", "no slices")
    tan_phis = [math.tan(math.radians(s.phi)) for s in slices]
    denom = sum(s.weight * s.sin_alpha for s in slices)
    if denom <= 1e-9:
        return FoSResult(None, False, 0, "bishop", "non-positive driving moment")

    fos = 1.5
    for it in range(1, max_iter + 1):
        num = 0.0
        for s, tan_phi in zip(slices, tan_phis):
            m_alpha = s.cos_alpha + s.sin_alpha * tan_phi / fos
            if abs(m_alpha) < 1e-4:
                m_alpha = 1e-4 if m_alpha >= 0 else -1e-4
            num += (s.cohesion * s.width + (s.weight - s.u * s.width) * tan_phi) / m_alpha
        new_fos = num / denom
        if new_fos <= 0 or not math.isfinite(new_fos):
            return FoSResult(None, False, it, "bishop", "diverged (non-physical FoS)")
        if abs(new_fos - fos) < tol:
            return FoSResult(new_fos, True, it, "bishop")
        fos = damping * new_fos + (1 - damping) * fos
    return FoSResult(fos, False, max_iter, "bishop", "did not converge within max_iter")


def fellenius_fos(slices: List[SliceResult]) -> FoSResult:
    """Ordinary Method of Slices (Fellenius, 1936) -- non-iterative,
    ignores interslice forces entirely (cruder than Bishop, but exact and
    independent, which is exactly why it's useful as a cross-check here:
    it should come out lower than Bishop's result, more so as pore
    pressure / base-angle steepness increase -- see validate.py). Uses
    each slice's own base-of-slice c/phi, same as bishop_fos."""
    if not slices:
        return FoSResult(None, False, 0, "fellenius", "no slices")
    num = 0.0
    den = 0.0
    for s in slices:
        tan_phi = math.tan(math.radians(s.phi))
        cos_a = s.cos_alpha if abs(s.cos_alpha) > 1e-6 else 1e-6
        l = s.width / cos_a
        num += s.cohesion * l + (s.weight * s.cos_alpha - s.u * l) * tan_phi
        den += s.weight * s.sin_alpha
    if den <= 1e-9:
        return FoSResult(None, False, 0, "fellenius", "non-positive driving moment")
    return FoSResult(num / den, True, 1, "fellenius")
