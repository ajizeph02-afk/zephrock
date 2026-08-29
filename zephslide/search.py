"""
search.py

Critical circular slip-surface search: a coarse grid over candidate circle
centers (xc, yc), each paired with a sweep of candidate exit points along
the toe region (which determines the radius R = distance from center to
exit point -- a standard, physically-motivated way to parametrize "how
deep" a trial circle is, used by classic chart-based methods too, rather
than gridding R blind), followed by local derivative-free refinement
(scipy Nelder-Mead) around the best coarse candidate to converge on the
true critical (minimum-FoS) circle.

This is a grid+refine search, not a global optimizer -- for the MVP's
circular-only, single-material scope the FoS surface is typically well-
behaved (one dominant basin), so this converges reliably in practice (see
validate.py), but it is not guaranteed to find a global minimum if the FoS
landscape has multiple comparable local minima. Flagged in the README as a
known limitation, same as Rocscience Slide's own grid-search mode carries
the same caveat.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

from .geometry import Slope
from .lem_core import find_valid_span, build_slices, bishop_fos, fellenius_fos, CircleGeometry, FoSResult

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


PENALTY_FOS = 50.0  # returned by the refine objective for invalid circles


@dataclass
class SearchResult:
    xc: float
    yc: float
    R: float
    x_left: float
    x_right: float
    fos: float
    fos_result: FoSResult
    geometry: CircleGeometry
    n_valid_evaluated: int
    n_total_tried: int
    refined: bool


def _evaluate(slope: Slope, xc: float, yc: float, R: float, n_slices: int, method: str):
    span = find_valid_span(slope, xc, yc, R)
    if span is None:
        return None
    x_left, x_right = span
    slices = build_slices(slope, xc, yc, R, x_left, x_right, n_slices)
    if len(slices) < 3:
        return None
    fos_res = bishop_fos(slices) if method == "bishop" else fellenius_fos(slices)
    if fos_res.fos is None:
        return None
    geom = CircleGeometry(xc, yc, R, x_left, x_right, slices)
    return geom, fos_res


def search_critical_circle(slope: Slope, n_slices: int = 30,
                            nx: int = 10, ny: int = 8, n_exit: int = 15,
                            method: str = "bishop", refine: bool = True,
                            verbose: bool = False) -> Optional[SearchResult]:
    total_h = slope.total_height
    toe_x, toe_y = slope.toe
    crest_x, crest_y = slope.crest

    xc_min, xc_max = toe_x - 0.5 * total_h, crest_x + 2.0 * total_h
    yc_min, yc_max = crest_y + 0.15 * total_h, crest_y + 4.0 * total_h
    exit_min, exit_max = toe_x - 0.5 * total_h, toe_x + 1.0 * total_h

    xcs = [xc_min + i * (xc_max - xc_min) / (nx - 1) for i in range(nx)] if nx > 1 else [0.5 * (xc_min + xc_max)]
    ycs = [yc_min + j * (yc_max - yc_min) / (ny - 1) for j in range(ny)] if ny > 1 else [0.5 * (yc_min + yc_max)]
    exits = [exit_min + k * (exit_max - exit_min) / (n_exit - 1) for k in range(n_exit)] if n_exit > 1 else [toe_x]

    best = None
    n_valid = 0
    n_total = 0
    for xc in xcs:
        for yc in ycs:
            for x_exit in exits:
                n_total += 1
                y_exit = slope.ground_height(x_exit)
                R = math.hypot(xc - x_exit, yc - y_exit)
                if R < 1e-6:
                    continue
                result = _evaluate(slope, xc, yc, R, n_slices, method)
                if result is None:
                    continue
                n_valid += 1
                geom, fos_res = result
                if best is None or fos_res.fos < best[1].fos:
                    best = (geom, fos_res)

    if best is None:
        if verbose:
            print("search_critical_circle: no valid circle found in the coarse grid -- "
                  "widen the grid bounds or check the slope geometry.")
        return None

    geom, fos_res = best
    refined = False

    if refine and _HAVE_SCIPY:
        # Re-parametrize the refine search as (xc, yc, x_exit) -> R, matching
        # the coarse search's own parametrization (keeps refinement in the
        # same, physically-bounded space rather than optimizing R directly).
        x_exit0 = _recover_exit_x(slope, geom)

        def objective(v):
            xc_, yc_, x_exit_ = v
            y_exit_ = slope.ground_height(x_exit_)
            R_ = math.hypot(xc_ - x_exit_, yc_ - y_exit_)
            if R_ < 1e-6:
                return PENALTY_FOS
            r = _evaluate(slope, xc_, yc_, R_, n_slices, method)
            if r is None:
                return PENALTY_FOS
            return r[1].fos

        x0 = [geom.xc, geom.yc, x_exit0]
        try:
            res = _scipy_minimize(objective, x0, method="Nelder-Mead",
                                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300, "maxfev": 400})
            if res.success or res.fun < fos_res.fos:
                xc_r, yc_r, x_exit_r = res.x
                y_exit_r = slope.ground_height(x_exit_r)
                R_r = math.hypot(xc_r - x_exit_r, yc_r - y_exit_r)
                r = _evaluate(slope, xc_r, yc_r, R_r, n_slices, method)
                if r is not None and r[1].fos < fos_res.fos:
                    geom, fos_res = r
                    refined = True
        except Exception:  # pragma: no cover -- refinement is best-effort
            pass
    elif refine and not _HAVE_SCIPY and verbose:  # pragma: no cover
        print("search_critical_circle: scipy not available, skipping refinement.")

    return SearchResult(xc=geom.xc, yc=geom.yc, R=geom.R, x_left=geom.x_left, x_right=geom.x_right,
                         fos=fos_res.fos, fos_result=fos_res, geometry=geom,
                         n_valid_evaluated=n_valid, n_total_tried=n_total, refined=refined)


def _recover_exit_x(slope: Slope, geom: CircleGeometry) -> float:
    """Best-effort recovery of an (xc,yc,x_exit) triple's x_exit from a
    CircleGeometry that only carries (xc,yc,R,x_left,x_right) -- used to
    seed the refine step from the coarse-grid winner. x_left/x_right are on
    the circle by construction, so either works; x_left (toe side, given
    this package's coordinate convention) is used."""
    return geom.x_left
