"""
model.py

Thin convenience wrapper tying geometry.Slope + search.search_critical_circle
together into one object with a simple .run() call and a printable summary
-- the same "small object model, one entry point" shape as Zephmatic's own
KinematicModel, deliberately, since this package's author (you) already
knows that pattern.
"""
from dataclasses import dataclass
from typing import Optional

from .geometry import Slope
from .search import search_critical_circle, SearchResult
from .lem_core import fellenius_fos, bishop_fos


@dataclass
class SlopeStabilityModel:
    slope: Slope
    n_slices: int = 30

    def run(self, method: str = "bishop", refine: bool = True,
            nx: int = 10, ny: int = 8, n_exit: int = 15,
            verbose: bool = False) -> Optional[SearchResult]:
        """Search for the critical circular slip surface and its FoS.
        method: 'bishop' (default, MVP's primary method) or 'fellenius'
        (Ordinary Method -- provided mainly for the cross-check in
        validate.py, but usable directly too)."""
        return search_critical_circle(self.slope, n_slices=self.n_slices, nx=nx, ny=ny,
                                       n_exit=n_exit, method=method, refine=refine, verbose=verbose)

    def cross_check(self, result: SearchResult) -> float:
        """Given a Bishop search result, recompute Fellenius/Ordinary FoS
        on that SAME critical circle's slices (not a separate search) --
        the standard sanity check: Fellenius should come out lower than
        Bishop (usually by a few percent up to ~20% for high pore
        pressure / steep base angles), never dramatically higher. See
        validate.py for the automated version of this check across
        several geometries."""
        fell = fellenius_fos(result.geometry.slices)
        return fell.fos

    def text_summary(self, result: Optional[SearchResult]) -> str:
        if result is None:
            return f"=== {self.slope.name}: no valid critical circle found ==="
        lines = [f"=== {self.slope.name}: critical circle search ==="]
        if self.slope.is_layered:
            layer_desc = ", ".join(
                f"{m.name or ('layer ' + str(i+1))} (c={m.cohesion:g}, phi={m.phi:g}, "
                f"gamma={m.unit_weight:g}, to depth {m.depth_to_bottom:g})"
                for i, m in enumerate(self.slope.layers))
            lines.append(f"  Materials ({len(self.slope.layers)} layers, shallow->deep): {layer_desc}")
        lines.append(f"  Method: {result.fos_result.method}  "
                     f"({'converged' if result.fos_result.converged else 'DID NOT CONVERGE'}, "
                     f"{result.fos_result.iterations} iteration(s))")
        lines.append(f"  Critical circle: center=({result.xc:.2f}, {result.yc:.2f}), R={result.R:.2f}")
        lines.append(f"  Slip surface span: x = {result.x_left:.2f} to {result.x_right:.2f}")
        lines.append(f"  Factor of Safety = {result.fos:.3f}"
                     f"{'  (refined)' if result.refined else ''}")
        fell = self.cross_check(result)
        if fell is not None:
            lines.append(f"  Cross-check (Ordinary/Fellenius, same circle): FoS = {fell:.3f}"
                         f"  (Bishop {'higher' if result.fos >= fell else 'LOWER -- check inputs'}"
                         f" by {abs(result.fos - fell):.3f})")
        lines.append(f"  Grid search: {result.n_valid_evaluated}/{result.n_total_tried} candidate circles valid")
        return "\n".join(lines)
