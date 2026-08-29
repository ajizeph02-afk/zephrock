"""
geometry.py

Parametric benched-slope geometry: turns a list of (height, face_angle,
berm_width) bench specs -- the same vocabulary Zephmatic's quarry faces are
already described in (dip/height per bench) -- into a 2D ground-surface
profile (a monotonic-in-x polyline from a flat toe extension, up through
each bench face/berm, to a flat crest extension), plus a single-material
strength model (MVP scope: one homogeneous material, no layering yet).

Coordinate convention (fixed for this whole package): x increases to the
right, y increases upward. The toe sits at the LOW end (small x, small y);
the crest sits at the HIGH end (large x, large y) -- the slope rises
left-to-right. This is an arbitrary but fully self-consistent choice; every
other module in this package (lem_core.py, search.py) assumes it.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from zeph_common.geometry import Bench

__all__ = ["Material", "Bench", "Slope", "simple_slope"]


@dataclass
class Material:
    """A single material layer. All strength/unit-weight values must be in
    mutually consistent units (e.g. kN/m^3 for unit_weight with kPa for
    cohesion, or MN/m^3 with MPa -- same convention Zephmatic's planar_fos_full
    uses).

    ru: uniform pore-pressure ratio (Bishop 1955's own simplification --
    u = ru * unit_weight * (vertical height of soil above that point) at
    every point on the slip surface). 0.0 = fully dry. This is the same
    simplified water treatment Zephmatic's planar_fos_full() uses for its
    water_pressure_ratio parameter -- deliberately consistent with that
    existing convention rather than introducing a second one. For a layered
    slope, ru is evaluated using whichever layer the slip surface's base
    sits in at that point (see lem_core.build_slices).

    depth_to_bottom: ONLY used for a layered profile (Slope(materials=[...]));
    leave as None for the single-homogeneous-material case
    (Slope(material=...)). Measured vertically DOWN FROM THE SLOPE'S CREST
    ELEVATION (Slope.total_height) to this layer's lower boundary -- i.e.
    layers are horizontal (constant-elevation) bands, the natural convention
    for near-horizontal sedimentary/carbonate strata (this thesis's own
    rock type) and, not coincidentally, the same convention PySlope uses
    internally (`Material.RL = external_height - depth_to_bottom`, and
    `external_height` there is always exactly the crest's own elevation) --
    which is why a PySlope worked example checked against real Rocscience
    Slide output can be reproduced under this same convention with no unit
    or reference-point conversion (see external_validation.py). Materials
    passed to Slope(materials=...) must each have a distinct depth_to_bottom;
    the deepest one's layer extends infinitely below (acts as a basement/
    bedrock unit for slip surfaces that dip below every stated boundary).

    name: optional label (e.g. "weathered overburden", "competent
    limestone") -- purely cosmetic, used in plots/reports for a layered
    profile so results are traceable back to a named unit.
    """
    cohesion: float
    phi: float           # degrees
    unit_weight: float
    ru: float = 0.0
    depth_to_bottom: Optional[float] = None
    name: str = ""

    def __post_init__(self):
        if self.cohesion < 0:
            raise ValueError("cohesion must be >= 0")
        if not (0 <= self.phi < 90):
            raise ValueError("phi must be in [0, 90) degrees")
        if self.unit_weight <= 0:
            raise ValueError("unit_weight must be > 0")
        if not (0 <= self.ru < 1):
            raise ValueError("ru must be in [0, 1)")
        if self.depth_to_bottom is not None and self.depth_to_bottom <= 0:
            raise ValueError("depth_to_bottom must be > 0 if given")


# Bench used to be defined here directly; it is now the single canonical
# definition in zeph_common.geometry (see that module's docstring) and is
# just re-exported under this name so `from zephslide.geometry import
# Bench` and `from zephslide import Bench` keep working unchanged.


@dataclass
class Slope:
    """A parametric benched slope: one or more Bench entries stacked from
    toe (first entry) to crest (last entry), a material model, and flat
    ground extensions on either side (auto-sized to 2x total height if not
    given -- generous enough that the circular-search grid in search.py has
    room to place both entry and exit points on flat ground when that's
    where the critical circle wants them, without the profile's own edges
    getting in the way).

    Material model -- pass exactly one of:
      material=Material(...)             a single homogeneous material
                                          (depth_to_bottom left as None)
      materials=[Material(...), ...]     a layered profile: each Material
                                          MUST have a distinct depth_to_bottom
                                          (see Material's own docstring for
                                          the horizontal-layer convention).
    """
    benches: List[Bench]
    material: Optional[Material] = None
    materials: Optional[List[Material]] = None
    toe_extension: Optional[float] = None
    crest_extension: Optional[float] = None
    name: str = "Slope"

    def __post_init__(self):
        if not self.benches:
            raise ValueError("Slope needs at least one Bench")
        if self.material is not None and self.materials is not None:
            raise ValueError("pass either material= (single) or materials= (layered), not both")
        if self.material is None and self.materials is None:
            raise ValueError("Slope needs a material= (single homogeneous) or materials= (layered list)")
        if self.material is not None:
            self._layers = [self.material]
        else:
            layers = list(self.materials)
            if not layers:
                raise ValueError("materials= list must not be empty")
            for m in layers:
                if m.depth_to_bottom is None:
                    raise ValueError(
                        "every Material in a layered profile (materials=) needs depth_to_bottom set")
            depths = [m.depth_to_bottom for m in layers]
            if len(depths) != len(set(depths)):
                raise ValueError("materials= entries must have unique depth_to_bottom values")
            self._layers = sorted(layers, key=lambda m: m.depth_to_bottom)
        total_h = sum(b.height for b in self.benches)
        if self.toe_extension is None:
            self.toe_extension = 2.0 * total_h
        if self.crest_extension is None:
            self.crest_extension = 2.0 * total_h
        self._profile = self._build_profile()

    # -- geometry construction -------------------------------------------
    def _build_profile(self) -> List[Tuple[float, float]]:
        pts = [(-self.toe_extension, 0.0), (0.0, 0.0)]  # toe at (0, 0)
        x, y = 0.0, 0.0
        for i, b in enumerate(self.benches):
            dx = b.height / math.tan(math.radians(b.face_angle))
            x += dx
            y += b.height
            pts.append((x, y))
            is_last = (i == len(self.benches) - 1)
            if b.berm_width > 0 and not is_last:
                x += b.berm_width
                pts.append((x, y))
        pts.append((x + self.crest_extension, y))
        return pts

    @property
    def profile(self) -> List[Tuple[float, float]]:
        return self._profile

    @property
    def layers(self) -> List[Material]:
        """Materials sorted shallow -> deep. Length 1 for a homogeneous
        slope (the material=... case)."""
        return self._layers

    @property
    def is_layered(self) -> bool:
        return len(self._layers) > 1

    def layer_bounds(self) -> List[Tuple[float, float, "Material"]]:
        """(top_elev, bottom_elev, Material) bands, shallow -> deep. The
        shallowest band's top is +inf (it clips against whatever the actual
        ground surface is at a given x automatically -- no need to track a
        separate "top of model" reference); the deepest band's bottom is
        -inf (an effectively infinite basement/bedrock unit)."""
        layers = self._layers
        if len(layers) == 1:
            return [(math.inf, -math.inf, layers[0])]
        crest_y = self.total_height
        bottoms = [crest_y - m.depth_to_bottom for m in layers]
        bottoms[-1] = -math.inf
        tops = [math.inf] + bottoms[:-1]
        return list(zip(tops, bottoms, layers))

    def material_at_elevation(self, y: float) -> Material:
        """The Material occupying absolute elevation y -- see Material's
        own docstring for the horizontal-layer (crest-referenced
        depth_to_bottom) convention. Always returns the single material for
        a homogeneous slope."""
        for top, bottom, m in self.layer_bounds():
            if y >= bottom:
                return m
        return self._layers[-1]

    @property
    def toe(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    @property
    def crest(self) -> Tuple[float, float]:
        total_h = sum(b.height for b in self.benches)
        total_run = sum(b.height / math.tan(math.radians(b.face_angle)) for b in self.benches) \
            + sum(b.berm_width for b in self.benches[:-1])
        return (total_run, total_h)

    @property
    def total_height(self) -> float:
        return sum(b.height for b in self.benches)

    def ground_height(self, x: float) -> float:
        """Interpolate the ground surface's y at a given x. Flat beyond
        either end of the profile (matches the toe/crest extensions)."""
        pts = self._profile
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return max(y0, y1)
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        raise RuntimeError("x outside profile domain unexpectedly")  # pragma: no cover

    def x_domain(self) -> Tuple[float, float]:
        return self._profile[0][0], self._profile[-1][0]


def simple_slope(height: float, face_angle: float, material: Optional[Material] = None,
                  materials: Optional[List[Material]] = None,
                  toe_extension: Optional[float] = None,
                  crest_extension: Optional[float] = None,
                  name: str = "Simple slope") -> Slope:
    """Convenience constructor for the single-bench (classic textbook
    "simple slope") special case -- one face, no berm. Pass either
    material= (homogeneous) or materials= (layered) -- see Slope's own
    docstring."""
    return Slope(benches=[Bench(height=height, face_angle=face_angle, berm_width=0.0)],
                 material=material, materials=materials, toe_extension=toe_extension,
                 crest_extension=crest_extension, name=name)
