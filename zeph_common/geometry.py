"""
zeph_common.geometry
=====================
Shared geometry vocabulary for the Zeph toolkit (Zephmatic + Zephslide +
Zephflac).

Bench is the one data structure that turned out, on inspection, to be
genuinely byte-for-byte identical across all three tools' independently
built codebases -- same three fields, same validation rules, same
docstring convention (toe-to-crest bench stack, height/face_angle/
berm_width, berm ignored on the topmost bench). It is consolidated here
as the single canonical definition; each tool's own module re-exports it
(`from zeph_common.geometry import Bench`) so existing import paths
inside this package (`zephslide.geometry.Bench`, `zephflac.mesh.Bench`,
`zephmatic.kinematic_model.Bench`) keep working unchanged.

Material (Zephslide's limit-equilibrium strength model) and
MohrCoulombMaterial (Zephflac's numerical elastic-plastic model) are
DELIBERATELY NOT unified here, even though they share field-naming
conventions (cohesion, phi/phi_deg, depth_to_bottom, name) by design.
They carry genuinely different physics -- Zephslide's Material needs
only cohesion/phi/unit_weight/ru for a slice-based limit-equilibrium
solve; Zephflac's MohrCoulombMaterial additionally needs an elastic
modulus/Poisson's ratio/density and a dilation angle for a stress-strain
continuum solve. Forcing them into one shared dataclass would either
silently drop fields one tool needs or push irrelevant parameters onto
the other -- see zephslide/geometry.py and zephflac/material.py for each
tool's real definition. Zephmatic has no material-strength class at all
(kinematic screening only needs a friction angle, already carried on its
own Slope/DiscontinuitySet).
"""
from dataclasses import dataclass


@dataclass
class Bench:
    """One bench of a stepped quarry/pit wall or embankment: a face of the
    given height, rising at face_angle (degrees from horizontal -- 90 =
    vertical), followed by a flat berm of berm_width (ignored for the
    topmost bench in a wall's bench list, since a toe/crest flat
    extension takes over past either end).

    height: bench height, length units (m).
    face_angle: bench face angle from horizontal, degrees, in (0, 90].
    berm_width: flat berm width above this bench, length units (m),
        >= 0. Ignored for the last (topmost) bench in a list.
    """
    height: float
    face_angle: float
    berm_width: float = 0.0

    def __post_init__(self):
        if self.height <= 0:
            raise ValueError("bench height must be > 0")
        if not (0 < self.face_angle <= 90):
            raise ValueError("face_angle must be in (0, 90] degrees")
        if self.berm_width < 0:
            raise ValueError("berm_width must be >= 0")
