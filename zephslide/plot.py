"""
plot.py

One function, plot_slope(), rendering the ground profile, the critical
slip circle (full circle outline + the used arc highlighted), the slice
boundaries, and a results box -- the single reference figure for a
Zephslide run, matching Zephmatic's own "one plotting function, toggle
arguments" convention.
"""
import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .geometry import Slope
from .search import SearchResult


def plot_slope(slope: Slope, result: Optional[SearchResult], save_path: str,
               show_slices: bool = True, title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 7))

    profile = slope.profile
    toe_x, _ = slope.toe
    crest_x, crest_y = slope.crest
    total_h = slope.total_height

    # trim the flat extensions for display (full extensions are only needed
    # for search headroom, not for a readable plot)
    disp_margin = max(0.6 * total_h, 5.0)
    x_disp_min = toe_x - disp_margin
    x_disp_max = crest_x + disp_margin

    xs = [p[0] for p in profile]
    ys = [p[1] for p in profile]
    ax.plot(xs, ys, color="#3b2f2f", linewidth=2.2, zorder=5, label="Ground surface")
    ax.fill_between(xs, ys, min(ys) - 0.15 * total_h, color="#d9c9a8", alpha=0.5, zorder=1)

    if slope.is_layered:
        disp_margin_for_labels = max(0.6 * total_h, 5.0)
        x_label = toe_x - disp_margin_for_labels + 0.3
        for top, bottom, mat in slope.layer_bounds():
            if bottom == -math.inf:
                continue  # deepest layer has no finite lower boundary to draw
            ax.axhline(y=bottom, color="#5a4632", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1.5)
            label = mat.name if mat.name else f"c={mat.cohesion:g}, phi={mat.phi:g}"
            ax.text(x_label, bottom, f"  {label}", fontsize=7, va="bottom", ha="left",
                    color="#5a4632", alpha=0.85, zorder=1.6)

    if result is not None:
        geom = result.geometry
        xc, yc, R = geom.xc, geom.yc, geom.R

        # full circle (light, for context)
        theta = [i * 2 * math.pi / 400 for i in range(400)]
        cx = [xc + R * math.cos(t) for t in theta]
        cy = [yc + R * math.sin(t) for t in theta]
        ax.plot(cx, cy, color="#888888", linewidth=0.8, linestyle=":", zorder=2, label="Trial circle (full)")

        # used arc (the actual slip surface), highlighted
        arc_x = [s.x_mid for s in geom.slices]
        # recompute arc using slice edges for a smooth curve
        n = 300
        ax_x = [geom.x_left + i * (geom.x_right - geom.x_left) / n for i in range(n + 1)]
        ax_y = []
        for x in ax_x:
            d2 = R * R - (x - xc) ** 2
            ax_y.append(yc - math.sqrt(max(d2, 0.0)))
        ax.plot(ax_x, ax_y, color="#a83232", linewidth=2.4, zorder=6, label="Critical slip surface")

        # shade the sliding mass
        mass_x = ax_x + ax_x[::-1]
        mass_y = ax_y + [slope.ground_height(x) for x in ax_x[::-1]]
        ax.fill(mass_x, mass_y, color="#e8a33d", alpha=0.35, zorder=3, label="Sliding mass")

        ax.plot([xc], [yc], marker="+", color="#a83232", markersize=10, zorder=7)

        if show_slices:
            for s_i in range(len(geom.slices) + 1):
                x_edge = geom.x_left + s_i * (geom.x_right - geom.x_left) / len(geom.slices)
                d2 = R * R - (x_edge - xc) ** 2
                y_base = yc - math.sqrt(max(d2, 0.0))
                y_top = slope.ground_height(x_edge)
                ax.plot([x_edge, x_edge], [y_base, y_top], color="#a83232", linewidth=0.5,
                        alpha=0.5, zorder=4)

        result_txt = (f"FoS (Bishop) = {result.fos:.3f}\n"
                     f"circle: center=({xc:.1f}, {yc:.1f}), R={R:.1f}\n"
                     f"{'converged' if result.fos_result.converged else 'NOT CONVERGED'}"
                     f"{'  (refined)' if result.refined else ''}")
        ax.text(0.02, 0.98, result_txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#a83232"))

    ax.set_xlim(x_disp_min, x_disp_max)
    y_min_disp = min(ys) - 0.1 * total_h
    y_max_disp = max(ys) + (0.6 * total_h if result is None else max(0.3 * total_h, (result.yc - crest_y) * 0.15))
    if result is not None:
        y_max_disp = max(y_max_disp, min(result.yc, crest_y + 1.2 * total_h))
    ax.set_ylim(y_min_disp, y_max_disp)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("elevation")
    ax.set_title(title or f"Zephslide -- {slope.name}")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
