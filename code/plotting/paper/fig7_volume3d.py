#!/usr/bin/env python3
"""
Fig. 7 of the letter: the toy of Fig. 6 in coordinates aligned with the spin axis.  The type-1 density depends only
on c = cos(theta_P), the cosine of the angle to the spin axis, so the azimuth collapses: every equiprobable contour
is a circle c = const (a line of length 2 pi in the (c, phi_P) plane), the band y1 < y < y2 is an interval Delta c,
Delta V = 2 pi Delta c, and |grad f| = |df/dc|, so Delta V ~ 2 pi Delta y / |df/dc| (Eq. 5) can be read off.

The toy is compare_y.py's: spin 1 -> 1/2 + 1/2 with helicity fractions (F_0, F_-, F_+) = (0.7, 0.3, 0); type 0 is
isotropic, 1/4pi.  A rotation keeps solid angles, so V(y), Delta V and dN/dy are those of Fig. 6.

Run: .venv/bin/python code/plotting/paper/fig7_volume3d.py      -> results/paper_figs/fig7_toy_collapsed.pdf / .png
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, CODE)
from plotting.paper import make_paper_figs as M  # noqa: E402  (fonts, colours)
from plotting.paper.make_paper_figs import plt  # noqa: E402
import repo_paths  # noqa: E402

F0, FM, FP = 0.7, 0.3, 0.0
Y1, Y2 = 0.04, 0.06
RED = M.RATIO_COLOR


def f(c):
    return (0.75 * F0 * (1 - c ** 2) + 0.375 * FM * (1 - c) ** 2 + 0.375 * FP * (1 + c) ** 2) / (2 * np.pi)


def df(c):
    return (-1.5 * F0 * c - 0.75 * FM * (1 - c) + 0.75 * FP * (1 + c)) / (2 * np.pi)


def main():
    c_peak = brentq(df, -0.99, 0.99)
    root = lambda y: brentq(lambda c: f(c) - y, c_peak, 1.0)        # the falling branch (y < f(-1))
    c1, c2, cb = root(Y2), root(Y1), root(0.5 * (Y1 + Y2))
    yb, dc = 0.5 * (Y1 + Y2), root(Y1) - root(Y2)
    dV, slope = 2 * np.pi * dc, df(cb)
    nums = {"c_band": [c1, c2], "delta_c": dc, "delta_V": dV, "s": [-np.log(Y1), -np.log(Y2)],
            "delta_s": np.log(Y2 / Y1), "ybar": yb, "c_at_ybar": cb, "inv_grad": 1 / abs(slope), "path": 2 * np.pi,
            "volume_side": yb * dV / (Y2 - Y1), "surface_side": yb * 2 * np.pi / abs(slope)}

    P, T, B, R, TOP = 2.6, 0.80, 0.62, 0.15, 0.12
    W, H = T + P + R, B + P + TOP
    fig = plt.figure(figsize=(W, H))
    ax = M.place(fig, T, B, P, P, W, H)
    c = np.linspace(-1, 1, 801)
    ax.axhspan(Y1, Y2, color=RED, alpha=0.12, lw=0)
    # the area under f shaded white-to-navy in y, as the density panels of Figs. 2-4
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("tt", ["#ffffff", M.NAVY])
    yc = f(c)
    for k in range(len(c) - 1):
        ax.fill_between(c[k:k + 2], 0, yc[k:k + 2], color=cmap(0.1 + 0.6 * yc[k] / yc.max()), lw=0, zorder=1)
    ax.plot(c, yc, color=M.NAVY, lw=1.6, zorder=3)
    seg = np.linspace(c1, c2, 50)
    ax.plot(seg, f(seg), color=RED, lw=3.0, zorder=4, solid_capstyle="butt")
    for cc, yy in ((c1, Y2), (c2, Y1)):
        ax.plot([cc, cc], [0, yy], color=M.INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
    ax.fill_between([c1, c2], 0, 0.006, facecolor="none", edgecolor=M.INK, hatch="////", lw=0.6, zorder=3)
    ax.annotate(rf"$\Delta V = 2\pi\,\Delta c = {dV:.2f}$", xy=(0.5 * (c1 + c2), 0.004), xytext=(0.30, 0.010),
                ha="right", va="center", arrowprops=dict(arrowstyle="-", color=M.INK, lw=0.6))
    # the tangent at ybar: |grad f| = |df/dc|
    tc = np.array([cb - 0.22, cb + 0.22])
    ax.plot(tc, yb + slope * (tc - cb), color=M.INK, lw=0.9, zorder=5)
    ax.plot([cb], [yb], "o", color=M.INK, ms=3, zorder=6)
    ax.annotate(r"$|\nabla f| = \left|\frac{\mathrm{d}f}{\mathrm{d}c}\right|$", xy=(cb - 0.16, yb - 0.16 * slope),
                xytext=(0.22, 0.027), ha="right", va="center", arrowprops=dict(arrowstyle="-", color=M.INK, lw=0.6))
    # Delta y on the right
    xr = -0.94
    ax.annotate("", xy=(xr, Y1), xytext=(xr, Y2), arrowprops=dict(arrowstyle="<->", color=RED, lw=0.9,
                                                                   shrinkA=0, shrinkB=0))
    ax.text(xr + 0.05, 0.5 * (Y1 + Y2), rf"$\Delta y = {Y2 - Y1:.2f}$", ha="left", va="center", color=RED)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 0.115)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_xlabel(r"$c = \cos\theta_P$")
    ax.set_ylabel(r"$y$", rotation=0, ha="right", va="center", labelpad=6)
    M.finish(ax, no_values_y=False)
    out = repo_paths.results_dir("paper_figs")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig7_toy_collapsed.{ext}"), dpi=200)
    json.dump(nums, open(os.path.join(out, "fig7_toy_collapsed.json"), "w"), indent=1)
    print(json.dumps(nums, indent=1), f"\nwidth {W:.2f} in")


if __name__ == "__main__":
    main()
