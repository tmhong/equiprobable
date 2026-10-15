#!/usr/bin/env python3
"""
Fig. 7 of the letter: the toy of Fig. 6 around its spin axis.  The type-1 density depends only on c = cos(theta_P),
the cosine of the angle to the spin axis, so it is drawn as a surface of revolution over the equal-area disk
r = sqrt(2 (1 - c)) (Lambert projection: r dr dphi = dc dphi = dOmega, so areas on the floor are solid angles).
The band y1 < y < y2 of Fig. 6 is the red ring on the surface; its projection is the hatched ring of area Delta V;
the arrows are grad f (radial).  Every equiprobable contour is a circle.

Run: .venv/bin/python code/plotting/paper/fig7_crater.py      -> results/plotting/paper/fig7_toy_crater.pdf / .png (+ .json numbers)
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, CODE)
from plotting.paper import make_paper_figs as M  # noqa: E402
from plotting.paper.make_paper_figs import plt  # noqa: E402
import repo_paths  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

F0, FM, FP = 0.7, 0.3, 0.0
Y1, Y2 = 0.04, 0.06
RED = "#c8102e"


def f_c(c):
    return (0.75 * F0 * (1 - c ** 2) + 0.375 * FM * (1 - c) ** 2 + 0.375 * FP * (1 + c) ** 2) / (2 * np.pi)


def df_dc(c):
    return (-1.5 * F0 * c - 0.75 * FM * (1 - c) + 0.75 * FP * (1 + c)) / (2 * np.pi)


def f_r(r):
    return f_c(1 - r ** 2 / 2)


def main():
    c_peak = brentq(df_dc, -0.99, 0.99)
    r_of = lambda y: np.sqrt(2 * (1 - brentq(lambda c: f_c(c) - y, c_peak, 1.0)))   # inner wall
    r1, r2, rb = r_of(Y1), r_of(Y2), r_of(0.5 * (Y1 + Y2))
    yb = 0.5 * (Y1 + Y2)
    dV = np.pi * (r2 ** 2 - r1 ** 2)
    grad = abs(df_dc(1 - rb ** 2 / 2)) * rb                  # |df/dr| = |df/dc| r
    nums = {"r_band": [r1, r2], "delta_V": dV, "ybar": yb, "r_bar": rb, "path": 2 * np.pi * rb,
            "inv_grad": 1 / grad, "volume_side": yb * dV / (Y2 - Y1), "surface_side": yb * 2 * np.pi * rb / grad}

    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_axes([-0.06, -0.04, 1.08, 1.1], projection="3d")
    cmap = LinearSegmentedColormap.from_list("tt", ["#ffffff", M.NAVY])
    # one surface (so matplotlib sorts it as a whole), on radial rings that include the band's edges r1 and r2
    rr = np.unique(np.r_[np.linspace(0, 2, 121), np.linspace(r1, r2, 12)])
    R, PH = np.meshgrid(rr, np.linspace(0, 2 * np.pi, 181), indexing="ij")
    X, Y, Z = R * np.cos(PH), R * np.sin(PH), f_r(R)
    rc = 0.5 * (R[1:, 1:] + R[:-1, :-1])                    # each facet's radius
    face = cmap(0.12 + 0.8 * f_r(rc) / Z.max())
    face[..., 3] = 0.62
    inband = (rc > r1) & (rc < r2)
    face[inband] = (0.78, 0.06, 0.18, 1.0)
    ax.plot_surface(X, Y, Z, facecolors=face, rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False)
    t = np.linspace(0, 2 * np.pi, 361)
    for rr, yy in ((r1, Y1), (r2, Y2)):
        ax.plot(rr * np.cos(t), rr * np.sin(t), yy, color="black", lw=0.9, zorder=6)
        ax.plot(rr * np.cos(t), rr * np.sin(t), 0 * t, color="black", lw=0.9, zorder=4)
    # hatched ring on the floor
    ring = np.where((R >= r1) & (R <= r2), 1.0, np.nan)
    ax.contourf(X, Y, ring, levels=[0.5, 1.5], zdir="z", offset=0, colors=[(0.55, 0.55, 0.55, 0.3)])
    ax.contourf(X, Y, ring, levels=[0.5, 1.5], zdir="z", offset=0, colors="none", hatches=["////"])
    # grad f: radial, outward on the inner wall, drawn flat on the floor just outside the ring
    for a in np.radians([195, 235, 275, 315, 355, 30, 160]):        # the front half (the back shows through)
        p0, p1 = (r2 + 0.06) * np.array([np.cos(a), np.sin(a)]), (r2 + 0.42) * np.array([np.cos(a), np.sin(a)])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [0, 0], color=RED, lw=1.4, zorder=7)
        for s in (1, -1):
            b = p1 - 0.13 * np.array([np.cos(a + s * 0.45), np.sin(a + s * 0.45)])
            ax.plot([p1[0], b[0]], [p1[1], b[1]], [0, 0], color=RED, lw=1.4, zorder=7)
    ax.text(1.55 * np.cos(4.9), 1.55 * np.sin(4.9), 0.0, r"$\nabla f$", color=RED, ha="center", va="center", zorder=9)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(0, 0.11)
    ax.set_box_aspect((1, 1, 0.62))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zticks([0, 0.05, 0.1]); ax.set_zticklabels(["$0$", "$0.05$", "$0.1$"])
    ax.set_zlabel(r"$y = f_1$", labelpad=6)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.set_facecolor((1, 1, 1, 0))
        a_.pane.set_edgecolor((0.85, 0.85, 0.85, 1))
        a_._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    ax.view_init(elev=52, azim=-62)
    # labels in the figure plane, with leaders to points of the 3-D scene
    from mpl_toolkits.mplot3d import proj3d
    fig.canvas.draw()
    def at(x, y, z):
        u, v, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        return u, v
    rm = 0.5 * (r1 + r2)
    ax.annotate(rf"$\Delta V = {dV:.2f}$", xy=at(rm * np.cos(np.radians(300)), rm * np.sin(np.radians(300)), 0),
                xytext=(0.80, 0.10), textcoords="axes fraction", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
    ax.annotate(rf"${Y1:g} < y < {Y2:g}$", xy=at(rm * np.cos(np.radians(140)), rm * np.sin(np.radians(140)), f_r(rm)),
                xytext=(0.04, 0.86), textcoords="axes fraction", ha="left", va="center", color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    out = repo_paths.results_dir("paper_figs")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig7_toy_crater.{ext}"), dpi=200)
    json.dump(nums, open(os.path.join(out, "fig7_toy_crater.json"), "w"), indent=1)
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
