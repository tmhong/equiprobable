#!/usr/bin/env python3
"""
Fig. 7 option: the toy density of Fig. 6 on the actual sphere of decay directions, as relief on a globe.  Around the
spin axis P (here along z) the type-1 density depends only on c = cos(theta_P); the globe's radius is 1 + K f(c),
coloured by f, so the zero of f along P is a crater.  The band y1 < y < y2 is the red ring; its area on the unit
sphere is the solid angle Delta V; the arrows are grad f along the surface (away from P).

Run: .venv/bin/python code/plotting/paper/fig7_globe.py      -> results/plotting/paper/fig7_toy_globe.pdf / .png
"""
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
RED = (0.78, 0.06, 0.18, 1.0)


def f_c(c):
    return (0.75 * F0 * (1 - c ** 2) + 0.375 * FM * (1 - c) ** 2 + 0.375 * FP * (1 + c) ** 2) / (2 * np.pi)


def df_dc(c):
    return (-1.5 * F0 * c - 0.75 * FM * (1 - c) + 0.75 * FP * (1 + c)) / (2 * np.pi)


def main():
    fmax = f_c(brentq(df_dc, -0.99, 0.99))
    K = 0.45 / fmax                                     # largest relief: 0.45 of the radius
    c_peak = brentq(df_dc, -0.99, 0.99)
    th = lambda y: np.arccos(brentq(lambda c: f_c(c) - y, c_peak, 1.0))
    t1, t2 = sorted((th(Y2), th(Y1)))                   # the band, in the angle to P (t1 < t2)
    dV = 2 * np.pi * (np.cos(t1) - np.cos(t2))
    tt = np.unique(np.r_[np.linspace(0, np.pi, 151), np.linspace(t1, t2, 10)])
    T, P = np.meshgrid(tt, np.linspace(0, 2 * np.pi, 181), indexing="ij")
    F = f_c(np.cos(T))
    rho = 1 + K * F
    X, Y, Z = rho * np.sin(T) * np.cos(P), rho * np.sin(T) * np.sin(P), rho * np.cos(T)
    cmap = LinearSegmentedColormap.from_list("tt", ["#ffffff", M.NAVY])
    tc = 0.5 * (T[1:, 1:] + T[:-1, :-1])
    face = cmap(0.08 + 0.85 * f_c(np.cos(tc)) / fmax)
    face[(tc > t1) & (tc < t2)] = RED

    fig = plt.figure(figsize=(5.0, 5.0))
    ax = fig.add_axes([-0.1, -0.1, 1.2, 1.2], projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=face, rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False)
    s = np.linspace(0, 2 * np.pi, 361)
    for t in (t1, t2):                                   # the band's edges
        r = 1 + K * f_c(np.cos(t))
        ax.plot(r * np.sin(t) * np.cos(s), r * np.sin(t) * np.sin(s), r * np.cos(t) + 0 * s, color="black", lw=0.8,
                zorder=10)
    # the spin axis through the crater
    ax.plot([0, 0], [0, 0], [1.0, 1.75], color=M.INK, lw=1.4, zorder=11)
    ax.quiver(0, 0, 1.65, 0, 0, 0.15, color=M.INK, arrow_length_ratio=0.9, lw=1.4)
    ax.text(0.08, 0, 1.8, r"$P$", ha="left", va="center", zorder=12)
    # grad f along the surface, away from P, just beyond the band
    tm = t2 + 0.12
    for a in np.radians([-150, -110, -70, -30, 10]):
        p0 = np.array([np.sin(tm) * np.cos(a), np.sin(tm) * np.sin(a), np.cos(tm)]) * (1 + K * f_c(np.cos(tm)) + 0.01)
        e_t = np.array([np.cos(tm) * np.cos(a), np.cos(tm) * np.sin(a), -np.sin(tm)])     # direction of growing angle
        p1 = p0 + 0.30 * e_t
        ax.plot(*np.c_[p0, p1], color=RED[:3], lw=1.5, zorder=12)
        n_out = p1 / np.linalg.norm(p1)
        side = np.cross(n_out, e_t)
        for sgn in (1, -1):
            b = p1 - 0.09 * e_t + sgn * 0.05 * side
            ax.plot(*np.c_[p1, b], color=RED[:3], lw=1.5, zorder=12)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim * 0.8, lim * 1.2)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    fig.text(0.04, 0.93, rf"$\Delta V = {dV:.2f}$ sr  (red ring, ${Y1:g} < y < {Y2:g}$)", ha="left", va="top",
             color=RED[:3], fontsize=M.FONT)
    fig.text(0.04, 0.86, r"$\nabla f$: arrows", ha="left", va="top", color=RED[:3], fontsize=M.FONT)
    out = repo_paths.results_dir("paper_figs")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig7_toy_globe.{ext}"), dpi=200)
    print(f"Delta V = {dV:.3f}; band at theta_P = {np.degrees(t1):.1f} to {np.degrees(t2):.1f} deg")


if __name__ == "__main__":
    main()
