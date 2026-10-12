#!/usr/bin/env python3
"""
Fig. 7 of the letter: the toy density of Fig. 6, y = f_1(x, phi) with x = cos theta, as a surface over the
(x, phi) plane.  The band y1 < y < y2 of Fig. 6 is the red part of the surface; its projection onto the plane is the
hatched ring of area Delta V; the arrows are grad f on that ring (uphill, normal to the contours).

The toy is compare_y.py's: a spin-1 particle decaying to two spin-1/2 ones with helicity fractions
(F_0, F_-, F_+) = (0.7, 0.3, 0) along the spin axis P at (theta, phi) = (120, 180) degrees.

Run: .venv/bin/python paper_figs/fig7_volume3d.py      -> paper_figs/fig7_volume3d.pdf / .png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_paper_figs as M  # noqa: E402  (fonts, colours)
from make_paper_figs import plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

F0, FM, FP = 0.7, 0.3, 0.0
POL = np.radians([120.0, 180.0])
Y1, Y2 = 0.05, 0.07
RED = "#c8102e"


def f(x, ph):
    st = np.sqrt(np.clip(1 - x ** 2, 0, None))
    tp, pp = POL
    c = np.sin(tp) * np.cos(pp) * st * np.cos(ph) + np.sin(tp) * np.sin(pp) * st * np.sin(ph) + np.cos(tp) * x
    return (0.75 * F0 * (1 - c ** 2) + 0.375 * FM * (1 - c) ** 2 + 0.375 * FP * (1 + c) ** 2) / (2 * np.pi)


def main():
    n = 181
    x = np.linspace(-1, 1, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n - 1)
    X, P = np.meshgrid(x, ph, indexing="ij")
    Z = f(X, P)
    # Delta V on a fine grid
    xf = np.linspace(-1, 1, 2001); pf = np.linspace(0, 2 * np.pi, 4001)
    Zf = f(*np.meshgrid(xf, pf, indexing="ij"))
    dA = (xf[1] - xf[0]) * (pf[1] - pf[0])
    dV = ((Zf > Y1) & (Zf < Y2)).sum() * dA

    fig = plt.figure(figsize=(5.6, 5.0))
    ax = fig.add_axes([0.0, 0.03, 0.9, 0.95], projection="3d")
    cmap = LinearSegmentedColormap.from_list("tt", ["#ffffff", M.NAVY])
    # the surface, with the band cut out, and the band itself on a finer grid so its edges follow the contours
    Zs = np.where((Z > Y1) & (Z < Y2), np.nan, Z)
    ax.plot_surface(X, P, Zs, facecolors=cmap(0.12 + 0.8 * Z / Z.max()), rstride=1, cstride=2, linewidth=0,
                    shade=False, antialiased=False, alpha=0.5)
    xb_, pb_ = np.linspace(-1, 1, 401), np.linspace(0, 2 * np.pi, 801)
    XB, PB = np.meshgrid(xb_, pb_, indexing="ij")
    ZB = f(XB, PB)
    ZB = np.where((ZB > Y1) & (ZB < Y2), ZB, np.nan)
    ax.plot_surface(XB, PB, ZB, color=RED, rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False)
    band = (Z > Y1) & (Z < Y2)
    # the band's edges on the surface and on the floor
    tmp = plt.figure().add_subplot()
    for lev in (Y1, Y2):
        for seg in tmp.contour(XB, PB, f(XB, PB), levels=[lev]).allsegs[0]:
            zz = np.full(len(seg), lev)
            ax.plot(seg[:, 0], seg[:, 1], zz, color="black", lw=1.0, zorder=5)
            ax.plot(seg[:, 0], seg[:, 1], 0 * zz, color="black", lw=0.9, zorder=4)
    mid = max(tmp.contour(XB, PB, f(XB, PB), levels=[0.5 * (Y1 + Y2)]).allsegs[0], key=len)
    plt.close(tmp.figure)
    # hatched ring on the floor
    ring = np.where(band, 1.0, np.nan)
    ax.contourf(X, P, ring, levels=[0.5, 1.5], zdir="z", offset=0, colors=[(0.55, 0.55, 0.55, 0.3)])
    ax.contourf(X, P, ring, levels=[0.5, 1.5], zdir="z", offset=0, colors="none", hatches=["////"])
    # grad f on the ring (uphill), drawn as flat arrows on the floor
    gx, gp = np.gradient(Z, x, ph)
    for x0, p0 in mid[np.linspace(0, len(mid) - 1, 10, endpoint=False).astype(int)]:
        i, j = np.argmin(abs(x - x0)), np.argmin(abs(ph - p0))
        g = np.array([gx[i, j], gp[i, j]])
        d = g / np.hypot(g[0] / 0.25, g[1] / 0.8)              # a fixed length on the page, in (x, phi) units
        tip = np.array([x0, p0]) + d
        ax.plot([x0, tip[0]], [p0, tip[1]], [0, 0], color=RED, lw=1.4, zorder=6)
        for sgn in (1, -1):                                     # arrow head: two short barbs
            ang = np.arctan2(d[1] / 0.8, d[0] / 0.25) + np.pi + sgn * 0.45
            hb = tip + np.array([0.08 * np.cos(ang), 0.25 * np.sin(ang)])
            ax.plot([tip[0], hb[0]], [tip[1], hb[1]], [0, 0], color=RED, lw=1.4, zorder=6)
    ax.text(-0.5, np.pi, 0.003, r"$\nabla f$", color=RED, ha="center", va="center", fontsize=M.FONT, zorder=10)
    # Delta y bracket next to the y axis, Delta V label beside the ring
    xb, pb = 1.0, 2 * np.pi
    ax.plot([xb, xb], [pb, pb], [Y1, Y2], color="black", lw=1.4)
    for yy in (Y1, Y2):
        ax.plot([xb, xb], [pb - 0.15, pb], [yy, yy], color="black", lw=1.4)
    ax.text(xb, pb - 0.3, 0.5 * (Y1 + Y2), rf"$\Delta y = {Y2 - Y1:.2f}$", ha="right", va="center")
    k = np.nanargmax(np.where(band, X + 0.2 * np.cos(P - np.pi), -np.inf))
    ax.text(X.flat[k] + 0.12, P.flat[k] - 0.6, 0.0, rf"$\Delta V = {dV:.2f}$", ha="left", va="center", zorder=10)

    ax.set_xlim(-1, 1); ax.set_ylim(0, 2 * np.pi); ax.set_zlim(0, 0.11)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([0, np.pi, 2 * np.pi]); ax.set_zticks([0, 0.05, 0.1])
    ax.set_yticklabels(["$0$", r"$\pi$", r"$2\pi$"]); ax.set_xticklabels(["$-1$", "$0$", "$1$"])
    ax.set_zticklabels(["$0$", "$0.05$", "$0.1$"])
    ax.set_xlabel(r"$x = \cos\theta$", labelpad=4)
    ax.set_ylabel(r"$\phi$", labelpad=4)
    ax.set_zlabel(r"$y = f_1(x,\phi)$", labelpad=12)
    ax.view_init(elev=28, azim=-58)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig7_volume3d.{ext}"), dpi=200)
    print(f"Delta V = {dV:.3f} for {Y1} < y < {Y2}")


if __name__ == "__main__":
    main()
