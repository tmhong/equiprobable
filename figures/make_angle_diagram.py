#!/usr/bin/env python3
"""
Diagram of the top-decay angles (θ, φ, θ', φ') as defined in main.tex, drawn as TikZ.

Top rest frame:  z = top direction in the lab, y ∝ beam × z, x = y × z.
W rest frame:    z' = W direction, y' = z × z' (in the xy plane), x' = y' × z'.

Projection: z to the right, x up, y = z × x out of the page, so the page is the x-z plane,
i.e. the plane spanned by the beam axis and the top momentum.

Run:  python figures/make_angle_diagram.py   ->  figures/angle_diagram.{tex,pdf,png}
"""
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- angles drawn (degrees) and layout -------------------------------------
THETA, PHI = 52.0, 40.0          # W in the top rest frame
THETA_P, PHI_P = 55.0, 30.0      # lepton in the W rest frame
BEAM_ANGLE = 35.0                # beam direction relative to -z, inside the x-z plane
L_AXIS, L_W, L_AXIS_P, L_LEP = 2.6, 3.4, 1.45, 1.6
# screen projection of the unit vectors (x up, y out of the page, z right)
PROJ = {"x": (0.0, 1.0), "y": (-0.55, -0.42), "z": (1.0, 0.0)}

ROYAL, GOLD, RED = "003594", "FFB81C", "C8102E"


def unit_vectors():
    t, p = np.radians(THETA), np.radians(PHI)
    ex, ey, ez = np.eye(3)
    zp = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])       # W direction
    yp = np.array([-np.sin(p), np.cos(p), 0.0])                                   # z × z' / |..|
    xp = np.cross(yp, zp)
    tp, pp = np.radians(THETA_P), np.radians(PHI_P)
    lep = np.sin(tp) * np.cos(pp) * xp + np.sin(tp) * np.sin(pp) * yp + np.cos(tp) * zp
    return ex, ey, ez, xp, yp, zp, lep


def c(v):
    """3D vector -> TikZ coordinate (x, y, z) in the frame's own axes."""
    return f"({v[0]:.4f},{v[1]:.4f},{v[2]:.4f})"


def arc(center, u, v, ang, r, n=30):
    """Points of an arc from unit vector u toward v (orthonormal u, v) through angle `ang`."""
    ts = np.linspace(0, ang, n)
    return " -- ".join(c(center + r * (np.cos(t) * u + np.sin(t) * v)) for t in ts)


def right_angle(center, u, v, s=0.22):
    return f"{c(center + s * u)} -- {c(center + s * u + s * v)} -- {c(center + s * v)}"


def build():
    ex, ey, ez, xp, yp, zp, lep = unit_vectors()
    O = np.zeros(3)
    t, p = np.radians(THETA), np.radians(PHI)
    tp, pp = np.radians(THETA_P), np.radians(PHI_P)
    beam = np.array([-np.sin(np.radians(BEAM_ANGLE)), 0.0, np.cos(np.radians(BEAM_ANGLE))])
    w = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    w_xy = np.array([np.cos(p), np.sin(p), 0.0])
    # in the W rest frame use its own axes as the drawing basis (z' right, x' up, y' out)
    l = np.array([np.sin(tp) * np.cos(pp), np.sin(tp) * np.sin(pp), np.cos(tp)])
    l_xy = np.array([np.cos(pp), np.sin(pp), 0.0])
    lines = []
    add = lines.append
    add(r"\documentclass[border=6pt]{standalone}")
    add(r"\usepackage{tikz,amsmath,bm}")
    add(r"\usetikzlibrary{arrows.meta}")
    add(rf"\definecolor{{royal}}{{HTML}}{{{ROYAL}}}\definecolor{{gold}}{{HTML}}{{{GOLD}}}"
        rf"\definecolor{{lep}}{{HTML}}{{{RED}}}")
    add(r"\begin{document}")
    (xx, xy), (yx, yy), (zx, zy) = PROJ["x"], PROJ["y"], PROJ["z"]
    add(rf"\begin{{tikzpicture}}[x={{({xx}cm,{xy}cm)}}, y={{({yx}cm,{yy}cm)}}, z={{({zx}cm,{zy}cm)}}, "
        r"font=\small, >={Latex[length=2.2mm]}, scale=1.2]")
    add(r"\tikzset{vec/.style={->, line width=1.2pt}, axis/.style={->, line width=0.7pt, black!75},"
        r" arc/.style={->, line width=0.8pt}, ra/.style={line width=0.5pt, black!70}}")

    def frame(origin_shift, xl, yl, zl, xz_label, xz_color):
        """Shaded x-z plane, gold x-y quadrant, axes and right-angle markers of one frame."""
        a = L_AXIS * 0.9
        add(rf"\begin{{scope}}[shift={{{origin_shift}}}]")
        add(rf"\fill[{xz_color}!10] {c(-0.45 * a * ex)} -- {c(a * ex - 0.45 * a * ez)} -- {c(a * ex + a * ez)} -- "
            rf"{c(-0.45 * a * ex + a * ez)} -- cycle;")
        add(rf"\node[{xz_color}!80, font=\footnotesize, anchor=north east, align=right] at {c(a * ex + a * ez)} "
            rf"{{{xz_label}}};")
        add(rf"\fill[gold!25, opacity=0.85] {c(O)} -- {c(1.3 * ex)} -- {c(1.3 * ex + 1.3 * ey)} -- {c(1.3 * ey)} -- cycle;")
        add(rf"\draw[axis] {c(O)} -- {c(L_AXIS * ex)} node[anchor=south] {{{xl}}};")
        add(rf"\draw[axis] {c(O)} -- {c(L_AXIS * 0.9 * ey)} node[anchor=north east] {{{yl}}};")
        add(rf"\draw[axis] {c(O)} -- {c(L_AXIS * ez)} node[anchor=west] {{{zl}}};")
        for u, v in ((ex, ey), (ey, ez), (ez, ex)):
            add(rf"\draw[ra] {right_angle(O, u, v)};")

    def angles(vec, vxy, th, ph, color, tl, pl, label, length):
        add(rf"\draw[dotted, black!55] {c(O)} -- {c(length * np.sin(th) * vxy)} -- {c(length * vec)};")
        add(rf"\draw[vec, {color}] {c(O)} -- {c(length * vec)} node[anchor=south west] {{{label}}};")
        add(rf"\draw[arc, {color}] {arc(O, ex, ey, ph, 0.95)};")
        add(rf"\node[{color}] at {c(1.2 * (np.cos(ph / 2) * ex + np.sin(ph / 2) * ey))} {{{pl}}};")
        add(rf"\draw[arc, {color}] {arc(O, ez, vxy, th, 1.3)};")
        add(rf"\node[{color}] at {c(1.55 * (np.cos(th / 2) * ez + np.sin(th / 2) * vxy))} {{{tl}}};")

    # ---------------- left: top rest frame
    frame("(0,0)", r"$x=y\times z$", r"$y\propto\hat z_\mathrm{beam}\times z$", r"$z=\hat p_t^{\,\mathrm{lab}}$",
          r"$x$--$z$ plane:\\beam and $\hat p_t^{\,\mathrm{lab}}$", "royal")
    add(rf"\draw[dashed, black!55, line width=0.7pt] {c(1.4 * beam)} -- {c(-2.0 * beam)} "
        r"node[anchor=south east, black!65] {beam};")
    add(rf"\draw[vec, black!55] {c(O)} -- {c(-0.9 * w)} node[anchor=north] {{$b$}};")
    angles(w, w_xy, t, p, "royal", r"$\theta$", r"$\phi$", r"$W$", L_W * 0.8)
    add(r"\end{scope}")
    add(r"\node[font=\bfseries] at (3.3,0,0.6) {top rest frame};")

    # ---------------- right: W rest frame
    shift = "(0,0,7.2)"
    frame(shift, r"$x'=y'\times z'$", r"$y'=\frac{z\times z'}{|z\times z'|}$", r"$z'=\hat p_W$",
          r"$x'$--$z'$ plane:\\contains $\hat p_t^{\,\mathrm{lab}}$ ($z$)", "royal")
    angles(l, l_xy, tp, pp, "lep", r"$\theta'$", r"$\phi'$", r"$\ell$", L_LEP * 1.4)
    add(r"\end{scope}")
    add(r"\node[font=\bfseries] at (3.3,0,7.8) {$W$ rest frame};")

    # boost arrow between the panels
    add(r"\draw[->, line width=1pt, royal!70, bend left=20] (1.4,0,3.4) to node[above, font=\footnotesize, align=center]"
        r" {boost along $\hat p_W$\\(standard boost)} (1.4,0,6.4);")

    # ---------------- legend
    add(r"\node[anchor=north west, align=left, font=\footnotesize, text width=15.5cm] at (-2.3,0,-1.8) {"
        r"\textbf{Top rest frame}: $z=\hat p_t^{\,\mathrm{lab}}$, $y\propto\hat z_\mathrm{beam}\times z$ (normal to the"
        r" plane of the beam and the top), $x=y\times z$; $(\theta,\phi)$ is the direction of the $W$, $\phi$ measured"
        r" from $x$ in the $xy$ plane (gold).\\"
        r"\textbf{$W$ rest frame}: $z'=\hat p_W$, $y'=z\times z'/|z\times z'|$ lies in the $xy$ plane, $x'=y'\times z'$;"
        r" $(\theta',\phi')$ is the direction of the charged lepton, i.e.\ $(x',y',z')=R(\phi,\theta,0)\,(x,y,z)$.\\"
        r"Hadronic top: the same construction gives $(\tilde\theta,\tilde\phi,\tilde\theta',\tilde\phi')$, with the"
        r" down-type quark in place of $\ell$.};")
    add(r"\end{tikzpicture}")
    add(r"\end{document}")
    return "\n".join(lines) + "\n"


def main():
    tex = os.path.join(HERE, "angle_diagram.tex")
    with open(tex, "w") as f:
        f.write(build())
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "angle_diagram.tex"],
                         cwd=HERE, stdout=subprocess.PIPE, text=True)
    if res.returncode:
        print("\n".join(res.stdout.splitlines()[-20:]))
        raise SystemExit("pdflatex failed")
    for ext in ("aux", "log"):
        p = os.path.join(HERE, f"angle_diagram.{ext}")
        if os.path.exists(p):
            os.remove(p)
    try:
        import pymupdf
        pymupdf.open(os.path.join(HERE, "angle_diagram.pdf"))[0].get_pixmap(dpi=150).save(
            os.path.join(HERE, "angle_diagram.png"))
    except ImportError:
        pass
    print(f"wrote {tex[:-4]}.pdf")


if __name__ == "__main__":
    main()
