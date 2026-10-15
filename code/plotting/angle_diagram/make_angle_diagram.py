#!/usr/bin/env python3
"""
Diagram of the top-decay angles (θ, φ, θ', φ') as defined in main.tex, drawn as TikZ.

Top rest frame:  z = top direction in the lab, y ∝ beam × z, x = y × z.
W rest frame:    z' = W direction, y' = z × z' (in the xy plane), x' = y' × z'.

Projection: z to the right, x up, y = z × x out of the page, so the page is the x-z plane,
i.e. the plane spanned by the beam axis and the top momentum.

Run:  python code/plotting/angle_diagram/make_angle_diagram.py   ->  code/plotting/angle_diagram/angle_diagram.tex,
                                                                    results/figures/angle_diagram.{pdf,png}
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))     # code/plotting/angle_diagram: the generated .tex (kept in git)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # code/
import repo_paths  # noqa: E402

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


def build(legend=True):
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
        + (r"font=\small" if legend else r"font=\large") + r", >={Latex[length=2.2mm]}, scale=1.2]")
    add(r"\tikzset{vec/.style={->, line width=1.2pt}, axis/.style={->, line width=0.7pt, black!75},"
        r" arc/.style={->, line width=0.8pt}, ra/.style={line width=0.5pt, black!70}}")

    def frame(origin_shift, xl, yl, zl, edge, edge_label, xz_color="royal"):
        """Shaded x-z plane anchored on a reference line (beam / top direction), gold xy quarter-disc,
        axes and right-angle markers.  No text is placed inside or across the shaded areas."""
        lo, hi, a = -1.1, 2.3, 2.35           # band: along the edge line [lo, hi], height a along +x
        add(rf"\begin{{scope}}[shift={{{origin_shift}}}]")
        add(rf"\fill[{xz_color}!10] {c(lo * edge)} -- {c(hi * edge)} -- {c(hi * edge + a * ex)} -- "
            rf"{c(lo * edge + a * ex)} -- cycle;")
        add(rf"\draw[dashed, black!60, line width=0.8pt] {c(lo * edge)} -- {c(hi * edge)} "
            rf"node[anchor=north west, black!70, font=\footnotesize] {{{edge_label}}};")
        # xy plane (perpendicular to z): gold quarter disc from x toward y
        R = 2.05
        disc = " -- ".join(c(R * (np.cos(u) * ex + np.sin(u) * ey)) for u in np.linspace(0, np.pi / 2, 30))
        add(rf"\fill[gold!30, opacity=0.9] {c(O)} -- {disc} -- cycle;")
        add(rf"\draw[axis] {c(O)} -- {c(L_AXIS * ex)} node[anchor=south] {{{xl}}};")
        add(rf"\draw[axis] {c(O)} -- {c(L_AXIS * 0.95 * ey)} node[anchor=north east] {{{yl}}};")
        add(rf"\draw[axis] {c(O)} -- {c(L_AXIS * ez)} node[anchor=west] {{{zl}}};")
        for u, v in ((ex, ey), (ey, ez), (ez, ex)):
            add(rf"\draw[ra] {right_angle(O, u, v)};")

    def angles(vec, vxy, th, ph, color, tl, pl, label, perp_label, length):
        tip, foot = length * vec, length * np.sin(th) * vxy
        # projection onto the xy plane (solid gold) and the drop line parallel to z (dashed)
        add(rf"\draw[->, line width=1.1pt, gold!75!black] {c(O)} -- {c(foot)} "
            rf"node[anchor=east, gold!60!black, font=\footnotesize, xshift=-2pt] {{{perp_label}}};")
        add(rf"\draw[dashed, line width=0.7pt, black!60] {c(foot)} -- {c(tip)};")
        add(rf"\draw[ra] {right_angle(foot, ez, -vxy, 0.16)};")
        add(rf"\draw[vec, {color}] {c(O)} -- {c(tip)} node[anchor=south west] {{{label}}};")
        # phi: from x to the projection, inside the xy plane
        add(rf"\draw[arc, gold!65!black] {arc(O, ex, ey, ph, 1.05)};")
        add(rf"\node[gold!55!black] at {c(1.6 * (np.cos(ph / 2) * ex + np.sin(ph / 2) * ey) + 0.15 * ez)} {{{pl}}};")
        # theta: from z to the vector
        add(rf"\draw[arc, {color}] {arc(O, ez, vxy, th, 1.3)};")
        add(rf"\node[{color}] at {c(1.55 * (np.cos(th / 2) * ez + np.sin(th / 2) * vxy))} {{{tl}}};")

    # ---------------- left: top rest frame
    frame("(0,0)", r"$x=y\times z$", r"$y\propto\hat z_\mathrm{beam}\times z$", r"$z=\hat p_t^{\,\mathrm{lab}}$",
          beam, r"beam axis")
    add(rf"\draw[vec, black!55] {c(O)} -- {c(-0.8 * w)} node[anchor=north west] {{$b$}};")
    angles(w, w_xy, t, p, "royal", r"$\theta$", r"$\phi$", r"$W$", r"$W_\perp$", L_W * 0.85)
    add(r"\end{scope}")
    add(r"\node[font=\bfseries] at (3.5,0,0.6) {top rest frame};")

    # ---------------- right: W rest frame
    shift = "(0,0,7.4)"
    frame(shift, r"$x'=y'\times z'$", r"$y'\propto z\times z'$", r"$z'=\hat p_W$",
          np.array([-np.sin(t), 0.0, np.cos(t)]), r"top direction $z$")
    angles(l, l_xy, tp, pp, "lep", r"$\theta'$", r"$\phi'$", r"$\ell$", r"$\ell_\perp$", L_LEP * 1.45)
    add(r"\end{scope}")
    add(r"\node[font=\bfseries] at (3.5,0,8.0) {$W$ rest frame};")

    # boost arrow between the panels
    add(r"\draw[->, line width=1pt, royal!70, bend left=20] (1.4,0,3.4) to node[above, font=\footnotesize, align=center]"
        r" {boost along $\hat p_W$\\(standard boost)} (1.4,0,6.4);")

    # ---------------- legend (left out of the slide version: the slide text carries it)
    if legend:
        add(r"\node[anchor=north west, align=left, font=\footnotesize, text width=16cm] at (-2.4,0,-1.9) {"
            r"\tikz\fill[royal!10] (0,0) rectangle (0.32,0.2); $x$--$z$ plane (contains the beam and $\hat p_t^{\,\mathrm{lab}}$;"
            r" right: the $x'$--$z'$ plane, contains $\hat p_W$ and the top direction)\quad"
            r"\tikz\fill[gold!30] (0,0) rectangle (0.32,0.2); $xy$ plane, perpendicular to $z$\\[2pt]"
            r"$\theta$: angle between $z$ and the $W$.\quad"
            r"$\phi$: \textbf{azimuth of the $W$ around $z$}: project the $W$ onto the $xy$ plane ($W_\perp$);"
            r" $\phi$ is the angle from $x$ to $W_\perp$, toward $y$.\\"
            r"$(\theta',\phi')$: the same for the charged lepton in the $W$ rest frame, with $(x',y',z')$:"
            r" $z'=\hat p_W$, $y'=z\times z'/|z\times z'|$ (in the $xy$ plane), $x'=y'\times z'$,"
            r" i.e.\ $(x',y',z')=R(\phi,\theta,0)\,(x,y,z)$.\\"
            r"The beam is not along $z$ ($z$ is the top direction); it lies in the $x$--$z$ plane with a negative"
            r" $x$ component, since $y\propto\hat z_\mathrm{beam}\times z$.\quad "
            r"Hadronic top: same construction with the down-type quark in place of $\ell$.};")
    add(r"\end{tikzpicture}")
    add(r"\end{document}")
    return "\n".join(lines) + "\n"


def main():
    out = repo_paths.results_dir("figures")          # pdf, png (and the LaTeX aux/log)
    for name, legend in (("angle_diagram", True), ("angle_diagram_slide", False)):
        tex = os.path.join(HERE, f"{name}.tex")
        with open(tex, "w") as f:
            f.write(build(legend))
        res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}",
                              f"{name}.tex"], cwd=HERE, stdout=subprocess.PIPE, text=True)
        if res.returncode:
            print("\n".join(res.stdout.splitlines()[-20:]))
            raise SystemExit("pdflatex failed")
        for ext in ("aux", "log"):
            p = os.path.join(out, f"{name}.{ext}")
            if os.path.exists(p):
                os.remove(p)
        try:
            import pymupdf
            pymupdf.open(os.path.join(out, f"{name}.pdf"))[0].get_pixmap(dpi=150).save(os.path.join(out, f"{name}.png"))
        except ImportError:
            pass
        print(f"wrote {os.path.join(out, name)}.pdf")


if __name__ == "__main__":
    main()
