#!/usr/bin/env python3
"""
Two-panel diagram of the top-decay angles (θ, φ in the top rest frame; θ', φ' in the W rest frame), as TikZ, in the
visual language of the paper figure angle_defn.pdf: flat planes in perspective (Pitt Gold / Pitt Royal / gray, 80%
opacity, no outlines), thin black vectors, the mother direction dashed through the vertex, small angle arcs at the
vertex, φ drawn as the angle between the two planes hinged on that direction, and a small axis triad on the side.

Conventions (as make_angle_diagram.py):
  top rest frame:  z = top direction in the lab, y ∝ beam × z, x = y × z; the gold plane is the x-z plane (it
                   contains the beam), the blue plane contains z and the W; φ is the angle between them.
  W rest frame:    z' = W direction, y' ∝ z × z', x' = y' × z'; the gold plane is the x'-z' plane (it contains the
                   top direction z), the gray plane contains z' and the lepton; φ' is the angle between them.

Sized for one PRL column (8.6 cm); the printed font size is the size after scaling to it.
Run:  .venv/bin/python code/figures/make_angle_diagram_slide.py   ->  code/figures/angle_diagram_slide.tex,
                                                                   results/figures/angle_diagram_slide.{pdf,png}
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))     # code/figures: the generated .tex (kept in git)
sys.path.insert(0, os.path.dirname(HERE))
import repo_paths  # noqa: E402

STEM = "angle_diagram_slide"
FONT_PT, FINAL_WIDTH_CM = 14.4, 8.6        # \Large; one PRL column

# screen images (cm) of the frame's unit vectors: z to the right and slightly down, x towards the viewer (down-left),
# y up, so the x-z plane is a sheet seen from above, as in the paper figure
EX, EY, EZ = np.array([-0.70, -0.62]), np.array([0.0, 1.0]), np.array([1.0, -0.35])
PANELS = {
    "top": dict(th=52.0, ph=75.0, plane="royal", plane_op=0.80, on_plane="white", vec="W", back="b", ref="beam",
                ref_ang=40.0, prime=""),
    "W": dict(th=55.0, ph=75.0, plane="black!30", plane_op=0.80, on_plane="black", vec=r"\ell", back=r"\nu", ref="z",
              ref_ang=52.0, prime="'"),
}
Z0, Z1 = -1.55, 2.35        # extent of the planes along z
R_GOLD, R_DEC = 1.55, 1.45  # extent of the gold plane (along +x, and 0.55 of it along -x) and of the decay plane
L_VEC, L_BACK = 1.95, 1.55
R_TH, R_PHI, Z_PHI = 0.80, 0.60, Z1 + 0.45     # theta arc radius; phi arc radius and its position along z
DX = 6.6 
LAB_HW = 0.30               # cm: half-width of a theta label plus clearance                   # cm between the two panel origins


def scr(v):
    return v[0] * EX + v[1] * EY + v[2] * EZ


def c(v):
    p = scr(v) if len(v) == 3 else v
    return "({:.4f},{:.4f})".format(*p)


def poly(pts):
    return " -- ".join(c(p) for p in pts)


def u_az(ph):
    return np.array([np.cos(np.radians(ph)), np.sin(np.radians(ph)), 0.0])


def panel(d):
    p = d["prime"]
    th, ph = d["th"], d["ph"]
    ez = np.array([0.0, 0.0, 1.0])
    u = u_az(ph)
    v = L_VEC * (np.sin(np.radians(th)) * u + np.cos(np.radians(th)) * ez)
    vb = -L_BACK * v / np.linalg.norm(v)
    ex = np.array([1.0, 0, 0])
    out = []
    # gold reference plane (x-z) and the decay plane (z and the vector), hinged on the z axis
    gold = [Z0 * ez - 0.55 * R_GOLD * ex, Z1 * ez - 0.55 * R_GOLD * ex, Z1 * ez + R_GOLD * ex, Z0 * ez + R_GOLD * ex]
    dec = [Z0 * ez, Z1 * ez, Z1 * ez + R_DEC * u, Z0 * ez + R_DEC * u]
    out.append(rf"\fill[gold, opacity=0.80] {poly(gold)} -- cycle;")
    out.append(rf"\fill[{d['plane']}, opacity={d['plane_op']}] {poly(dec)} -- cycle;")
    # reference line in the gold plane: the beam (top frame) or the top direction z (W frame), through the vertex
    a = np.radians(d["ref_ang"])
    rdir = np.cos(a) * ez * (-1 if not p else 1) + np.sin(a) * ex * (1 if not p else -1)
    if not p:
        out.append(rf"\draw[thin] {c(-2.1 * rdir)} -- {c(2.1 * rdir)} node[pos=1, anchor=north, inner sep=2pt] {{beam}};")
    else:
        out.append(rf"\draw[dashed, thin] (0,0) -- {c(2.0 * rdir)} node[anchor=south, inner sep=1pt] {{$z$}};")
    # the mother direction: dashed through the vertex
    out.append(rf"\draw[dashed, thin] {c(Z0 * ez)} -- {c((Z1 + 1.05) * ez)} node[anchor=west] {{$z{p}$}};")
    # vectors
    out.append(rf"\draw[vec] (0,0) -- {c(v)} node[anchor=south, inner sep=2pt, font=\LARGE] {{${d['vec']}$}};")
    out.append(rf"\draw[vec] (0,0) -- {c(vb)} node[anchor=north, inner sep=2pt, font=\LARGE] {{${d['back']}$}};")
    # theta: from z to the vector, in the decay plane
    arc_th = [R_TH * (np.sin(np.radians(t)) * u + np.cos(np.radians(t)) * ez) for t in np.linspace(0, th, 25)]
    # label on the screen bisector between the line crossing the arc (beam / z ray, upper half) and the vector, far
    # enough out that a label of half-width LAB_HW fits in the gap
    up = (-rdir if not p else rdir)
    a1, a2 = (np.arctan2(*scr(x)[::-1]) for x in (up, v))
    gap = abs(a2 - a1)
    r_lab = max(np.linalg.norm(scr(R_TH * (np.sin(np.radians(th)) * u + np.cos(np.radians(th)) * ez))) + 0.30,
                LAB_HW / np.sin(gap / 2))
    mid = r_lab * np.array([np.cos((a1 + a2) / 2), np.sin((a1 + a2) / 2)])   # nearer the vector: clear of the beam / z line
    out.append(rf"\draw[angle, {d['on_plane']}] {poly(arc_th)};")
    out.append(rf"\node[{d['on_plane']}] at {c(mid)} {{$\theta{p}$}};")   # screen position
    # phi: between the gold and the decay plane, around the z axis at z = Z_PHI
    arc_ph = [Z_PHI * ez + R_PHI * u_az(a_) for a_ in np.linspace(0, ph, 25)]
    out.append(rf"\draw[angle] {poly(arc_ph)};")
    out.append(rf"\node[anchor=south west, inner sep=1pt] at {c(Z_PHI * ez + R_PHI * u_az(ph))} {{$\phi{p}$}};")
    return out


def triad(origin, p):
    o = np.array(origin)
    L = 0.55
    out = []
    for e, lab, anc in ((EX, "x", "north east"), (EY, "y", "south"), (EZ, "z", "west")):
        out.append(rf"\draw[->, thin] ({o[0]:.3f},{o[1]:.3f}) -- ({o[0] + L * e[0]:.3f},{o[1] + L * e[1]:.3f}) "
                   rf"node[anchor={anc}, inner sep=1pt, font=\large] {{${lab}{p}$}};")
    return out


def build():
    L = [r"\documentclass[border=4pt]{standalone}",
         r"\usepackage{tikz,amsmath}",
         r"\usetikzlibrary{arrows.meta}",
         r"\definecolor{royal}{HTML}{003594}\definecolor{gold}{HTML}{FFB81C}",
         r"\begin{document}",
         r"\begin{tikzpicture}[font=\Large, >={Latex[length=2.4mm, width=1.8mm]}, thin/.style={line width=0.8pt},"
         r" vec/.style={->, line width=1.0pt}, angle/.style={line width=1.0pt}]"]
    for i, (name, d) in enumerate(PANELS.items()):
        L += [rf"\begin{{scope}}[shift={{({i * DX}cm,0cm)}}]"] + panel(d) + triad((2.0, -2.45), d["prime"]) + [r"\end{scope}"]
    L += [r"\node[font=\Large\bfseries] at (0.4cm,2.9cm) {top rest frame};",
          rf"\node[font=\Large\bfseries] at ({DX + 0.4}cm,2.9cm) {{$W$ rest frame}};",
          r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(L) + "\n"


def main():
    out = repo_paths.results_dir("figures")          # pdf, png (and the LaTeX aux/log)
    open(os.path.join(HERE, STEM + ".tex"), "w").write(build())
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}", STEM + ".tex"],
                   cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    import pymupdf
    doc = pymupdf.open(os.path.join(out, STEM + ".pdf"))
    w_cm = doc[0].rect.width / 72 * 2.54
    doc[0].get_pixmap(dpi=200).save(os.path.join(out, STEM + ".png"))
    f = FINAL_WIDTH_CM / w_cm
    print(f"{STEM}.pdf: {w_cm:.1f} cm wide; in one PRL column (x{f:.2f}) the labels are {FONT_PT * f:.1f} pt")


if __name__ == "__main__":
    main()
