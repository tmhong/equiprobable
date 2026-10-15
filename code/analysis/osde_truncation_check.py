#!/usr/bin/env python3
"""
Why the gen/reco projections of f_tt̄ miss the data: the basis order, not the fit.

For each angle, compare the histogram with
  (1) the projection of the fitted 10-D f (results/osde/f_ttbar_<level>.npz),
  (2) a direct 1-D orthogonal expansion of the same data truncated at the SAME order as the
      paper's basis (cos θ: Legendre l<=1; cos θ': l<=2; φ, φ': Fourier |m|<=1).  It must agree
      with (1), which checks the code,
  (3) a higher-order 1-D expansion (Legendre l<=10, Fourier |m|<=6), which shows that the
      acceptance edges need higher orders than the basis contains.

Output: results/osde/truncation_<level>.png and results/osde/truncation_<level>.json
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.polynomial import legendre  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import repo_paths  # noqa: E402
from osde_fit import ANGLE_TEX  # noqa: E402
from lib import osde, style  # noqa: E402

# angle: (kind, order of the paper's basis)
ANGLES = {"cos_theta": ("leg", 1), "cos_theta_prime": ("leg", 2), "phi_prime": ("four", 1),
          "cos_theta_tilde": ("leg", 1), "cos_theta_prime_tilde": ("leg", 2), "phi_prime_tilde": ("four", 1)}
HIGH = {"leg": 10, "four": 6}


def expansion_1d(v, kind, order, grid):
    """Orthogonal-series density estimate of the 1-D sample v, evaluated on grid."""
    if kind == "leg":
        out = np.zeros_like(grid)
        for l in range(order + 1):
            pl = legendre.Legendre.basis(l)
            out += (2 * l + 1) / 2 * np.mean(pl(v)) * pl(grid)
        return out
    out = np.full_like(grid, 1 / (2 * np.pi))
    for m in range(1, order + 1):
        out += (np.mean(np.cos(m * v)) * np.cos(m * grid) + np.mean(np.sin(m * v)) * np.sin(m * grid)) / np.pi
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="tt̄ fit sample (angle cache)")
    p.add_argument("--osde", default=os.path.join(repo_paths.RESULTS, "osde"))
    p.add_argument("--levels", default="parton,parton_trig,gen,reco")
    p.add_argument("--tag", default="down_truth")
    args = p.parse_args()

    z = np.load(os.path.join(args.train, f"angles_{args.tag}.npz"))
    for lvl in args.levels.split(","):
        f = osde.load_density(os.path.join(args.osde, f"f_ttbar_{lvl}.npz"))
        style.use()
        fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.0))
        res = {}
        for ax, (a, (kind, order)) in zip(axs.ravel(), ANGLES.items()):
            v = z[f"{lvl}/{a}"]
            lo, hi = (-1.0, 1.0) if kind == "leg" else (-np.pi, np.pi)
            edges = np.linspace(lo, hi, 26)
            h, _ = np.histogram(v, edges)
            dens, err = h / len(v) / np.diff(edges), np.sqrt(h) / len(v) / np.diff(edges)
            cen = 0.5 * (edges[1:] + edges[:-1])
            grid = np.linspace(lo, hi, 301)
            proj = f.projection(a, grid)
            same = expansion_1d(v, kind, order, grid)
            high = expansion_1d(v, kind, HIGH[kind], grid)
            # the 10-D projection carries the Y-normalization factor ∫f (≈1): compare shapes
            norm = np.trapezoid(proj, grid)
            diff = float(np.max(np.abs(proj / norm - same)) / np.max(same))

            def chi2(curve):
                fine = np.interp(cen, grid, curve)
                return float(np.sum(((dens - fine) / err) ** 2))
            res[a] = {"order": order, "max_rel_diff_f_vs_same_order_1d": diff,
                      "chi2_same_order": chi2(same), "chi2_high_order": chi2(high), "ndf": len(cen)}
            ax.errorbar(cen, dens, yerr=err, fmt="o", ms=3, color="#1f4e9c", label=f"{lvl}-level $t\\bar t$ (fit sample)")
            ax.plot(grid, proj, color="k", lw=2.2, label=r"projection of 10-D $f_{t\bar t}$")
            ax.plot(grid, same, color="#e9a03b", lw=1.4, ls="--",
                    label=f"1-D expansion, same order ({'$l$' if kind == 'leg' else '$|m|$'}$\\leq${order})")
            ax.plot(grid, high, color="#d1495b", lw=1.4,
                    label=f"1-D expansion, {'$l$' if kind == 'leg' else '$|m|$'}$\\leq${HIGH[kind]}")
            ax.set_xlabel(ANGLE_TEX[a])
            ax.set_xlim(lo, hi)
            ax.set_ylim(bottom=0)
            ax.set_title(f"$\\chi^2$: {res[a]['chi2_same_order']:.0f} vs {res[a]['chi2_high_order']:.0f} (ndf {len(cen)})", fontsize=18)
        axs[0, 0].legend(fontsize=14, frameon=True, framealpha=0.9, loc="lower left")
        fig.suptitle(f"{lvl} level: the basis order limits what $f_{{t\\bar t}}$ can describe "
                     "(black = dashed orange: code consistent)", fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(args.osde, f"truncation_{lvl}.png"), dpi=110)
        plt.close(fig)
        with open(os.path.join(args.osde, f"truncation_{lvl}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        print(lvl, json.dumps({a: (round(r["max_rel_diff_f_vs_same_order_1d"], 4), round(r["chi2_same_order"]),
                                   round(r["chi2_high_order"])) for a, r in res.items()}))


if __name__ == "__main__":
    main()
