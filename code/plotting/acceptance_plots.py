#!/usr/bin/env python3
"""
Effect of the event selection (acceptance) on the ten angles.

The parton-level angles are used throughout, so the angle value of an event never
changes: any shape difference between "no cuts" and "after cuts" is acceptance only,
not resolution or wrong jet assignment.  For comparison, the angles measured at the
selected level are also overlaid.

Outputs per sample (in --out, default results/acceptance):
  acceptance_<label>_shapes.png      unit-area parton angles: no cuts / gen selection /
                                     reco selection, + reco-measured angles; efficiency below
  acceptance_<label>_cutflow.png     reco efficiency vs (parton) angle after each cumulative cut
  acceptance_<label>.json            the efficiencies, per angle and bin

Example:
  python code/plotting/acceptance_plots.py --sample tt=samples/ttbar_bblvqq/run_seed1001 --sample hh=samples/hh_bblvqq/run_seed2001
"""
import argparse
import json
import os
import runpy
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)          # code/
sys.path.insert(0, CODE)

import repo_paths  # noqa: E402
from plotting import make_plots  # noqa: E402
from analysis.osde_fit import ANGLE_TEX, domain  # noqa: E402
from lib import style  # noqa: E402

STYLE = {  # name: (color, linestyle)
    "no cuts": ("#1f1f1f", "-"),
    "gen selection": ("#d1495b", "-"),
    "reco selection": ("#1f4e9c", "-"),
    "reco-measured, reco selection": ("#1f4e9c", ":"),
}
CUT_COLORS = ["#8ab17d", "#e9a03b", "#d1495b", "#1f4e9c"]


def cumulative_cuts(ev):
    """[(label, mask)] cumulative reco/gen selection steps."""
    steps = [("1 lepton", ev.nleptons == 1), (r"$\geq$4 jets", ev.njets >= 4),
             (r"$\geq$1 b-jet", ev.nbjets >= 1), (r"$\chi^2$ reco", np.asarray(ev.reco_ok))]
    out, m = [], np.ones(len(ev), bool)
    for name, cut in steps:
        m = m & np.asarray(cut)
        out.append((name, m.copy()))
    return out


def efficiency(x, mask, edges):
    n_all, _ = np.histogram(x, edges)
    n_pass, _ = np.histogram(x[mask], edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = n_pass / n_all
        err = np.sqrt(eff * (1 - eff) / n_all)
    return eff, err


GROUPS = [("", ["Y", "cos_Theta", "cos_theta", "phi", "cos_theta_prime", "phi_prime"], (2, 3)),
          ("_had", ["cos_theta_tilde", "phi_tilde", "cos_theta_prime_tilde", "phi_prime_tilde"], (2, 2))]


def shapes_figure(P, G, R, label, path, nbins):
    """Unit-area parton angles for no cuts / gen sel / reco sel (+ reco-measured), with the
    efficiency below.  Two figures: <path> (production + leptonic top), <stem>_had (hadronic)."""
    style.use()
    sel = {"gen selection": np.asarray(G.reco_ok), "reco selection": np.asarray(R.reco_ok)}
    effs = {}
    stem, ext = os.path.splitext(path)
    for suffix, names, (nr, nc) in GROUPS:
        fig = plt.figure(figsize=(4.7 * nc, 4.4 * nr))
        outer = fig.add_gridspec(nr, nc, hspace=0.38, wspace=0.3)
        for i, a in enumerate(names):
            inner = outer[i // nc, i % nc].subgridspec(2, 1, height_ratios=[2.0, 1], hspace=0.0)
            ax, rax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            lo, hi = domain(a)
            edges = np.linspace(lo, hi, nbins + 1)
            x = np.clip(np.asarray(P[a], float), lo, hi - 1e-9)
            curves = {"no cuts": x, **{k: x[m] for k, m in sel.items()},
                      "reco-measured, reco selection": np.clip(np.asarray(R[a], float)[sel["reco selection"]],
                                                               lo, hi - 1e-9)}
            for name, v in curves.items():
                h, _ = np.histogram(v, edges, density=True)
                c, ls = STYLE[name]
                ax.stairs(h, edges, color=c, ls=ls, lw=2, label=name)
            ax.set_xlim(lo, hi)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelbottom=False)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, prune="lower"))
            effs[a] = {}
            for name, m in sel.items():
                e, er = efficiency(x, m, edges)
                c, _ = STYLE[name]
                rax.errorbar(0.5 * (edges[1:] + edges[:-1]), e, yerr=er, fmt="o-", ms=3, lw=1.5, color=c)
                effs[a][name] = {"edges": edges.tolist(), "eff": np.nan_to_num(e).tolist()}
            rax.set_xlim(lo, hi)
            rax.set_ylim(0, None)
            rax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2, prune="upper"))
            rax.set_xlabel(ANGLE_TEX[a] + " (parton)")
            if i % nc == 0:
                ax.set_ylabel("unit area")
                rax.set_ylabel("eff.")
        h, lab = fig.axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=4, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.96))
        fig.suptitle(f"{label}: acceptance (parton-level angles; dotted = measured at reco)"
                     + (" — hadronic top" if suffix else ""), y=1.0)
        fig.savefig(stem + suffix + ext, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return effs


def cutflow_figure(P, R, label, path, nbins):
    """Reco efficiency vs (parton) angle after each cumulative cut; two figures as above."""
    style.use()
    steps = cumulative_cuts(R)
    stem, ext = os.path.splitext(path)
    for suffix, names, (nr, nc) in GROUPS:
        fig, axs = plt.subplots(nr, nc, figsize=(4.7 * nc, 3.6 * nr))
        for ax, a in zip(np.ravel(axs), names):
            lo, hi = domain(a)
            edges = np.linspace(lo, hi, nbins + 1)
            x = np.clip(np.asarray(P[a], float), lo, hi - 1e-9)
            for (name, m), c in zip(steps, CUT_COLORS):
                e, er = efficiency(x, m, edges)
                ax.errorbar(0.5 * (edges[1:] + edges[:-1]), e, yerr=er, fmt="o-", ms=3, lw=1.5, color=c,
                            label=f"{name} ({100 * m.mean():.1f}%)")
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1)
            ax.set_xlabel(ANGLE_TEX[a] + " (parton)")
        np.ravel(axs)[0].set_ylabel("reco efficiency")
        np.ravel(axs)[0].legend(fontsize=15, frameon=True, framealpha=0.85, title="cumulative cuts", title_fontsize=15)
        fig.suptitle(f"{label}: reco efficiency vs angle, cut by cut" + (" — hadronic top" if suffix else ""))
        fig.tight_layout()
        fig.savefig(stem + suffix + ext, dpi=110)
        plt.close(fig)
    return [(n, float(m.mean())) for n, m in steps]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--config", default=os.path.join(HERE, "plot_config.py"))
    p.add_argument("--nbins", type=int, default=20)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--out", default=os.path.join(repo_paths.RESULTS, "acceptance"))
    args = p.parse_args()

    cfg = runpy.run_path(args.config)
    opts = {"had_fermion": "down", "jet_assignment": "truth", **cfg.get("ANGLES", {})}
    os.makedirs(args.out, exist_ok=True)
    for spec in args.sample:
        label, _, path = spec.partition("=")
        ev, _ = make_plots.load_sample(label, path, ["parton", "gen", "reco"], args.max_events, opts)
        shown = {"tt": r"$t\bar t$", "hh": r"$hh$"}.get(label, label)
        P, G, R = ev["parton"], ev["gen"], ev["reco"]
        effs = shapes_figure(P, G, R, shown, os.path.join(args.out, f"acceptance_{label}_shapes.png"), args.nbins)
        steps = cutflow_figure(P, R, shown, os.path.join(args.out, f"acceptance_{label}_cutflow.png"), args.nbins)
        with open(os.path.join(args.out, f"acceptance_{label}.json"), "w") as f:
            json.dump({"sample": path, "n": len(P), "cutflow_reco": steps,
                       "eff_gen": float(np.mean(G.reco_ok)), "eff_reco": float(np.mean(R.reco_ok)),
                       "per_angle": effs}, f, indent=1)
        print(f"[acceptance] {label}: N={len(P)}  gen sel {100 * np.mean(G.reco_ok):.1f}%  "
              f"reco sel {100 * np.mean(R.reco_ok):.1f}%")
    print(f"[acceptance] outputs in {args.out}")


if __name__ == "__main__":
    main()
