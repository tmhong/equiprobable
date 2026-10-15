#!/usr/bin/env python3
"""
Neutrino p_z resolution: Δp_z = p_z(ν, reconstructed) − p_z(ν, parton), for events with a
reconstructed tt̄ system.

Panels: (1) gen and reco level; (2) reco, all events vs events whose jets were all assigned
correctly; (3) choice between the two m_W-constrained solutions: the top-mass χ² choice
(default, as in the paper) vs the smaller-|p_z| rule vs always taking the other solution.

Example:
  python code/plotting/nu_pz_plot.py --sample samples/ttbar_bblvqq/run_seed12345 --out results/plotting/plots/nu_pz.png
"""
import argparse
import json
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import awkward as ak  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)          # code/
sys.path.insert(0, CODE)

import repo_paths  # noqa: E402
from plotting import make_plots  # noqa: E402
from lib import physics, summary, style  # noqa: E402


def width(d):
    q16, q50, q84 = np.percentile(d, [16, 50, 84])
    return q50, (q84 - q16) / 2


def both_solutions(ev):
    """The two m_W-constrained p_z solutions and whether the discriminant was negative."""
    lep = ev.lep
    lpx, lpy, lpz, le = (ak.to_numpy(ak.fill_none(getattr(lep, c), np.nan)).astype(float)
                         for c in ("px", "py", "pz", "E"))
    mpx, mpy = ev.met * np.cos(ev.met_phi), ev.met * np.sin(ev.met_phi)
    sols = physics.neutrino_pz(lpx, lpy, lpz, le, mpx, mpy)
    return sols, np.isclose(sols[0], sols[1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--config", default=os.path.join(HERE, "plot_config.py"))
    p.add_argument("--out", default=repo_paths.results("plots", "nu_pz.png"))
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    cfg = runpy.run_path(args.config)
    opts = {"had_fermion": "down", "jet_assignment": "truth", **cfg.get("ANGLES", {})}
    ev, _ = make_plots.load_sample("nu", args.sample, ["parton", "gen", "reco"], args.max_events, opts)
    P = ev["parton"]
    truth = np.asarray(P.nu.pz, float)
    bins = np.linspace(-400, 400, 81)
    res = {}

    style.use()

    figs = [plt.subplots(figsize=(9.5, 6.6)) for _ in range(3)]
    axs = [a for _, a in figs]
    for lvl, c in (("gen", "#d1495b"), ("reco", "#1f4e9c")):
        E = ev[lvl]
        ok = np.asarray(E.reco_ok)
        d = (np.asarray(E.nu.pz) - truth)[ok]
        m, w = width(d)
        res[lvl] = {"median": m, "half_width_68": w, "rms": float(d.std()), "n": int(ok.sum())}
        axs[0].hist(np.clip(d, bins[0], bins[-1]), bins, histtype="step", density=True, lw=1.6, color=c,
                    label=f"{lvl}: median {m:+.1f}, 68% half-width {w:.0f} GeV")
    axs[0].set_title("gen vs reco", fontsize=18)

    R = ev["reco"]
    ok = np.asarray(R.reco_ok)
    good = summary.correct_assignment(R, P)
    for mask, lab, c in ((ok, "all", "#1f4e9c"), (good, "all jets correctly assigned", "#2a9d8f"),
                         (ok & ~good, "some jet wrong", "#e9a03b")):
        d = (np.asarray(R.nu.pz) - truth)[mask]
        m, w = width(d)
        res[f"reco_{lab}"] = {"median": m, "half_width_68": w, "n": int(mask.sum())}
        axs[1].hist(np.clip(d, bins[0], bins[-1]), bins, histtype="step", density=True, lw=1.6, color=c,
                    label=f"{lab} ({mask.sum()}): {w:.0f} GeV")
    axs[1].set_title("reco: effect of the jet assignment", fontsize=18)

    sols, degenerate = both_solutions(R)
    chosen = np.asarray(R.nu.pz)
    other = np.where(np.isclose(chosen, sols[0]), sols[1], sols[0])
    smaller = np.where(np.abs(sols[0]) < np.abs(sols[1]), sols[0], sols[1])
    two = ok & ~degenerate                         # only events where the choice matters
    for val, lab, c in ((chosen, "top-mass $\\chi^2$ choice (default)", "#1f4e9c"),
                        (smaller, "smaller $|p_z|$", "#6a4c93"), (other, "the other solution", "#999999")):
        d = (val - truth)[two]
        m, w = width(d)
        axs[2].hist(np.clip(d, bins[0], bins[-1]), bins, histtype="step", density=True, lw=1.6, color=c,
                    label=f"{lab}: {w:.0f} GeV")
    closer = np.abs(chosen - truth) <= np.abs(other - truth)
    res["choice"] = {"frac_complex": float(np.mean(degenerate[ok])),
                     "frac_chi2_picks_closer": float(np.mean(closer[two])),
                     "frac_smaller_picks_closer": float(np.mean((np.abs(smaller - truth) <= np.abs(
                         np.where(smaller == sols[0], sols[1], sols[0]) - truth))[two]))}
    axs[2].set_title(f"reco: choice of solution ({two.sum()} events)", fontsize=18)

    for ax in axs:
        ax.set_xlabel(r"$\Delta p_z = p_z^{\nu,\mathrm{reco}} - p_z^{\nu,\mathrm{parton}}$ [GeV]")
        ax.set_ylabel("normalized")
        ax.legend(fontsize=14, frameon=True, framealpha=0.9, loc="upper left")
        ax.set_xlim(bins[0], bins[-1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    stem, ext = os.path.splitext(args.out)
    for (fig, _), key in zip(figs, "abc"):
        fig.tight_layout()
        fig.savefig(f"{stem}_{key}{ext}", dpi=110)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
