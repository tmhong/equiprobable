#!/usr/bin/env python3
"""
Compare the three bases for f_tt̄ — the paper's M functions (results/osde/), spherical-harmonic
products (results/osde_sh/), the extended block basis (results/osde_ext/) — full and CV-truncated, per level.

Figures of merit (all on the independent tt̄ sample, which no fit has seen):
  held-out ISE   ∫(f̂ - f)^2 - ∫f^2 = Σ|ĉ|^2 - 2 <f̂>_indep   (orthonormal bases; lower = closer to the
                 true density; comparable across bases at the same level)
  projection χ²  of the ten 1-D projections against the fit sample (worst angle and Y)
  f <= 0         fraction of independent tt̄ events with negative density
  AUC            hh vs tt̄ (from <dir>/simplify_summary.json)

Output: results/osde_sh/basis_comparison.json and results/osde_sh/basis_comparison.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import osde_fit  # noqa: E402
import repo_paths  # noqa: E402
from plotting import osde, style  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402

BASES = [("paper (M functions)", "osde"), ("spherical harmonics", "osde_sh"), ("extended", "osde_ext")]   # dirs in results/
OUT = os.path.join(repo_paths.RESULTS, "osde_sh")
LEVELS = ["parton", "parton_trig", "gen", "reco"]


def main():
    tr = np.load(os.path.join(repo_paths.SAMPLES, "ttbar_bblvqq/run_seed1001/angles_down_truth.npz"))
    te = np.load(os.path.join(repo_paths.SAMPLES, "ttbar_bblvqq/run_seed12345/angles_down_truth.npz"))
    get = lambda z, l: {a: z[f"{l}/{a}"] for a in ANGLE_NAMES}  # noqa: E731
    res, fvals = {}, {}
    for lab, d in BASES:
        d = os.path.join(repo_paths.RESULTS, d)
        simp = json.load(open(os.path.join(d, "simplify_summary.json")))["levels"] \
            if os.path.exists(os.path.join(d, "simplify_summary.json")) else {}
        for lvl in LEVELS:
            for kind, vkey in (("", "all"), ("_simplified", "cv_best")):
                path = os.path.join(d, f"f_ttbar_{lvl}{kind}.npz")
                if not os.path.exists(path) or f"{lvl}/Y" not in te.files:
                    continue
                f = osde.load_density(path)
                fi = f(get(te, lvl))
                fvals[(lab, lvl, kind)] = (fi, float(np.sum(np.abs(f.c) ** 2)))
                ise = float(np.sum(np.abs(f.c) ** 2) - 2 * fi.mean())
                ise_err = float(2 * fi.std() / np.sqrt(len(fi)))
                c2 = osde_fit.plot_projections(f, {"train": get(tr, lvl), "test": get(te, lvl)}, lvl,
                                               os.path.join(d, f"projections_ttbar{kind}_{lvl}.png"))
                worst = max((v[0] / v[1], a) for a, v in c2.items() if a != "Y")
                v = simp.get(lvl, {}).get("variants", {}).get(vkey, {})
                res[f"{lab}|{lvl}|{'CV' if kind else 'all'}"] = {
                    "basis": lab, "level": lvl, "truncation": "CV" if kind else "all",
                    "terms": int(np.count_nonzero(f.c)), "heldout_ise": ise, "heldout_ise_err": ise_err,
                    "worst_chi2ndf": worst[0], "worst_angle": worst[1], "Y_chi2ndf": c2["Y"][0] / c2["Y"][1],
                    "frac_negative_indep": float(np.mean(fi <= 0)), "auc": v.get("auc")}
                r = res[f"{lab}|{lvl}|{'CV' if kind else 'all'}"]
                print(f"{lab:22s} {lvl:12s} {r['truncation']:4s} terms {r['terms']:6d}  ISE {ise:+.3e} ± {ise_err:.1e}  "
                      f"worst χ²/ndf {r['worst_chi2ndf']:7.1f} ({r['worst_angle']})  Y {r['Y_chi2ndf']:7.1f}  "
                      f"f<=0 {100 * r['frac_negative_indep']:.1f}%  AUC {r['auc']}")
    # paired differences vs the paper basis (same events -> much smaller errors)
    for key, r in res.items():
        ref = fvals.get(("paper (M functions)", r["level"], "" if r["truncation"] == "all" else "_simplified"))
        mine = fvals.get((r["basis"], r["level"], "" if r["truncation"] == "all" else "_simplified"))
        if ref and mine:
            d = mine[0] - ref[0]
            r["delta_ise_vs_paper"] = float(mine[1] - ref[1] - 2 * d.mean())
            r["delta_ise_vs_paper_err"] = float(2 * d.std() / np.sqrt(len(d)))
    with open(os.path.join(OUT, "basis_comparison.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    plot(res)


def plot(res):
    """Left: paired held-out ISE difference vs the paper basis (CV) per level; right: worst projection χ²/ndf."""
    style.use()
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.0))
    cols = {"paper (M functions)": "#1f4e9c", "spherical harmonics": "#d1495b", "extended": "#2a9d8f"}
    lv = [l for l in LEVELS if any(k.split("|")[1] == l for k in res)]
    x = np.arange(len(lv))
    for i, (lab, _) in enumerate(BASES):
        for j, trunc in enumerate(("all", "CV")):
            vals, errs, worst = [], [], []
            for l in lv:
                r = res.get(f"{lab}|{l}|{trunc}", {})
                ref = res.get(f"paper (M functions)|{l}|CV", {})
                # paired difference vs paper-CV: (this - paper-all) + (paper-all vs paper-CV) is not paired, so use
                # the difference to the paper basis at the same truncation, shifted by paper(trunc) - paper(CV)
                pa = res.get(f"paper (M functions)|{l}|{trunc}", {})
                d = r.get("delta_ise_vs_paper", 0.0) + pa.get("heldout_ise", np.nan) - ref.get("heldout_ise", np.nan)
                vals.append(1e6 * d)
                errs.append(1e6 * (r.get("delta_ise_vs_paper_err", 0.0) or 0.0))
                worst.append(r.get("worst_chi2ndf", np.nan))
            off = (i - 1) * 0.26 + (j - 0.5) * 0.11
            kw = dict(color=cols[lab], mfc=cols[lab] if trunc == "CV" else "white", ms=11, mew=2.5)
            axs[0].errorbar(x + off, vals, yerr=errs, fmt="o" if trunc == "CV" else "s", capsize=3,
                            label=f"{lab.split(' (')[0]}, {'CV-truncated' if trunc == 'CV' else 'all terms'}", **kw)
            axs[1].plot(x + off, worst, "o" if trunc == "CV" else "s", **kw)
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("parton_trig", "parton+trig") for l in lv])
    axs[0].axhline(0, color="0.5", lw=1)
    axs[0].set_ylabel(r"$\Delta$ held-out ISE [$10^{-6}$]")
    axs[0].set_title("vs paper basis (CV): lower = closer to truth", fontsize=16)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=15, frameon=False,
               bbox_to_anchor=(0.5, 1.13))
    axs[1].set_yscale("log")
    axs[1].axhline(1, color="0.5", lw=1)
    axs[1].set_ylabel(r"worst projection $\chi^2$/ndf")
    axs[1].set_title("1-D projections (excluding $Y$)", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "basis_comparison.png"), dpi=120, bbox_inches="tight")
    print(f"wrote {os.path.join(OUT, 'basis_comparison.png')}")


if __name__ == "__main__":
    main()
