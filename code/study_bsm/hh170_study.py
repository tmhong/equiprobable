#!/usr/bin/env python3
"""
Study A': hh with m_H = 170 GeV, so that both W bosons are on shell (user's idea).

Isolates what the W part of the mass handle does: the W Breit-Wigner factors go blind, the "top"
factors (m_Wb, b from the other Higgs) still see the signal, and after the chi2 reconstruction a
genuinely on-shell hadronic W makes the signal look more tt̄-like.  What is left is the angles:
the W helicity structure of a scalar -> WW decay is not V-A, and the pseudo-top direction differs.

Same scores as the hh(125) study, nothing retrained: f_tt̄ (paper basis, all terms and |c|/σ>10),
the Breit-Wigner factors, and their sum, at parton, gen and reco; AUC (bootstrap error) and the
signal efficiency at tt̄ efficiency 1% and 10%; selection efficiencies; per-variable 1-D likelihood-
ratio AUCs at parton level; a figure of m_W,had and m_t,had (parton and reco) for tt̄, hh(125), hh(170).
Output: results/study_bsm/hh170_study.json, results/study_bsm/fig_hh170.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "study_salvage"))
import make_plots  # noqa: E402
import repo_paths  # noqa: E402
from htt_study import lr_auc, eff_at  # noqa: E402
from htt_test import auc_err, extract  # noqa: E402
from salvage import surprisal  # noqa: E402
from simplest import PDG, bw_surprisal, cauchy_surprisal  # noqa: E402
from plotting import osde, style  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402

import argparse  # noqa: E402

_p = argparse.ArgumentParser()
_p.add_argument("--tag", default="170", help="variant tag: outputs hh<tag>_study.json / fig_hh<tag>.png")
_p.add_argument("--sample", default="samples/hh_bblvqq/run_m170_seed2170")
_p.add_argument("--label", default=r"$hh$, $m_H=170$ ($WW$ on shell)")
_a = _p.parse_args()
TAG = _a.tag
SIG = {"hh125": "samples/hh_bblvqq/run_seed2001", f"hh{TAG}": _a.sample}
TT_FIT, TT_IND = "samples/ttbar_bblvqq/run_seed1001", "samples/ttbar_bblvqq/run_seed12345"
OPTS = {"had_fermion": "down", "jet_assignment": "truth"}


def main():
    style.use()
    out_dir = repo_paths.results_dir("study_bsm")
    out = {"levels": {}, "selection": {}}
    for k, p in SIG.items():
        ev, _ = make_plots.load_sample(k, os.path.join(repo_paths.REPO, p), ["parton", "gen", "reco"], None, OPTS)
        out["selection"][k] = {"n": len(ev["parton"]), "trigger": float(np.mean(ev["parton"].lep_trigger)),
                               "gen_reco_ok": float(np.mean(np.asarray(ev["gen"].reco_ok))),
                               "reco_reco_ok": float(np.mean(np.asarray(ev["reco"].reco_ok)))}
    keep = {}
    for level, fname in (("parton", "f_ttbar_parton.npz"), ("gen", "f_ttbar_gen.npz"), ("reco", "f_ttbar_reco.npz")):
        D = {"tt_fit": extract(os.path.join(repo_paths.REPO, TT_FIT), level, 30000), "tt": extract(os.path.join(repo_paths.REPO, TT_IND), level)}
        for k, p in SIG.items():
            D[k] = extract(os.path.join(repo_paths.REPO, p), level)
        keep[level] = D
        ref = D["tt_fit"]
        f_all = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", fname))
        f_10 = f_all.truncated(f_all.significance() > 10)
        masses = list(D["tt"]["m"])

        def s_mass(d, which):
            s = 0
            for k in which:
                x = d["m"][k]
                if level == "parton":
                    s = s + bw_surprisal(x, *PDG[k])
                else:
                    r = ref["m"][k]; r = r[np.isfinite(r)]
                    M = np.median(r); q1, q3 = np.percentile(r, [25, 75])
                    s = s + cauchy_surprisal(x, M, 0.5 * (q3 - q1))
            return s

        wm = [m for m in masses if m.startswith("m_w")]
        tm = [m for m in masses if m.startswith("m_t")]

        def scores(d):
            fr_all, fr_10 = f_all(ref["ang"]), f_10(ref["ang"])
            s10 = surprisal(f_10(d["ang"]), fr_10)
            return {"angles, all terms": surprisal(f_all(d["ang"]), fr_all), "angles, >10 sigma": s10,
                    "BW: W masses only": s_mass(d, wm), "BW: top masses only": s_mass(d, tm), "BW: all masses": s_mass(d, masses),
                    "angles (>10 sigma) + BW": s10 + s_mass(d, masses)}

        S_tt = scores(D["tt"])
        res = {"n_tt_test": len(D["tt"]["mtt"]), "masses_used": masses}
        for k in SIG:
            S = scores(D[k])
            lab = np.r_[np.zeros(len(D["tt"]["mtt"])), np.ones(len(D[k]["mtt"]))]
            r = {"n": len(D[k]["mtt"]), "scores": {}}
            for name in S:
                a, e = auc_err(lab, np.r_[S_tt[name], S[name]])
                r["scores"][name] = {"auc": a, "err": e, "eff_at_tt_1pct": eff_at(S[name], S_tt[name], 0.01),
                                     "eff_at_tt_10pct": eff_at(S[name], S_tt[name], 0.10)}
            if level == "parton":
                r["single_variable_lr_auc"] = {a: lr_auc(D["tt"]["ang"][a], D[k]["ang"][a]) for a in ANGLE_NAMES}
                for m in masses:
                    r["single_variable_lr_auc"][m] = lr_auc(D["tt"]["m"][m], D[k]["m"][m])
            r["mass_medians"] = {m: float(np.nanmedian(D[k]["m"][m])) for m in masses}
            res[k] = r
            print(level, k, json.dumps({n: (round(v["auc"], 3), round(100 * v["eff_at_tt_1pct"], 1)) for n, v in r["scores"].items()}), flush=True)
        out["levels"][level] = res
    json.dump(out, open(os.path.join(out_dir, f"hh{TAG}_study.json"), "w"), indent=1)

    cols = {"tt": "#1f4e9c", "hh125": "#d1495b", f"hh{TAG}": "#2a9d8f"}
    labs = {"tt": r"$t\bar t$", "hh125": r"$hh$, $m_h=125$", f"hh{TAG}": _a.label}
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    for row, level in enumerate(("parton", "reco")):
        D = keep[level]
        for ax, m, xl, rng in zip(axs[row], ("m_w_had", "m_t_had"), [r"$m_{W,\mathrm{had}}$ [GeV]", r"$m_{t,\mathrm{had}}$ [GeV]"], [(0, 160), (50, 400)]):
            bins = np.linspace(*rng, 61)
            for k in ("tt", "hh125", f"hh{TAG}"):
                ax.hist(D[k]["m"][m], bins, density=True, histtype="step", lw=2.4, color=cols[k], label=labs[k])
            ax.set_yscale("log"); ax.set_ylim(1e-5, 30 * ax.get_ylim()[1]); ax.set_xlabel(xl + f" ({level})")
            ax.legend(fontsize=12, loc="upper center")
        axs[row][0].set_ylabel("unit area")
    fig.suptitle(r"$hh$ with on-shell $W$s: the $W$ mass handle disappears, the top-mass handle stays")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"fig_hh{TAG}.png"), dpi=110)
    print("done")


if __name__ == "__main__":
    main()
