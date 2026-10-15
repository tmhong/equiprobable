#!/usr/bin/env python3
"""
Study B2: new physics in the top decay vertex, a right-handed W helicity fraction F_R > 0 in regular tt̄.

The textbook case for a density method: the SM (V-A) decay density vanishes at cos θ' -> +1 and a
right-handed W populates exactly that region.  Current limits (F_R < ~0.03) make it a benchmark, not a
discovery channel; the point is to see what the two uses of f_tt̄ do with new physics in the decay.

Signal emulated by reweighting the independent tt̄ sample event by event with the parton-level W-decay
angles (wtb_reweight.py: rho(c) = 3/4(1-c²)F_0 + 3/8(1-c)²F_L + 3/8(1+c)²F_R on both sides, F_0 kept,
F_L -> F_L - F_R), so that gen and reco scores of the same events carry the parton weights.  For the
coefficient fit the weighted sample is fitted with event weights: ĉ = Σ w B* / Σ w with the weighted-variance error (the
effective number of events Σw²/Σw² is reported).

1. f_tt̄ as an anomaly score (angles all / >10σ, + BW masses): AUC and signal efficiency at tt̄ efficiency
   1% and 10%, at parton, gen, reco, for F_R = 0.02 ... 0.3.
2. The coefficient test (pulls of ĉ_data - ĉ_tt̄ over the 14,400 real degrees of freedom; chi2 per physics
   category; the terms that respond) at parton, gen and reco, for F_R = 0.01 ... 0.3.
Outputs: results/study_bsm/wtb_study.json, results/study_bsm/fig_wtb.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "study_salvage"))
import make_plots  # noqa: E402
import repo_paths  # noqa: E402
from htt_study import term_groups  # noqa: E402
from wtb_reweight import F_SM, eff_at, weights  # noqa: E402
from salvage import surprisal  # noqa: E402
from simplest import PDG, bw_surprisal, cauchy_surprisal  # noqa: E402
from lib import osde, style  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
FR_SCORE = [0.02, 0.05, 0.1, 0.3]
FR_COEF = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3]


def weighted_fit(X, w, chunk=20000):
    """OSDE with event weights: ĉ = Σ w B* / Σ w, error from the weighted variance (Σ w² |B|² / W² - |ĉ|² Σ w² / W²)."""
    g = osde.TTbarDensity()
    n = len(X["Y"])
    s1 = np.zeros(g.shape, complex)
    s2 = np.zeros(g.shape)
    for lo in range(0, n, chunk):
        ml, mh, p, h = g.factors({k: v[lo:lo + chunk] for k, v in X.items()})
        wk = w[lo:lo + chunk]
        s1 += np.einsum("kj,ki,kl,kn->jiln", ml.conj(), mh.conj(), p * wk[:, None], h, optimize=True)
        s2 += np.einsum("kj,ki,kl,kn->jiln", abs(ml) ** 2, abs(mh) ** 2, p ** 2 * (wk ** 2)[:, None], h ** 2, optimize=True)
    W, W2 = float(np.sum(w)), float(np.sum(w ** 2))
    g.c = s1 / W
    g.c_err = np.sqrt(np.clip(s2 / W ** 2 - abs(g.c) ** 2 * W2 / W ** 2, 0, None))
    g.n_fit = int(W ** 2 / W2)
    return g


def coef_pulls(f_ref, X, w, phys, groups):
    g = weighted_fit(X, w)
    dc = (g.c - f_ref.c).ravel()
    err = np.sqrt(g.c_err.ravel() ** 2 + f_ref.c_err.ravel() ** 2)
    pull = dc / np.where(err > 0, err, np.inf)
    partner = f_ref.conjugate_partner()
    k = np.arange(len(pull))
    selfc, first = partner == k, k < partner
    dofs = np.r_[pull[selfc].real, np.sqrt(2) * pull[first].real, np.sqrt(2) * pull[first].imag]
    term_of = np.r_[k[selfc], k[first], k[first]]
    out = {"n_data": len(X["Y"]), "n_eff": g.n_fit, "pull_std": float(np.std(dofs)), "max_abs_pull": float(np.max(np.abs(dofs))), "groups": {}}
    for name, mask in [("all terms", np.ones(len(pull), bool))] + [(n, m) for n, m in groups.items() if "SM-" not in n]:
        m = mask[term_of]
        chi2, ndf = float(np.sum(dofs[m] ** 2)), int(m.sum())
        out["groups"][name] = {"chi2": chi2, "ndf": ndf, "z": float((chi2 - ndf) / np.sqrt(2 * ndf)) if ndf else 0.0}
    best = np.zeros(len(pull))
    np.maximum.at(best, term_of, np.abs(dofs))
    out["top_terms"] = [{"term": osde.describe(f_ref, int(i))[0], "physics": phys[i], "pull": float(best[i]),
                         "ref_signif": float(f_ref.significance().ravel()[i])} for i in np.argsort(-best)[:8]]
    return out


def main():
    style.use()
    out_dir = repo_paths.results_dir("study_bsm")
    ev, _ = make_plots.load_sample("tt", os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed12345"), ["parton", "gen", "reco"], None, OPTS)
    ref, _ = make_plots.load_sample("ref", os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed1001"), ["parton", "gen", "reco"], 30000, OPTS)
    P = ev["parton"]
    cl, ch = np.asarray(P.cos_theta_prime, float), np.asarray(P.cos_theta_prime_tilde, float)
    out = {"F_SM": F_SM, "levels": {}}
    keep = {}
    for lvl in ("parton", "gen", "reco"):
        e, r = ev[lvl], ref[lvl]
        ok = np.ones(len(e), bool) if lvl == "parton" else np.asarray(e.reco_ok)
        ang = {a: np.asarray(e[a], float) for a in ANGLE_NAMES}
        ok &= np.all([np.isfinite(v) for v in ang.values()], 0)
        ang = {a: v[ok] for a, v in ang.items()}
        rok = np.ones(len(r), bool) if lvl == "parton" else np.asarray(r.reco_ok)
        rang = {a: np.asarray(r[a], float)[rok] for a in ANGLE_NAMES}
        fin = np.all([np.isfinite(x) for x in rang.values()], 0)
        rang = {a: v[fin] for a, v in rang.items()}
        f_all = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", f"f_ttbar_{lvl}.npz"))
        f_10 = f_all.truncated(f_all.significance() > 10)
        masses = {"m_w_had": e.w_had.mass, "m_t_lep": e.t_lep.mass, "m_t_had": e.t_had.mass}
        if lvl == "parton":
            masses["m_w_lep"] = e.w_lep.mass
        sm = 0
        for k, v in masses.items():
            x = np.asarray(v, float)[ok]
            if lvl == "parton":
                sm = sm + bw_surprisal(x, *PDG[k])
            else:
                rr = np.asarray({"m_w_had": r.w_had.mass, "m_t_lep": r.t_lep.mass, "m_t_had": r.t_had.mass}[k], float)[rok]
                rr = rr[np.isfinite(rr)]
                M = np.median(rr); q1, q3 = np.percentile(rr, [25, 75])
                sm = sm + cauchy_surprisal(x, M, 0.5 * (q3 - q1))
        scores = {}
        for name, f in (("angles, all terms", f_all), ("angles, >10 sigma", f_10)):
            s = surprisal(f(ang), f(rang))
            scores[name] = s
            scores[name + " + BW"] = s + sm
        scores["BW masses only"] = sm
        res = {"n": int(ok.sum()), "score": {}, "coef": {}}
        for fr in FR_SCORE:
            w = weights(cl[ok], ch[ok], fr)
            entry = {"n_eff": float(np.sum(w) ** 2 / np.sum(w ** 2)),
                     "frac_cosprime_gt_0.8_sm": float(np.mean(ang["cos_theta_prime"] > 0.8)),
                     "frac_cosprime_gt_0.8_bsm": float(np.sum(w * (ang["cos_theta_prime"] > 0.8)) / np.sum(w))}
            for name, s in scores.items():
                lab = np.r_[np.zeros(len(s)), np.ones(len(s))]
                entry[name] = {"auc": roc_auc_score(lab, np.r_[s, s], sample_weight=np.r_[np.ones(len(s)), w]),
                               "eff_at_bkg_1pct": eff_at(s, w, s, 0.01), "eff_at_bkg_10pct": eff_at(s, w, s, 0.10)}
            res["score"][str(fr)] = entry
        phys, groups = term_groups(f_all)
        for fr in FR_COEF:
            w = weights(cl[ok], ch[ok], fr) if fr > 0 else np.ones(ok.sum())
            c = coef_pulls(f_all, ang, w, phys, groups)
            c["F_R"] = fr
            res["coef"][str(fr)] = c
            print(f"{lvl} F_R={fr}: n_eff {c['n_eff']}  pull std {c['pull_std']:.2f}  max|pull| {c['max_abs_pull']:.1f}  "
                  + ", ".join(f"{g} {v['z']:+.1f}" for g, v in c["groups"].items() if g in ("all terms", "W helicity", "spin correlation", "production angle"))
                  + "  top: " + "; ".join(f"{t['term']} ({t['pull']:.1f})" for t in c["top_terms"][:2]), flush=True)
        out["levels"][lvl] = res
        keep[lvl] = (ang, scores)
        for fr, en in res["score"].items():
            print(f"   score F_R={fr}: " + ", ".join(f"{n} {v['auc']:.3f}/{100 * v['eff_at_bkg_1pct']:.1f}%" for n, v in en.items() if isinstance(v, dict)), flush=True)
    json.dump(out, open(os.path.join(out_dir, "wtb_study.json"), "w"), indent=1)

    # figure: cos theta' (parton) SM vs F_R; the surprisal tail; the coefficient-test significance vs F_R
    ang, scores = keep["parton"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
    bins = np.linspace(-1, 1, 41)
    axs[0].hist(ang["cos_theta_prime"], bins, density=True, histtype="step", lw=2.6, color="#1f4e9c", label="SM ($V\\!-\\!A$)")
    for fr, col in ((0.1, "#e9a03b"), (0.3, "#d1495b")):
        w = weights(cl[:len(ang["Y"])], ch[:len(ang["Y"])], fr)
        axs[0].hist(ang["cos_theta_prime"], bins, weights=w, density=True, histtype="step", lw=2.4, color=col, label=rf"$F_R={fr}$")
    axs[0].set_xlabel(r"$\cos\theta'$ (lepton, parton)"); axs[0].set_ylabel("unit area"); axs[0].legend(fontsize=14, frameon=False)
    s = scores["angles, >10 sigma"]
    lo, hi = np.percentile(s, [0.5, 99.9]); sb = np.linspace(lo, hi, 50)
    axs[1].hist(s, sb, density=True, histtype="step", lw=2.6, color="#1f4e9c", label="SM")
    w = weights(cl[:len(s)], ch[:len(s)], 0.3)
    axs[1].hist(s, sb, weights=w, density=True, histtype="step", lw=2.4, color="#d1495b", label=r"$F_R=0.3$")
    axs[1].set_yscale("log"); axs[1].set_xlabel(r"surprisal $-\ln f_{t\bar t}$ (angles, parton)"); axs[1].legend(fontsize=14, frameon=False)
    for lvl, col, mk in (("parton", "#1f4e9c", "o"), ("gen", "#2a9d8f", "s"), ("reco", "#d1495b", "^")):
        c = out["levels"][lvl]["coef"]
        frs = [float(k) for k in c]
        axs[2].plot(frs, [c[str(f)]["groups"]["W helicity"]["z"] for f in frs], mk + "-", lw=2.2, ms=9, color=col, label=rf"$\chi^2$, $W$ helicity terms, {lvl}")
        axs[2].plot(frs, [c[str(f)]["max_abs_pull"] for f in frs], mk + "--", lw=1.6, ms=7, color=col, alpha=0.6)
    axs[2].axhline(5, color="0.6", lw=1, ls=":"); axs[2].set_xscale("symlog", linthresh=0.01)
    axs[2].set_xlabel(r"$F_R$"); axs[2].set_ylabel(r"significance [$\sigma$] (dashed: largest single pull)"); axs[2].legend(fontsize=12, frameon=False)
    fig.suptitle(r"A right-handed $W$ in $t\bar t$: where the density is in its element")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_wtb.png"), dpi=110)
    print("done")


if __name__ == "__main__":
    main()
