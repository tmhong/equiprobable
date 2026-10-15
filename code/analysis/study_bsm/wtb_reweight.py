#!/usr/bin/env python3
"""
BSM benchmark where an unsupervised tt̄ density should be in its element: new physics in the top
decay vertex that gives the W a right-handed helicity fraction F_R > 0.  In the SM (V-A) the W
decay distribution vanishes at cos θ' -> +1; a right-handed W populates exactly that region, where
f_tt̄ is smallest.

Emulated by reweighting the independent tt̄ sample event by event (no new generation):
   w = rho(cos θ'_lep; F_new) / rho(cos θ'_lep; F_SM) x rho(cos θ~'_had; F_new) / rho(cos θ~'_had; F_SM)
   rho(c) = 3/4 (1-c²) F_0 + 3/8 (1-c)² F_L + 3/8 (1+c)² F_R          (LO W-decay distribution)
using the PARTON-level angles, so that gen and reco scores of the same events carry the parton
weights.  F_SM = (0.703, 0.296, 0.001) (LO; the small F_R regulates the ratio at c -> 1); the new
physics keeps F_0 and moves F_L -> F_L - F_R.  This reweights the W-decay distributions only; the
polarization / spin-correlation interference terms are left at their SM values.

Scores (existing fits, nothing retrained): f_tt̄ paper basis all terms and |c|/σ>10, and with the
Breit-Wigner mass factors (study_salvage), at parton, gen and reco level.  Metrics: weighted AUC
(signal = reweighted sample, background = unweighted), and the signal efficiency at tt̄ efficiency
1% and 10% (tail enrichment: what an anomaly score is for).  Also the effective sample size of the
reweighted signal.  Output: results/study_bsm/wtb_reweight.json
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "study_salvage"))
import make_plots  # noqa: E402
import repo_paths  # noqa: E402
from salvage import surprisal  # noqa: E402
from simplest import PDG, bw_surprisal, cauchy_surprisal  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
F_SM = (0.703, 0.296, 0.001)
FR_VALUES = [0.02, 0.05, 0.1, 0.3]


def rho(c, F):
    f0, fl, fr = F
    return 0.75 * (1 - c ** 2) * f0 + 0.375 * (1 - c) ** 2 * fl + 0.375 * (1 + c) ** 2 * fr


def weights(cl, ch, fr):
    Fn = (F_SM[0], F_SM[1] + F_SM[2] - fr, fr)
    return rho(cl, Fn) / rho(cl, F_SM) * rho(ch, Fn) / rho(ch, F_SM)


def eff_at(score_sig, w_sig, score_bkg, bkg_eff):
    """signal efficiency when a cut on the score (high = signal-like) keeps bkg_eff of the background."""
    cut = np.quantile(score_bkg, 1 - bkg_eff)
    return float(np.sum(w_sig[score_sig > cut]) / np.sum(w_sig))


def main():
    ev, _ = make_plots.load_sample("tt", os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed12345"), ["parton", "gen", "reco"], None, OPTS)
    ref, _ = make_plots.load_sample("ref", os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed1001"), ["parton", "gen", "reco"], 30000, OPTS)
    P = ev["parton"]
    cl, ch = np.asarray(P.cos_theta_prime, float), np.asarray(P.cos_theta_prime_tilde, float)
    out = {"F_SM": F_SM, "levels": {}}
    for lvl in ("parton", "gen", "reco"):
        e, r = ev[lvl], ref[lvl]
        ok = np.ones(len(e), bool) if lvl == "parton" else np.asarray(e.reco_ok)
        ang = {a: np.asarray(e[a], float) for a in ANGLE_NAMES}
        ok &= np.all([np.isfinite(v) for v in ang.values()], 0)
        ang = {a: v[ok] for a, v in ang.items()}
        rok = np.ones(len(r), bool) if lvl == "parton" else np.asarray(r.reco_ok)
        rang = {a: np.asarray(r[a], float)[rok] for a in ANGLE_NAMES}
        rang = {a: v[np.all([np.isfinite(x) for x in rang.values()], 0)] for a, v in rang.items()}
        f_all = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", f"f_ttbar_{lvl}.npz"))
        f_10 = f_all.truncated(f_all.significance() > 10)
        # mass factors (labelling-independent)
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
            fr_ = f(rang)
            s = surprisal(f(ang), fr_)
            scores[name] = s
            scores[name.replace("angles", "angles") + " + BW"] = s + sm
        scores["BW masses only"] = sm
        res = {"n": int(ok.sum()), "scores": {}}
        for fr in FR_VALUES:
            w = weights(cl[ok], ch[ok], fr)
            neff = float(np.sum(w) ** 2 / np.sum(w ** 2))
            entry = {"n_eff": neff, "frac_cosprime_gt_0.8_sm": float(np.mean(ang["cos_theta_prime"] > 0.8)),
                     "frac_cosprime_gt_0.8_bsm": float(np.sum(w * (ang["cos_theta_prime"] > 0.8)) / np.sum(w))}
            for name, s in scores.items():
                lab = np.r_[np.zeros(len(s)), np.ones(len(s))]
                sw = np.r_[np.ones(len(s)), w]
                entry[name] = {"auc": roc_auc_score(lab, np.r_[s, s], sample_weight=sw),
                               "eff_at_bkg_1pct": eff_at(s, w, s, 0.01), "eff_at_bkg_10pct": eff_at(s, w, s, 0.10)}
            res["scores"][f"F_R={fr}"] = entry
        out["levels"][lvl] = res
        print(lvl, "n", res["n"])
        for fr, entry in res["scores"].items():
            print(f"  {fr}: n_eff {entry['n_eff']:.0f}; cos θ'>0.8: SM {100 * entry['frac_cosprime_gt_0.8_sm']:.1f}% BSM {100 * entry['frac_cosprime_gt_0.8_bsm']:.1f}%")
            for name in scores:
                v = entry[name]
                print(f"     {name:28s} AUC {v['auc']:.3f}  eff@1% {100 * v['eff_at_bkg_1pct']:.1f}%  eff@10% {100 * v['eff_at_bkg_10pct']:.1f}%")
    json.dump(out, open(os.path.join(repo_paths.results_dir("study_bsm"), "wtb_reweight.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
