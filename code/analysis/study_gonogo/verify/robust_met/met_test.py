#!/usr/bin/env python3
"""MET hypothesis for the stop at reco level (scratch).

R = eps_S/eps_tt at fixed eps_tt for simple MET handles (MET, m_T(l,MET), no real W-constrained neutrino p_z),
paired bootstrap against f_tt (CV-truncated, the primary detector); then conditional tests:
  - f_tt enrichment for the stop after reweighting tt to the stop's (MET, m_T) distribution
  - f_tt enrichment within m_T < m_W (real neutrino solution) and m_T > m_W
  - supervised BDT: (MET, m_T) vs angles vs angles + (MET, m_T)
Run: OMP_NUM_THREADS=4 .venv/bin/python code/study_gonogo/verify/robust_met/met_test.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "study_gonogo"))
import analyze as A  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
import repo_paths  # noqa: E402

OUT = repo_paths.results_dir("study_gonogo", "verify", "robust_met")   # also extract_met.py's met_<key>.npz
WPS = (0.005, 0.01, 0.02, 0.05, 0.10)
NBOOT = 300


def lt(ref, x):
    return -np.log(A.tail_fraction(ref, x))


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(cw, q)]


def main():
    f_all = osde.load_density(os.path.join(repo_paths.RESULTS, A.F_FILES["reco"]))
    f_cv = f_all.truncated(f_all.significance() > A.T_CV)
    tt = A.load("tt_fit", "reco")
    keys = [k for k in ("stop", "etat", "H350", "tt_indep") if os.path.exists(os.path.join(OUT, f"met_{k}.npz"))]
    Mtt = dict(np.load(os.path.join(OUT, "met_tt_fit.npz")))
    assert bool(Mtt["aligned"])
    assert np.allclose(Mtt["ang/Y"], tt["X"]["Y"])

    def handles(M, d, is_tt):
        h = {"met": M["met"], "mtw": M["mtw"], "nu_noreal": -M["nu_disc"]}
        h["f_cv"] = -f_cv.loo(d["X"]) if is_tt else -f_cv(d["X"])
        h["bw"] = A.bw_score(tt, d)
        return h

    H_tt = handles(Mtt, tt, True)
    # combinations (sum of -ln tt tail fractions)
    def combos(h):
        h["met+mtw"] = lt(H_tt["met"], h["met"]) + lt(H_tt["mtw"], h["mtw"])
        h["f_cv+bw"] = lt(H_tt["f_cv"], h["f_cv"]) + lt(H_tt["bw"], h["bw"])
        h["bw+met+mtw"] = lt(H_tt["bw"], h["bw"]) + h["met+mtw"]
        h["f_cv+met+mtw"] = lt(H_tt["f_cv"], h["f_cv"]) + h["met+mtw"]
        return h
    H_tt = combos(H_tt)
    out = {"frac_noreal": {"tt_fit": float(np.mean(Mtt["nu_disc"] < 0))},
           "check_noreal_equiv_mtw_gt_mW": float(np.mean((Mtt["nu_disc"] < 0) == (Mtt["mtw"] > 80.4)))}
    print("tt: frac(no real nu pz) = %.3f; agreement of (disc<0) with (m_T > m_W): %.4f"
          % (out["frac_noreal"]["tt_fit"], out["check_noreal_equiv_mtw_gt_mW"]))
    print("tt: MET quantiles 50/90/99%%: %s; m_T: %s" % (np.percentile(Mtt["met"], [50, 90, 99]).round(1),
                                                       np.percentile(Mtt["mtw"], [50, 90, 99]).round(1)))
    rng = np.random.default_rng(7)
    nb = len(tt["mtt"])
    for k in keys:
        M = dict(np.load(os.path.join(OUT, f"met_{k}.npz")))
        d = A.load(k, "reco")
        assert bool(M["aligned"]) and np.allclose(M["ang/Y"], d["X"]["Y"])
        H = combos(handles(M, d, False))
        ns = len(d["mtt"])
        names = list(H)
        reps = {(m, wp): [] for m in names for wp in WPS}
        for b in range(NBOOT + 1):
            ib = np.arange(nb) if b == 0 else rng.integers(0, nb, nb)
            is_ = np.arange(ns) if b == 0 else rng.integers(0, ns, ns)
            for m in names:
                cuts = np.quantile(H_tt[m][ib], [1 - w for w in WPS])
                for wp, c in zip(WPS, cuts):
                    reps[(m, wp)].append(np.mean(H[m][is_] > c) / wp)
        reps = {kk: np.array(v) for kk, v in reps.items()}
        r = {"n": ns, "frac_noreal": float(np.mean(M["nu_disc"] < 0)),
             "R_noreal_cut": float(np.mean(M["nu_disc"] < 0) / np.mean(Mtt["nu_disc"] < 0)),
             "eps_tt_noreal_cut": float(np.mean(Mtt["nu_disc"] < 0)), "R": {}, "minus_f_cv": {}}
        for m in names:
            r["R"][m] = {f"{wp:g}": [float(reps[(m, wp)][0]), float(np.std(reps[(m, wp)][1:]))] for wp in WPS}
            dd = {f"{wp:g}": (reps[(m, wp)] - reps[("f_cv", wp)]) for wp in WPS}
            r["minus_f_cv"][m] = {w: [float(v[0]), float(np.std(v[1:]))] for w, v in dd.items()}
        print(f"\n=== {k} reco n={ns}: frac no-real-nu {r['frac_noreal']:.3f} (tt {r['eps_tt_noreal_cut']:.3f}),"
              f" R of that cut {r['R_noreal_cut']:.2f}")
        print("   handle         " + "  ".join(f"R@{w:g}".rjust(12) for w in WPS) + "   | minus f_cv @1%, @10%")
        for m in names:
            print(f"   {m:14s} " + "  ".join(f"{r['R'][m][f'{w:g}'][0]:6.2f}+-{r['R'][m][f'{w:g}'][1]:.2f}" for w in WPS)
                  + f"   | {r['minus_f_cv'][m]['0.01'][0]:+.2f}+-{r['minus_f_cv'][m]['0.01'][1]:.2f}, "
                    f"{r['minus_f_cv'][m]['0.1'][0]:+.2f}+-{r['minus_f_cv'][m]['0.1'][1]:.2f}")
        # ---- conditional: reweight tt to the signal's (MET, m_T) distribution, then R of f_cv
        qe = np.quantile(Mtt["met"], np.linspace(0, 1, 11))
        qm = np.quantile(Mtt["mtw"], np.linspace(0, 1, 11))
        qe[0], qe[-1], qm[0], qm[-1] = -np.inf, np.inf, -np.inf, np.inf
        bt = (np.digitize(Mtt["met"], qe[1:-1]) * 10 + np.digitize(Mtt["mtw"], qm[1:-1]))
        bs = (np.digitize(M["met"], qe[1:-1]) * 10 + np.digitize(M["mtw"], qm[1:-1]))
        ht = np.bincount(bt, minlength=100) / len(bt)
        hs = np.bincount(bs, minlength=100) / len(bs)
        w = np.where(ht[bt] > 0, hs[bt] / np.maximum(ht[bt], 1e-12), 0.0)
        cond = {}
        for wp in WPS:
            c_un = np.quantile(H_tt["f_cv"], 1 - wp)
            c_rw = wquantile(H_tt["f_cv"], w, 1 - wp)
            es = np.mean(H["f_cv"] > c_rw)
            cond[f"{wp:g}"] = {"R_unweighted": float(np.mean(H["f_cv"] > c_un) / wp),
                               "R_tt_reweighted_to_signal_met_mtw": float(es / wp)}
        # bootstrap error for the reweighted 1% and 10%
        for wp in (0.01, 0.1):
            vals = []
            for b in range(200):
                ib = rng.integers(0, nb, nb)
                is_ = rng.integers(0, ns, ns)
                vals.append(np.mean(H["f_cv"][is_] > wquantile(H_tt["f_cv"][ib], w[ib], 1 - wp)) / wp)
            cond[f"{wp:g}"]["err"] = float(np.std(vals))
        r["conditional_on_met_mtw"] = cond
        print("   f_cv R after reweighting tt to the signal's (MET, m_T) [10x10 tt-quantile bins]: " +
              "; ".join(f"{w}: {v['R_unweighted']:.2f} -> {v['R_tt_reweighted_to_signal_met_mtw']:.2f}"
                        + (f"+-{v['err']:.2f}" if 'err' in v else "") for w, v in cond.items()))
        # ---- within the real / non-real neutrino categories (tt quantile from the same category)
        cat = {}
        for name, mt, ms in (("mT<mW", Mtt["mtw"] <= 80.4, M["mtw"] <= 80.4), ("mT>mW", Mtt["mtw"] > 80.4, M["mtw"] > 80.4)):
            cat[name] = {"frac_tt": float(mt.mean()), "frac_sig": float(ms.mean())}
            for wp in (0.01, 0.1):
                c = np.quantile(H_tt["f_cv"][mt], 1 - wp)
                cat[name][f"R@{wp:g}"] = float(np.mean(H["f_cv"][ms] > c) / wp)
                cat[name][f"R@{wp:g}_err_stat"] = float(np.sqrt(np.mean(H["f_cv"][ms] > c) * ms.sum()) / ms.sum() / wp)
        r["by_nu_category"] = cat
        print("   f_cv R within categories: " + "; ".join(
            f"{n}: tt frac {v['frac_tt']:.2f}, sig frac {v['frac_sig']:.2f}, R@1% {v['R@0.01']:.2f}+-{v['R@0.01_err_stat']:.2f}, "
            f"R@10% {v['R@0.1']:.2f}" for n, v in cat.items()))
        # ---- f_cv tail composition
        c1 = np.quantile(H_tt["f_cv"], 0.99)
        tail_s, tail_t = H["f_cv"] > c1, H_tt["f_cv"] > c1
        r["tail_composition_1pct"] = {"sig_frac_mT>mW_in_tail": float(np.mean(M["mtw"][tail_s] > 80.4)),
                                      "sig_frac_mT>mW_all": float(np.mean(M["mtw"] > 80.4)),
                                      "tt_frac_mT>mW_in_tail": float(np.mean(Mtt["mtw"][tail_t] > 80.4)),
                                      "tt_frac_mT>mW_all": float(np.mean(Mtt["mtw"] > 80.4)),
                                      "sig_median_met_tail": float(np.median(M["met"][tail_s])),
                                      "sig_median_met_all": float(np.median(M["met"])),
                                      "tt_median_met_tail": float(np.median(Mtt["met"][tail_t])),
                                      "tt_median_met_all": float(np.median(Mtt["met"]))}
        print("   f_cv 1% tail composition: " + ", ".join(f"{a} {b:.3g}" for a, b in r["tail_composition_1pct"].items()))
        # Spearman correlation of f_cv score with MET / m_T
        from scipy.stats import spearmanr
        r["spearman"] = {"sig_fcv_met": float(spearmanr(H["f_cv"], M["met"])[0]),
                         "sig_fcv_mtw": float(spearmanr(H["f_cv"], M["mtw"])[0]),
                         "tt_fcv_met": float(spearmanr(H_tt["f_cv"], Mtt["met"])[0]),
                         "tt_fcv_mtw": float(spearmanr(H_tt["f_cv"], Mtt["mtw"])[0])}
        print("   Spearman(f_cv score, .): " + ", ".join(f"{a} {b:+.3f}" for a, b in r["spearman"].items()))
        # ---- supervised: which inputs carry the separation
        if k == "stop":
            Xa = lambda dd: np.column_stack([dd["X"][x] for x in ANGLE_NAMES])  # noqa: E731
            F = {"met,mtw": (np.column_stack([Mtt["met"], Mtt["mtw"]]), np.column_stack([M["met"], M["mtw"]])),
                 "angles": (Xa(tt), Xa(d)),
                 "angles+met,mtw": (np.column_stack([Xa(tt), Mtt["met"], Mtt["mtw"]]),
                                    np.column_stack([Xa(d), M["met"], M["mtw"]])),
                 "masses+mtt": (np.column_stack([tt["m"][x] for x in A.MASSES] + [tt["mtt"]]),
                                np.column_stack([d["m"][x] for x in A.MASSES] + [d["mtt"]])),
                 "masses+mtt+met,mtw": (np.column_stack([tt["m"][x] for x in A.MASSES] + [tt["mtt"], Mtt["met"], Mtt["mtw"]]),
                                        np.column_stack([d["m"][x] for x in A.MASSES] + [d["mtt"], M["met"], M["mtw"]])),
                 "all (angles+masses+mtt+met,mtw)": (
                     np.column_stack([Xa(tt)] + [tt["m"][x] for x in A.MASSES] + [tt["mtt"], Mtt["met"], Mtt["mtw"]]),
                     np.column_stack([Xa(d)] + [d["m"][x] for x in A.MASSES] + [d["mtt"], M["met"], M["mtw"]]))}
            bd = {}
            for name, (Ft, Fs) in F.items():
                st, ss = A.bdt_scores(Ft, Fs)
                bd[name] = {f"{wp:g}": float(np.mean(ss > np.quantile(st, 1 - wp)) / wp) for wp in WPS}
                bd[name]["auc"] = A.auc(ss, st)
                print(f"   BDT {name:32s} AUC {bd[name]['auc']:.3f}  " +
                      "  ".join(f"R@{wp:g} {bd[name][f'{wp:g}']:.2f}" for wp in WPS), flush=True)
            r["bdt"] = bd
        out[k] = r
    json.dump(out, open(os.path.join(OUT, "met_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
