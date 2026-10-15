#!/usr/bin/env python3
"""
One estimator for every number in the paper: f_tt̄ truncated to |ĉ|/σ > T (T = 2.5, the 5-fold least-squares
cross-validation optimum on tt̄ alone at every level), with the tt̄ events scored by 5-fold cross-fitting.

For each level (parton_trig, gen, reco = chi2 jets, matched = ΔR-matched jets) the tt̄ fit sample is split into five
folds (event i -> fold i % 5, as in osde.fold_sums).  Fold model k is fitted on the other four folds, its terms are
selected on those four folds (|ĉ|/σ > T, constant always kept), and it scores
  * the tt̄ events of fold k (out of fold), and
  * the signal events assigned to fold k (signal event j -> fold j % 5),
so tt̄ and signal see the same model noise.  The all-terms cross-fit is kept as a cross-check, and the full-sample
truncated fit scores the independent tt̄ sample.  At the matched level the chi2-assignment angles of the same matched
events are scored with the reco-level fold models (tt̄: the fold of the event in the reco fit sample).

Writes results/study_cv/scores_<level>.npz and results/study_cv/cv_summary.json.
Run: OMP_NUM_THREADS=8 .venv/bin/python code/analysis/study_cv/cv_scores.py [--levels ...] [--t 2.5]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, CODE)
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
import repo_paths  # noqa: E402

K = 5
OUT = repo_paths.results_dir("study_cv")
CACHE = repo_paths.results_dir("study_gonogo", "cache", create=False)
TM = repo_paths.results_dir("study_truthmatch", create=False)
F_FILES = {"parton_trig": "osde/f_ttbar_parton_trig.npz", "gen": "osde/f_ttbar_gen.npz", "reco": "osde/f_ttbar_reco.npz",
           "matched": "study_truthmatch/osde_dRmatch/f_ttbar_reco.npz"}  # under results/
MASSES = ("m_w_had", "m_t_lep", "m_t_had")


def cached(key, level):
    z = np.load(os.path.join(CACHE, f"{key}.npz"))
    return {a: z[f"{level}/{a}"] for a in ANGLE_NAMES}, {m: z[f"{level}/{m}"] for m in MASSES + ("mtt",)}


def matched(run):
    z = np.load(os.path.join(TM, f"data_{run}.npz"))
    ok = z["ok"]
    return z, ok


def fold_models(f_all, X, t):
    """Five out-of-fold models (truncated and all-terms) from the per-fold sums of the fit sample X."""
    s1, s2, counts = f_all.fold_sums(X, k=K)
    tot1, tot2, ntot = s1.sum(0), s2.sum(0), counts.sum()
    # the five folds together must reproduce the stored full-sample fit (same basis, same events)
    assert int(ntot) == f_all.n_fit and np.allclose(tot1 / ntot, f_all.c, atol=1e-12, rtol=1e-9), "fold sums != stored fit"
    models = []
    for k in range(K):
        n = ntot - counts[k]
        c = (tot1 - s1[k]) / n
        err = np.sqrt(np.clip((tot2 - s2[k]) / n - abs(c) ** 2, 0, None) / n)
        sig = np.abs(c) / np.where(err > 0, err, np.inf)
        sig.flat[0] = np.inf
        keep = sig > t
        g_t, g_a = f_all.truncated(keep), f_all.truncated(np.ones_like(keep))
        g_t.c, g_a.c = np.where(keep, c, 0), c
        g_t.c_err = g_a.c_err = err
        g_t.n_fit = g_a.n_fit = int(n)
        models.append((g_t, g_a, int(keep.sum())))
    return models


def score(models, X, folds):
    """Scores of events X with the model of their fold: (truncated, all terms)."""
    yt, ya = np.zeros(len(folds)), np.zeros(len(folds))
    for k, (g_t, g_a, _) in enumerate(models):
        m = folds == k
        if m.any():
            Xk = {a: v[m] for a, v in X.items()}
            yt[m], ya[m] = g_t(Xk), g_a(Xk)
    return yt, ya


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", default="parton_trig,gen,reco,matched")
    p.add_argument("--t", type=float, default=2.5)
    a = p.parse_args()
    summ_path = os.path.join(OUT, "cv_summary.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    summ["threshold"] = a.t
    reco_models = None
    for level in a.levels.split(","):
        f_all = osde.load_density(os.path.join(repo_paths.RESULTS, F_FILES[level]))
        out = {}
        if level != "matched":
            Xtt, Mtt = cached("tt_fit", level)
            sigs = {"hh125": cached("hh125", level), "HH170": cached("HH170_all", level)}
            Xind, _ = cached("tt_indep", level)
        else:
            z, ok = matched("run_seed1001")
            Xtt = {x: z[f"m_{x}"][ok] for x in ANGLE_NAMES}
            Mtt = {m: z[f"m_{m}"][ok] for m in MASSES + ("mtt",) if f"m_{m}" in z}
            zh, okh = matched("run_m170_all")
            sigs = {"HH170": ({x: zh[f"m_{x}"][okh] for x in ANGLE_NAMES}, {m: zh[f"m_{m}"][okh] for m in MASSES + ("mtt",) if f"m_{m}" in zh})}
            z1, ok1 = matched("run_seed2001")                                 # hh(125), for the pooled upsilon scale
            sigs["hh125"] = ({x: z1[f"m_{x}"][ok1] for x in ANGLE_NAMES}, {m: z1[f"m_{m}"][ok1] for m in MASSES if f"m_{m}" in z1})
            Xind = None
        n = len(Xtt["Y"])
        models = fold_models(f_all, Xtt, a.t)
        fold_tt = np.arange(n) % K
        out["tt_cv"], out["tt_all_cv"] = score(models, Xtt, fold_tt)
        out["tt_all_loo"] = f_all.loo(Xtt)                                     # the old estimator, for reference
        for m, v in Mtt.items():
            out[f"tt_{m}"] = v
        for key, (Xs, Ms) in sigs.items():
            fs = np.arange(len(Xs["Y"])) % K
            out[f"{key}_cv"], out[f"{key}_all_cv"] = score(models, Xs, fs)
            out[f"{key}_all_full"] = f_all(Xs)                                 # the old estimator, for reference
            for m, v in Ms.items():
                out[f"{key}_{m}"] = v
        f_full = f_all.truncated(f_all.significance() > a.t)                   # full-sample truncated fit
        if Xind is not None:
            out["tt_indep_full"] = f_full(Xind)
        if level == "reco":
            reco_models = models
        if level == "matched":
            # chi2 assignment on the same matched events, scored with the reco-level fold models
            if reco_models is None:
                f_reco = osde.load_density(os.path.join(repo_paths.RESULTS, F_FILES["reco"]))
                Xr, _ = cached("tt_fit", "reco")
                reco_models = fold_models(f_reco, Xr, a.t)
            chi_ok = z["chi_ok"]
            pos = np.cumsum(chi_ok) - 1                                        # position in the reco fit sample
            both = ok & chi_ok
            Xc = {x: z[f"c_{x}"][both] for x in ANGLE_NAMES}
            out["tt_chi2_cv"], out["tt_chi2_all_cv"] = score(reco_models, Xc, pos[both] % K)
            out["tt_chi2_in_matched"] = both[ok]                                # which matched tt events have a chi2 reco
            bh = okh & zh["chi_ok"]
            Xch = {x: zh[f"c_{x}"][bh] for x in ANGLE_NAMES}
            out["HH170_chi2_cv"], out["HH170_chi2_all_cv"] = score(reco_models, Xch, np.arange(bh.sum()) % K)
        np.savez_compressed(os.path.join(OUT, f"scores_{level}.npz"), **out)
        summ[level] = {"n_tt": n, "kept_per_fold": [m[2] for m in models], "kept_full": int((f_full.c != 0).sum()),
                       **{f"n_{k}": int(len(v[0]["Y"])) for k, v in sigs.items()}}
        print(level, json.dumps(summ[level]), flush=True)
        json.dump(summ, open(summ_path, "w"), indent=1)


if __name__ == "__main__":
    main()
