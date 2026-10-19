#!/usr/bin/env python3
"""
Scores with the tt̄ density as the prior over the candidate list (the study_vae idea, first test: a fixed encoder).

For each selected event, candidates i = 1..K (candidates.py) carry the ten angles z_i and a chi2_i.  The encoder
weights are w_i = exp(-chi2_i / 2T) / sum (T = 1 is the chi2 likelihood; larger T flattens it).  The prior is the
tt̄ density f, shifted so it is positive: y' = f + y0, with y0 the magnitude of the lowest f on the tt̄ fit sample
(rounded up, as for upsilon in the paper); the few candidates still below zero are floored at y0/100.  Shifting adds
a constant to f, so without a finite volume (Y is unbounded) the shifted f is not renormalized: scores move by a
constant, which does not change any ranking or AUC.  Surprisal s = -ln y'.  Scores (high = less tt̄-like):

  best      s of the lowest-chi2 candidate (the paper's choice of assignment)
  Eq_s      E_q[s] = sum_i w_i s_i          (the cross-entropy term of the latent KL)
  kl        sum_i w_i ln w_i + E_q[s]       (the latent KL with a discrete q)
  marginal  -ln sum_i w_i y'_i              (the likelihood averaged over the candidates)

with two priors: f_parton (parton level after the trigger: the physical latent space) and f_reco (fit to the chi2
reco angles, the paper's reco-level density).  AUC of each signal against the independent tt̄ test sample (run_seed12345;
the densities are fit on run_seed1001), with bootstrap errors; tt_pu0 (50k, no pileup, ATLAS PU0 card) is a cross-check.

Writes results/analysis/study_vae/scores.json and scores.md.
Run: OMP_NUM_THREADS=6 .venv/bin/python code/analysis/study_vae/score.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CODE)
import repo_paths  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

T_CUT = 2.5                                   # |c|/sigma threshold of the paper
PRIORS = {"parton": ("osde/f_ttbar_parton_trig.npz", "parton_trig"), "reco": ("osde/f_ttbar_reco.npz", "reco")}
TEMPS = (1.0, 2.0, 4.0, np.inf)
OUT = repo_paths.results_dir("study_vae")


def shift_y0(y):
    """Magnitude of the lowest value, rounded up to two significant figures (make_paper_figs.npad_upsilon)."""
    low = float(np.min(y))
    if low >= 0:
        return 0.0
    tenth = 10.0 ** (np.floor(np.log10(-low)) - 1)
    return float(np.ceil(-low / tenth) * tenth)


def auc(bkg, sig):
    """P(score_sig > score_bkg), ties counted half."""
    x = np.r_[sig, bkg]
    r = x.argsort(kind="stable").argsort() + 1.0
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)       # average ranks for ties
    first = np.cumsum(cnt) - cnt
    r = (first + (cnt + 1) / 2.0)[inv]
    return (r[:len(sig)].sum() - len(sig) * (len(sig) + 1) / 2) / (len(sig) * len(bkg))


def auc_err(bkg, sig, n=200, seed=3):
    rng = np.random.default_rng(seed)
    return float(np.std([auc(bkg[rng.integers(0, len(bkg), len(bkg))], sig[rng.integers(0, len(sig), len(sig))])
                         for _ in range(n)]))


def scores(cand, f, y0, temp):
    ok = cand["reco_ok"]
    chi2 = cand["chi2"][ok]
    X = {a: cand[a][ok] for a in ANGLE_NAMES}
    valid = np.isfinite(chi2) & np.all([np.isfinite(X[a]) for a in ANGLE_NAMES], axis=0)
    y = np.full(chi2.shape, np.nan)
    y[valid] = f({a: X[a][valid] for a in ANGLE_NAMES})
    yp = y + y0
    n_floor = int(np.sum(valid & (yp <= 0)))
    yp = np.where(valid, np.maximum(yp, y0 / 100.0), np.nan)
    s = -np.log(yp)
    if np.isinf(temp):
        logw = np.where(valid, 0.0, -np.inf)
    else:
        logw = np.where(valid, -(chi2 - np.nanmin(np.where(valid, chi2, np.nan), axis=1, keepdims=True)) / (2 * temp), -np.inf)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True)
    ws = np.where(valid, w * s, 0.0)
    out = {"best": s[:, 0],
           "Eq_s": ws.sum(axis=1),
           "kl": np.where(valid, w * np.log(np.where(w > 0, w, 1.0)), 0.0).sum(axis=1) + ws.sum(axis=1),
           "marginal": -np.log(np.where(valid, w * yp, 0.0).sum(axis=1))}
    return out, n_floor, valid.sum(axis=1)


def main():
    cands = {k: dict(np.load(os.path.join(OUT, f"cand_{k}.npz"))) for k in ("tt_test", "tt_pu0", "HH170", "hh125")}
    cache = np.load(repo_paths.results("study_gonogo", "cache", "tt_fit.npz"))
    res = {"n_selected": {k: int(c["reco_ok"].sum()) for k, c in cands.items()},
           "n_candidates_mean": {}, "priors": {}, "auc": {}}
    for pname, (ffile, level) in PRIORS.items():
        f_all = osde.load_density(repo_paths.results(ffile))
        f = f_all.truncated(f_all.significance() > T_CUT)
        y_ref = f({a: cache[f"{level}/{a}"] for a in ANGLE_NAMES})
        y0 = shift_y0(y_ref)
        res["priors"][pname] = {"file": ffile, "kept_terms": int((f.c != 0).sum()), "y0": y0}
        for temp in TEMPS:
            tkey = "inf" if np.isinf(temp) else f"{temp:g}"
            sc = {}
            for k, c in cands.items():
                sc[k], nfl, nval = scores(c, f, y0, temp)
                res["priors"][pname].setdefault("n_floored", {})[k] = nfl
                res["n_candidates_mean"][k] = float(np.mean(nval))
            for sname in sc["tt_test"]:
                if temp != 1.0 and sname == "best":
                    continue                                   # does not depend on the weights
                for sig in ("HH170", "hh125"):
                    for bk in ("tt_test", "tt_pu0"):
                        b, s_ = sc[bk][sname], sc[sig][sname]
                        key = f"{pname}/{sname}/T={tkey}/{sig}_vs_{bk}"
                        res["auc"][key] = [round(auc(b, s_), 4), round(auc_err(b, s_), 4)]
        print(pname, json.dumps(res["priors"][pname]), flush=True)
    json.dump(res, open(os.path.join(OUT, "scores.json"), "w"), indent=1)

    lines = ["# study_vae: tt̄ density as the prior over the candidate list (fixed chi2 encoder)", "",
             f"Selected events: {res['n_selected']}; mean candidates per event: "
             f"{ {k: round(v, 2) for k, v in res['n_candidates_mean'].items()} }", "",
             "AUC against the independent tt̄ test sample (run_seed12345); in brackets against tt_pu0 (50k).", "",
             "| prior | score | T | HH170 | hh125 |", "|---|---|---|---|---|"]
    for pname in PRIORS:
        for sname in ("best", "Eq_s", "kl", "marginal"):
            for temp in TEMPS:
                tkey = "inf" if np.isinf(temp) else f"{temp:g}"
                if sname == "best" and temp != 1.0:
                    continue
                cells = []
                for sig in ("HH170", "hh125"):
                    a1, e1 = res["auc"][f"{pname}/{sname}/T={tkey}/{sig}_vs_tt_test"]
                    a2, _ = res["auc"][f"{pname}/{sname}/T={tkey}/{sig}_vs_tt_pu0"]
                    cells.append(f"{a1:.3f} ± {e1:.3f} ({a2:.3f})")
                lines.append(f"| {pname} | {sname} | {'-' if sname == 'best' else tkey} | {cells[0]} | {cells[1]} |")
    open(os.path.join(OUT, "scores.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
