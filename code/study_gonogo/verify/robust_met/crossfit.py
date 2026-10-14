#!/usr/bin/env python3
"""Honest (cross-fitted) version of the truncation scan at reco (scratch).

The study's tt reference for f_tt is leave-one-out on the fit sample, but the truncation |c|/sigma > T selects terms
on the same events, so tt events score systematically 'more tt-like' than unseen events (the null tt_indep and the
no-spin control both show R > 1).  Here: K-fold cross-fit, repeated; in each fold the OSDE is fit on the other folds,
truncated with that fit's own significances, and scores the held-out tt events and the signal events assigned to that
fold; scores are averaged over the repetitions.  bw / mtt_dens / flow / window are taken from scan.py's saved scores.

Run: OMP_NUM_THREADS=4 .venv/bin/python code/study_gonogo/verify/robust_met/crossfit.py
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
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402
import repo_paths  # noqa: E402

OUT = repo_paths.results_dir("study_gonogo", "verify", "robust_met")   # also scan.py's and stop_nochi.py's outputs
TS = (0.0, 1.0, float(np.sqrt(2)), 2.0, 2.5, 3.0, 4.0, 5.0)
WPS = (0.005, 0.01, 0.02, 0.05, 0.10)
K, REPS, NBOOT = 5, 3, 300


def tname(T):
    return "all" if T == 0 else f"{T:.3g}"


def main():
    f0 = osde.load_density(os.path.join(repo_paths.RESULTS, A.F_FILES["reco"]))
    tt = A.load("tt_fit", "reco")
    D = {k: A.load(k, "reco") for k in ("etat", "stop", "H350", "nospin", "tt_indep")}
    z = np.load(os.path.join(OUT, "stop_nochi.npz"))
    D["stop_nochi"] = {"X": {a: z[f"reco/{a}"] for a in ANGLE_NAMES}, "m": {k: z[f"reco/{k}"] for k in A.MASSES},
                       "mtt": z["reco/mtt"]}
    n = len(tt["mtt"])
    S_tt = {tname(T): np.zeros(n) for T in TS}
    S = {k: {tname(T): np.zeros(len(d["mtt"])) for T in TS} for k, d in D.items()}
    nterms = {tname(T): [] for T in TS}
    for rep in range(REPS):
        rng = np.random.default_rng(1000 + rep)
        fold = rng.permutation(n) % K
        sfold = {k: rng.permutation(len(d["mtt"])) % K for k, d in D.items()}
        for f in range(K):
            tr = fold != f
            g = osde.TTbarDensity(int(f0.n_legendre), int(f0.n_hermite), float(f0.y_scale), int(f0.j1max),
                                  int(f0.j2max)).fit({a: v[tr] for a, v in tt["X"].items()})
            sig = g.significance()
            for T in TS:
                h = g if T == 0 else g.truncated(sig > T)
                nterms[tname(T)].append(int((np.abs(h.c) > 0).sum()))
                S_tt[tname(T)][~tr] += -h({a: v[~tr] for a, v in tt["X"].items()}) / REPS
                for k, d in D.items():
                    m = sfold[k] == f
                    S[k][tname(T)][m] += -h({a: v[m] for a, v in d["X"].items()}) / REPS
        print(f"rep {rep} done", flush=True)
    # comparison scores
    sv = np.load(os.path.join(OUT, "scores_reco.npz"))
    for m in ("bw", "mtt_dens", "flow"):
        S_tt[m] = sv[f"tt/{m}"]
        for k in ("etat", "stop", "H350", "nospin", "tt_indep"):
            S[k][m] = sv[f"{k}/{m}"]
    S["stop_nochi"]["bw"] = A.bw_score(tt, D["stop_nochi"])
    S["stop_nochi"]["mtt_dens"] = A.mtt_density_score(tt, D["stop_nochi"])
    lt = lambda ref, x: -np.log(A.tail_fraction(ref, x))  # noqa: E731
    for T in TS:
        t = tname(T)
        S_tt[f"{t}+bw"] = lt(S_tt[t], S_tt[t]) + lt(S_tt["bw"], S_tt["bw"])
        for k in D:
            S[k][f"{t}+bw"] = lt(S_tt[t], S[k][t]) + lt(S_tt["bw"], S[k]["bw"])
    out = {"nterms_mean": {t: float(np.mean(v)) for t, v in nterms.items()}, "signals": {}}
    print("terms kept (mean over folds):", out["nterms_mean"])
    rng = np.random.default_rng(9)
    for k, d in D.items():
        meths = [m for m in S[k] if m in S_tt]
        ns = len(d["mtt"])
        reps = {(m, w): [] for m in meths + ["mtt_window"] for w in WPS}
        for b in range(NBOOT + 1):
            ib = np.arange(n) if b == 0 else rng.integers(0, n, n)
            is_ = np.arange(ns) if b == 0 else rng.integers(0, ns, ns)
            for m in meths:
                cuts = np.quantile(S_tt[m][ib], [1 - w for w in WPS])
                for w, c in zip(WPS, cuts):
                    reps[(m, w)].append(np.mean(S[k][m][is_] > c) / w)
            for w in WPS:
                reps[("mtt_window", w)].append(A.window_eff(d["mtt"][is_], tt["mtt"][ib], w) / w)
        reps = {kk: np.array(v) for kk, v in reps.items()}
        res = {"R": {m: {f"{w:g}": [float(reps[(m, w)][0]), float(np.std(reps[(m, w)][1:]))] for w in WPS}
                     for m in meths + ["mtt_window"]}, "rule": {}}
        print(f"\n=== {k} (cross-fit, reco, n={ns}) R at " + " ".join(f"{w:g}" for w in WPS))
        for m in meths + ["mtt_window"]:
            print(f"  {m:10s} " + "  ".join(f"{reps[(m, w)][0]:5.2f}+-{np.std(reps[(m, w)][1:]):.2f}" for w in WPS))
        if k in ("etat", "stop", "H350") or (k == "stop_nochi"):
            passes = []
            for P in [tname(T) for T in TS] + [f"{tname(T)}+bw" for T in TS]:
                for w in WPS:
                    R = reps[(P, w)]
                    r0, sr = R[0], np.std(R[1:])
                    c = {"R": float(r0), "R_err": float(sr), "a": bool(r0 >= 2 and (r0 - 1) > 3 * sr)}
                    for m in ("bw", "mtt_dens", "mtt_window") + (("flow",) if "flow" in meths else ()):
                        dd = R - reps[(m, w)]
                        thr = -2 if m == "flow" else 2
                        ok = dd[0] > thr * np.std(dd[1:]) if m != "flow" else dd[0] > -2 * np.std(dd[1:])
                        c[m] = [float(dd[0]), float(np.std(dd[1:])), bool(ok)]
                    c["pass"] = bool(c["a"] and all(c[m][2] for m in ("bw", "mtt_dens", "mtt_window"))
                                     and (c["flow"][2] if "flow" in c else True))
                    res["rule"].setdefault(P, {})[f"{w:g}"] = c
                    if c["pass"]:
                        passes.append((P, f"{w:g}", round(c["R"], 2), round(c["R_err"], 2)))
            res["passes"] = passes
            print(f"  PASSING cells (cross-fit): {passes}")
        out["signals"][k] = res
    json.dump(out, open(os.path.join(OUT, "crossfit_reco.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
