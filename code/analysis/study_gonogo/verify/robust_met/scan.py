#!/usr/bin/env python3
"""Adversarial robustness scan of the go/no-go rule (scratch; does not touch the study outputs).

Variants: truncation threshold T of f_tt (0 = all terms, 1, sqrt2 (Kronmal-Tarter), 2, 2.5 (primary), 3, 4, 5),
working points 0.5%, 1%, 2%, 5%, 10%, and f_T + BW (sum of -ln tail fractions) as a primary detector.
The rule of CRITERIA.md is applied to every (signal, T, wp, f or f+BW) cell, with paired bootstrap.
Also: f_T+BW minus BW (do angles add to the masses?), and the tt_indep null for each variant.

Run: OMP_NUM_THREADS=4 .venv/bin/python code/analysis/study_gonogo/verify/robust_met/scan.py [--level reco] [--nboot 300]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # code/
sys.path.insert(0, ROOT)
from analysis.study_gonogo import analyze as A  # noqa: E402
from lib import osde  # noqa: E402
import repo_paths  # noqa: E402

OUT = repo_paths.results_dir("study_gonogo", "verify", "robust_met")
TS = (0.0, 1.0, float(np.sqrt(2)), 2.0, 2.5, 3.0, 4.0, 5.0)
WPS = (0.005, 0.01, 0.02, 0.05, 0.10)
SIGS = ("etat", "stop", "H350", "nospin", "tt_indep")


def tname(T):
    return "all" if T == 0 else f"{T:.3g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="reco")
    p.add_argument("--nboot", type=int, default=300)
    p.add_argument("--seeds", default="1,2,3")
    a = p.parse_args()
    torch.set_num_threads(4)
    level = a.level
    f_all = osde.load_density(repo_paths.results(A.F_FILES[level]))
    sig = f_all.significance()
    tt = A.load([A.TT_RUN], level) if level == "matched" else A.load("tt_fit", level)
    assert len(tt["mtt"]) == f_all.n_fit
    D = {k: A.sample(k, level) for k in SIGS}
    S_tt, S = {}, {k: {} for k in SIGS}
    nterms = {}
    for T in TS:
        f = f_all if T == 0 else f_all.truncated(sig > T)
        nterms[tname(T)] = int((np.abs(f.c) > 0).sum())
        S_tt[f"f_{tname(T)}"] = -f.loo(tt["X"])
        for k, d in D.items():
            S[k][f"f_{tname(T)}"] = -f(d["X"])
    S_tt["bw"] = A.bw_score(tt, tt)
    S_tt["mtt_dens"] = A.mtt_density_score(tt, tt)
    for k, d in D.items():
        S[k]["bw"] = A.bw_score(tt, d)
        S[k]["mtt_dens"] = A.mtt_density_score(tt, d)
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[{level}] training flows, seeds {seeds}", flush=True)
    ft, fs = A.flow_scores(tt["X"], {k: d["X"] for k, d in D.items()}, seeds)
    S_tt["flow"] = ft
    for k in D:
        S[k]["flow"] = fs[k]
    lt = lambda ref, x: -np.log(A.tail_fraction(ref, x))  # noqa: E731
    for T in TS:
        n = f"f_{tname(T)}"
        S_tt[f"{n}+bw"] = lt(S_tt[n], S_tt[n]) + lt(S_tt["bw"], S_tt["bw"])
        for k in D:
            S[k][f"{n}+bw"] = lt(S_tt[n], S[k][n]) + lt(S_tt["bw"], S[k]["bw"])
    methods = list(S_tt)

    rng = np.random.default_rng(5)
    nb = len(tt["mtt"])
    out = {"level": level, "n_tt": nb, "nterms": nterms, "wps": WPS, "signals": {}}
    for k, d in D.items():
        ns = len(d["mtt"])
        reps = {(m, wp): [] for m in methods + ["mtt_window"] for wp in WPS}
        for b in range(a.nboot + 1):
            ib = np.arange(nb) if b == 0 else rng.integers(0, nb, nb)
            is_ = np.arange(ns) if b == 0 else rng.integers(0, ns, ns)
            for m in methods:
                sb, ss = S_tt[m][ib], S[k][m][is_]
                cuts = np.quantile(sb, [1 - wp for wp in WPS])
                for wp, c in zip(WPS, cuts):
                    reps[(m, wp)].append(np.mean(ss > c) / wp)
            for wp in WPS:
                reps[("mtt_window", wp)].append(A.window_eff(d["mtt"][is_], tt["mtt"][ib], wp) / wp)
        reps = {kk: np.array(v) for kk, v in reps.items()}
        res = {"n": ns, "R": {}, "rule": {}, "combo_minus_bw": {}}
        for m in methods + ["mtt_window"]:
            res["R"][m] = {f"{wp:g}": [float(reps[(m, wp)][0]), float(np.std(reps[(m, wp)][1:]))] for wp in WPS}
        prim = [f"f_{tname(T)}" for T in TS] + [f"f_{tname(T)}+bw" for T in TS]
        for P in prim:
            res["rule"][P] = {}
            for wp in WPS:
                R = reps[(P, wp)]
                r0, sr = R[0], np.std(R[1:])
                c = {"R": float(r0), "R_err": float(sr), "a": bool(r0 >= 2 and (r0 - 1) > 3 * sr)}
                for m in ("bw", "mtt_dens", "mtt_window"):
                    dd = R - reps[(m, wp)]
                    c[f"b_{m}"] = [float(dd[0]), float(np.std(dd[1:])), bool(dd[0] > 2 * np.std(dd[1:]))]
                dd = R - reps[("flow", wp)]
                c["c_flow"] = [float(dd[0]), float(np.std(dd[1:])), bool(dd[0] > -2 * np.std(dd[1:]))]
                c["pass"] = bool(c["a"] and all(c[f"b_{m}"][2] for m in ("bw", "mtt_dens", "mtt_window")) and c["c_flow"][2])
                res["rule"][P][f"{wp:g}"] = c
        for T in TS:
            n = f"f_{tname(T)}"
            res["combo_minus_bw"][n] = {}
            for wp in WPS:
                dd = reps[(f"{n}+bw", wp)] - reps[("bw", wp)]
                res["combo_minus_bw"][n][f"{wp:g}"] = [float(dd[0]), float(np.std(dd[1:]))]
        out["signals"][k] = res
        print(f"  {k}: done", flush=True)
    json.dump(out, open(os.path.join(OUT, f"scan_{level}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"scores_{level}.npz"),
                        **{f"tt/{m}": v for m, v in S_tt.items()},
                        **{f"{k}/{m}": v for k in S for m, v in S[k].items()})
    # summary
    print(f"\nterms kept: {nterms}")
    for k in SIGS:
        r = out["signals"][k]
        print(f"\n=== {k} ({level}, n={r['n']}) R at wp " + " ".join(f"{w:g}" for w in WPS))
        for m in methods + ["mtt_window"]:
            print(f"  {m:12s} " + "  ".join(f"{r['R'][m][f'{w:g}'][0]:6.2f}+-{r['R'][m][f'{w:g}'][1]:.2f}" for w in WPS))
        if k in A.PHYSICAL:
            passes = [(P, w) for P, v in r["rule"].items() for w, c in v.items() if c["pass"]]
            print(f"  PASSING cells: {passes}")
            for P, v in r["rule"].items():
                for w, c in v.items():
                    if c["R"] >= 1.8:
                        print(f"   near/at: {P:10s} wp {w:5s} R {c['R']:.2f}+-{c['R_err']:.2f} a={c['a']} "
                              f"bw {c['b_bw'][0]:+.2f}+-{c['b_bw'][1]:.2f} mttd {c['b_mtt_dens'][0]:+.2f}+-{c['b_mtt_dens'][1]:.2f} "
                              f"win {c['b_mtt_window'][0]:+.2f}+-{c['b_mtt_window'][1]:.2f} flow {c['c_flow'][0]:+.2f}+-{c['c_flow'][1]:.2f} "
                              f"pass={c['pass']}")
            print("  f_T+bw minus bw: " + "; ".join(
                f"{n}: " + ",".join(f"{v[f'{w:g}'][0]:+.2f}({v[f'{w:g}'][1]:.2f})" for w in WPS)
                for n, v in r["combo_minus_bw"].items()))


if __name__ == "__main__":
    main()
