#!/usr/bin/env python3
"""Why does a 23-term f beat the 14,400-term f?  Test the estimation-noise explanation.

Each ĉ_k = mean of B_k over N fit events has noise σ_k ~ 1/sqrt(N); terms whose true value is ~0
contribute only noise, and the sum over thousands of them puts a random wobble on f̂(x) that is
unrelated to tt̄ vs hh.  Predictions if this is right:
  (1) the gain from truncation, AUC(trunc) - AUC(all), shrinks as N grows;
  (2) two fits on disjoint subsamples of the same size agree on the truncated f but not on the full f.
Fit on N parton-level tt̄ events (disjoint subsamples of the 100k fit sample), score the independent
10k tt̄ sample and 20k hh (no leave-one-out needed).  Paired bootstrap for the AUC differences.
Output: results/study_salvage/truncation_noise.json, truncation_noise.png
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
ROOT = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, ROOT)
import repo_paths  # noqa: E402
from lib import osde, style  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

LVL = "parton"
fit_all = np.load(os.path.join(repo_paths.SAMPLES, "ttbar_bblvqq/run_seed1001/angles_down_truth.npz"))
ind = np.load(os.path.join(repo_paths.SAMPLES, "ttbar_bblvqq/run_seed12345/angles_down_truth.npz"))
hh = np.load(os.path.join(repo_paths.SAMPLES, "hh_bblvqq/run_seed2001/angles_down_truth.npz"))
get = lambda z, m=slice(None): {a: z[f"{LVL}/{a}"][m] for a in ANGLE_NAMES}  # noqa: E731
Xi, Xh = get(ind), get(hh)
lab = np.r_[np.zeros(len(Xi["Y"])), np.ones(len(Xh["Y"]))]
ntot = len(fit_all[f"{LVL}/Y"])
rng = np.random.default_rng(3)
boot = [rng.integers(0, len(lab), len(lab)) for _ in range(200)]


def auc(score, idx=None):
    return roc_auc_score(lab if idx is None else lab[idx], score if idx is None else score[idx])


rows = []
for N in (12500, 25000, 50000, 100000):
    for rep in range(min(2, ntot // N)):
        sl = slice(rep * N, (rep + 1) * N)
        f = osde.TTbarDensity().fit(get(fit_all, sl))
        variants = {"all": f, "10 sigma": f.truncated(f.significance() > 10), "3 sigma": f.truncated(f.significance() > 3)}
        sc = {k: -np.r_[g(Xi), g(Xh)] for k, g in variants.items()}           # low f = hh-like
        r = {"N": N, "rep": rep}
        for k, s in sc.items():
            r[f"auc {k}"] = auc(s)
            r[f"terms {k}"] = int(np.count_nonzero(variants[k].c))
        for k in ("10 sigma", "3 sigma"):
            d = [auc(sc[k], b) - auc(sc["all"], b) for b in boot]
            r[f"gain {k}"], r[f"gain {k} err"] = float(np.mean(d)), float(np.std(d))
        r["_scores"] = {k: v for k, v in sc.items()}
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "_scores"}), flush=True)
# (2) reproducibility: correlation of the scores from the two disjoint fits at the same N
for N in sorted({r["N"] for r in rows}):
    pair = [r for r in rows if r["N"] == N]
    if len(pair) == 2:
        for k in ("all", "10 sigma"):
            a, b = pair[0]["_scores"][k], pair[1]["_scores"][k]
            pair[0][f"rep corr {k}"] = float(np.corrcoef(a, b)[0, 1])
        print(f"N={N}: score correlation between disjoint fits: all {pair[0]['rep corr all']:.3f}, "
              f"10 sigma {pair[0]['rep corr 10 sigma']:.3f}", flush=True)
for r in rows:
    r.pop("_scores")
json.dump(rows, open(os.path.join(repo_paths.results_dir("study_salvage"), "truncation_noise.json"), "w"), indent=1)

style.use()
fig, ax = plt.subplots(figsize=(10, 6.4))
for k, c in (("10 sigma", "#1f4e9c"), ("3 sigma", "#2a9d8f")):
    Ns = sorted({r["N"] for r in rows})
    m = [np.mean([r[f"gain {k}"] for r in rows if r["N"] == N]) for N in Ns]
    e = [np.mean([r[f"gain {k} err"] for r in rows if r["N"] == N]) for N in Ns]
    ax.errorbar(Ns, m, e, marker="o", ms=9, lw=2.2, capsize=4, color=c, label=f"keep terms with |c|/σ > {k.split()[0]}")
ax.axhline(0, color="0.5", lw=1)
ax.set_xscale("log")
ax.set_xlabel(r"$t\bar t$ events in the fit, $N$")
ax.set_ylabel("AUC(truncated) − AUC(all 14,400 terms)")
ax.set_title("parton level: the truncation gain shrinks with more fit events", fontsize=18)
ax.legend(fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(repo_paths.results_dir("study_salvage"), "truncation_noise.png"), dpi=110)
