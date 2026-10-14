#!/usr/bin/env python3
"""Critic: independent supervised angle-only ceiling (5-fold cross-fit, two seeds) at reco for etat, H350,
nospin, stop, tt_indep; compare with results.json bdt_angles (2-fold)."""
import json, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from plotting.angles import ANGLE_NAMES
import repo_paths
C = os.path.join(repo_paths.RESULTS, "study_gonogo/cache")
A = lambda k: np.column_stack([np.load(f"{C}/{k}.npz")[f"reco/{a}"] for a in ANGLE_NAMES])
tt = A("tt_fit"); res = json.load(open(os.path.join(repo_paths.RESULTS, "study_gonogo/results.json")))
def R(s, b, wp): return float(np.mean(s > np.quantile(b, 1 - wp)) / wp)
for k in ("etat", "stop", "H350", "nospin", "tt_indep"):
    X = A(k); out = []
    for seed in (5, 6):
        rng = np.random.default_rng(seed); K = 5
        ft, fs = rng.permutation(len(tt)) % K, rng.permutation(len(X)) % K
        ot, os_ = np.zeros(len(tt)), np.zeros(len(X))
        for f in range(K):
            clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.15,
                                                 class_weight="balanced", random_state=seed)
            clf.fit(np.vstack([tt[ft != f], X[fs != f]]), np.r_[np.zeros((ft != f).sum()), np.ones((fs != f).sum())])
            ot[ft == f] = clf.predict_proba(tt[ft == f])[:, 1]; os_[fs == f] = clf.predict_proba(X[fs == f])[:, 1]
        out.append((R(os_, ot, .01), R(os_, ot, .1)))
    o = np.mean(out, axis=0); r = res["levels"]["reco"][k]["scores"]["bdt_angles"]
    print(f"{k:9s} n={len(X):6d}  5-fold R@1%/10% {o[0]:.2f}/{o[1]:.2f} (seeds {out[0][0]:.2f},{out[1][0]:.2f})   results.json 2-fold {r['R@0.01']:.2f}+-{r['R@0.01_err']:.2f}/{r['R@0.1']:.2f}", flush=True)
