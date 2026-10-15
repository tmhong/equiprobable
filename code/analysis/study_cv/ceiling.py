#!/usr/bin/env python3
"""Supervised ceiling on the same ten angles (BDT trained with the signal labels, 5-fold out-of-fold predictions),
at each level, on the events of study_cv/cv_scores.py.  Writes results/study_cv/ceiling.json."""
import json, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from lib.angles import ANGLE_NAMES  # noqa: E402
import repo_paths  # noqa: E402
OUT = repo_paths.results_dir("study_cv")
C = repo_paths.results_dir("study_gonogo", "cache", create=False); TM = repo_paths.results_dir("study_truthmatch", create=False)

def arr(X): return np.column_stack([X[a] for a in ANGLE_NAMES])
def cached(key, lv):
    z = np.load(os.path.join(C, f"{key}.npz")); return arr({a: z[f"{lv}/{a}"] for a in ANGLE_NAMES})
def matched(run):
    z = np.load(os.path.join(TM, f"data_{run}.npz")); ok = z["ok"]; return arr({a: z[f"m_{a}"][ok] for a in ANGLE_NAMES})

def ceiling(Xb, Xs, seed=0):
    rng = np.random.default_rng(seed)
    fb, fs = rng.permutation(len(Xb)) % 5, rng.permutation(len(Xs)) % 5
    ob, os_ = np.zeros(len(Xb)), np.zeros(len(Xs))
    for f in range(5):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
        clf.fit(np.vstack([Xb[fb != f], Xs[fs != f]]), np.r_[np.zeros((fb != f).sum()), np.ones((fs != f).sum())])
        ob[fb == f] = clf.predict_proba(Xb[fb == f])[:, 1]; os_[fs == f] = clf.predict_proba(Xs[fs == f])[:, 1]
    return round(float(roc_auc_score(np.r_[np.zeros(len(ob)), np.ones(len(os_))], np.r_[ob, os_])), 3)

out = {}
for lv in ("parton_trig", "gen", "reco"):
    tt = cached("tt_fit", lv)
    out[lv] = {"hh125": ceiling(tt, cached("hh125", lv)), "HH170": ceiling(tt, cached("HH170_all", lv))}
    print(lv, out[lv], flush=True)
out["matched"] = {"HH170": ceiling(matched("run_seed1001"), matched("run_m170_all"))}
print("matched", out["matched"])
json.dump(out, open(os.path.join(OUT, "ceiling.json"), "w"), indent=1)
