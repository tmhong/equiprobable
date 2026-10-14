#!/usr/bin/env python3
"""
How much do the angles add on top of the Breit-Wigner mass factors?

Unsupervised (the model itself): already in reco_extension.json (masses only vs angles + BW).
Here the supervised check: a BDT on the masses alone, on the angles alone, and on both, trained
on the tt̄ fit sample (30k) + half of hh and tested on the independent tt̄ sample + the other half
of hh, with a paired bootstrap on AUC(both) - AUC(masses).  If a joint model cannot use the angles
either, the angles carry no extra information for hh at that level.
Output: results/study_salvage/angles_vs_bw.json
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import repo_paths  # noqa: E402
from reco_extension import load, HH, SUFFIX  # noqa: E402  (SIGNAL_SAMPLE / OUT_SUFFIX, see reco_extension.py)
from plotting.angles import ANGLE_NAMES  # noqa: E402

MASSES = ["m_w_had", "m_t_lep", "m_t_had"]


def X(d, which):
    cols = []
    if "angles" in which:
        cols += [d["ang"][a] for a in ANGLE_NAMES]
    if "masses" in which:
        cols += [d["m"][k] for k in MASSES]
    return np.column_stack(cols)


def main():
    levels = ["gen", "reco"]
    out_dir = repo_paths.results_dir("study_salvage")
    ref = load(os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed1001"), levels, 30000)
    tt = load(os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed12345"), levels)
    hh = load(os.path.join(repo_paths.REPO, HH), levels)
    rx = json.load(open(os.path.join(out_dir, f"reco_extension{SUFFIX}.json")))
    out = {}
    rng = np.random.default_rng(5)
    for l in levels:
        nh = len(hh[l]["ang"]["Y"])
        tr_h = np.arange(nh) < nh // 2
        lab = np.r_[np.zeros(len(tt[l]["ang"]["Y"])), np.ones((~tr_h).sum())]
        res = {"unsupervised masses only": rx[l]["masses only"],
               "unsupervised angles(>10 sigma) + BW": rx[l]["truncation"][">10 sigma"]["angles+BW"]}
        scores = {}
        for which in ("masses", "angles", "angles+masses"):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=1)
            Xtr = np.vstack([X(ref[l], which), X(hh[l], which)[tr_h]])
            ytr = np.r_[np.zeros(len(ref[l]["ang"]["Y"])), np.ones(tr_h.sum())]
            ok = np.all(np.isfinite(Xtr), 1)
            clf.fit(Xtr[ok], ytr[ok])
            Xte = np.vstack([X(tt[l], which), X(hh[l], which)[~tr_h]])
            Xte = np.nan_to_num(Xte, nan=0.0)
            scores[which] = clf.predict_proba(Xte)[:, 1]
            res[f"BDT {which}"] = roc_auc_score(lab, scores[which])
        boot = [rng.integers(0, len(lab), len(lab)) for _ in range(200)]
        d = [roc_auc_score(lab[b], scores["angles+masses"][b]) - roc_auc_score(lab[b], scores["masses"][b]) for b in boot]
        res["BDT (angles+masses) - (masses): paired"] = [float(np.mean(d)), float(np.std(d))]
        out[l] = res
        print(l, json.dumps(res, indent=1), flush=True)
    json.dump(out, open(os.path.join(out_dir, f"angles_vs_bw{SUFFIX}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
