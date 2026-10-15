#!/usr/bin/env python3
"""With the Breit-Wigner mass factors in, how many angular terms does f_tt̄ still need?
Truncate f(Ω) by coefficient significance |c|/σ(c) > k (k = 0 is all 14,400 terms; CV = the
cross-validated choice; 'none' = flat in the angles) and score  s(Ω) + Σ_k s_BW(m_k).
Output: results/study_salvage/angle_terms.json"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import repo_paths  # noqa: E402
from salvage import surprisal  # noqa: E402
from simplest import PDG, bw_surprisal, cauchy_surprisal  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

SUFFIX = os.environ.get("OUT_SUFFIX", "")   # e.g. "_h170": signal from results/study_washout_h170/, output angle_terms_h170.json
tt = np.load(os.path.join(repo_paths.RESULTS, "study_washout/mechanism/data_tt.npz"))
hh = np.load(os.path.join(repo_paths.RESULTS, f"study_washout{SUFFIX}/mechanism/data_hh.npz"))
out = {}
for lvl, pre, mk, masses in (("parton", "p_", None, list(PDG)), ("gen", "c_", "g_reco_ok", ["m_w_had", "m_t_lep", "m_t_had"])):
    mt = tt[mk].astype(bool) if mk else np.ones(len(tt["p_Y"]), bool)
    mh = hh[mk].astype(bool) if mk else np.ones(len(hh["p_Y"]), bool)
    Xt = {a: tt[pre + a][mt] for a in ANGLE_NAMES}; Xh = {a: hh[pre + a][mh] for a in ANGLE_NAMES}
    A = np.random.default_rng(7).random(mt.sum()) < 0.5
    lab = np.r_[np.zeros((~A).sum()), np.ones(mh.sum())]
    sm = 0
    for v in masses:
        xt, xh = tt[pre + v][mt], hh[pre + v][mh]
        if lvl == "parton":
            sm = sm + np.r_[bw_surprisal(xt[~A], *PDG[v]), bw_surprisal(xh, *PDG[v])]
        else:
            M = np.median(xt[A]); q1, q3 = np.percentile(xt[A], [25, 75]); W = 0.5 * (q3 - q1)
            sm = sm + np.r_[cauchy_surprisal(xt[~A], M, W), cauchy_surprisal(xh, M, W)]
    full = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", f"f_ttbar_{lvl}.npz"))
    cv = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", f"f_ttbar_{lvl}_simplified.npz"))
    variants = [("all", full)] + [(f">{k} sigma", full.truncated(full.significance() > k)) for k in (3, 5, 10, 20, 40)] \
        + [("CV", cv)]
    res = {"none (masses only)": {"terms": 0, "angles_only": 0.5, "with_masses": roc_auc_score(lab, sm)}}
    for name, f in variants:
        ft, fh = f.loo(Xt), f(Xh)
        sa = np.r_[surprisal(ft[~A], ft[A]), surprisal(fh, ft[A])]
        res[name] = {"terms": int(np.count_nonzero(f.c)), "angles_only": roc_auc_score(lab, sa),
                     "with_masses": roc_auc_score(lab, sa + sm)}
    out[lvl] = res
    for k, v in res.items():
        print(f"{lvl:7s} {k:20s} terms {v['terms']:6d}  angles {v['angles_only']:.3f}  angles+BW {v['with_masses']:.3f}")
json.dump(out, open(os.path.join(repo_paths.results_dir("study_salvage"), f"angle_terms{SUFFIX}.json"), "w"), indent=1)
