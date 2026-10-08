#!/usr/bin/env python3
"""
Simplest momentum extension of f_tt̄: multiply by one Breit-Wigner-shaped factor per resonance mass.
  parton: the physical relativistic Breit-Wigner, no fit:  BW(m) ∝ 1 / ((m² - M²)² + M²Γ²)
  gen   : the same shape with M, W from tt̄ (median, half-IQR) = a Cauchy (no coefficients)
Score = surprisal of the angles + Σ surprisal of each mass.  Also: the score from the masses alone,
and adding the masses one at a time.  Output: study_salvage/simplest.json
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from salvage import surprisal  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402

PDG = {"m_w_lep": (80.4, 2.085), "m_w_had": (80.4, 2.085), "m_t_lep": (172.5, 1.42), "m_t_had": (172.5, 1.42)}


def bw_surprisal(m, M, G):
    """-ln of the normalized relativistic Breit-Wigner in m (m dm-measure, up to a constant)."""
    return np.log((m ** 2 - M ** 2) ** 2 + M ** 2 * G ** 2) - np.log(m)


def cauchy_surprisal(m, M, W):
    return np.log((m - M) ** 2 + W ** 2)


def main():
    tt = np.load(os.path.join(ROOT, "study_washout/mechanism/data_tt.npz"))
    hh = np.load(os.path.join(ROOT, "study_washout/mechanism/data_hh.npz"))
    out = {}
    for lvl, pre, mkey, masses, fname in (("parton", "p_", None, list(PDG), "f_ttbar_parton.npz"),
                                          ("gen", "c_", "g_reco_ok", ["m_w_had", "m_t_lep", "m_t_had"], "f_ttbar_gen.npz")):
        mt = tt[mkey].astype(bool) if mkey else np.ones(len(tt["p_Y"]), bool)
        mh = hh[mkey].astype(bool) if mkey else np.ones(len(hh["p_Y"]), bool)
        f = osde.load_density(os.path.join(ROOT, "osde", fname))
        fa_t = f.loo({a: tt[pre + a][mt] for a in ANGLE_NAMES})
        fa_h = f({a: hh[pre + a][mh] for a in ANGLE_NAMES})
        rng = np.random.default_rng(7)
        A = rng.random(mt.sum()) < 0.5          # half A: sets M, W at gen and y0; half B: test
        lab = np.r_[np.zeros((~A).sum()), np.ones(mh.sum())]
        s_ang = np.r_[surprisal(fa_t[~A], fa_t[A]), surprisal(fa_h, fa_t[A])]
        s_m = {}
        for v in masses:
            xt, xh = tt[pre + v][mt], hh[pre + v][mh]
            if lvl == "parton":
                M, G = PDG[v]
                s_m[v] = np.r_[bw_surprisal(xt[~A], M, G), bw_surprisal(xh, M, G)]
            else:
                M = np.median(xt[A]); q1, q3 = np.percentile(xt[A], [25, 75]); W = 0.5 * (q3 - q1)
                s_m[v] = np.r_[cauchy_surprisal(xt[~A], M, W), cauchy_surprisal(xh, M, W)]
        res = {"angles": roc_auc_score(lab, s_ang), "masses_only": roc_auc_score(lab, sum(s_m.values()))}
        cum = s_ang.copy()
        for v in masses:
            res[f"single {v}"] = roc_auc_score(lab, s_m[v])
            cum = cum + s_m[v]
            res[f"angles + ... + {v}"] = roc_auc_score(lab, cum)
        out[lvl] = res
        print(lvl, json.dumps({k: round(v, 3) for k, v in res.items()}, indent=1))
    json.dump(out, open(os.path.join(HERE, "simplest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
