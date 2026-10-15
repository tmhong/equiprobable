#!/usr/bin/env python3
"""Momentum (resonance-mass) extension of f_tt̄ and the angular-term truncation at gen AND reco.

Same recipe as simplest.py / angle_terms.py, done the clean way for both levels:
  - tt̄ test events: the independent sample (seed12345), never used in any fit -> no leave-one-out
  - M, W of the Cauchy factor (median, half-IQR): from 30k events of the fit sample (seed1001)
  - f(Ω): results/analysis/osde/f_ttbar_<level>.npz, truncated by coefficient significance; 'CV' = the paper method
  - masses: m_W,had, m_t,lep, m_t,had of the χ² reconstruction (m_W,lep is fixed to m_W by the ν solution)
Score = surprisal(angles) + Σ surprisal(mass).  Output: results/analysis/study_salvage/reco_extension.json
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import repo_paths  # noqa: E402
from plotting import make_plots  # noqa: E402
from salvage import surprisal  # noqa: E402
from simplest import cauchy_surprisal  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
# signal sample (SIGNAL_SAMPLE, relative to the project) and output suffix (OUT_SUFFIX, e.g. "_h170");
# defaults = hh(125) and the original file names
HH_DEFAULT = "samples/hh_bblvqq/run_seed2001"
HH = os.environ.get("SIGNAL_SAMPLE", HH_DEFAULT)
SUFFIX = os.environ.get("OUT_SUFFIX", "")
if os.path.normpath(HH) != HH_DEFAULT and not SUFFIX:
    sys.exit("SIGNAL_SAMPLE is not hh(125): set OUT_SUFFIX (e.g. _h170) so the hh(125) outputs are not overwritten")
MASSES = {"m_w_had": lambda e: e.w_had.mass, "m_t_lep": lambda e: e.t_lep.mass, "m_t_had": lambda e: e.t_had.mass,
          "m_w_lep": lambda e: e.w_lep.mass}


def load(path, levels, nmax=None):
    ev, _ = make_plots.load_sample("x", path, levels, nmax, OPTS)
    out = {}
    for l in levels:
        e = ev[l]
        ok = np.asarray(e.reco_ok) & np.all([np.isfinite(np.asarray(e[a], float)) for a in ANGLE_NAMES], 0)
        out[l] = {"ang": {a: np.asarray(e[a], float)[ok] for a in ANGLE_NAMES},
                  "m": {k: np.asarray(fn(e), float)[ok] for k, fn in MASSES.items()}}
    return out


def main():
    levels = ["gen", "reco"]
    ref = load(os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed1001"), levels, 30000)
    tt = load(os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed12345"), levels)
    hh = load(os.path.join(repo_paths.REPO, HH), levels)
    out = {}
    for l in levels:
        lab = np.r_[np.zeros(len(tt[l]["ang"]["Y"])), np.ones(len(hh[l]["ang"]["Y"]))]
        res = {"n_tt": int((lab == 0).sum()), "n_hh": int(lab.sum()),
               "m_w_lep_spread_tt": float(np.nanstd(tt[l]["m"]["m_w_lep"]))}
        sm, single = 0, {}
        for k in ("m_w_had", "m_t_lep", "m_t_had"):
            x = ref[l]["m"][k]; x = x[np.isfinite(x)]
            M = np.median(x); q1, q3 = np.percentile(x, [25, 75]); W = 0.5 * (q3 - q1)
            s = np.r_[cauchy_surprisal(tt[l]["m"][k], M, W), cauchy_surprisal(hh[l]["m"][k], M, W)]
            single[k] = {"M": float(M), "W": float(W), "auc": roc_auc_score(lab, s)}
            sm = sm + s
        res["masses"] = single
        res["masses only"] = roc_auc_score(lab, sm)
        full = osde.load_density(repo_paths.results("osde", f"f_ttbar_{l}.npz"))
        cv = osde.load_density(repo_paths.results("osde", f"f_ttbar_{l}_simplified.npz"))
        fref = full(ref[l]["ang"])
        variants = [("all", full)] + [(f">{k} sigma", full.truncated(full.significance() > k)) for k in (3, 5, 10, 20, 40)] \
            + [("CV", cv)]
        res["truncation"] = {}
        for name, f in variants:
            ft, fh, fr = f(tt[l]["ang"]), f(hh[l]["ang"]), (fref if name == "all" else f(ref[l]["ang"]))
            sa = np.r_[surprisal(ft, fr), surprisal(fh, fr)]
            res["truncation"][name] = {"terms": int(np.count_nonzero(f.c)), "angles": roc_auc_score(lab, sa),
                                       "angles+BW": roc_auc_score(lab, sa + sm)}
        out[l] = res
        print(l, json.dumps(res, indent=1), flush=True)
    json.dump(out, open(os.path.join(repo_paths.results_dir("study_salvage"), f"reco_extension{SUFFIX}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
