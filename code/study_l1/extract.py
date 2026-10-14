#!/usr/bin/env python3
"""
Input stage of the L1 feasibility study: read the reco-level samples once with the project's
own loader (make_plots.load_sample, angle options down-type / truth) and cache, per event,

  * the raw objects an L1 algorithm could see: the pT-ordered jets (pT > 25 GeV, |eta| < 2.5;
    up to NJ = 8 kept: pt, eta, phi, mass, btag, Delphes truth flavour), the leading lepton
    (pt, eta, phi, mass, flavour), MET and its phi, and the object counts;
  * the offline reference outputs: preselection flag, reco_ok, chi2, the chi2-chosen
    four-vectors (b_had, b_lep, q1, q2, nu), the truth-matched down-type W_had jet, the three
    masses and the ten angles.

Output: results/study_l1/cache/<name>.npz (one file per sample).  Never modifies existing files.

  python code/study_l1/extract.py       # all three samples (skips existing cache files)
"""
import os
import sys
import time

import awkward as ak
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path.insert(0, ROOT)
import make_plots  # noqa: E402
import repo_paths  # noqa: E402
from plotting.angles import ANGLE_NAMES, as_array  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
NJ = 8
CACHE = repo_paths.results_dir("study_l1", "cache", create=False)
SAMPLES = {
    # name: (path, max events)
    "tt_test": ("samples/ttbar_bblvqq/run_seed12345", None),     # independent tt̄ (10k): test
    "tt_calib": ("samples/ttbar_bblvqq/run_seed1001", 30000),    # tt̄ fit sample: calibration (as reco_extension.py)
    "hh": ("samples/hh_bblvqq/run_seed2001", None),              # hh (20k): test
}


def dense(x, n, fill=np.nan):
    x = ak.pad_none(x[:, :n], n, clip=True)
    return ak.to_numpy(ak.fill_none(x, fill)).astype(float)


def extract(name, path, nmax):
    t0 = time.time()
    ev, _ = make_plots.load_sample(name, os.path.join(repo_paths.REPO, path), ["parton", "reco"], nmax, OPTS)
    R = ev["reco"]
    out = {}
    j = R.jets
    for k in ("pt", "eta", "phi", "mass"):
        out[f"jet_{k}"] = dense(getattr(j, k), NJ)
    out["jet_btag"] = dense(j.btag, NJ, 0).astype(bool)
    out["jet_flavor"] = dense(j.flavor, NJ, 0).astype(int)
    out["njets"] = np.asarray(R.njets)
    out["nbjets"] = np.asarray(R.nbjets)
    out["nleptons"] = np.asarray(R.nleptons)
    lep = R.lep
    for k in ("pt", "eta", "phi", "mass"):
        out[f"lep_{k}"] = ak.to_numpy(ak.fill_none(getattr(lep, k), np.nan)).astype(float)
    out["lep_flavour"] = ak.to_numpy(ak.fill_none(lep.flavour, 0)).astype(int)
    out["met"] = np.asarray(R.met, float)
    out["met_phi"] = np.asarray(R.met_phi, float)
    out["presel"] = np.asarray(R.presel, bool)
    out["reco_ok"] = np.asarray(R.reco_ok, bool)
    out["reco_chi2"] = np.asarray(R.reco_chi2, float)
    for k in ("b_had", "b_lep", "q1", "q2", "nu", "q_down", "w_had", "t_had", "t_lep", "w_lep"):
        out[f"ref_{k}"] = as_array(R[k])                     # (n, 4) px, py, pz, E (NaN if not reco_ok)
    for a in ANGLE_NAMES:
        out[f"ref_{a}"] = np.asarray(R[a], float)
    # parton-level quarks used by the truth assignment of the down-type W_had jet
    P = ev["parton"]
    out["parton_q_down"] = as_array(P.q_down)
    out["parton_q_up"] = as_array(P.q_up)
    np.savez_compressed(os.path.join(CACHE, f"{name}.npz"), **out)
    print(f"[{name}] {len(out['met'])} events, presel {out['presel'].sum()}, reco_ok {out['reco_ok'].sum()} "
          f"({time.time() - t0:.0f} s)", flush=True)


def main():
    os.makedirs(CACHE, exist_ok=True)
    for name, (path, nmax) in SAMPLES.items():
        if os.path.exists(os.path.join(CACHE, f"{name}.npz")) and "--force" not in sys.argv:
            print(f"[{name}] cached")
            continue
        extract(name, path, nmax)


if __name__ == "__main__":
    main()
