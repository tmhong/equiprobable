#!/usr/bin/env python3
"""Reco-level MET handles for a sample, with exactly the selection of study_gonogo/extract.py (reco_ok & finite
angles), so that the rows line up with results/study_gonogo/cache/<key>.npz reco arrays.  Scratch for the robustness check.

Run: OMP_NUM_THREADS=4 .venv/bin/python code/analysis/study_gonogo/verify/robust_met/extract_met.py KEY=PATH ...   (PATH relative to the repo top)
"""
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # code/
sys.path.insert(0, ROOT)
from plotting import make_plots  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
from lib.physics import M_W  # noqa: E402
import repo_paths  # noqa: E402

OUT = repo_paths.results_dir("study_gonogo", "verify", "robust_met")

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}


def one(key, path):
    ev, _ = make_plots.load_sample(os.path.basename(os.path.normpath(path)), path, ["reco"], None, OPTS)
    e = ev["reco"]
    ang = {a: np.asarray(e[a], float) for a in ANGLE_NAMES}
    ok = np.all([np.isfinite(v) for v in ang.values()], axis=0) & np.asarray(e.reco_ok)
    lep = e.lep
    f = lambda x: ak.to_numpy(ak.fill_none(x, np.nan)).astype(float)  # noqa: E731
    lpx, lpy, lpz, le = f(lep.px), f(lep.py), f(lep.pz), f(lep.E)
    met, mphi = np.asarray(e.met, float), np.asarray(e.met_phi, float)
    mpx, mpy = met * np.cos(mphi), met * np.sin(mphi)
    lpt2 = lpx ** 2 + lpy ** 2
    mu = 0.5 * M_W ** 2 + lpx * mpx + lpy * mpy
    a = mu * lpz / lpt2
    disc = a ** 2 - (le ** 2 * (mpx ** 2 + mpy ** 2) - mu ** 2) / lpt2      # same as physics.neutrino_pz
    jets = e.jets
    ht = ak.to_numpy(ak.sum(jets.pt, axis=1)).astype(float)
    out = {"met": met, "met_phi": mphi, "mtw": np.asarray(e.mtw, float), "nu_disc": disc,
           "lep_pt": f(lep.pt), "ht": ht, "njets": np.asarray(e.njets, float), "nbjets": np.asarray(e.nbjets, float),
           "reco_chi2": np.asarray(e.reco_chi2, float),
           "nu_pz": np.asarray(e.nu.pz, float), "t_lep_pt": np.asarray(e.t_lep.pt, float),
           "t_had_pt": np.asarray(e.t_had.pt, float)}
    out = {k: v[ok] for k, v in out.items()}
    for a_, v in ang.items():
        out[f"ang/{a_}"] = v[ok]
    out["n_all"] = np.array(len(e))
    # alignment check with the go/no-go cache
    z = np.load(os.path.join(repo_paths.RESULTS, "study_gonogo", "cache", f"{key}.npz"))
    same = all(len(z[f"reco/{x}"]) == len(out[f"ang/{x}"]) and np.allclose(z[f"reco/{x}"], out[f"ang/{x}"])
               for x in ANGLE_NAMES)
    print(f"{key}: {ok.sum()} of {len(e)} events, identical to study_gonogo cache: {same}", flush=True)
    out["aligned"] = np.array(same)
    np.savez_compressed(os.path.join(OUT, f"met_{key}.npz"), **out)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        k, p = arg.split("=", 1)
        one(k, os.path.join(repo_paths.REPO, p))
