#!/usr/bin/env python3
"""Stop at reco level with the two neutralinos' true pT subtracted from the Delphes MET vector (scratch).

Everything else (jets, lepton, chi2 assignment, W-mass-constrained neutrino, angles, selection) is the standard
pipeline (make_plots.load_sample with jet_assignment=truth, had_fermion=down); physics.build_reco is wrapped at
run time only.  If the reco-level angular anomaly of the stop comes from the neutralinos corrupting the neutrino
reconstruction, f_tt's R should fall towards its parton value (0.76) here.

Run: OMP_NUM_THREADS=4 .venv/bin/python code/analysis/study_gonogo/verify/robust_met/stop_nochi.py
"""
import os
import sys

import awkward as ak
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # code/
sys.path.insert(0, ROOT)
from plotting import make_plots  # noqa: E402
from lib import physics  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
import repo_paths  # noqa: E402

OUT = repo_paths.results_dir("study_gonogo", "verify", "robust_met")
ORIG = physics.build_reco
STATS = {"n_chi": []}


def build_reco_nochi(d):
    ev = ORIG(d)
    p = d["Particle"]
    sel = (abs(p.pid) == 1000022) & (p.status == 1)
    STATS["n_chi"].append(ak.to_numpy(ak.sum(sel, axis=1)))
    cpx = ak.to_numpy(ak.sum(p.pt[sel] * np.cos(p.phi[sel]), axis=1))
    cpy = ak.to_numpy(ak.sum(p.pt[sel] * np.sin(p.phi[sel]), axis=1))
    mpx = ev.met * np.cos(ev.met_phi) - cpx
    mpy = ev.met * np.sin(ev.met_phi) - cpy
    ev.met_orig = ev.met.copy()
    ev.met = np.hypot(mpx, mpy)
    ev.met_phi = np.arctan2(mpy, mpx)
    ev.mtw = np.sqrt(np.clip(2 * ak.to_numpy(ak.fill_none(ev.lep.pt, 0.0)) * ev.met *
                             (1 - np.cos(ak.to_numpy(ak.fill_none(ev.lep.phi, 0.0)) - ev.met_phi)), 0, None))
    physics._reconstruct_ttbar(ev)
    return ev


def main():
    physics.build_reco = build_reco_nochi
    path = os.path.join(repo_paths.SAMPLES, "stop_bblvqq/run_seed4002")
    ev, _ = make_plots.load_sample("run_seed4002", path, ["reco"], None,
                                   {"had_fermion": "down", "jet_assignment": "truth"})
    e = ev["reco"]
    n_chi = np.concatenate(STATS["n_chi"])
    print("neutralinos per event:", np.bincount(n_chi), flush=True)
    ang = {a: np.asarray(e[a], float) for a in ANGLE_NAMES}
    ok = np.all([np.isfinite(v) for v in ang.values()], axis=0) & np.asarray(e.reco_ok)
    out = {f"reco/{a}": v[ok] for a, v in ang.items()}
    out["reco/m_w_had"] = np.asarray(e.w_had.mass, float)[ok]
    out["reco/m_t_lep"] = np.asarray(e.t_lep.mass, float)[ok]
    out["reco/m_t_had"] = np.asarray(e.t_had.mass, float)[ok]
    out["reco/mtt"] = np.asarray((e.t_lep + e.t_had).mass, float)[ok]
    out["reco/n_all"] = np.array(len(e))
    out["met"] = np.asarray(e.met, float)[ok]
    out["met_orig"] = np.asarray(e.met_orig, float)[ok]
    out["mtw"] = np.asarray(e.mtw, float)[ok]
    # which events coincide with the standard selection (for a paired comparison)
    z = np.load(repo_paths.results("study_gonogo", "cache", "stop.npz"))
    print(f"selected {ok.sum()} of {len(e)} (standard pipeline: {len(z['reco/Y'])})", flush=True)
    np.savez_compressed(os.path.join(OUT, "stop_nochi.npz"), **out)


if __name__ == "__main__":
    main()
