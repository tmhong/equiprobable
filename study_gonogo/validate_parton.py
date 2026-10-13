#!/usr/bin/env python3
"""Parton-level sanity checks of the go/no-go signal samples against QCD tt̄.

D-like spin correlation: <cos phi>, phi = angle between the charged lepton (in the leptonic top's rest frame) and
the down-type quark (in the hadronic top's rest frame).  With analyzing powers k_l = 1, k_d ~ 0.97:
SM tt̄ at 13 TeV ~ +0.07, no correlation 0, a spin singlet (eta_t) ~ +0.32, a 3P0 scalar (H -> tt̄) ~ -0.11.
Also m_tt̄, |cos Theta| and the mean W-helicity (cos theta' of the lepton) for each sample.

Run: .venv/bin/python study_gonogo/validate_parton.py NAME=PATH ...
"""
import os
import sys

import numpy as np
import vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_plots  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}


def unit3(v):
    a = np.stack([np.asarray(v.px), np.asarray(v.py), np.asarray(v.pz)], 1)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main():
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        ev, xs = make_plots.load_sample(name, os.path.join(ROOT, path), ["parton"], 20000, OPTS)
        e = ev["parton"]
        cm = -(e.t_lep + e.t_had).to_beta3()                   # tt̄ rest frame first, then each top's frame
        tl, th = e.t_lep.boost(cm), e.t_had.boost(cm)
        lep = e.lep.boost(cm).boost(-tl.to_beta3())
        qd = e.q_down.boost(cm).boost(-th.to_beta3())
        tl, th = e.t_lep, e.t_had
        cphi = np.sum(unit3(lep) * unit3(qd), 1)
        mtt = np.asarray((tl + th).mass, float)
        ok = np.isfinite(cphi) & np.isfinite(mtt)
        n = ok.sum()
        print(f"{name:12s} n={n:6d}  sigma={xs}  <cos phi>={cphi[ok].mean():+.3f}+-{cphi[ok].std() / np.sqrt(n):.3f}  "
              f"m_tt median={np.median(mtt[ok]):6.1f} q10={np.percentile(mtt[ok], 10):6.1f} q90={np.percentile(mtt[ok], 90):6.1f}  "
              f"<|cosTheta|>={np.mean(np.abs(np.asarray(e.cos_Theta)[ok])):.3f}  <cos theta'>={np.mean(np.asarray(e.cos_theta_prime)[ok]):+.3f}  "
              f"m_tlep med={np.median(np.asarray(tl.mass)[ok]):.1f}")


if __name__ == "__main__":
    main()
