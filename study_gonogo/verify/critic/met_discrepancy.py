#!/usr/bin/env python3
"""Critic: why do robust_met (stop f_tt R@1% 1.75->0.96) and physics_skeptic (1.75->1.55) disagree on the
neutralino-MET subtraction?  Compares LHE vs Delphes final-state neutralino pT per event, and the two
oracle score vectors on the same 16599 reco events."""
import os, sys
import awkward as ak
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from plotting import io, osde
from plotting.angles import ANGLE_NAMES

path = os.path.join(ROOT, "samples/stop_bblvqq/run_seed4002")
parts, _ = io.read_lhe(os.path.join(path, "parton_events.lhe.gz"), None)
inv = (parts.status == 1) & (abs(parts.pid) == 1000022)
lx = ak.to_numpy(ak.sum(parts.px[inv], axis=1)); ly = ak.to_numpy(ak.sum(parts.py[inv], axis=1))
print("LHE events", len(lx), "chi per event", np.bincount(ak.to_numpy(ak.sum(inv, axis=1))))
d = io.read_delphes(os.path.join(path, "delphes_events.root"), entry_start=0, max_events=20000)
p = d["Particle"]
sel = (abs(p.pid) == 1000022) & (p.status == 1)
dx = ak.to_numpy(ak.sum(p.pt[sel] * np.cos(p.phi[sel]), axis=1)); dy = ak.to_numpy(ak.sum(p.pt[sel] * np.sin(p.phi[sel]), axis=1))
n = len(dx)
res = np.hypot(lx[:n] - dx, ly[:n] - dy)
print(f"first {n} events: |pT(chichi)| LHE median {np.median(np.hypot(lx[:n], ly[:n])):.1f}, Delphes median {np.median(np.hypot(dx, dy)):.1f}")
print(f"|LHE - Delphes| chichi pT vector: median {np.median(res):.2f}, 90% {np.quantile(res, .9):.2f} GeV")
print("corr px LHE vs Delphes", np.corrcoef(lx[:n], dx)[0, 1])
# check alignment: per-event individual neutralino pz/E match?
lpz = ak.to_numpy(ak.sum(parts.pz[inv], axis=1))[:n]
dpz = ak.to_numpy(ak.sum((p.pt * np.sinh(p.eta))[sel], axis=1))
print("corr pz LHE vs Delphes", np.corrcoef(lpz, dpz)[0, 1], "median |dpz|", np.median(abs(lpz - dpz)))
# ISR recoil of the full event: Delphes pT of hard system vs 0
# Score comparison on the 16599 reco events
f_all = osde.load_density(os.path.join(ROOT, "osde/f_ttbar_reco.npz"))
f_cv = f_all.truncated(f_all.significance() > 2.5)
zr = np.load(os.path.join(ROOT, "study_gonogo/verify/robust_met/stop_nochi.npz"))
s_rob = -f_cv({a: zr[f"reco/{a}"] for a in ANGLE_NAMES})
zp = np.load(os.path.join(ROOT, "study_gonogo/verify/physics_skeptic/stop_met_scores.npz"))
s_phy, s_nom, s_tt = zp["reco/oracle"], zp["reco/nominal"], zp["reco/tt"]
def R(s, b, wp): return float(np.mean(s > np.quantile(b, 1 - wp)) / wp)
print("n", len(s_rob), len(s_phy), len(s_nom))
for wp in (0.01, 0.1):
    print(f"R@{wp}: nominal {R(s_nom, s_tt, wp):.3f}  robust-oracle {R(s_rob, s_tt, wp):.3f}  physics-oracle {R(s_phy, s_tt, wp):.3f}")
print("corr(robust oracle, physics oracle)", np.corrcoef(s_rob, s_phy)[0, 1], " corr(nominal, physics oracle)", np.corrcoef(s_nom, s_phy)[0, 1],
      " corr(nominal, robust oracle)", np.corrcoef(s_nom, s_rob)[0, 1])
print("frac events with identical score: rob==nom", np.mean(np.isclose(s_rob, s_nom)), " phy==nom", np.mean(np.isclose(s_phy, s_nom)))
print("met_orig median", np.median(zr["met_orig"]), "robust oracle met median", np.median(zr["met"]))
