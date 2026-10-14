#!/usr/bin/env python3
"""Critic: cross-check physics_skeptic's m_tt-conditional R (threshold set within tt m_tt deciles) at reco and parton_trig."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from plotting import osde
from plotting.angles import ANGLE_NAMES
C = os.path.join(ROOT, "study_gonogo/cache")
FF = {"reco": "osde/f_ttbar_reco.npz", "parton_trig": "osde/f_ttbar_parton_trig.npz"}
for lev, fpath in FF.items():
    if not os.path.exists(os.path.join(ROOT, fpath)):
        print("missing", fpath); continue
    f = osde.load_density(os.path.join(ROOT, fpath)); f = f.truncated(f.significance() > 2.5)
    z = np.load(f"{C}/tt_fit.npz"); stt = -f.loo({a: z[f"{lev}/{a}"] for a in ANGLE_NAMES}); mtt = z[f"{lev}/mtt"]
    edges = np.quantile(mtt, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
    btt = np.digitize(mtt, edges[1:-1])
    for k in ("etat", "H350", "stop", "tt_indep", "nospin"):
        zs = np.load(f"{C}/{k}.npz"); s = -f({a: zs[f"{lev}/{a}"] for a in ANGLE_NAMES}); bs = np.digitize(zs[f"{lev}/mtt"], edges[1:-1])
        line = []
        for wp in (0.01, 0.1):
            thr = np.array([np.quantile(stt[btt == b], 1 - wp) for b in range(10)])
            inc = np.mean(s > np.quantile(stt, 1 - wp)) / wp
            cond = np.mean(s > thr[bs]) / wp
            line.append(f"R@{wp:g} incl {inc:.2f} cond {cond:.2f}")
        print(f"{lev:11s} {k:9s} " + "   ".join(line), flush=True)
