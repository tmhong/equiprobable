import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch_float import *
D = {k: load(k) for k in ("tt_test", "tt_calib", "hh")}
f = osde.load_density(os.path.join(ROOT, "osde", "f_ttbar_reco.npz")); f = f.truncated(f.significance() > 10)
for k, d in D.items():
    idx = np.flatnonzero((d["nleptons"] == 1) & (d["njets"] >= 4))
    o = chi2_float(d, idx, 6, False, True)
    X = angles_float(o, "softer")
    fv = f(X)
    T = o["t_lep"] + o["t_had"]
    ms = {"mWh": mass(o["w_had"]), "mTl": mass(o["t_lep"]), "mTh": mass(o["t_had"])}
    print(k, "f range", fv.min(), np.percentile(fv, [1, 50, 99]), fv.max())
    print("   masses >512:", {kk: float((v > 512).mean()) for kk, v in ms.items()}, " >256:", {kk: float((v > 256).mean()) for kk, v in ms.items()})
    print("   masses 99.5%:", {kk: float(np.percentile(v, 99.5)) for kk, v in ms.items()})
    print("   E_T max", T[:, 3].max(), "E_T>4096 frac", (T[:, 3] > 4096).mean(), "|Y|max", np.abs(X["Y"]).max(), "|Y|>4", (np.abs(X["Y"]) > 4).mean())
    print("   chi2min pct", np.percentile(o["chi2"], [50, 90, 99, 99.9]), "max", o["chi2"].max())
    J = d["jet_pt"][idx, :6] * np.cosh(d["jet_eta"][idx, :6])
    print("   jet E max", np.nanmax(J), "frac jets E>4096", np.nanmean(J > 4096), " lep pt min", d["lep_pt"][idx].min())
    # gamma of tops, slowest top momentum
    for nm in ("t_lep", "t_had"):
        t = o[nm]; P = np.linalg.norm(t[:, :3], axis=1); m = mass(t)
        print(f"   {nm}: gamma max {np.max(t[:,3]/m):.1f} 99.9% {np.percentile(t[:,3]/m, 99.9):.1f}  P min {P.min():.2f} pT min {np.hypot(t[:,0], t[:,1]).min():.2f}  m min {m.min():.1f}")
    Tm = mass(T); print(f"   ttbar gamma max {np.max(T[:,3]/Tm):.1f}")
