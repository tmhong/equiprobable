import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch_float import *

D = {k: load(k) for k in ("tt_test", "tt_calib", "hh")}
f = osde.load_density(os.path.join(ROOT, "osde", "f_ttbar_reco.npz"))
f = f.truncated(f.significance() > 10)

def evset(d, kind):
    if kind == "offline":
        return np.flatnonzero(d["reco_ok"])
    return np.flatnonzero((d["nleptons"] == 1) & (d["njets"] >= 4))

def run(nj=6, btag=True, massless=False, mode="truth", kind="offline"):
    out = {}
    for k, d in D.items():
        idx = evset(d, kind)
        o = chi2_float(d, idx, nj, btag, massless)
        assert o["ok"].all(), (k, (~o["ok"]).sum())
        X = angles_float(o, mode, d["parton_q_down"][idx], d["parton_q_up"][idx])
        ms = {"m_w_had": mass(o["w_had"]), "m_t_lep": mass(o["t_lep"]), "m_t_had": mass(o["t_had"])}
        # agreement with the offline reference assignment (offline set only)
        same = None
        if kind == "offline":
            ref = np.stack([d["ref_b_had"][idx], d["ref_b_lep"][idx], d["ref_q1"][idx], d["ref_q2"][idx]], 1)
            mine = np.stack([o["b_had"], o["b_lep"], o["q1"], o["q2"]], 1)
            # compare jet directions (massless changes E): use px, py
            same = np.all(np.abs(ref[..., :2] - mine[..., :2]) < 1e-2, axis=(1, 2)).mean()
        out[k] = (X, ms, same, o["chi2"])
    Xc, mc, _, _ = out["tt_calib"]
    MW = {kk: (np.median(x), 0.5 * np.subtract(*np.percentile(x, [75, 25]))) for kk, x in mc.items()}
    fr = f(Xc)
    ft, fh = f(out["tt_test"][0]), f(out["hh"][0])
    lab = np.r_[np.zeros(len(ft)), np.ones(len(fh))]
    sa = np.r_[surprisal(ft, fr), surprisal(fh, fr)]
    sm = sum(np.r_[cauchy_s(out["tt_test"][1][kk], *MW[kk]), cauchy_s(out["hh"][1][kk], *MW[kk])] for kk in MW)
    chi = np.r_[out["tt_test"][3], out["hh"][3]]
    print(f"nj={nj} btag={btag!s:5} massless={massless!s:5} {mode:6s} {kind:8s} n_tt={len(ft)} n_hh={len(fh)} | "
          f"AUC ang {roc_auc_score(lab, sa):.4f} mass {roc_auc_score(lab, sm):.4f} ang+BW {roc_auc_score(lab, sa+sm):.4f} "
          f"chi2min {roc_auc_score(lab, chi):.4f} | same-assign tt {out['tt_test'][2]} hh {out['hh'][2]} | "
          + " ".join(f"{kk}:{v[0]:.1f}/{v[1]:.1f}" for kk, v in MW.items()))









run(nj=6, btag=False, massless=True, mode="softer", kind="l1")
run(nj=5, btag=False, massless=True, mode="softer", kind="l1")
run(nj=4, btag=False, massless=True, mode="softer", kind="l1")
