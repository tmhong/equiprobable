"""Scratch: reproduce the offline reference with an independent float implementation."""
import os
import sys
from itertools import permutations

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path.insert(0, ROOT)
import repo_paths  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES, production_angles, decay_angles  # noqa: E402

M_W, M_TOP, SIG_W, SIG_T = 80.4, 172.5, 10.0, 15.0


def load(name):
    z = np.load(os.path.join(repo_paths.RESULTS, "study_l1", "cache", f"{name}.npz"))
    return {k: z[k] for k in z.files}


def p4_from_ptetaphim(pt, eta, phi, m):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)
    return np.stack([px, py, pz, e], -1)


def mass(p):
    return np.sqrt(np.clip(p[..., 3] ** 2 - p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2, 0, None))


def nu_pz(lep, mex, mey, mw=M_W):
    lpx, lpy, lpz, le = lep[:, 0], lep[:, 1], lep[:, 2], lep[:, 3]
    lpt2 = lpx ** 2 + lpy ** 2
    mu = 0.5 * mw ** 2 + lpx * mex + lpy * mey
    a = mu * lpz / lpt2
    disc = a ** 2 - (le ** 2 * (mex ** 2 + mey ** 2) - mu ** 2) / lpt2
    root = np.sqrt(np.clip(disc, 0, None))
    return np.stack([a - root, a + root])          # (2, n)


def chi2_float(d, idx, njmax=6, use_btag=True, massless=False):
    """Same algorithm as plotting.physics._reconstruct_ttbar (generalized)."""
    pt = d["jet_pt"][idx, :njmax]; eta = d["jet_eta"][idx, :njmax]; phi = d["jet_phi"][idx, :njmax]
    m = np.zeros_like(pt) if massless else d["jet_mass"][idx, :njmax]
    J = p4_from_ptetaphim(pt, eta, phi, m)                       # (n, nj, 4), NaN for missing
    lm = np.zeros(len(idx)) if massless else d["lep_mass"][idx]
    L = p4_from_ptetaphim(d["lep_pt"][idx], d["lep_eta"][idx], d["lep_phi"][idx], lm)
    mex, mey = d["met"][idx] * np.cos(d["met_phi"][idx]), d["met"][idx] * np.sin(d["met_phi"][idx])
    pz = nu_pz(L, mex, mey)
    nue = np.sqrt(mex ** 2 + mey ** 2 + pz ** 2)
    NU = np.stack([np.broadcast_to(mex, pz.shape), np.broadcast_to(mey, pz.shape), pz, nue], -1)  # (2, n, 4)
    perms = np.array([p for p in permutations(range(njmax), 4) if p[2] < p[3]])
    bh, bl, q1, q2 = perms.T
    W = J[:, q1] + J[:, q2]
    TH = W + J[:, bh]
    c_had = ((mass(W) - M_W) / SIG_W) ** 2 + ((mass(TH) - M_TOP) / SIG_T) ** 2
    chi2 = np.concatenate([c_had + ((mass(L[:, None] + NU[s][:, None] + J[:, bl]) - M_TOP) / SIG_T) ** 2
                           for s in (0, 1)], axis=1)
    if use_btag:
        jb = d["jet_btag"][idx, :njmax]
        need = np.minimum(d["nbjets"][idx], 2)
        nt = np.tile(jb[:, bh].astype(int) + jb[:, bl].astype(int), 2)
        chi2 = np.where((nt == need[:, None]) & np.isfinite(chi2), chi2, np.inf)
    else:
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
    best = np.argmin(chi2, axis=1)
    bchi2 = chi2[np.arange(len(idx)), best]
    isol, ip = best // len(perms), best % len(perms)
    sel = perms[ip]
    rows = np.arange(len(idx))
    o = {"ok": np.isfinite(bchi2), "chi2": bchi2, "isol": isol, "sel": sel,
         "b_had": J[rows, sel[:, 0]], "b_lep": J[rows, sel[:, 1]], "q1": J[rows, sel[:, 2]], "q2": J[rows, sel[:, 3]],
         "nu": NU[isol, rows], "lep": L}
    o["w_had"] = o["q1"] + o["q2"]; o["t_had"] = o["w_had"] + o["b_had"]
    o["w_lep"] = o["lep"] + o["nu"]; o["t_lep"] = o["w_lep"] + o["b_lep"]
    return o


def eta_phi(p):
    pt = np.hypot(p[:, 0], p[:, 1])
    return np.arcsinh(p[:, 2] / pt), np.arctan2(p[:, 1], p[:, 0])


def dr(a, b):
    ea, pa = eta_phi(a); eb, pb = eta_phi(b)
    dphi = (pa - pb + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((ea - eb) ** 2 + dphi ** 2)


def angles_float(o, mode, pq_down=None, pq_up=None):
    q1, q2 = o["q1"], o["q2"]
    if mode == "truth":
        swap = dr(q1, pq_down) + dr(q2, pq_up) < dr(q1, pq_up) + dr(q2, pq_down)
    else:   # softer: q2 (lower pT) is the down-type
        swap = np.zeros(len(q1), bool)
    down = np.where(swap[:, None], q1, q2)
    X = {}
    X["Y"], X["cos_Theta"] = production_angles(o["t_lep"], o["t_had"])
    X["cos_theta"], X["phi"], X["cos_theta_prime"], X["phi_prime"] = decay_angles(o["t_lep"], o["w_lep"], o["lep"])
    (X["cos_theta_tilde"], X["phi_tilde"], X["cos_theta_prime_tilde"], X["phi_prime_tilde"]) = \
        decay_angles(o["t_had"], o["w_had"], down)
    return X


def surprisal(y, y_ref):
    y0 = 2.5 * min(np.min(y_ref), 0) if np.min(y_ref) < 0 else 0.0
    return -np.log(np.clip(y - y0, 1e-300 + (np.min(y_ref) - y0) * 1e-3, None))


def cauchy_s(m, M, W):
    return np.log((m - M) ** 2 + W ** 2)


if __name__ == "__main__":
    D = {k: load(k) for k in ("tt_test", "tt_calib", "hh")}
    f = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", "f_ttbar_reco.npz"))
    f = f.truncated(f.significance() > 10)
    res = {}
    for k, d in D.items():
        idx = np.flatnonzero(d["presel"])
        o = chi2_float(d, idx)
        ok = o["ok"]
        print(k, "presel", len(idx), "ok", ok.sum(), "ref reco_ok", d["reco_ok"].sum(),
              "same ok", np.array_equal(idx[ok], np.flatnonzero(d["reco_ok"])))
        idx = idx[ok]
        o = {kk: (v[ok] if (hasattr(v, "shape") and v.shape[:1] == ok.shape) else v) for kk, v in o.items()}
        o["nu"] = o["nu"]
        X = angles_float(o, "truth", d["parton_q_down"][idx], d["parton_q_up"][idx])
        # compare with the reference
        for a in ANGLE_NAMES:
            ref = d[f"ref_{a}"][idx]
            dd = np.abs(X[a] - ref)
            if a.startswith("phi"):
                dd = np.minimum(dd, 2 * np.pi - dd)
            print(f"   {a:24s} max|diff| {np.nanmax(dd):.2e}")
        for kk in ("t_had", "t_lep", "w_had"):
            print(f"   {kk} max|diff|", np.nanmax(np.abs(o[kk] - d[f'ref_{kk}'][idx])))
        res[k] = (X, {"m_w_had": mass(o["w_had"]), "m_t_lep": mass(o["t_lep"]), "m_t_had": mass(o["t_had"])})
    # calibration
    Xc, mc = res["tt_calib"]
    MW = {}
    for kk, x in mc.items():
        M = np.median(x); q1, q3 = np.percentile(x, [25, 75]); MW[kk] = (M, 0.5 * (q3 - q1))
    print("M, W:", MW)
    fr = f(Xc)
    ft, fh = f(res["tt_test"][0]), f(res["hh"][0])
    lab = np.r_[np.zeros(len(ft)), np.ones(len(fh))]
    sa = np.r_[surprisal(ft, fr), surprisal(fh, fr)]
    sm = sum(np.r_[cauchy_s(res["tt_test"][1][kk], *MW[kk]), cauchy_s(res["hh"][1][kk], *MW[kk])] for kk in MW)
    print("AUC angles", roc_auc_score(lab, sa), "masses", roc_auc_score(lab, sm), "angles+BW", roc_auc_score(lab, sa + sm))
    print("min f calib", fr.min(), "y0", 2.5 * fr.min())
