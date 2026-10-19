#!/usr/bin/env python3
"""
A factorized density beyond the ten angles, as the prior of study_vae:

    f(x) = f(angles | m_tt̄) * h(m_tt̄) [* g(m_W^had, m_t^had, m_t^lep)]

  f(angles | m_tt̄)   the paper's OSDE density of the ten angles, fit separately in NB quantile bins of m_tt̄ on the
                     parton-level (after trigger) tt̄ fit sample, truncated at |c|/sigma > 2.5 per bin, normalized per bin
  h(m_tt̄)            a one-dimensional kernel density of ln m_tt̄ (Gaussian kernel, Scott's bandwidth) on the same sample
                     ("parton"), or on the chi2-reconstructed m_tt̄ of the reco-level tt̄ fit sample ("reco")
  g(masses)          a three-dimensional histogram of the three reconstructed resonance masses of the reco-level tt̄ fit
                     sample (the chi2 assignment), smoothed by a Gaussian kernel of Scott's bandwidth per axis, i.e. a
                     histogram-then-kernel density in the three dimensions where that is affordable

A histogram in all 14 dimensions is not possible (10^5 events); the factorization keeps the kernel estimates in
one and three dimensions, and the angles in the OSDE basis.  The angular density is shifted by a common y0 (the
largest |min f| over the bin fits on their fit samples) to be positive, as in score.py; h and g are positive.

Scores (high = less tt̄-like), per candidate of candidates.py, for the best (lowest chi2) candidate and for E_q[s]
with chi2 weights (T = 1):   s = -ln(f + y0) - ln h - ln g   for each combination of factors in COMBOS.
AUC of HH(170) and hh(125) against the independent tt̄ test sample (run_seed12345), with tt_pu0 (50k) in brackets.

Note: the single-factor AUCs do not depend on the y0 shift, but the products do, at the level of the bootstrap error
(angles * g for HH170: 0.629 at y0 -> 0, 0.626 nominal, 0.611 at y0 -> inf); the ordering of the combinations does not change.
Verified independently (recomputation, code review, physics critique) on 2026-10-19.
Writes results/analysis/study_vae/factorized.json and factorized.md.
Run: OMP_NUM_THREADS=6 .venv/bin/python code/analysis/study_vae/factorized.py [--bins 4]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CODE)
import repo_paths  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
from analysis.study_vae.score import auc, auc_err, shift_y0  # noqa: E402

T_CUT = 2.5
MASSES = ("m_w_had", "m_t_had", "m_t_lep")
SAMPLES = ("tt_test", "tt_pu0", "HH170", "hh125")
COMBOS = {                                   # name: (angular density, h, g)
    "angles": ("uncond", None, False),
    "angles|mtt": ("cond", None, False),
    "h(mtt) parton": (None, "parton", False),
    "h(mtt) reco": (None, "reco", False),
    "angles * h": ("uncond", "parton", False),
    "angles|mtt * h": ("cond", "parton", False),
    "g(masses)": (None, None, True),
    "angles * g": ("uncond", None, True),
    "angles|mtt * h * g": ("cond", "parton", True),
    "angles|mtt * h(reco) * g": ("cond", "reco", True),
}
OUT = repo_paths.results_dir("study_vae")


def kde1(x_fit, x):
    """Gaussian kernel density of x (one dimension) from the sample x_fit, Scott's bandwidth."""
    x_fit = x_fit[np.isfinite(x_fit)]
    bw = 1.06 * np.std(x_fit) * len(x_fit) ** (-0.2)
    out = np.zeros(len(x))
    for lo in range(0, len(x), 2000):
        d = (x[lo:lo + 2000, None] - x_fit[None, :]) / bw
        out[lo:lo + 2000] = np.exp(-0.5 * d ** 2).sum(axis=1) / (len(x_fit) * bw * np.sqrt(2 * np.pi))
    return out


class SmoothedHistogram:
    """A histogram in d dimensions smoothed by a Gaussian kernel (Scott's bandwidth per axis), linearly interpolated."""

    def __init__(self, data, nbins=60, pad=3.0):
        data = data[np.all(np.isfinite(data), axis=1)]
        n, d = data.shape
        self.bw = np.std(data, axis=0) * n ** (-1.0 / (d + 4))
        lo, hi = data.min(axis=0) - pad * self.bw, data.max(axis=0) + pad * self.bw
        edges = [np.linspace(lo[i], hi[i], nbins + 1) for i in range(d)]
        H, _ = np.histogramdd(data, bins=edges)
        widths = np.array([e[1] - e[0] for e in edges])
        H = gaussian_filter(H, sigma=self.bw / widths, mode="constant")
        H /= H.sum() * np.prod(widths)                      # a density
        centers = [0.5 * (e[1:] + e[:-1]) for e in edges]
        self.floor = H[H > 0].min()
        self.interp = RegularGridInterpolator(centers, H, bounds_error=False, fill_value=0.0)

    def __call__(self, x):
        return np.maximum(self.interp(x), self.floor)        # never zero: outside the grid = the smallest nonzero bin


def fit_conditional(cache, level, nbins):
    """OSDE fits of the ten angles in quantile bins of m_tt̄ on the tt̄ fit sample: (edges, [density per bin])."""
    X = {a: cache[f"{level}/{a}"] for a in ANGLE_NAMES}
    m = cache[f"{level}/mtt"]
    edges = np.quantile(m, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    fits, info = [], []
    for b in range(nbins):
        sel = (m >= edges[b]) & (m < edges[b + 1])
        f = osde.TTbarDensity().fit({a: v[sel] for a, v in X.items()})
        f = f.truncated(f.significance() > T_CUT)
        fits.append(f)
        info.append({"n": int(sel.sum()), "kept_terms": int((f.c != 0).sum()),
                     "y0": shift_y0(f({a: v[sel] for a, v in X.items()}))})
    return edges, fits, info


def evaluate_angles(fits, edges, X, m):
    """f(angles | bin of m) for candidates: X {name: (n, K)}, m (n, K)."""
    n, k = m.shape
    out = np.full((n, k), np.nan)
    b = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, len(fits) - 1)
    for i, f in enumerate(fits):
        sel = (b == i) & np.isfinite(m)
        if sel.any():
            out[sel] = f({a: X[a][sel] for a in ANGLE_NAMES})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bins", type=int, default=4, help="quantile bins of m_tt for the conditional angular density")
    a = p.parse_args()
    cache = np.load(repo_paths.results("study_gonogo", "cache", "tt_fit.npz"))
    indep = np.load(repo_paths.results("study_gonogo", "cache", "tt_indep.npz"))
    cands = {k: dict(np.load(os.path.join(OUT, f"cand_{k}.npz"))) for k in SAMPLES}

    # the densities, all from the tt̄ fit sample (run_seed1001)
    f_un = osde.load_density(repo_paths.results("osde", "f_ttbar_parton_trig.npz"))
    f_un = f_un.truncated(f_un.significance() > T_CUT)
    X_fit = {a: cache[f"parton_trig/{a}"] for a in ANGLE_NAMES}
    y0_un = shift_y0(f_un(X_fit))
    edges, fits, info = fit_conditional(cache, "parton_trig", a.bins)
    y0 = max([y0_un] + [i["y0"] for i in info])
    m_parton, m_reco = cache["parton_trig/mtt"], cache["reco/mtt"]
    g = SmoothedHistogram(np.stack([cache[f"reco/{m}"] for m in MASSES], axis=1))
    res = {"bins": a.bins, "mtt_edges": [float(e) for e in edges[1:-1]], "per_bin": info, "y0": y0,
           "kept_terms_unconditional": int((f_un.c != 0).sum()), "g_bandwidth_GeV": [float(b) for b in g.bw],
           "h_bandwidth_ln_mtt": {"parton": float(1.06 * np.std(np.log(m_parton)) * len(m_parton) ** -0.2),
                                  "reco": float(1.06 * np.std(np.log(m_reco)) * len(m_reco) ** -0.2)},
           "n_selected": {k: int(c["reco_ok"].sum()) for k, c in cands.items()}, "auc": {}}
    # consistency: the best candidate's masses of the independent sample equal the cached reco masses
    c = cands["tt_test"]; ok = c["reco_ok"]
    same = all(np.allclose(c[m][ok, 0], indep[f"reco/{m}"], atol=1e-4) for m in MASSES + ("mtt",))
    res["best_candidate_masses_match_cache"] = bool(same)
    print("best-candidate masses of tt_test match the tt_indep cache:", same, flush=True)

    # per-candidate factors
    fac = {}
    for k, c in cands.items():
        ok = c["reco_ok"]
        chi2 = c["chi2"][ok]
        X = {a: c[a][ok] for a in ANGLE_NAMES}
        m = c["mtt"][ok]
        valid = np.isfinite(chi2) & np.isfinite(m) & np.all([np.isfinite(X[a]) for a in ANGLE_NAMES], axis=0)
        n, kk = chi2.shape
        flat = {a: X[a][valid] for a in ANGLE_NAMES}
        ang = {"uncond": np.full((n, kk), np.nan), "cond": np.full((n, kk), np.nan)}
        ang["uncond"][valid] = f_un(flat)
        ang["cond"][valid] = evaluate_angles(fits, edges, X, np.where(valid, m, np.nan))[valid]
        h = {"parton": np.full((n, kk), np.nan), "reco": np.full((n, kk), np.nan)}
        h["parton"][valid] = kde1(np.log(m_parton), np.log(m[valid])) / m[valid]     # density in m from ln m
        h["reco"][valid] = kde1(np.log(m_reco), np.log(m[valid])) / m[valid]
        gm = np.full((n, kk), np.nan)
        gm[valid] = g(np.stack([c[mm][ok][valid] for mm in MASSES], axis=1))
        fac[k] = dict(chi2=chi2, valid=valid, ang=ang, h=h, g=gm)
        res.setdefault("n_floored", {})[k] = {kind: int(np.sum(valid & (ang[kind] + y0 <= 0))) for kind in ang}

    def score(fk, combo):
        kind, hk, use_g = COMBOS[combo]
        chi2, valid = fk["chi2"], fk["valid"]
        s = np.zeros(chi2.shape)
        if kind:
            s = s - np.log(np.maximum(fk["ang"][kind] + y0, y0 / 100.0))
        if hk:
            s = s - np.log(fk["h"][hk])
        if use_g:
            s = s - np.log(fk["g"])
        s = np.where(valid, s, np.nan)
        logw = np.where(valid, -(chi2 - np.nanmin(np.where(valid, chi2, np.nan), axis=1, keepdims=True)) / 2, -np.inf)
        w = np.exp(logw); w /= w.sum(axis=1, keepdims=True)
        return {"best": s[:, 0], "Eq_s": np.where(valid, w * s, 0.0).sum(axis=1)}

    lines = [f"# study_vae: factorized prior f(angles | m_tt) h(m_tt) g(masses), {a.bins} m_tt bins", "",
             f"m_tt bin edges (GeV): {[round(e) for e in res['mtt_edges']]}; kept terms per bin: "
             f"{[i['kept_terms'] for i in info]} (unconditional {res['kept_terms_unconditional']}); "
             f"g bandwidths (GeV): {[round(b, 1) for b in res['g_bandwidth_GeV']]}", "",
             "AUC against the independent tt̄ test sample; in brackets against tt_pu0.", "",
             "| density | score | HH170 | hh125 |", "|---|---|---|---|"]
    for combo in COMBOS:
        sc = {k: score(fk, combo) for k, fk in fac.items()}
        for sname in ("best", "Eq_s"):
            cells = []
            for sig in ("HH170", "hh125"):
                vals = []
                for bk in ("tt_test", "tt_pu0"):
                    b_, s_ = sc[bk][sname], sc[sig][sname]
                    m_ = np.isfinite(b_), np.isfinite(s_)
                    v = [round(auc(b_[m_[0]], s_[m_[1]]), 4), round(auc_err(b_[m_[0]], s_[m_[1]]), 4)]
                    res["auc"][f"{combo}/{sname}/{sig}_vs_{bk}"] = v
                    vals.append(v)
                cells.append(f"{vals[0][0]:.3f} ± {vals[0][1]:.3f} ({vals[1][0]:.3f})")
            lines.append(f"| {combo} | {sname} | {cells[0]} | {cells[1]} |")
        print(lines[-2], flush=True)
    json.dump(res, open(os.path.join(OUT, "factorized.json"), "w"), indent=1)
    open(os.path.join(OUT, "factorized.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
