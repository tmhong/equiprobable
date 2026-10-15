#!/usr/bin/env python3
"""
Feasibility test: put the off-shell (momentum) information into the density method by
extending f_tt̄ from the ten angles to the resonance masses.

The paper's angles are coordinates on the narrow-width manifold (both W and both tops on
shell).  Off shell, the phase space has extra directions: m_W,lep, m_W,had, m_t,lep, m_t,had
(+ m_tt̄).  Here f_tt̄ = f_angles(Ω) × g(m), with
  f_angles  the existing paper-basis OSDE (results/osde/f_ttbar_<level>.npz, fit on 100k tt̄; tt̄ events
            scored leave-one-out because they were in that fit)
  g(m)      an OSDE in the masses: each mass mapped to u = 1/2 + arctan((m - M)/W)/π (Cauchy
            CDF, M = median, W = half the interquartile range of tt̄), orthonormal Legendre
            products in u up to total degree D (D by held-out ISE), and the Jacobian du/dm so
            that g is a density in m.
Score = total surprisal  s = -ln(f_angles - y0_a) - ln(g - y0_g)  (y0 = 2.5 × min, as for t).
The product assumes angles and masses independent for tt̄; a supervised BDT on the same
inputs is the reference for what they contain.

Input: results/study_washout/mechanism/data_{tt,hh}.npz (40k tt̄ from the fit sample, 20k hh).
Output: results/study_salvage/salvage.json, printed table.
"""
import itertools
import json
import os
import sys

import numpy as np
from numpy.polynomial import legendre
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, ROOT)
import repo_paths  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

LEVELS = {
    # level: (prefix, event mask key or None, mass variables, osde file)
    "parton": ("p_", None, ["m_w_lep", "m_w_had", "m_t_lep", "m_t_had"], "f_ttbar_parton.npz"),
    "gen": ("c_", "g_reco_ok", ["m_w_had", "m_t_lep", "m_t_had"], "f_ttbar_gen.npz"),
    # + the b-pair mass (h_m_bb, the two b-tagged/χ² b jets; a function of the full tt̄ phase space)
    "gen+m_bb": ("c_", "g_reco_ok", ["m_w_had", "m_t_lep", "m_t_had", "h_m_bb"], "f_ttbar_gen.npz"),
}


class MassDensity:
    """Orthonormal Legendre expansion in Cauchy-mapped masses (total degree <= D)."""

    def __init__(self, degree):
        self.D = degree

    def _u(self, M):
        return 0.5 + np.arctan((M - self.mid) / self.w) / np.pi

    def _basis(self, M):
        u = np.clip(self._u(M), 1e-9, 1 - 1e-9)
        P = [np.stack([np.sqrt(2 * n + 1) * legendre.legval(2 * u[:, k] - 1, np.eye(n + 1)[n])
                       for n in range(self.D + 1)], 1) for k in range(M.shape[1])]
        return np.stack([np.prod([P[k][:, n] for k, n in enumerate(idx)], 0) for idx in self.idx], 1)

    def fit(self, M):
        self.mid = np.median(M, 0)
        q1, q3 = np.percentile(M, [25, 75], 0)
        self.w = 0.5 * (q3 - q1)
        self.idx = [i for i in itertools.product(range(self.D + 1), repeat=M.shape[1]) if sum(i) <= self.D]
        self.c = self._basis(M).mean(0)
        return self

    def density_u(self, M):
        return self._basis(M) @ self.c

    def __call__(self, M):
        jac = np.prod(self.w / (np.pi * ((M - self.mid) ** 2 + self.w ** 2)), 1)
        return self.density_u(M) * jac

    def heldout_ise(self, M):
        return float(np.sum(self.c ** 2) - 2 * self.density_u(M).mean())


def surprisal(y, y_ref):
    y0 = 2.5 * min(np.min(y_ref), 0) if np.min(y_ref) < 0 else 0.0
    # events below the reference minimum are clipped just above y0 (maximally surprising)
    return -np.log(np.clip(y - y0, 1e-300 + (np.min(y_ref) - y0) * 1e-3, None))


def bdt_auc(Xtt, Xhh, seed=1):
    rng = np.random.default_rng(seed)
    X = np.vstack([Xtt, Xhh]); y = np.r_[np.zeros(len(Xtt)), np.ones(len(Xhh))]
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    k = len(y) // 2
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
    clf.fit(X[:k], y[:k])
    return roc_auc_score(y[k:], clf.predict_proba(X[k:])[:, 1])


def main():
    tt = np.load(os.path.join(repo_paths.RESULTS, "study_washout/mechanism/data_tt.npz"))
    hh = np.load(os.path.join(repo_paths.RESULTS, "study_washout/mechanism/data_hh.npz"))
    out = {}
    for lvl, (pre, mkey, masses, fname) in LEVELS.items():
        sel = lambda z: (z[mkey].astype(bool) if mkey else np.ones(len(z["p_Y"]), bool))  # noqa: E731
        mt, mh = sel(tt), sel(hh)
        ang = lambda z, m: {a: z[pre + a][m] for a in ANGLE_NAMES}  # noqa: E731
        col = lambda z, v, m: z[v if v.startswith("h_") else pre + v][m]  # noqa: E731
        Mt = np.stack([col(tt, v, mt) for v in masses], 1)
        Mh = np.stack([col(hh, v, mh) for v in masses], 1)
        ok_t, ok_h = np.all(np.isfinite(Mt), 1), np.all(np.isfinite(Mh), 1)
        # angles part: existing f (fit on 100k tt̄ incl. these events) -> tt̄ leave-one-out
        f = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", fname))
        fa_t, fa_h = f.loo(ang(tt, mt)), f(ang(hh, mh))
        # split tt̄: half A fits g, half B tests
        rng = np.random.default_rng(7)
        isA = rng.random(len(Mt)) < 0.5
        A, B = ok_t & isA, ok_t & ~isA
        # choose degree on a validation part of A
        vA = rng.random(len(Mt)) < 0.25
        best = min(((MassDensity(D).fit(Mt[A & ~vA]).heldout_ise(Mt[A & vA]), D) for D in (2, 4, 6, 8, 10, 12, 14, 16)))
        g = MassDensity(best[1]).fit(Mt[A])
        g_t, g_h = g(Mt[B]), g(Mh[ok_h])
        lab = np.r_[np.zeros(B.sum()), np.ones(ok_h.sum())]
        sa = np.r_[surprisal(fa_t[B], fa_t[A]), surprisal(fa_h[ok_h], fa_t[A])]
        sm = np.r_[surprisal(g_t, g(Mt[A])), surprisal(g_h, g(Mt[A]))]
        res = {"n_tt_test": int(B.sum()), "n_hh": int(ok_h.sum()), "mass_vars": masses, "degree": best[1],
               "auc_angles": roc_auc_score(lab, sa), "auc_masses": roc_auc_score(lab, sm),
               "auc_angles_x_masses": roc_auc_score(lab, sa + sm)}
        Xa_t = np.column_stack([tt[pre + a][mt] for a in ANGLE_NAMES]); Xa_h = np.column_stack([hh[pre + a][mh] for a in ANGLE_NAMES])
        res["bdt_angles"] = bdt_auc(Xa_t[ok_t], Xa_h[ok_h])
        res["bdt_angles_masses"] = bdt_auc(np.column_stack([Xa_t, Mt])[ok_t], np.column_stack([Xa_h, Mh])[ok_h])
        # independence check for tt̄: can a BDT tell real (angles, masses) pairs from shuffled pairs?
        perm = rng.permutation(ok_t.sum())
        real = np.column_stack([Xa_t, Mt])[ok_t]
        shuf = np.column_stack([Xa_t[ok_t], Mt[ok_t][perm]])
        res["tt_independence_auc"] = bdt_auc(real, shuf)
        out[lvl] = res
        print(f"[{lvl}] " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))
    with open(os.path.join(repo_paths.results_dir("study_salvage"), "salvage.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
