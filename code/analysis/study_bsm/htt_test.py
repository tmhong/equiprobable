#!/usr/bin/env python3
"""
BSM test: does the unsupervised f_tt̄ catch a signal that an hh-trained supervised classifier misses?

Signal: g g -> H(500 GeV) -> t t̄ -> b b̄ l nu q q' (heft model, SM-like couplings, width by MG5),
samples/Htt_bblvqq/run_seed3002.  Unlike hh it has on-shell tops and Ws: it differs from QCD tt̄
only in the spin structure (scalar: isotropic cos Θ, like-helicity tops), the rapidity and m_tt̄.

Compared at parton and reco level, hh vs tt̄ and H->tt̄ vs tt̄, same test events:
  unsupervised  f_tt̄ (paper basis, |c|/σ>10 terms) on the angles; the Breit-Wigner mass factors;
                a 1-D tt̄ density of m_tt̄ (histogram of the fit sample); combinations (surprisals add)
  supervised    BDT trained on hh vs tt̄ (angles; angles + masses + m_tt̄), applied to BOTH signals
  ceiling       BDT trained on H->tt̄ vs tt̄ (what a dedicated search could do)
Output: results/study_bsm/htt_test.json
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, ROOT)
import repo_paths  # noqa: E402
from plotting import make_plots  # noqa: E402
from analysis.study_salvage.salvage import surprisal  # noqa: E402
from analysis.study_salvage.simplest import PDG, bw_surprisal, cauchy_surprisal  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
SAMPLES = {"tt_fit": "samples/ttbar_bblvqq/run_seed1001", "tt": "samples/ttbar_bblvqq/run_seed12345",
           "hh": "samples/hh_bblvqq/run_seed2001", "Htt": "samples/Htt_bblvqq/run_seed3002"}


def extract(path, level, nmax=None):
    """angles, resonance masses and m_tt̄ for events with a tt̄ system (parton: all; reco: reco_ok)."""
    levels = ["parton"] if level == "parton" else ["parton", level]
    ev, _ = make_plots.load_sample("x", path, levels, nmax, OPTS)
    e = ev[level]
    ok = np.ones(len(e), bool) if level == "parton" else np.asarray(e.reco_ok)
    ang = {a: np.asarray(e[a], float) for a in ANGLE_NAMES}
    ok &= np.all([np.isfinite(v) for v in ang.values()], 0)
    m = {"m_w_had": e.w_had.mass, "m_t_lep": e.t_lep.mass, "m_t_had": e.t_had.mass}
    if level == "parton":
        m["m_w_lep"] = e.w_lep.mass
    m = {k: np.asarray(v, float) for k, v in m.items()}
    mtt = np.asarray((e.t_lep + e.t_had).mass, float)
    return {"ang": {a: v[ok] for a, v in ang.items()}, "m": {k: v[ok] for k, v in m.items()}, "mtt": mtt[ok]}


class HistDensity:
    """1-D density from a histogram (overflow into the edge bins), evaluated by lookup."""

    def __init__(self, x, lo, hi, nb=40):
        self.edges = np.linspace(lo, hi, nb + 1)
        h, _ = np.histogram(np.clip(x, lo, hi - 1e-6), self.edges, density=True)
        self.h = np.maximum(h, 1e-6)

    def surprisal(self, x):
        i = np.clip(np.digitize(np.clip(x, self.edges[0], self.edges[-1] - 1e-6), self.edges) - 1, 0, len(self.h) - 1)
        return -np.log(self.h[i])


def feats(d, which):
    cols = [d["ang"][a] for a in ANGLE_NAMES]
    if which == "all":
        cols += [d["m"][k] for k in sorted(d["m"])] + [d["mtt"]]
    return np.column_stack(cols)


def bdt(Xs, ys, seed=0):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
    return clf.fit(np.vstack(Xs), np.r_[tuple(ys)])


def auc_err(lab, score, n=100, seed=0):
    rng = np.random.default_rng(seed)
    a = roc_auc_score(lab, score)
    bs = [roc_auc_score(lab[i], score[i]) for i in (rng.integers(0, len(lab), len(lab)) for _ in range(n))]
    return float(a), float(np.std(bs))


def main():
    out = {}
    for level, fname in (("parton", "f_ttbar_parton.npz"), ("reco", "f_ttbar_reco.npz")):
        D = {k: extract(os.path.join(repo_paths.REPO, p), level, 30000 if k == "tt_fit" else None) for k, p in SAMPLES.items()}
        n = {k: len(v["mtt"]) for k, v in D.items()}
        half = {k: np.arange(n[k]) < n[k] // 2 for k in ("hh", "Htt", "tt_fit")}   # train / test halves
        f = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", fname))
        f = f.truncated(f.significance() > 10)
        fref = f(D["tt_fit"]["ang"])
        ref = D["tt_fit"]
        mtt_den = HistDensity(ref["mtt"], 300, 1500)
        # mass factors: PDG Breit-Wigner at parton, Cauchy with tt̄ median / half-IQR at reco
        def s_mass(d):
            s = 0
            for k, x in d["m"].items():
                if level == "parton":
                    s = s + bw_surprisal(x, *PDG[k])
                else:
                    r = ref["m"][k]; r = r[np.isfinite(r)]
                    M = np.median(r); q1, q3 = np.percentile(r, [25, 75])
                    s = s + cauchy_surprisal(x, M, 0.5 * (q3 - q1))
            return s
        unsup = {}
        for k, d in D.items():
            if k == "tt_fit":
                continue
            sa = surprisal(f(d["ang"]), fref)
            unsup[k] = {"angles": sa, "masses": s_mass(d), "mtt": mtt_den.surprisal(d["mtt"])}
        # supervised: hh-trained and Htt-trained BDTs, angles only and all inputs
        models = {}
        for sig in ("hh", "Htt"):
            for which in ("angles", "all"):
                models[(sig, which)] = bdt([feats(D[sig], which)[half[sig]], feats(D["tt_fit"], which)[half["tt_fit"]]],
                                           [np.ones(half[sig].sum()), np.zeros(half["tt_fit"].sum())])
        res = {"n_test": {"tt": n["tt"], "hh": int((~half["hh"]).sum()), "Htt": int((~half["Htt"]).sum())},
               "mtt_median": {k: float(np.median(D[k]["mtt"])) for k in ("tt", "hh", "Htt")}}
        for sig in ("hh", "Htt"):
            t = ~half[sig]
            lab = np.r_[np.zeros(n["tt"]), np.ones(t.sum())]
            r = {}
            u_tt, u_s = unsup["tt"], unsup[sig]
            combos = {"f_tt angles (>10 sigma)": ("angles",), "BW masses": ("masses",), "m_tt density": ("mtt",),
                      "angles + BW": ("angles", "masses"), "angles + BW + m_tt": ("angles", "masses", "mtt")}
            for name, parts in combos.items():
                sc = np.r_[sum(u_tt[p] for p in parts), sum(u_s[p][t] for p in parts)]
                r["unsupervised | " + name] = auc_err(lab, sc)
            for (trained, which), clf in models.items():
                sc = np.r_[clf.predict_proba(feats(D["tt"], which))[:, 1], clf.predict_proba(feats(D[sig], which)[t])[:, 1]]
                r[f"supervised trained on {trained} | {'angles' if which == 'angles' else 'angles + masses + m_tt'}"] = auc_err(lab, sc)
            res[f"{sig} vs tt"] = r
        out[level] = res
        print(level, json.dumps(res, indent=1), flush=True)
    with open(os.path.join(repo_paths.results_dir("study_bsm"), "htt_test.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
