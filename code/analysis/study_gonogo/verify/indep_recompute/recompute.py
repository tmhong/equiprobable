#!/usr/bin/env python3
"""Independent recomputation of the go/no-go numbers (does NOT import study_gonogo/analyze.py).

Only the OSDE basis functions (TTbarDensity.factors) and the stored coefficients are taken from the repo; the
density evaluation, the leave-one-out correction, the mass scores, the working points, the m_tt windows, the AUCs
and the bootstrap are written here from scratch.

Run: OMP_NUM_THREADS=4 .venv/bin/python code/study_gonogo/verify/indep_recompute/recompute.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from lib import osde  # noqa: E402
import repo_paths  # noqa: E402

RES = repo_paths.RESULTS                                    # inputs: FFILE, the caches, results.json
OUT = repo_paths.results_dir("study_gonogo", "verify", "indep_recompute")

ANG = ["Y", "cos_Theta", "cos_theta", "phi", "cos_theta_prime", "phi_prime",
       "cos_theta_tilde", "phi_tilde", "cos_theta_prime_tilde", "phi_prime_tilde"]
MASS = ["m_w_had", "m_t_lep", "m_t_had"]
SIG = ["etat", "stop", "H350", "nospin", "tt_indep"]
RUNS = {"tt_fit": ["run_seed1001"], "tt_indep": ["run_seed12345"], "etat": ["run_seed4001"],
        "stop": ["run_seed4002"], "H350": ["run_m350_seed3003"], "nospin": ["run_seed4003", "run_seed4004"]}
FFILE = {"parton_trig": "osde/f_ttbar_parton_trig.npz", "gen": "osde/f_ttbar_gen.npz",
         "reco": "osde/f_ttbar_reco.npz", "matched": "study_truthmatch/osde_dRmatch/f_ttbar_reco.npz"}  # under results/
WPS = [0.01, 0.10]
NBOOT = 300


def get(key, level):
    if level != "matched":
        z = np.load(os.path.join(RES, "study_gonogo/cache", key + ".npz"))
        return {"X": {a: np.asarray(z[f"{level}/{a}"], float) for a in ANG},
                "m": {k: np.asarray(z[f"{level}/{k}"], float) for k in MASS},
                "mtt": np.asarray(z[f"{level}/mtt"], float), "n_all": int(z[f"{level}/n_all"])}
    X = {a: [] for a in ANG}
    m = {k: [] for k in MASS}
    mtt, n_all = [], 0
    for r in RUNS[key]:
        z = np.load(os.path.join(RES, "study_truthmatch", f"data_{r}.npz"))
        ok = z["ok"].astype(bool)
        for a in ANG:
            X[a].append(z["m_" + a][ok])
        for k in MASS:
            m[k].append(z["m_" + k][ok])
        mtt.append(z["m_mtt"][ok])
        n_all += int(z["n_generated"])
    return {"X": {a: np.concatenate(v).astype(float) for a, v in X.items()},
            "m": {k: np.concatenate(v).astype(float) for k, v in m.items()},
            "mtt": np.concatenate(mtt).astype(float), "n_all": n_all}


# ---------------------------------------------------------------- OSDE evaluation (own contraction)
def basis(f, X):
    return f.factors(X)  # (M_lep (n,24) c, M_had (n,24) c, P (n,5), H (n,5))


def f_eval(c, fac, chunk=8000):
    ml, mh, p, h = fac
    out = np.empty(len(p))
    for lo in range(0, len(p), chunk):
        s = slice(lo, lo + chunk)
        ph = p[s][:, :, None] * h[s][:, None, :]                       # (k, l, n)
        T = np.tensordot(ph.reshape(len(ph), -1), c.reshape(c.shape[0] * c.shape[1], -1).T, axes=(1, 0))
        T = T.reshape(len(ph), c.shape[0], c.shape[1])                  # (k, j, i)
        out[s] = np.real(np.sum(T * ml[s][:, :, None] * mh[s][:, None, :], axis=(1, 2)))
    return out


def self_term(mask, fac, chunk=8000):
    ml, mh, p, h = fac
    out = np.empty(len(p))
    w = mask.astype(float)
    for lo in range(0, len(p), chunk):
        s = slice(lo, lo + chunk)
        a = (np.abs(ml[s]) ** 2)[:, :, None] * (np.abs(mh[s]) ** 2)[:, None, :]       # (k,j,i)
        b = (p[s] ** 2)[:, :, None] * (h[s] ** 2)[:, None, :]                         # (k,l,n)
        out[s] = np.einsum("kji,jiln,kln->k", a, w, b)
    return out


def own_fit(fac):
    ml, mh, p, h = fac
    n = len(p)
    c = np.einsum("kj,ki,kl,kn->jiln", ml.conj(), mh.conj(), p, h, optimize=True) / n
    s2 = np.einsum("kj,ki,kl,kn->jiln", abs(ml) ** 2, abs(mh) ** 2, p ** 2, h ** 2, optimize=True) / n
    err = np.sqrt(np.clip(s2 - abs(c) ** 2, 0, None) / n)
    return c, err


# ---------------------------------------------------------------- metrics (own)
def thr(b, eb):
    """Cut value keeping exactly k = round(eb n) tt events (midpoint between the k-th and (k+1)-th largest)."""
    sb = np.sort(b)[::-1]
    k = int(round(eb * len(sb)))
    return 0.5 * (sb[k - 1] + sb[k]), k / len(sb)


def R_at(s, b, eb):
    t, e = thr(b, eb)
    return float(np.mean(s > t) / e)


def auc(s, b):
    r = rankdata(np.r_[b, s])
    rs = r[len(b):].sum()
    return float((rs - len(s) * (len(s) + 1) / 2) / (len(s) * len(b)))


def bw(tt, d):
    out = np.zeros(len(d["mtt"]))
    for k in MASS:
        M = np.median(tt["m"][k])
        q1, q3 = np.quantile(tt["m"][k], [0.25, 0.75])
        G = (q3 - q1) / 2
        out -= np.log(G / np.pi / ((d["m"][k] - M) ** 2 + G ** 2))     # -ln Cauchy pdf
    return out


def mtt_dens(tt, d, lo=300.0, hi=1500.0, nb=40):
    e = np.linspace(lo, hi, nb + 1)
    w = (hi - lo) / nb
    cnt = np.bincount(np.clip(((np.clip(tt["mtt"], lo, hi - 1e-9) - lo) // w).astype(int), 0, nb - 1), minlength=nb)
    dens = np.maximum(cnt / (len(tt["mtt"]) * w), 1e-6)
    idx = np.clip(((np.clip(d["mtt"], lo, hi - 1e-9) - lo) // w).astype(int), 0, nb - 1)
    return -np.log(dens[idx])


def win_counts(sorted_x, lo, hi):
    return np.searchsorted(sorted_x, hi, "right") - np.searchsorted(sorted_x, lo, "left")


def windows(mb, eb):
    mb = np.sort(mb)
    n = len(mb)
    k = max(int(round(eb * n)), 1)
    return mb[: n - k + 1], mb[k - 1:], k / n


def window_full(ms, mb, eb):
    """Window with exactly round(eb n) tt events maximizing the signal count on the full signal (CRITERIA literal)."""
    lo, hi, e = windows(mb, eb)
    c = win_counts(np.sort(ms), lo, hi)
    i = int(np.argmax(c))
    return c[i] / len(ms) / e, (lo[i], hi[i])


def window_halfsig(ms, mb, eb, rng, nsplit=10):
    """Window chosen on a random half of the signal, eps_S on the other half (tt windows from the full tt)."""
    lo, hi, e = windows(mb, eb)
    out = []
    for _ in range(nsplit):
        perm = rng.permutation(len(ms))
        A, B = np.sort(ms[perm[: len(ms) // 2]]), np.sort(ms[perm[len(ms) // 2:]])
        for a, b in ((A, B), (B, A)):
            i = int(np.argmax(win_counts(a, lo, hi)))
            out.append(win_counts(b, lo[i:i + 1], hi[i:i + 1])[0] / len(b) / e)
    return float(np.mean(out))


def window_fullcv(ms, mb, eb, rng, nsplit=10):
    """Fully cross-validated: window defined on tt half A + signal half A; eps_S and eps_tt measured on the B halves."""
    out = []
    for _ in range(nsplit):
        ps, pb = rng.permutation(len(ms)), rng.permutation(len(mb))
        sA, sB = ms[ps[: len(ms) // 2]], ms[ps[len(ms) // 2:]]
        bA, bB = mb[pb[: len(mb) // 2]], mb[pb[len(mb) // 2:]]
        for (sa, ba, sb, bb) in ((sA, bA, sB, bB), (sB, bB, sA, bA)):
            lo, hi, _ = windows(ba, eb)
            i = int(np.argmax(win_counts(np.sort(sa), lo, hi)))
            es = win_counts(np.sort(sb), lo[i:i + 1], hi[i:i + 1])[0] / len(sb)
            et = win_counts(np.sort(bb), lo[i:i + 1], hi[i:i + 1])[0] / len(bb)
            out.append(es / max(et, 1e-12))
    return float(np.mean(out))


def window_evenodd(ms, mb, eb):
    """Same split as described in analyze.py's docstring (even/odd halves of the signal), for a like-for-like check."""
    lo, hi, e = windows(mb, eb)
    A, B = np.sort(ms[0::2]), np.sort(ms[1::2])
    out = []
    for a, b in ((A, B), (B, A)):
        i = int(np.argmax(win_counts(a, lo, hi)))
        out.append(win_counts(b, lo[i:i + 1], hi[i:i + 1])[0] / len(b) / e)
    return float(np.mean(out))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    ref = json.load(open(os.path.join(RES, "study_gonogo/results.json")))
    out = {}
    for level in ["reco", "matched", "parton_trig", "gen"]:
        full_rule = level in ("reco", "matched")
        f = osde.load_density(os.path.join(RES, FFILE[level]))
        tt = get("tt_fit", level)
        n = len(tt["mtt"])
        L = out.setdefault(level, {})
        L["n_tt"], L["n_fit"] = n, int(f.n_fit)
        fac_tt = basis(f, tt["X"])
        # (0) the tt events are the fit sample: refit and compare coefficients
        c_re, err_re = own_fit(fac_tt)
        L["refit_max_abs_dc"] = float(np.max(np.abs(c_re - f.c)))
        L["refit_max_rel_dc_over_err"] = float(np.max(np.abs(c_re - f.c) / np.where(f.c_err > 0, f.c_err, np.inf)))
        L["refit_max_abs_derr"] = float(np.max(np.abs(err_re - f.c_err)))
        sig = np.abs(f.c) / np.where(f.c_err > 0, f.c_err, np.inf)
        m_cv = sig > 2.5
        m_all = np.abs(f.c) > 0
        c_cv = np.where(m_cv, f.c, 0)
        L["n_terms_cv"], L["n_terms_all"] = int(m_cv.sum()), int(m_all.sum())
        # (1) LOO on tt, own formula
        S_tt = {}
        for name, c, msk in (("osde_cv", c_cv, m_cv), ("osde_all", f.c, m_all)):
            fin = f_eval(c, fac_tt)
            loo = (n * fin - self_term(msk, fac_tt)) / (n - 1)
            S_tt[name] = -loo
            S_tt[name + "_insample"] = -fin
            # cross-check with the class methods
            g = f.truncated(msk)
            L[f"{name}_max_dev_vs_class_loo"] = float(np.max(np.abs(loo - g.loo(tt["X"]))))
            L[f"{name}_max_dev_vs_class_call"] = float(np.max(np.abs(fin - g(tt["X"]))))
        # brute-force LOO for a few events (refit without event k, same truncation mask)
        if level in ("reco", "matched"):
            ml, mh, p, h = fac_tt
            S1 = np.einsum("kj,ki,kl,kn->jiln", ml.conj(), mh.conj(), p, h, optimize=True)
            bf = []
            for k in [0, 17, n // 2, n - 1]:
                Bk = np.einsum("j,i,l,n->jiln", ml[k].conj(), mh[k].conj(), p[k], h[k])
                ck = (S1 - Bk) / (n - 1)
                fk = (ml[k:k + 1], mh[k:k + 1], p[k:k + 1], h[k:k + 1])
                for name, msk in (("osde_cv", m_cv), ("osde_all", m_all)):
                    bf.append(abs(f_eval(np.where(msk, ck, 0), fk)[0] + S_tt[name][k]))
            L["bruteforce_loo_max_dev"] = float(max(bf))
        S_tt["bw"] = bw(tt, tt)
        S_tt["mtt_dens"] = mtt_dens(tt, tt)
        L["frac_y_le_0_tt"] = {"osde_cv": float(np.mean(S_tt["osde_cv"] >= 0)), "osde_all": float(np.mean(S_tt["osde_all"] >= 0))}

        for key in SIG:
            d = get(key, level)
            fac = basis(f, d["X"])
            S = {"osde_cv": -f_eval(c_cv, fac), "osde_all": -f_eval(f.c, fac), "bw": bw(tt, d), "mtt_dens": mtt_dens(tt, d)}
            r = {"n": len(d["mtt"]), "eff_sel": len(d["mtt"]) / d["n_all"], "scores": {}}
            rng = np.random.default_rng(20261013)
            for mth in S:
                rr = {"auc": auc(S[mth], S_tt[mth])}
                for eb in WPS:
                    rr[f"R@{eb:g}"] = R_at(S[mth], S_tt[mth], eb)
                r["scores"][mth] = rr
            # in-sample (no LOO) tt reference, to show what LOO does
            r["scores"]["osde_cv_noLOO"] = {f"R@{eb:g}": R_at(S["osde_cv"], S_tt["osde_cv_insample"], eb) for eb in WPS}
            r["scores"]["osde_cv_noLOO"]["auc"] = auc(S["osde_cv"], S_tt["osde_cv_insample"])
            if full_rule:
                wz = {}
                for eb in WPS:
                    Rf, (wlo, whi) = window_full(d["mtt"], tt["mtt"], eb)
                    wz[f"full@{eb:g}"] = Rf
                    wz[f"full_window@{eb:g}"] = [float(wlo), float(whi)]
                    wz[f"evenodd@{eb:g}"] = window_evenodd(d["mtt"], tt["mtt"], eb)
                    wz[f"halfsig_random@{eb:g}"] = window_halfsig(d["mtt"], tt["mtt"], eb, rng)
                    wz[f"fullcv@{eb:g}"] = window_fullcv(d["mtt"], tt["mtt"], eb, rng)
                r["mtt_window"] = wz
                # bootstrap (own seed), paired over both samples
                if key in ("etat", "stop", "H350"):
                    rb = np.random.default_rng(777)
                    reps = {eb: {m: [] for m in ("osde_cv", "bw", "mtt_dens", "win_evenodd", "win_full")} for eb in WPS}
                    for b in range(NBOOT):
                        ib, is_ = rb.integers(0, n, n), rb.integers(0, len(d["mtt"]), len(d["mtt"]))
                        for eb in WPS:
                            for m in ("osde_cv", "bw", "mtt_dens"):
                                reps[eb][m].append(R_at(S[m][is_], S_tt[m][ib], eb))
                            reps[eb]["win_evenodd"].append(window_evenodd(d["mtt"][is_], tt["mtt"][ib], eb))
                            reps[eb]["win_full"].append(window_full(d["mtt"][is_], tt["mtt"][ib], eb)[0])
                    rule = {}
                    for eb in WPS:
                        v = {m: np.array(x) for m, x in reps[eb].items()}
                        R0 = r["scores"]["osde_cv"][f"R@{eb:g}"]
                        sR = float(np.std(v["osde_cv"]))
                        c = {"R": R0, "R_err": sR, "a": bool(R0 >= 2 and R0 - 1 > 3 * sR)}
                        cen = {"bw": r["scores"]["bw"][f"R@{eb:g}"], "mtt_dens": r["scores"]["mtt_dens"][f"R@{eb:g}"],
                               "win_evenodd": wz[f"evenodd@{eb:g}"], "win_full": wz[f"full@{eb:g}"]}
                        for m in cen:
                            sd = float(np.std(v["osde_cv"] - v[m]))
                            c[f"b_{m}"] = {"diff": R0 - cen[m], "sigma": sd, "pass": bool(R0 - cen[m] > 2 * sd),
                                           "R_other_err": float(np.std(v[m]))}
                        # (c) uses results.json flow central value and analyze.py's paired sigma (flow not retrained here)
                        refrule = ref["levels"][level][key]["rule"][f"{eb:g}"]
                        c["c_flow_from_results"] = refrule["c_vs_flow"]
                        c["pass_with_evenodd_window"] = bool(c["a"] and c["b_bw"]["pass"] and c["b_mtt_dens"]["pass"]
                                                             and c["b_win_evenodd"]["pass"] and refrule["c_vs_flow"]["pass"])
                        c["pass_with_full_window"] = bool(c["a"] and c["b_bw"]["pass"] and c["b_mtt_dens"]["pass"]
                                                          and c["b_win_full"]["pass"] and refrule["c_vs_flow"]["pass"])
                        rule[f"{eb:g}"] = c
                    r["rule"] = rule
            L[key] = r
            print(f"[{level}] {key:8s} n={r['n']:6d} " + "  ".join(
                f"{m} {r['scores'][m]['R@0.01']:.2f}/{r['scores'][m]['R@0.1']:.2f} (auc {r['scores'][m]['auc']:.3f})"
                for m in r["scores"]) + (("  win full %.2f/%.2f evenodd %.2f/%.2f fullcv %.2f/%.2f" % (
                    r["mtt_window"]["full@0.01"], r["mtt_window"]["full@0.1"], r["mtt_window"]["evenodd@0.01"],
                    r["mtt_window"]["evenodd@0.1"], r["mtt_window"]["fullcv@0.01"], r["mtt_window"]["fullcv@0.1"]))
                    if full_rule else ""), flush=True)
        json.dump(out, open(os.path.join(OUT, "recompute.json"), "w"), indent=1)
        print(f"  level {level} done, {time.time() - t0:.0f}s; checks:",
              {k: v for k, v in L.items() if not isinstance(v, dict)}, flush=True)
    json.dump(out, open(os.path.join(OUT, "recompute.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
