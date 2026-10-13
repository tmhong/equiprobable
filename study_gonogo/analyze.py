#!/usr/bin/env python3
"""
Go/no-go analysis for a PRL: does f_tt̄ flag spin-structure anomalies at reconstruction level better than the mass
handles and as well as a flow?  The decision rule is fixed in study_gonogo/CRITERIA.md (committed before results).

Inputs
  study_gonogo/cache/<key>.npz         extract.py: angles, masses, m_tt̄ at parton_trig / gen / reco (chi2 jets)
  study_truthmatch/data_<run>.npz      build_matched.py: the same with ΔR-matched jets ("matched")
  osde/f_ttbar_<level>.npz             the paper's tt̄ fits;  study_truthmatch/osde_dRmatch/  the matched-jet fit
Scores (higher = more anomalous; tt̄ scores are leave-one-out or out-of-fold on the tt̄ fit sample)
  osde_cv     f_tt̄ keeping |c|/sigma > 2.5 (primary)        osde_all    all 14,400 terms
  bw          Breit-Wigner (Cauchy) mass surprisal           mtt_dens    1-D tt̄ density of m_tt̄ (surprisal)
  mtt_window  best m_tt̄ window at the working point, chosen on one half of the signal, evaluated on the other
  flow        RealNVP on the ten angles, 5-fold out-of-fold, seed ensemble
  osde_cv+bw  -ln(tail fraction) of osde_cv plus that of bw
  bdt_*       supervised ceilings (2-fold), not part of the rule
Outputs: study_gonogo/results.json, results.md, fig_enrichment.png

Run: OMP_NUM_THREADS=8 .venv/bin/python study_gonogo/analyze.py [--levels reco,matched] [--nboot 300]
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from ml_compare import RealNVP  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402

TT_RUN = "run_seed1001"
SIGNALS = {
    "etat": dict(runs=["run_seed4001"], sigma_pb=8.8, label=r"$\eta_t(343)$"),
    "stop": dict(runs=["run_seed4002"], sigma_pb=10.57, label=r"$\tilde t\tilde t$ (300, 120)"),
    "H350": dict(runs=["run_m350_seed3003"], sigma_pb=None, label=r"$H(350)\to t\bar t$"),
    "nospin": dict(runs=["run_seed4003", "run_seed4004"], sigma_pb=None, label=r"$t\bar t$, no spin corr."),
    "tt_indep": dict(runs=["run_seed12345"], sigma_pb=None, label=r"$t\bar t$ (independent)"),
}
PHYSICAL = ("etat", "stop", "H350")
F_FILES = {"parton_trig": "osde/f_ttbar_parton_trig.npz", "gen": "osde/f_ttbar_gen.npz", "reco": "osde/f_ttbar_reco.npz",
           "matched": "study_truthmatch/osde_dRmatch/f_ttbar_reco.npz"}
MASSES = ("m_w_had", "m_t_lep", "m_t_had")
WPS = (0.01, 0.10)
T_CV = 2.5
SIGMA_TT, BR, LUMI = 832.0, 0.288, 140.0          # pb (NNLO+NNLL), BR(e/mu + jets, PDG), fb^-1
RULE_METHODS = ("bw", "mtt_dens", "mtt_window")
METHODS = ("osde_cv", "osde_all", "bw", "mtt_dens", "mtt_window", "flow", "osde_cv+bw", "bdt_angles", "bdt_all")


# ----------------------------------------------------------------------------- data
def load(key_or_runs, level):
    """{'X': angles, 'm': masses, 'mtt', 'n_all'} for a sample (cache key) or a list of runs (matched level)."""
    if level != "matched":
        z = np.load(os.path.join(HERE, "cache", f"{key_or_runs}.npz"))
        d = {"X": {a: z[f"{level}/{a}"] for a in ANGLE_NAMES}, "m": {k: z[f"{level}/{k}"] for k in MASSES},
             "mtt": z[f"{level}/mtt"], "n_all": int(z[f"{level}/n_all"])}
        if f"{level}/same_as_fit_cache" in z:
            d["same_as_fit_cache"] = bool(z[f"{level}/same_as_fit_cache"])
        return d
    parts = []
    for run in key_or_runs:
        z = np.load(os.path.join(ROOT, "study_truthmatch", f"data_{run}.npz"))
        ok = z["ok"]
        parts.append({"X": {a: z[f"m_{a}"][ok] for a in ANGLE_NAMES}, "m": {k: z[f"m_{k}"][ok] for k in MASSES},
                      "mtt": z["m_mtt"][ok], "n_all": int(z["n_generated"]) if "n_generated" in z else len(ok)})
    return {"X": {a: np.concatenate([p["X"][a] for p in parts]) for a in ANGLE_NAMES},
            "m": {k: np.concatenate([p["m"][k] for p in parts]) for k in MASSES},
            "mtt": np.concatenate([p["mtt"] for p in parts]), "n_all": sum(p["n_all"] for p in parts)}


def sample(key, level):
    return load(SIGNALS[key]["runs"] if level == "matched" else key, level)


# ----------------------------------------------------------------------------- scores
def tail_fraction(ref, x):
    """Fraction of the tt̄ reference with a score >= x (an empirical p-value), floored at 1/(N+1)."""
    r = np.sort(ref)
    return np.maximum(1.0 - np.searchsorted(r, x, side="left") / len(r), 1.0 / (len(r) + 1))


def bw_score(tt, d):
    s = np.zeros(len(d["mtt"]))
    for k in MASSES:
        r = tt["m"][k][np.isfinite(tt["m"][k])]
        M = np.median(r)
        q1, q3 = np.percentile(r, [25, 75])
        W = 0.5 * (q3 - q1)
        s += np.log(np.pi * W) + np.log1p(((d["m"][k] - M) / W) ** 2)          # -ln(Cauchy density)
    return np.nan_to_num(s, nan=1e9)


def mtt_density_score(tt, d, lo=300.0, hi=1500.0, nb=40):
    edges = np.linspace(lo, hi, nb + 1)
    h, _ = np.histogram(np.clip(tt["mtt"], lo, hi - 1e-6), edges, density=True)
    h = np.maximum(h, 1e-6)
    i = np.clip(np.digitize(np.clip(np.nan_to_num(d["mtt"], nan=lo), lo, hi - 1e-6), edges) - 1, 0, nb - 1)
    return -np.log(h[i])


def train_flow(x_tr, x_va, seed, epochs=40):
    """ml_compare.train_flow with an explicit seed (the paper's RealNVP baseline: 8 couplings, width 128)."""
    torch.manual_seed(seed)
    flow = RealNVP(x_tr.shape[1])
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    xt, xv = torch.as_tensor(x_tr, dtype=torch.float32), torch.as_tensor(x_va, dtype=torch.float32)
    best, state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        flow.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(perm), 512):
            opt.zero_grad()
            loss = -flow.log_prob(xt[perm[i:i + 512]]).mean()
            loss.backward()
            opt.step()
        flow.eval()
        with torch.no_grad():
            vl = float(-flow.log_prob(xv).mean())
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in flow.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    flow.load_state_dict(state)
    return flow


def flow_scores(X_tt, X_sigs, seeds, k=5):
    """-log p of a RealNVP trained on tt̄ only: tt̄ out-of-fold; each signal event scored by the model of its fold.
    Ensemble over seeds (mean -log p)."""
    A = lambda X: np.column_stack([X[a] for a in ANGLE_NAMES])  # noqa: E731
    xtt = A(X_tt)
    n = len(xtt)
    rng = np.random.default_rng(11)
    fold_tt = rng.permutation(n) % k
    xs = {key: A(X) for key, X in X_sigs.items()}
    fold_s = {key: rng.permutation(len(x)) % k for key, x in xs.items()}
    out_tt = np.zeros(n)
    out_s = {key: np.zeros(len(x)) for key, x in xs.items()}
    for seed in seeds:
        for f in range(k):
            tr = np.where(fold_tt != f)[0]
            va_mask = np.random.default_rng(seed + f).random(len(tr)) < 0.1
            mu, sd = xtt[tr].mean(0), xtt[tr].std(0) + 1e-8
            model = train_flow((xtt[tr[~va_mask]] - mu) / sd, (xtt[tr[va_mask]] - mu) / sd, seed=1000 * seed + f)
            with torch.no_grad():
                te = fold_tt == f
                out_tt[te] += -model.log_prob(torch.as_tensor((xtt[te] - mu) / sd, dtype=torch.float32)).numpy() / len(seeds)
                for key, x in xs.items():
                    m = fold_s[key] == f
                    out_s[key][m] += -model.log_prob(torch.as_tensor((x[m] - mu) / sd, dtype=torch.float32)).numpy() / len(seeds)
    return out_tt, out_s


def bdt_scores(F_tt, F_s, seed=0):
    """Supervised ceiling, 2-fold out-of-fold predictions for both classes."""
    rng = np.random.default_rng(seed)
    ft, fs = rng.permutation(len(F_tt)) % 2, rng.permutation(len(F_s)) % 2
    o_tt, o_s = np.zeros(len(F_tt)), np.zeros(len(F_s))
    for f in (0, 1):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
        X = np.vstack([F_tt[ft != f], F_s[fs != f]])
        y = np.r_[np.zeros((ft != f).sum()), np.ones((fs != f).sum())]
        clf.fit(np.nan_to_num(X), y)
        o_tt[ft == f] = clf.predict_proba(np.nan_to_num(F_tt[ft == f]))[:, 1]
        o_s[fs == f] = clf.predict_proba(np.nan_to_num(F_s[fs == f]))[:, 1]
    return o_tt, o_s


# ----------------------------------------------------------------------------- metrics
def eff_at(a_s, a_b, eb):
    return float(np.mean(a_s > np.quantile(a_b, 1 - eb)))


def window_eff(m_s, m_b, eb):
    """Best m_tt̄ window holding a fraction eb of tt̄, chosen on one half of the signal and evaluated on the other
    (both ways, averaged): an optimistic bump hunt without the look-elsewhere effect."""
    mb = np.sort(m_b[np.isfinite(m_b)])
    n = len(mb)
    k = max(int(round(eb * n)), 1)
    lo, hi = mb[:n - k + 1], mb[k - 1:]
    ms = np.where(np.isfinite(m_s), m_s, -1.0)
    halves = (ms[0::2], ms[1::2])
    effs = []
    for a, b in ((0, 1), (1, 0)):
        sa, sb = np.sort(halves[a]), np.sort(halves[b])
        cnt = np.searchsorted(sa, hi, "right") - np.searchsorted(sa, lo, "left")
        i = int(np.argmax(cnt))
        effs.append((np.searchsorted(sb, hi[i], "right") - np.searchsorted(sb, lo[i], "left")) / len(sb))
    return float(np.mean(effs))


def auc(a_s, a_b):
    return float(roc_auc_score(np.r_[np.zeros(len(a_b)), np.ones(len(a_s))], np.r_[a_b, a_s]))


def evaluate(S_tt, S_s, mtt_tt, mtt_s, nboot, seed=5):
    """Enrichment R = eps_S/eps_B at each working point for every score, with bootstrap replicas (paired)."""
    rng = np.random.default_rng(seed)
    nb, ns = len(mtt_tt), len(mtt_s)
    res = {m: {"auc": auc(S_s[m], S_tt[m]) if m != "mtt_window" else None} for m in S_tt}
    res["mtt_window"] = {"auc": None}
    reps = {(m, wp): [] for m in list(S_tt) + ["mtt_window"] for wp in WPS}
    for b in range(nboot + 1):
        ib = np.arange(nb) if b == 0 else rng.integers(0, nb, nb)
        is_ = np.arange(ns) if b == 0 else rng.integers(0, ns, ns)
        for wp in WPS:
            for m in S_tt:
                reps[(m, wp)].append(eff_at(S_s[m][is_], S_tt[m][ib], wp) / wp)
            reps[("mtt_window", wp)].append(window_eff(mtt_s[is_], mtt_tt[ib], wp) / wp)
    for (m, wp), v in reps.items():
        v = np.array(v)
        res[m][f"R@{wp:g}"] = float(v[0])
        res[m][f"R@{wp:g}_err"] = float(np.std(v[1:]))
    return res, {k: np.array(v) for k, v in reps.items()}


def verdict(reps, wp):
    """The CRITERIA.md checks for the primary detector at one working point."""
    R = reps[("osde_cv", wp)]
    r0, sr = R[0], np.std(R[1:])
    checks = {"R": float(r0), "R_err": float(sr), "a_enrichment": bool(r0 >= 2 and (r0 - 1) > 3 * sr)}
    for m in RULE_METHODS:
        d = R - reps[(m, wp)]
        checks[f"b_vs_{m}"] = {"diff": float(d[0]), "sigma": float(np.std(d[1:])),
                               "pass": bool(d[0] > 2 * np.std(d[1:]))}
    d = R - reps[("flow", wp)]
    checks["c_vs_flow"] = {"diff": float(d[0]), "sigma": float(np.std(d[1:])), "pass": bool(d[0] > -2 * np.std(d[1:]))}
    checks["pass"] = bool(checks["a_enrichment"] and all(checks[f"b_vs_{m}"]["pass"] for m in RULE_METHODS)
                          and checks["c_vs_flow"]["pass"])
    return checks


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", default="parton_trig,gen,reco,matched")
    p.add_argument("--signals", default=",".join(SIGNALS))
    p.add_argument("--nboot", type=int, default=300)
    p.add_argument("--quick", action="store_true", help="one flow seed, 50 bootstraps (testing)")
    a = p.parse_args()
    torch.set_num_threads(8)
    levels = a.levels.split(",")
    sigs = [s for s in a.signals.split(",") if s]
    out_path = os.path.join(HERE, "results.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out.setdefault("levels", {})
    for level in levels:
        f_all = osde.load_density(os.path.join(ROOT, F_FILES[level]))
        f_cv = f_all.truncated(f_all.significance() > T_CV)
        tt = load([TT_RUN], level) if level == "matched" else load("tt_fit", level)
        n_tt = len(tt["mtt"])
        assert n_tt == f_all.n_fit, (level, n_tt, f_all.n_fit)
        if level != "matched":
            assert tt.get("same_as_fit_cache", False), f"{level}: tt̄ events differ from the fit sample"
        else:
            z = np.load(os.path.join(ROOT, "samples/ttbar_bblvqq", TT_RUN, "angles_down_dRmatch.npz"))
            assert all(np.allclose(z[f"reco/{x}"], tt["X"][x]) for x in ANGLE_NAMES), "matched tt̄ differs from the fit"
        D = {k: sample(k, level) for k in sigs}
        print(f"[{level}] tt̄ {n_tt} (of {tt['n_all']}), " + ", ".join(f"{k} {len(d['mtt'])}" for k, d in D.items()), flush=True)

        S_tt = {"osde_cv": -f_cv.loo(tt["X"]), "osde_all": -f_all.loo(tt["X"]), "bw": bw_score(tt, tt),
                "mtt_dens": mtt_density_score(tt, tt)}
        S = {k: {"osde_cv": -f_cv(d["X"]), "osde_all": -f_all(d["X"]), "bw": bw_score(tt, d),
                 "mtt_dens": mtt_density_score(tt, d)} for k, d in D.items()}
        seeds = [1] if (a.quick or level in ("parton_trig", "gen")) else [1, 2, 3]
        ft, fs = flow_scores(tt["X"], {k: d["X"] for k, d in D.items()}, seeds)
        S_tt["flow"] = ft
        for k in D:
            S[k]["flow"] = fs[k]
        S_tt["osde_cv+bw"] = -np.log(tail_fraction(S_tt["osde_cv"], S_tt["osde_cv"])) - np.log(tail_fraction(S_tt["bw"], S_tt["bw"]))
        for k in D:
            S[k]["osde_cv+bw"] = -np.log(tail_fraction(S_tt["osde_cv"], S[k]["osde_cv"])) - np.log(tail_fraction(S_tt["bw"], S[k]["bw"]))

        lev = out["levels"].setdefault(level, {})
        lev["n_tt"], lev["eff_sel_tt"] = n_tt, n_tt / tt["n_all"]
        lev["n_terms_cv"] = int((np.abs(f_cv.c) > 0).sum())
        lev["frac_y_le_0_tt"] = {"osde_cv": float(np.mean(S_tt["osde_cv"] >= 0)), "osde_all": float(np.mean(S_tt["osde_all"] >= 0))}
        for k, d in D.items():
            A = lambda dd: np.column_stack([dd["X"][x] for x in ANGLE_NAMES])  # noqa: E731
            Fall = lambda dd: np.column_stack([A(dd)] + [dd["m"][m] for m in MASSES] + [dd["mtt"]])  # noqa: E731
            S_tt_k = dict(S_tt)
            S_tt_k["bdt_angles"], S[k]["bdt_angles"] = bdt_scores(A(tt), A(d))
            S_tt_k["bdt_all"], S[k]["bdt_all"] = bdt_scores(Fall(tt), Fall(d))
            res, reps = evaluate(S_tt_k, S[k], tt["mtt"], d["mtt"], 50 if a.quick else a.nboot)
            r = {"n": len(d["mtt"]), "eff_sel": len(d["mtt"]) / d["n_all"], "scores": res,
                 "frac_y_le_0": {"osde_cv": float(np.mean(S[k]["osde_cv"] >= 0)), "osde_all": float(np.mean(S[k]["osde_all"] >= 0))}}
            if k in PHYSICAL:
                r["rule"] = {f"{wp:g}": verdict(reps, wp) for wp in WPS}
            sig_pb = SIGNALS[k]["sigma_pb"]
            if sig_pb:
                NB = SIGMA_TT * BR * 1000 * LUMI * lev["eff_sel_tt"]
                NS = sig_pb * BR * 1000 * LUMI * r["eff_sel"]
                y = {"N_S": NS, "N_B": NB, "S/B_inclusive": NS / NB}
                for m in ("osde_cv", "bw", "mtt_window", "flow", "osde_cv+bw", "bdt_all"):
                    for wp in WPS:
                        s_, b_ = NS * res[m][f"R@{wp:g}"] * wp, NB * wp
                        y[f"{m}@{wp:g}"] = {"S": s_, "B": b_, "S/B": s_ / b_, "Z_stat": s_ / np.sqrt(b_),
                                            "Z_5pct": s_ / np.sqrt(b_ + (0.05 * b_) ** 2), "Z_10pct": s_ / np.sqrt(b_ + (0.10 * b_) ** 2)}
                r["yields_140fb"] = y
            lev[k] = r
            line = "  ".join(f"{m} {res[m]['R@0.01']:.2f}/{res[m]['R@0.1']:.2f}" for m in METHODS if m in res)
            print(f"  {k:9s} n={r['n']:6d}  AUC cv {res['osde_cv']['auc']:.3f} flow {res['flow']['auc']:.3f}  R@1%/10%: {line}", flush=True)
            if "rule" in r:
                print("     rule: " + "; ".join(f"wp {w}: {'PASS' if v['pass'] else 'fail'} (R {v['R']:.2f}+-{v['R_err']:.2f})"
                                             for w, v in r["rule"].items()), flush=True)
        json.dump(out, open(out_path, "w"), indent=1)

    # decision
    dec = {}
    for level, tag in (("reco", "GO"), ("matched", "CONDITIONAL GO")):
        if level not in out["levels"]:
            continue
        passes = [(k, w) for k in PHYSICAL if k in out["levels"][level] for w, v in out["levels"][level][k]["rule"].items() if v["pass"]]
        dec[level] = {"passes": passes, "verdict": tag if passes else "NO-GO"}
    out["decision"] = dec
    json.dump(out, open(out_path, "w"), indent=1)
    print("DECISION:", json.dumps(dec))


if __name__ == "__main__":
    main()
