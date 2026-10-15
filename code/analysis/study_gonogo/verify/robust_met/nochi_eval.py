#!/usr/bin/env python3
"""R for the stop at reco: standard vs neutralino pT subtracted from MET (same events, paired bootstrap)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # code/
sys.path.insert(0, ROOT)
from analysis.study_gonogo import analyze as A
from lib import osde
from lib.angles import ANGLE_NAMES
import repo_paths
OUT = repo_paths.results_dir("study_gonogo", "verify", "robust_met")   # also extract_met.py's and stop_nochi.py's outputs
WPS = (0.005, 0.01, 0.02, 0.05, 0.1)
f_all = osde.load_density(os.path.join(repo_paths.RESULTS, A.F_FILES["reco"]))
f_cv = f_all.truncated(f_all.significance() > A.T_CV)
tt = A.load("tt_fit", "reco")
Mtt = dict(np.load(os.path.join(OUT, "met_tt_fit.npz")))
std = A.load("stop", "reco"); Mstd = dict(np.load(os.path.join(OUT, "met_stop.npz")))
z = np.load(os.path.join(OUT, "stop_nochi.npz"))
nc = {"X": {a: z[f"reco/{a}"] for a in ANGLE_NAMES}, "m": {k: z[f"reco/{k}"] for k in A.MASSES}, "mtt": z["reco/mtt"]}
# same events? (selection identical; jets/lepton unchanged so the hadronic-side W mass must agree)
print("events", len(nc["mtt"]), len(std["mtt"]), " m_W,had identical:", np.allclose(nc["m"]["m_w_had"], std["m"]["m_w_had"], equal_nan=True),
      " met_orig == standard met:", np.allclose(z["met_orig"], Mstd["met"]))
print("median MET: standard %.1f, neutralinos removed %.1f, tt %.1f" % (np.median(Mstd["met"]), np.median(z["met"]), np.median(Mtt["met"])))
def sc(d, is_tt, M):
    s = {"f_cv": -f_cv.loo(d["X"]) if is_tt else -f_cv(d["X"]),
         "f_all": -f_all.loo(d["X"]) if is_tt else -f_all(d["X"]),
         "bw": A.bw_score(tt, d), "mtw": M["mtw"], "met": M["met"]}
    return s
Stt = sc(tt, True, Mtt)
lt = lambda ref, x: -np.log(A.tail_fraction(ref, x))
Sa = sc(std, False, Mstd); Sb = sc(nc, False, {"mtw": z["mtw"], "met": z["met"]})
for S in (Stt, Sa, Sb):
    S["f_cv+bw"] = lt(Stt["f_cv"], S["f_cv"]) + lt(Stt["bw"], S["bw"])
# m_t,lep alone (Cauchy surprisal)
r = tt["m"]["m_t_lep"]; M0 = np.median(r); W0 = 0.5 * np.subtract(*np.percentile(r, [75, 25]))
for S, d in ((Stt, tt), (Sa, std), (Sb, nc)):
    S["bw_tlep"] = np.nan_to_num(np.log1p(((d["m"]["m_t_lep"] - M0) / W0) ** 2), nan=1e9)
names = list(Stt)
rng = np.random.default_rng(3); nb, ns = len(tt["mtt"]), len(std["mtt"])
reps = {(m, v, w): [] for m in names for v in "ab" for w in WPS}
for b in range(301):
    ib = np.arange(nb) if b == 0 else rng.integers(0, nb, nb)
    is_ = np.arange(ns) if b == 0 else rng.integers(0, ns, ns)
    for m in names:
        cuts = np.quantile(Stt[m][ib], [1 - w for w in WPS])
        for v, S in (("a", Sa), ("b", Sb)):
            for w, c in zip(WPS, cuts):
                reps[(m, v, w)].append(np.mean(S[m][is_] > c) / w)
reps = {k: np.array(v) for k, v in reps.items()}
print("R (standard -> neutralinos removed from MET) [paired diff]")
out = {}
for m in names:
    row = []
    for w in WPS:
        a, b_ = reps[(m, "a", w)], reps[(m, "b", w)]
        row.append(f"{a[0]:.2f}->{b_[0]:.2f} [{b_[0]-a[0]:+.2f}+-{np.std((b_-a)[1:]):.2f}]")
        out.setdefault(m, {})[f"{w:g}"] = [float(a[0]), float(b_[0]), float(np.std((b_ - a)[1:])), float(np.std(b_[1:]))]
    print(f"  {m:9s} " + "  ".join(f"@{w:g}: {s}" for w, s in zip(WPS, row)))
# supervised angles BDT on the neutralino-removed stop
Xa = lambda d: np.column_stack([d["X"][x] for x in ANGLE_NAMES])
for lab, d in (("standard", std), ("no-chi MET", nc)):
    st, ss = A.bdt_scores(Xa(tt), Xa(d))
    bd = {f"{w:g}": float(np.mean(ss > np.quantile(st, 1 - w)) / w) for w in WPS}
    out[f"bdt_angles_{lab}"] = bd
    print(f"  BDT angles ({lab}): AUC {A.auc(ss, st):.3f} " + " ".join(f"R@{k} {v:.2f}" for k, v in bd.items()))
json.dump(out, open(os.path.join(OUT, "nochi_eval.json"), "w"), indent=1)
