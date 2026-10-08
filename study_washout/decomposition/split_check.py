"""Where does the reco_ok-sample loss sit: events with all four partons truth-matched to distinct
jets vs the rest?  Uses the parton-level f_tt̄ (fit on half of the LHE tt̄) and per-subset LOO fits."""
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
from decompose import load, steps_for, ANGLE_NAMES, osde, RNG_SEED, OUT   # OUT: see OUT_SUFFIX in decompose.py

dt, dh = load("tt"), load("hh", OUT)
st, _ = steps_for(dt, load("tt_shower"))
sh, _ = steps_for(dh, load("hh_shower", OUT))
X0 = {a: dt[f"p_ang_{a}"] for a in ANGLE_NAMES}
half = np.random.default_rng(RNG_SEED).permutation(len(X0["Y"]))[: len(X0["Y"]) // 2]
fp = osde.TTbarDensity().fit({k: v[half] for k, v in X0.items()})
notfit = np.ones(len(X0["Y"]), bool); notfit[half] = False
mt_match = st["S1c_m post-ISR parton, reco_ok & matched"][1]
mh_match = sh["S1c_m post-ISR parton, reco_ok & matched"][1]
out = {}
for step in ["S1 LHE parton, reco_ok", "S1c post-ISR parton, reco_ok", "S1d pre-hadronization (FSR), reco_ok",
             "S2chi chi2 jets + gen lep + true nu", "S4 gen chi2 reconstruction"]:
    Xt, mt = st[step]; Xh, mh = sh[step]
    row = {}
    for sub, (ct, ch) in {"matched": (mt_match, mh_match), "unmatched": (~mt_match, ~mh_match)}.items():
        gt = mt & ct & notfit & np.all([np.isfinite(Xt[a]) for a in ANGLE_NAMES], 0)
        gh = mh & ch & np.all([np.isfinite(Xh[a]) for a in ANGLE_NAMES], 0)
        s_t = -fp({a: Xt[a][gt] for a in ANGLE_NAMES}); s_h = -fp({a: Xh[a][gh] for a in ANGLE_NAMES})
        auc_p = roc_auc_score(np.r_[np.zeros(len(s_t)), np.ones(len(s_h))], np.r_[s_t, s_h])
        gt2 = mt & ct & np.all([np.isfinite(Xt[a]) for a in ANGLE_NAMES], 0)
        Xs = {a: Xt[a][gt2] for a in ANGLE_NAMES}
        f = osde.TTbarDensity().fit(Xs)
        auc_l = roc_auc_score(np.r_[np.zeros(gt2.sum()), np.ones(gh.sum())], np.r_[-f.loo(Xs), -f({a: Xh[a][gh] for a in ANGLE_NAMES})])
        row[sub] = {"n_tt": int(gt2.sum()), "n_hh": int(gh.sum()), "auc_parton_f": round(auc_p, 4), "auc_loo": round(auc_l, 4)}
    out[step] = row
    print(f"{step:40s}", row)
json.dump(out, open(os.path.join(OUT, "split_check.json"), "w"), indent=1)
