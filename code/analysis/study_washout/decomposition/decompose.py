"""
Stepwise parton -> gen decomposition of the tt̄ / hh separation on the ten angles.

Reads cache_tt.npz / cache_hh.npz (from extract.py, in results/study_washout/decomposition) and, for each
step, computes the ten angles, then
  * AUC of f_tt̄ (paper OSDE basis) fit on a random half of the tt̄ events at that step,
    evaluated on the other half + all hh at that step (score = -f, hh = 1);
  * AUC of a supervised HistGradientBoostingClassifier on the ten angles (φ -> cos, sin),
    50/50 train/test split, class-balanced;
  * single-angle AUCs (same BDT on one angle).
Writes decomposition.json (next to the hh caches, see OUT_SUFFIX below) and prints a table.
"""
import os, sys, json, time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # code/
import repo_paths
from lib.angles import decay_angles, production_angles, ANGLE_NAMES
from lib import osde

# the tt̄ caches and the default outputs: results/study_washout/decomposition
BASE = repo_paths.results_dir("study_washout", "decomposition")
# OUT_SUFFIX (e.g. "_h170"): signal caches (cache_hh*.npz) and outputs in the sibling results/study_washout<SUFFIX>/decomposition;
# the tt̄ caches are always read from BASE.  Default: everything in BASE, as before.
SUFFIX = os.environ.get("OUT_SUFFIX", "")
OUT = repo_paths.results_dir("study_washout" + SUFFIX, "decomposition") if SUFFIX else BASE
M_TOP, M_W = 172.5, 80.4
RNG_SEED = 7
N_FIX = 8000          # tt̄ fit events for the equal-statistics f_tt̄ AUC
PHI = {"phi", "phi_prime", "phi_tilde", "phi_prime_tilde"}


def load(name, d=BASE):
    return dict(np.load(os.path.join(d, f"cache_{name}.npz")))


def angles(bl, bh, qd, qu, lep, nu):
    wl, wh = lep + nu, qd + qu
    tl, th = wl + bl, wh + bh
    Y, cT = production_angles(tl, th)
    a = decay_angles(tl, wl, lep)
    b = decay_angles(th, wh, qd)
    return dict(zip(ANGLE_NAMES, (Y, cT) + a + b))


def deta_phi_dr(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


def eta_phi(p):
    pt = np.hypot(p[:, 0], p[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = np.arcsinh(p[:, 2] / pt)
    return eta, np.arctan2(p[:, 1], p[:, 0])


def truth_match(d, ref, dr_max=0.4):
    """Nearest GenJet (ΔR < 0.4) to each of the reference partons ref[b_lep, b_had, q_down, q_up]
    (post-ISR Pythia partons); the four jets must be distinct."""
    idx, ok = [], np.ones(len(d["njets"]), bool)
    jeta, jphi = d["j_eta"], d["j_phi"]
    for k in ["b_lep", "b_had", "q_down", "q_up"]:
        e, f = eta_phi(ref[k])
        dr = deta_phi_dr(e[:, None], f[:, None], jeta, jphi)
        dr = np.where(np.isfinite(dr), dr, np.inf)
        i = np.argmin(dr, axis=1)
        ok &= dr[np.arange(len(i)), i] < dr_max
        idx.append(i)
    idx = np.stack(idx, 1)
    s = np.sort(idx, 1)
    ok &= np.all(s[:, 1:] != s[:, :-1], axis=1)
    J = np.stack([d["j_px"], d["j_py"], d["j_pz"], d["j_E"]], -1)       # (n, NJ, 4)
    rows = np.arange(len(ok))
    jets = {k: J[rows, idx[:, i]] for i, k in enumerate(["b_lep", "b_had", "q_down", "q_up"])}
    return ok, jets, idx


def nu_from_met(lep, bl, metx, mety):
    """W-mass-constrained ν pz from (metx, mety); the solution with m(ℓνb_lep) closest to m_t."""
    lpx, lpy, lpz, le = lep.T
    lpt2 = lpx ** 2 + lpy ** 2
    mu = 0.5 * M_W ** 2 + lpx * metx + lpy * mety
    a = mu * lpz / lpt2
    disc = a ** 2 - (le ** 2 * (metx ** 2 + mety ** 2) - mu ** 2) / lpt2
    root = np.sqrt(np.clip(disc, 0, None))
    best, bestd = None, None
    for pz in (a - root, a + root):
        nu = np.stack([metx, mety, pz, np.sqrt(metx ** 2 + mety ** 2 + pz ** 2)], -1)
        t = lep + nu + bl
        m = np.sqrt(np.clip(t[:, 3] ** 2 - np.sum(t[:, :3] ** 2, 1), 0, None))
        dm = np.abs(m - M_TOP)
        if best is None:
            best, bestd = nu, dm
        else:
            take = dm < bestd
            best = np.where(take[:, None], nu, best)
            bestd = np.where(take, dm, bestd)
    return best, disc < 0


def steps_for(d, sh):
    """{step: (angles dict, event mask)} for one sample (d: extract.py cache, sh: extract_shower.py cache)."""
    n = len(d["njets"])
    reco = d["g_reco_ok"]
    has_lep = np.all(np.isfinite(d["g_lep"]), 1)
    R = ["b_lep", "b_had", "q_down", "q_up", "lep", "nu"]
    h = {k: sh[f"h_{k}"] for k in R}             # post-ISR partons
    l = {k: sh[f"l_{k}"] for k in R}             # pre-hadronization (after FSR)
    match, J, _ = truth_match(d, h)
    m = reco & match & has_lep
    gl = d["g_lep"]
    metx, mety = d["g_met"] * np.cos(d["g_met_phi"]), d["g_met"] * np.sin(d["g_met_phi"])
    out = {}
    P = {a: d[f"p_ang_{a}"] for a in ANGLE_NAMES}
    H = angles(h["b_lep"], h["b_had"], h["q_down"], h["q_up"], h["lep"], h["nu"])
    out["S0 LHE parton, all"] = (P, np.ones(n, bool))
    out["S1 LHE parton, reco_ok"] = (P, reco)
    out["S1b post-ISR parton, all"] = (H, np.ones(n, bool))
    out["S1c post-ISR parton, reco_ok"] = (H, reco)
    out["S1d pre-hadronization (FSR), reco_ok"] = (angles(l["b_lep"], l["b_had"], l["q_down"], l["q_up"],
                                                          l["lep"], h["nu"]), reco)
    out["S1c_m post-ISR parton, reco_ok & matched"] = (H, m)
    # one ingredient at a time (on the reco_ok & matched events)
    out["R_lep post-ISR + gen lepton"] = (angles(h["b_lep"], h["b_had"], h["q_down"], h["q_up"], gl, h["nu"]), m)
    out["R_b post-ISR + matched b jets"] = (angles(J["b_lep"], J["b_had"], h["q_down"], h["q_up"], h["lep"], h["nu"]), m)
    out["R_q post-ISR + matched light jets"] = (angles(h["b_lep"], h["b_had"], J["q_down"], J["q_up"], h["lep"], h["nu"]), m)
    S2 = angles(J["b_lep"], J["b_had"], J["q_down"], J["q_up"], gl, h["nu"])
    out["S2 matched jets + gen lep + true nu"] = (S2, m)
    out["S2_all (S2 without reco_ok)"] = (S2, match & has_lep)
    # ν: transverse from truth, pz from the W constraint + m_t rule  /  full MET
    nu_tpt, _ = nu_from_met(gl, J["b_lep"], h["nu"][:, 0], h["nu"][:, 1])
    out["S3a S2 + nu pz from mW (true nu pT)"] = (angles(J["b_lep"], J["b_had"], J["q_down"], J["q_up"], gl, nu_tpt), m)
    nu_met, cplx = nu_from_met(gl, J["b_lep"], metx, mety)
    out["S3 S2 + nu from MET + mW"] = (angles(J["b_lep"], J["b_had"], J["q_down"], J["q_up"], gl, nu_met), m)
    # χ² jet choice with the true ν
    gj = {k: d[f"g_{k}"] for k in ["b_lep", "b_had", "q_down", "q_up"]}
    S2chi = angles(gj["b_lep"], gj["b_had"], gj["q_down"], gj["q_up"], gl, h["nu"])
    out["S2chi chi2 jets + gen lep + true nu"] = (S2chi, reco & has_lep)
    out["S2chi_m (matched events)"] = (S2chi, m)
    G = {a: d[f"g_ang_{a}"] for a in ANGLE_NAMES}
    out["S4 gen chi2 reconstruction"] = (G, reco)
    out["S4m (matched events)"] = (G, m)

    # diagnostics: is the χ² choice the truth-matched one?
    same = {}
    for k in ["b_lep", "b_had", "q_down", "q_up"]:
        same[k] = np.all(np.abs(gj[k] - J[k]) < 1e-3, 1)
    allc = same["b_lep"] & same["b_had"] & same["q_down"] & same["q_up"]
    qset = (same["q_down"] & same["q_up"]) | (np.all(np.abs(gj["q_down"] - J["q_up"]) < 1e-3, 1) &
                                              np.all(np.abs(gj["q_up"] - J["q_down"]) < 1e-3, 1))
    info = {"n": int(n), "reco_ok": float(reco.mean()), "matched": float((match & has_lep).mean()),
            "matched_given_reco_ok": float(m.sum() / reco.sum()),
            "chi2_all4_correct_given_matched": float(allc[m].mean()),
            "chi2_b_lep_correct": float(same["b_lep"][m].mean()),
            "chi2_b_had_correct": float(same["b_had"][m].mean()),
            "chi2_qpair_correct": float(qset[m].mean()),
            "nu_disc_negative_frac(S3)": float(cplx[m].mean())}
    return out, info


def features(X, names=ANGLE_NAMES):
    cols = []
    for a in names:
        if a in PHI:
            cols += [np.cos(X[a]), np.sin(X[a])]
        else:
            cols.append(X[a])
    return np.stack(cols, 1)


def hanley(auc, n1, n0):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return float(np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)))


def bdt_auc(Xt, Xh, rng, names=ANGLE_NAMES, max_iter=300):
    ft, fh = features(Xt, names), features(Xh, names)
    it, ih = rng.permutation(len(ft)), rng.permutation(len(fh))
    tt_tr, tt_te = it[: len(it) // 2], it[len(it) // 2:]
    hh_tr, hh_te = ih[: len(ih) // 2], ih[len(ih) // 2:]
    Xtr = np.concatenate([ft[tt_tr], fh[hh_tr]])
    ytr = np.r_[np.zeros(len(tt_tr)), np.ones(len(hh_tr))]
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                         class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)
    Xte = np.concatenate([ft[tt_te], fh[hh_te]])
    yte = np.r_[np.zeros(len(tt_te)), np.ones(len(hh_te))]
    s = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, s)
    return float(auc), hanley(auc, len(hh_te), len(tt_te))


def osde_auc(Xt, Xh, rng, f_fixed=None):
    n = len(Xt["Y"])
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    f = osde.TTbarDensity().fit({k: v[a] for k, v in Xt.items()})
    st = -f({k: v[b] for k, v in Xt.items()})
    sh = -f(Xh)
    auc = roc_auc_score(np.r_[np.zeros(len(st)), np.ones(len(sh))], np.r_[st, sh])
    res = {"auc": float(auc), "err": hanley(auc, len(sh), len(st))}
    if f_fixed is not None:      # the parton-level f_tt̄ applied at this step
        s2t, s2h = -f_fixed({k: v[b] for k, v in Xt.items()}), -f_fixed(Xh)
        res["auc_parton_f"] = float(roc_auc_score(np.r_[np.zeros(len(s2t)), np.ones(len(s2h))], np.r_[s2t, s2h]))
    # leave-one-out: fit on ALL tt̄ at this step, tt̄ scored without its own contribution
    f = osde.TTbarDensity().fit(Xt)
    st, sh = -f.loo(Xt), -f(Xh)
    res["auc_loo"] = float(roc_auc_score(np.r_[np.zeros(len(st)), np.ones(len(sh))], np.r_[st, sh]))
    # equal fit statistics at every step: N_FIX tt̄ fit events, the rest for evaluation (3 draws)
    v = []
    for _ in range(3):
        perm = rng.permutation(n)
        a, b = perm[:N_FIX], perm[N_FIX:]
        f = osde.TTbarDensity().fit({k: x[a] for k, x in Xt.items()})
        st, sh = -f({k: x[b] for k, x in Xt.items()}), -f(Xh)
        v.append(roc_auc_score(np.r_[np.zeros(len(st)), np.ones(len(sh))], np.r_[st, sh]))
    res["auc_nfix"], res["auc_nfix_spread"] = float(np.mean(v)), float(np.std(v))
    return res


def main():
    t0 = time.time()
    dt, dh = load("tt"), load("hh", OUT)
    st, it = steps_for(dt, load("tt_shower"))
    sh, ih = steps_for(dh, load("hh_shower", OUT))
    print(json.dumps({"tt": it, "hh": ih}, indent=1))
    # sanity: recomputing parton angles from the parton momenta reproduces the stored ones
    p = {k: dt[f"p_{k}"] for k in ["b_lep", "b_had", "q_down", "q_up", "lep", "nu"]}
    chk = angles(p["b_lep"], p["b_had"], p["q_down"], p["q_up"], p["lep"], p["nu"])
    maxdiff = max(float(np.nanmax(np.abs(np.angle(np.exp(1j * (chk[a] - dt[f"p_ang_{a}"]))))
                               if a in PHI else np.abs(chk[a] - dt[f"p_ang_{a}"]))) for a in ANGLE_NAMES)
    print("parton angle recomputation max diff:", maxdiff)

    # the parton-level f_tt̄ (fit on the first half of parton tt̄) for the "fixed density" column
    rng0 = np.random.default_rng(RNG_SEED)
    X0 = {a: dt[f"p_ang_{a}"] for a in ANGLE_NAMES}
    half = rng0.permutation(len(X0["Y"]))[: len(X0["Y"]) // 2]
    f_parton = osde.TTbarDensity().fit({k: v[half] for k, v in X0.items()})

    results = {"info": {"tt": it, "hh": ih, "parton_angle_recompute_maxdiff": maxdiff}, "steps": {}}
    for name in st:
        Xt_all, mt = st[name]
        Xh_all, mh = sh[name]
        good_t = mt & np.all([np.isfinite(Xt_all[a]) for a in ANGLE_NAMES], 0)
        good_h = mh & np.all([np.isfinite(Xh_all[a]) for a in ANGLE_NAMES], 0)
        Xt = {a: Xt_all[a][good_t] for a in ANGLE_NAMES}
        Xh = {a: Xh_all[a][good_h] for a in ANGLE_NAMES}
        rng = np.random.default_rng(RNG_SEED)
        r = {"n_tt": int(good_t.sum()), "n_hh": int(good_h.sum()),
             "nan_dropped_tt": int((mt & ~good_t).sum()), "nan_dropped_hh": int((mh & ~good_h).sum())}
        r["osde"] = osde_auc(Xt, Xh, rng, f_parton)
        r["bdt"] = bdt_auc(Xt, Xh, np.random.default_rng(RNG_SEED + 1))
        r["single"] = {a: bdt_auc(Xt, Xh, np.random.default_rng(RNG_SEED + 2), [a], max_iter=100)[0]
                       for a in ANGLE_NAMES}
        results["steps"][name] = r
        print(f"{name:42s} ntt={r['n_tt']:6d} nhh={r['n_hh']:6d}  f_tt={r['osde']['auc']:.3f}"
              f"±{r['osde']['err']:.3f} loo={r['osde']['auc_loo']:.3f} n8k={r['osde']['auc_nfix']:.3f} (parton f {r['osde']['auc_parton_f']:.3f})  "
              f"BDT={r['bdt'][0]:.3f}±{r['bdt'][1]:.3f}  "
              + " ".join(f"{a[:12]}={v:.3f}" for a, v in r["single"].items()), flush=True)
    json.dump(results, open(os.path.join(OUT, "decomposition.json"), "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
