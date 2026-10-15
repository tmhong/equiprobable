"""Numbers + figures for the washout mechanism study (reads data_hh.npz, data_tt.npz in results/study_washout/mechanism,
writes results.json there; see OUT_SUFFIX below)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.abspath(os.path.join(HERE, "../../.."))   # code/
sys.path.insert(0, CODE)
import repo_paths
from lib import angles as A, osde
# the tt̄ data and the default outputs: results/study_washout/mechanism
BASE = repo_paths.results_dir("study_washout", "mechanism")
# OUT_SUFFIX (e.g. "_h170"): signal data (data_hh.npz) and outputs in the sibling results/study_washout<SUFFIX>/mechanism;
# the tt̄ data (data_tt.npz) is always read from BASE.  Default: everything in BASE, as before.
SUFFIX = os.environ.get("OUT_SUFFIX", "")
OUT = repo_paths.results_dir("study_washout" + SUFFIX, "mechanism") if SUFFIX else BASE
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "legend.fontsize": 15, "lines.linewidth": 2.2,
                     "axes.spines.top": False, "axes.spines.right": False})
C_TT, C_HH = "#1f4e9c", "#d1495b"
hh = dict(np.load(os.path.join(OUT, "data_hh.npz"))); tt = dict(np.load(os.path.join(BASE, "data_tt.npz")))
NAMES = A.ANGLE_NAMES
LAB = {"Y": "Y", "cos_Theta": r"cos$\Theta$", "cos_theta": r"cos$\theta$", "phi": r"$\phi$",
       "cos_theta_prime": r"cos$\theta'$", "phi_prime": r"$\phi'$", "cos_theta_tilde": r"cos$\tilde\theta$",
       "phi_tilde": r"$\tilde\phi$", "cos_theta_prime_tilde": r"cos$\tilde\theta'$", "phi_prime_tilde": r"$\tilde\phi'$"}
RNG = {a: ((-3, 3) if a == "Y" else (-np.pi, np.pi) if a.startswith("phi") else (-1, 1)) for a in NAMES}
R = {}

def auc1d(x_h, x_t, lo, hi, nb=25):
    """AUC of the binned likelihood ratio (best 1D discriminant), 2-fold to avoid bias."""
    x_h, x_t = x_h[np.isfinite(x_h)], x_t[np.isfinite(x_t)]
    e = np.linspace(lo, hi, nb + 1)
    sc, y = [], []
    for k in (0, 1):
        tr_h, te_h = x_h[k::2], x_h[1 - k::2]; tr_t, te_t = x_t[k::2], x_t[1 - k::2]
        hh_, _ = np.histogram(np.clip(tr_h, lo, hi - 1e-9), e); ht_, _ = np.histogram(np.clip(tr_t, lo, hi - 1e-9), e)
        r = (hh_ + 1) / hh_.sum() / ((ht_ + 1) / ht_.sum())
        f = lambda x: r[np.clip(np.digitize(np.clip(x, lo, hi - 1e-9), e) - 1, 0, nb - 1)]
        sc += [f(te_h), f(te_t)]; y += [np.ones(len(te_h)), np.zeros(len(te_t))]
    return roc_auc_score(np.concatenate(y), np.concatenate(sc))

def bdt_auc(Xh, Xt):
    X = np.vstack([Xh, Xt]); y = np.r_[np.ones(len(Xh)), np.zeros(len(Xt))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1),
                          X, y, cv=StratifiedKFold(3, shuffle=True, random_state=1), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

def angs(d, pre, m):
    return np.stack([d[f"{pre}_{a}"][m] for a in NAMES], 1)

def fin(d, pre, m):
    return m & np.all(np.isfinite(np.stack([d[f"{pre}_{a}"] for a in NAMES], 1)), 1)

# ----- event classes
for d in (hh, tt):
    d["all"] = np.ones(len(d["g_reco_ok"]), bool)
    d["reco"] = d["g_reco_ok"].astype(bool)
    d["match"] = d["t_matched"].astype(bool)
    d["match_reco"] = d["match"] & d["reco"]
hh["whad_off"] = hh["p_m_w_had"] < hh["p_m_w_lep"]
R["n"] = {"hh": len(hh["all"]), "tt": len(tt["all"])}
R["eff"] = {s: {"reco_ok": float(d["reco"].mean()), "all4_matched+lep": float(d["match"].mean()),
                "reco_ok & all4 matched in leading 6": float((d["reco"] & d["t_in6"]).mean() / d["reco"].mean()),
                "chi2 picks right 4 jets | reco_ok": float((d["reco"] & d["t_chi_qq_ok"] & ((d["t_chi_bhad_ok"] & d["t_chi_blep_ok"]) | d["t_chi_bswap"])).mean() / d["reco"].mean()),
                "chi2 picks right W_had jets | reco_ok": float((d["reco"] & d["t_chi_qq_ok"]).mean() / d["reco"].mean()),
                "chi2 fully right (pairing too) | reco_ok": float((d["reco"] & d["t_chi_qq_ok"] & d["t_chi_bhad_ok"] & d["t_chi_blep_ok"]).mean() / d["reco"].mean())}
            for s, d in (("hh", hh), ("tt", tt))}
for s, d in (("hh", hh), ("tt", tt)):
    for k in ("b_had", "b_lep", "q_down", "q_up"):
        R["eff"][s][f"truth {k} pT>25,|eta|<2.5"] = float(np.mean((d[f"a_pt_{k}"] > 25) & (np.abs(d[f"a_eta_{k}"]) < 2.5)))

# ----- per-angle 1D separation at each step
STEPS = [("p", "all", "parton (LHE)"), ("a", "all", "parton after shower recoil"),
         ("a", "reco", "+ gen selection (reco_ok)"), ("t", "match", "truth-matched jets + true ν"),
         ("n", "match", "truth jets + m_W-constrained ν"), ("c", "reco", "χ² (gen)")]
R["auc1d"] = {}; R["bdt"] = {}
for pre, cls, lab in STEPS:
    mh, mt = fin(hh, pre, hh[cls]), fin(tt, pre, tt[cls])
    R["auc1d"][lab] = {a: auc1d(hh[f"{pre}_{a}"][mh], tt[f"{pre}_{a}"][mt], *RNG[a]) for a in NAMES}
    R["auc1d"][lab]["N_hh"] = int(mh.sum()); R["auc1d"][lab]["N_tt"] = int(mt.sum())
    R["bdt"][lab] = bdt_auc(angs(hh, pre, mh), angs(tt, pre, mt))
    print(f"{lab:40s} N={mh.sum():6d}/{mt.sum():6d} BDT={R['bdt'][lab]:.3f} " +
          " ".join(f"{a[:9]}={R['auc1d'][lab][a]:.3f}" for a in NAMES), flush=True)
# same matched events: parton angles (to separate selection from reconstruction)
mh, mt = fin(hh, "a", hh["match"]), fin(tt, "a", tt["match"])
R["bdt"]["parton-after-shower, matched-event subset"] = bdt_auc(angs(hh, "a", mh), angs(tt, "a", mt))
mh, mt = fin(hh, "c", hh["match_reco"]), fin(tt, "c", tt["match_reco"])
R["bdt"]["χ², matched-event subset"] = bdt_auc(angs(hh, "c", mh), angs(tt, "c", mt))
print(R["bdt"])

# ----- f_ttbar sanity (paper basis fits from results/osde/), tt̄ = independent sample seed12345 cached angles
R["f_auc"] = {}
for lvl, pre, cls in (("parton", "p", "all"), ("gen", "c", "reco")):
    f = osde.TTbarDensity.load(os.path.join(repo_paths.RESULTS, "osde", f"f_ttbar_{lvl}.npz"))
    z = np.load(os.path.join(repo_paths.SAMPLES, "ttbar_bblvqq/run_seed12345/angles_down_truth.npz"))
    Xt = {a: z[f"{lvl}/{a}"] for a in NAMES}
    m = fin(hh, pre, hh[cls]); Xh = {a: hh[f"{pre}_{a}"][m] for a in NAMES}
    yh, yt = np.real(f(Xh)), np.real(f(Xt))
    R["f_auc"][lvl] = float(roc_auc_score(np.r_[np.ones(len(yh)), np.zeros(len(yt))], -np.r_[yh, yt]))
print("f_tt AUC", R["f_auc"])

# ----- masses
def q(x, m=None):
    x = x if m is None else x[m]; x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [16, 50, 84])]
R["mass"] = {}
for s, d in (("hh", hh), ("tt", tt)):
    R["mass"][s] = {k: q(d[k], d["all"] if k[0] in "pa" else d["reco"]) for k in
                    ("p_m_t_lep", "p_m_t_had", "p_m_w_lep", "p_m_w_had", "c_m_t_lep", "c_m_t_had", "c_m_w_had", "c_m_w_lep", "p_m_bb")}
    R["mass"][s]["c_chi2"] = q(d["c_chi2"], d["reco"])
for k, m in (("W_had off-shell", hh["whad_off"]), ("W_lep off-shell", ~hh["whad_off"])):
    R["mass"]["hh " + k] = {"frac": float(m.mean()), "p_m_w_had": q(hh["p_m_w_had"], m), "c_m_w_had": q(hh["c_m_w_had"], m & hh["reco"]),
                            "p_m_w_lep": q(hh["p_m_w_lep"], m), "reco_ok": float(hh["reco"][m].mean())}
# hh fraction of chi2 events where the pseudo-top masses are 'top-like' at parton level
R["mass"]["hh frac |m(Wb)-mt|<30 both, parton"] = float(np.mean((abs(hh["p_m_t_lep"] - 172.5) < 30) & (abs(hh["p_m_t_had"] - 172.5) < 30)))
R["mass"]["tt frac |m(Wb)-mt|<30 both, parton"] = float(np.mean((abs(tt["p_m_t_lep"] - 172.5) < 30) & (abs(tt["p_m_t_had"] - 172.5) < 30)))
# neutrino: off-shell leptonic W* under m_W constraint
m = hh["match"] & ~hh["whad_off"]
R["nu"] = {"hh lep W* (matched): median |Δpz| m_W-ν vs true": float(np.nanmedian(np.abs(hh["n_nu_pz"] - hh["p_nu_pz"])[m])),
           "hh lep W on-shell (matched)": float(np.nanmedian(np.abs(hh["n_nu_pz"] - hh["p_nu_pz"])[hh["match"] & hh["whad_off"]])),
           "tt (matched)": float(np.nanmedian(np.abs(tt["n_nu_pz"] - tt["p_nu_pz"])[tt["match"]]))}

# ----- hypothesis-independent gen observables (same reco_ok events)
HV = ["h_m_bb", "h_dR_bb", "h_pt_bb", "h_m_jj", "h_m_ljj", "h_dR_l_jj", "h_mT_WW", "h_mtw", "h_met", "h_lep_pt", "h_HT", "h_njets", "h_nbjets", "h_dR_lb_min"]
R["hyp_indep"] = {}
mh, mt = hh["reco"], tt["reco"]
for v in HV:
    a, b = hh[v][mh], tt[v][mt]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    au = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    R["hyp_indep"][v] = max(au, 1 - au)
Xh = np.stack([hh[v][mh] for v in HV], 1); Xt = np.stack([tt[v][mt] for v in HV], 1)
R["bdt"]["gen: hypothesis-independent observables (reco_ok events)"] = bdt_auc(Xh, Xt)
Xh2 = np.stack([hh[v][mh] for v in ["h_m_bb", "h_dR_bb", "h_m_ljj", "h_dR_l_jj"]], 1); Xt2 = np.stack([tt[v][mt] for v in ["h_m_bb", "h_dR_bb", "h_m_ljj", "h_dR_l_jj"]], 1)
R["bdt"]["gen: m_bb, dR_bb, m_ljj, dR_l,jj only"] = bdt_auc(Xh2, Xt2)
cm = ["c_m_t_lep", "c_m_t_had", "c_m_w_had", "c_chi2"]
R["bdt"]["gen: χ² angles + χ² masses + chi2"] = bdt_auc(np.c_[angs(hh, "c", mh), np.stack([hh[k][mh] for k in cm], 1)],
                                                        np.c_[angs(tt, "c", mt), np.stack([tt[k][mt] for k in cm], 1)])
print(json.dumps({k: R[k] for k in ("eff", "hyp_indep", "bdt", "mass", "nu")}, indent=1, ensure_ascii=False))
json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, ensure_ascii=False)
