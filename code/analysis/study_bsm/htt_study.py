#!/usr/bin/env python3
"""
Study B: tt̄ versus a heavy scalar H -> tt̄ (new physics in the production, on-shell tops).

Signals: H(500 GeV, Γ=25 GeV) and H(350 GeV, Γ=10 GeV; just above 2 m_t = 346 GeV), heft ggH, signal only.
Background: the independent tt̄ sample; everything tt̄-derived is fitted on the fit sample.

1. What is different at parton level: per-variable likelihood-ratio AUC of the ten angles and m_tt̄.
2. f_tt̄ as an anomaly score: angles (all terms / >10σ), + Breit-Wigner masses, + a 1-D m_tt̄ density;
   AUC and the signal efficiency at tt̄ efficiency 1% and 10% (tail enrichment), at parton, gen, reco.
   Reference: a dedicated supervised BDT (angles + masses + m_tt̄).
3. The OSDE coefficients as a bump hunt in angle space (parton level): fit the coefficients on a "data"
   sample = independent tt̄ + a signal fraction ε, compare with the tt̄ reference coefficients
   (fit sample, 100k) term by term: pulls, global χ² over the 14,400 real degrees of freedom, the terms
   that respond.  ε = 0 calibrates the pulls (must be N(0,1)).
Outputs: results/analysis/study_bsm/htt_study.json, figures results/analysis/study_bsm/fig_*.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import repo_paths  # noqa: E402
from htt_test import HistDensity, auc_err, extract  # noqa: E402
from analysis.study_salvage.salvage import surprisal  # noqa: E402
from analysis.study_salvage.simplest import PDG, bw_surprisal, cauchy_surprisal  # noqa: E402
from lib import osde, style  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

SIGNALS = {"H500": ("samples/Htt_bblvqq/run_seed3002", r"$H(500)\to t\bar t$"),
           "H350": ("samples/Htt_bblvqq/run_m350_seed3003", r"$H(350)\to t\bar t$")}
TT_FIT, TT_IND = "samples/ttbar_bblvqq/run_seed1001", "samples/ttbar_bblvqq/run_seed12345"
EPS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
TEX = {"Y": "$Y$", "cos_Theta": r"$\cos\Theta$", "cos_theta": r"$\cos\theta$", "phi": r"$\phi$",
       "cos_theta_prime": r"$\cos\theta'$", "phi_prime": r"$\phi'$", "cos_theta_tilde": r"$\cos\tilde\theta$",
       "phi_tilde": r"$\tilde\phi$", "cos_theta_prime_tilde": r"$\cos\tilde\theta'$", "phi_prime_tilde": r"$\tilde\phi'$",
       "mtt": r"$m_{t\bar t}$"}


def lr_auc(xb, xs, nb=40, seed=0):
    """1-D likelihood-ratio AUC (histogram LR fit on half, tested on the other half)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile(np.r_[xb, xs], [0.5, 99.5])
    edges = np.linspace(lo, hi, nb + 1)
    ab, as_ = rng.random(len(xb)) < 0.5, rng.random(len(xs)) < 0.5
    hb, _ = np.histogram(np.clip(xb[ab], lo, hi - 1e-9), edges, density=True)
    hs, _ = np.histogram(np.clip(xs[as_], lo, hi - 1e-9), edges, density=True)
    lr = np.log((hs + 1e-6) / (hb + 1e-6))
    sc = lambda x: lr[np.clip(np.digitize(np.clip(x, lo, hi - 1e-9), edges) - 1, 0, nb - 1)]  # noqa: E731
    lab = np.r_[np.zeros((~ab).sum()), np.ones((~as_).sum())]
    return float(roc_auc_score(lab, np.r_[sc(xb[~ab]), sc(xs[~as_])]))


def eff_at(s_sig, s_bkg, bkg_eff):
    cut = np.quantile(s_bkg, 1 - bkg_eff)
    return float(np.mean(s_sig > cut))


GROUPS = [("production angle", "production angle"), ("rapidity", "rapidity"), ("W helicity", "W helicity"),
          ("top polarization", "top polarization"), ("spin correlation", "spin correlation"),
          ("pol. x W helicity interference", "interference")]


def term_groups(f_ref):
    """Physics category of every term (osde.describe), plus 'SM-significant' and 'SM-null' sets."""
    n = int(np.prod(f_ref.shape))
    phys = [osde.describe(f_ref, i)[1] for i in range(n)]
    sig = f_ref.significance().ravel()
    groups = {name: np.array([key in p for p in phys]) for name, key in GROUPS}
    groups["SM-significant (>3 sigma in tt)"] = sig > 3
    groups["SM-null (<1 sigma in tt)"] = (sig < 1) & np.array([p != "normalization" for p in phys])
    return phys, groups


def coefficient_test(f_ref, X_bkg, X_sig, eps, rng, phys, groups):
    """Fit OSDE on bkg + signal fraction eps; pulls of ĉ_data - ĉ_ref over the real degrees of freedom.
    A conjugate pair carries one complex number whose real and imaginary parts each have half the
    variance of |c|, so the component pulls are sqrt(2) x the components of the complex pull; a
    self-conjugate term is real and its pull is used as is."""
    n_b = len(X_bkg["Y"])
    n_s = int(round(eps / (1 - eps) * n_b)) if eps > 0 else 0
    pick = rng.choice(len(X_sig["Y"]), n_s, replace=False) if n_s else np.array([], int)
    X = {a: np.r_[X_bkg[a], X_sig[a][pick]] for a in ANGLE_NAMES}
    g = osde.TTbarDensity().fit(X)
    dc = (g.c - f_ref.c).ravel()
    err = np.sqrt(g.c_err.ravel() ** 2 + f_ref.c_err.ravel() ** 2)
    pull = dc / np.where(err > 0, err, np.inf)
    partner = f_ref.conjugate_partner()
    k = np.arange(len(pull))
    selfc, first = partner == k, k < partner
    dofs = np.r_[pull[selfc].real, np.sqrt(2) * pull[first].real, np.sqrt(2) * pull[first].imag]
    term_of = np.r_[k[selfc], k[first], k[first]]
    out = {"n_data": n_b + n_s, "n_signal": n_s, "pull_std": float(np.std(dofs)), "max_abs_pull": float(np.max(np.abs(dofs))), "groups": {}}
    for name, mask in [("all terms", np.ones(len(pull), bool))] + list(groups.items()):
        m = mask[term_of]
        chi2, ndf = float(np.sum(dofs[m] ** 2)), int(m.sum())
        out["groups"][name] = {"chi2": chi2, "ndf": ndf, "z": float((chi2 - ndf) / np.sqrt(2 * ndf)) if ndf else 0.0}
    # the terms that respond: largest |pull| per term (max over its components)
    best = np.zeros(len(pull))
    np.maximum.at(best, term_of, np.abs(dofs))
    top = []
    for i in np.argsort(-best)[:8]:
        top.append({"term": osde.describe(f_ref, int(i))[0], "physics": phys[i], "pull": float(best[i]),
                    "ref_signif": float(f_ref.significance().ravel()[i])})
    out["top_terms"] = top
    return out


def main():
    style.use()
    out_dir = repo_paths.results_dir("study_bsm")
    out = {"signals": {k: v[0] for k, v in SIGNALS.items()}, "levels": {}}
    data = {}
    for level in ("parton", "gen", "reco"):
        D = {"tt_fit": extract(os.path.join(repo_paths.REPO, TT_FIT), level, 30000), "tt": extract(os.path.join(repo_paths.REPO, TT_IND), level)}
        for k, (p, _) in SIGNALS.items():
            if os.path.exists(os.path.join(repo_paths.REPO, p)):
                D[k] = extract(os.path.join(repo_paths.REPO, p), level)
        data[level] = D
        ref = D["tt_fit"]
        f_all = osde.load_density(repo_paths.results("osde", f"f_ttbar_{level}.npz"))
        f_10 = f_all.truncated(f_all.significance() > 10)
        mtt_den = HistDensity(ref["mtt"], 300, 1500)

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

        def scores(d):
            fr_all, fr_10 = f_all(ref["ang"]), f_10(ref["ang"])
            sa, s10 = surprisal(f_all(d["ang"]), fr_all), surprisal(f_10(d["ang"]), fr_10)
            sm, smtt = s_mass(d), mtt_den.surprisal(d["mtt"])
            return {"angles, all terms": sa, "angles, >10 sigma": s10, "BW masses": sm, "m_tt density": smtt,
                    "angles (>10 sigma) + BW": s10 + sm, "angles (>10 sigma) + m_tt": s10 + smtt,
                    "angles (>10 sigma) + BW + m_tt": s10 + sm + smtt}

        S_tt = scores(D["tt"])
        res = {"n_tt_test": len(D["tt"]["mtt"]), "mtt_median_tt": float(np.median(D["tt"]["mtt"]))}
        for k in SIGNALS:
            if k not in D:
                continue
            d = D[k]
            S = scores(d)
            r = {"n": len(d["mtt"]), "mtt_median": float(np.median(d["mtt"])), "unsupervised": {}}
            lab = np.r_[np.zeros(len(D["tt"]["mtt"])), np.ones(len(d["mtt"]))]
            for name in S:
                a, e = auc_err(lab, np.r_[S_tt[name], S[name]])
                r["unsupervised"][name] = {"auc": a, "err": e, "eff_at_tt_1pct": eff_at(S[name], S_tt[name], 0.01),
                                           "eff_at_tt_10pct": eff_at(S[name], S_tt[name], 0.10)}
            # dedicated supervised reference: half signal + fit-sample tt̄ -> other half signal + independent tt̄
            half = np.arange(len(d["mtt"])) < len(d["mtt"]) // 2
            feats = lambda x: np.column_stack([x["ang"][a] for a in ANGLE_NAMES] + [x["m"][m] for m in sorted(x["m"])] + [x["mtt"]])  # noqa: E731
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=0)
            Xtr = np.vstack([feats(ref), feats(d)[half]]); ytr = np.r_[np.zeros(len(ref["mtt"])), np.ones(half.sum())]
            ok = np.all(np.isfinite(Xtr), 1); clf.fit(Xtr[ok], ytr[ok])
            Xte = np.nan_to_num(np.vstack([feats(D["tt"]), feats(d)[~half]]))
            lab2 = np.r_[np.zeros(len(D["tt"]["mtt"])), np.ones((~half).sum())]
            r["supervised reference (angles + masses + m_tt)"] = roc_auc_score(lab2, clf.predict_proba(Xte)[:, 1])
            if level == "parton":
                r["single_variable_lr_auc"] = {a: lr_auc(D["tt"]["ang"][a], d["ang"][a]) for a in ANGLE_NAMES}
                r["single_variable_lr_auc"]["mtt"] = lr_auc(D["tt"]["mtt"], d["mtt"])
                # the coefficient bump hunt
                rng = np.random.default_rng(42)
                if "phys" not in locals():
                    phys, groups = term_groups(f_all)
                r["coefficient_test"] = {str(e): coefficient_test(f_all, D["tt"]["ang"], d["ang"], e, rng, phys, groups) for e in EPS}
            res[k] = r
            print(level, k, json.dumps({kk: vv for kk, vv in r.items() if kk != "coefficient_test"}, indent=1)[:1500], flush=True)
            if "coefficient_test" in r:
                for e, c in r["coefficient_test"].items():
                    print(f"   eps {e}: n_s {c['n_signal']}  pull std {c['pull_std']:.2f}  max|pull| {c['max_abs_pull']:.1f}  z: "
                          + ", ".join(f"{gname} {gv['z']:+.1f}" for gname, gv in c["groups"].items()) + "  top: "
                          + "; ".join(f"{t['term']} ({t['pull']:.1f})" for t in c["top_terms"][:3]), flush=True)
        out["levels"][level] = res
    json.dump(out, open(os.path.join(out_dir, "htt_study.json"), "w"), indent=1)

    # ---- figures (parton level): what is different, and the coefficient pulls
    P = data["parton"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    cols = {"tt": "#1f4e9c", "H500": "#d1495b", "H350": "#2a9d8f"}
    labs = {"tt": r"$t\bar t$ (QCD)", "H500": SIGNALS["H500"][1], "H350": SIGNALS["H350"][1]}
    for k in ("tt", "H500", "H350"):
        if k not in P:
            continue
        axs[0].hist(P[k]["mtt"], np.linspace(300, 900, 61), density=True, histtype="step", lw=2.4, color=cols[k], label=labs[k])
        axs[1].hist(P[k]["ang"]["cos_Theta"], np.linspace(-1, 1, 41), density=True, histtype="step", lw=2.4, color=cols[k])
        c = P[k]["ang"]["cos_theta"] * P[k]["ang"]["cos_theta_tilde"]
        axs[2].hist(c, np.linspace(-1, 1, 41), density=True, histtype="step", lw=2.4, color=cols[k])
    axs[0].set_xlabel(r"$m_{t\bar t}$ [GeV]"); axs[0].set_ylabel("unit area"); axs[0].legend(fontsize=15, frameon=False)
    axs[1].set_xlabel(r"$\cos\Theta$"); axs[1].set_ylim(bottom=0)
    axs[2].set_xlabel(r"$\cos\theta\,\cos\tilde\theta$ (spin correlation)"); axs[2].set_ylim(bottom=0)
    fig.suptitle(r"parton level: what distinguishes $H\to t\bar t$ from QCD $t\bar t$")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_htt_parton.png"), dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6.0))
    for ax, k in zip(axs, ("H500", "H350")):
        if k not in out["levels"]["parton"]:
            continue
        ct = out["levels"]["parton"][k]["coefficient_test"]
        eps = [float(e) for e in ct]
        for gname, mk, col in (("all terms", "o-", "0.3"), ("production angle", "s-", "#1f4e9c"), ("spin correlation", "^-", "#d1495b"),
                               ("top polarization", "v-", "#2a9d8f"), ("SM-null (<1 sigma in tt)", "D-", "#e9a03b")):
            ax.plot(eps, [ct[str(e)]["groups"][gname]["z"] for e in eps], mk, lw=2.2, ms=8, color=col, label=rf"$\chi^2$ test: {gname}")
        ax.plot(eps, [ct[str(e)]["max_abs_pull"] for e in eps], "x--", lw=2.2, ms=10, color="k", label="largest single pull")
        ax.axhline(5, color="0.6", lw=1, ls=":"); ax.axhline(0, color="0.8", lw=1)
        ax.set_xlabel(r"signal fraction $\epsilon$ in 10k events"); ax.set_ylabel(r"significance [$\sigma$]")
        ax.set_title(labs[k], fontsize=18); ax.legend(fontsize=14, frameon=False)
    fig.suptitle(r"OSDE coefficients as a bump hunt: $\hat c_\mathrm{data}-\hat c_{t\bar t}$")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_coefficient_test.png"), dpi=110); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
