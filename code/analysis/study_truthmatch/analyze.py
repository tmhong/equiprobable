#!/usr/bin/env python3
"""
Reco level with perfect (Delta-R) jet matching vs the chi2 reco of the paper: AUCs, tt̄_R, the corner picture.

Inputs (under results/): study_truthmatch/data_*.npz (build_matched.py), study_truthmatch/osde_dRmatch/ (osde_fit.py
on the matched angles), osde_h170/ (the paper's chi2 reco fit).  Writes results/study_truthmatch/results.json and plots.

Run: OMP_NUM_THREADS=3 .venv/bin/python code/analysis/study_truthmatch/analyze.py [--fr 0.3]
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, CODE)
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
from analysis.study_bsm.wtb_reweight import weights  # noqa: E402
import repo_paths  # noqa: E402

OUT = repo_paths.results_dir("study_truthmatch")   # build_matched.py's data_*.npz, stats_*.json are here too
TT, TT_TEST, PU0, HH = "run_seed1001", "run_seed12345", "pu0_fit", "run_m170_all"
MASSES = ("m_w_had", "m_t_lep", "m_t_had")


def auc(sig, bkg, w_sig=None, w_bkg=None):
    """P(score_sig < score_bkg) + 0.5 P(=): low score = anomalous.  Weighted Mann-Whitney."""
    w_sig = np.ones(len(sig)) if w_sig is None else w_sig
    w_bkg = np.ones(len(bkg)) if w_bkg is None else w_bkg
    x = np.r_[sig, bkg]
    w = np.r_[w_sig, np.zeros(len(bkg))], np.r_[np.zeros(len(sig)), w_bkg]
    o = np.argsort(x, kind="mergesort")
    xs, ws, wb = x[o], w[0][o], w[1][o]
    # bkg weight strictly above each sig value (ties count half)
    cb = np.cumsum(wb)
    tot_b = cb[-1]
    uniq, start = np.unique(xs, return_index=True)
    end = np.r_[start[1:], len(xs)]
    a = 0.0
    for s, e in zip(start, end):
        sw = ws[s:e].sum()
        if sw == 0:
            continue
        below = cb[s - 1] if s > 0 else 0.0
        tie = cb[e - 1] - below
        a += sw * (tot_b - below - tie + 0.5 * tie)
    return float(a / (ws.sum() * tot_b))


def angles(D, prefix, sel):
    return {a: D[f"{prefix}_{a}"][sel] for a in ANGLE_NAMES}


def bw_score(D_ref, sel_ref, D, sel, prefix):
    """Product of Breit-Wigner (Cauchy) factors centred on the tt̄ median, half-width = half the tt̄ IQR."""
    y = np.ones(int(sel.sum()))
    for m in MASSES:
        x = D_ref[f"{prefix}_{m}"][sel_ref]
        M = np.median(x)
        q1, q3 = np.percentile(x, [25, 75])
        W = 0.5 * (q3 - q1)
        y = y * W / np.pi / ((D[f"{prefix}_{m}"][sel] - M) ** 2 + W ** 2)
    return y


def quantile(ref, x):
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fr", type=float, default=0.3)
    p.add_argument("--osde", default=os.path.join(OUT, "osde_dRmatch"))
    p.add_argument("--tag", default="")
    p.add_argument("--chi2-osde", default=os.path.join(repo_paths.RESULTS, "osde_h170"), help="the chi2 reco fit to compare with")
    a = p.parse_args()
    D = {k: np.load(os.path.join(OUT, f"data_{k}.npz")) for k in (TT, TT_TEST, PU0, HH)}
    stats = {k: json.load(open(os.path.join(OUT, f"stats_{k}.json"))) for k in D}
    ok = {k: D[k]["ok"] for k in D}                       # matched + preselection
    both = {k: D[k]["ok"] & D[k]["chi_ok"] for k in D}    # matched and chi2-reconstructed (all matched pass chi2)
    R = {"fr": a.fr, "matching": {k: {"n_all": stats[k]["n_all"], "n_matched": stats[k]["n_matched_presel"],
                                      "eff_matched": stats[k]["n_matched_presel"] / stats[k]["n_all"],
                                      "n_chi2": stats[k]["n_chi2_reco_ok"],
                                      "matched_of_chi2": stats[k]["n_matched_presel"] / stats[k]["n_chi2_reco_ok"],
                                      "chi2_all_roles_right_of_matched": stats[k]["chi2_all_roles_right_of_matched"]}
                                  for k in D}}

    # --- matched fit (LOO for the fit sample) ---
    fm = osde.load_density(os.path.join(a.osde, "f_ttbar_reco.npz"))
    sm = np.load(os.path.join(a.osde, "scores_reco.npz"))
    y_tt = fm.loo(angles(D[TT], "m", ok[TT]))
    assert np.corrcoef(y_tt, sm["train"])[0, 1] > 0.999, "stored LOO scores do not match the recomputed ones"
    y_hh = fm(angles(D[HH], "m", ok[HH]))
    y_test = fm(angles(D[TT_TEST], "m", ok[TT_TEST]))
    y_pu0 = fm(angles(D[PU0], "m", ok[PU0]))
    w_r = weights(D[TT]["p_cos_theta_prime"][ok[TT]], D[TT]["p_cos_theta_prime_tilde"][ok[TT]], a.fr)

    # --- chi2 fit of the paper, on all chi2 events and on the matched events only ---
    fc = osde.load_density(os.path.join(a.chi2_osde, "f_ttbar_reco.npz"))
    sc = np.load(os.path.join(a.chi2_osde, "scores_reco.npz"))
    yc_tt_m = fc.loo(angles(D[TT], "c", both[TT]))          # these events are in the chi2 fit sample: LOO
    yc_hh_m = fc(angles(D[HH], "c", both[HH]))
    yc_tt_all = fc.loo(angles(D[TT], "c", D[TT]["chi_ok"]))
    assert np.corrcoef(yc_tt_all, sc["train"])[0, 1] > 0.999, "chi2 LOO mismatch with osde_h170"
    w_r_all = weights(D[TT]["p_cos_theta_prime"][D[TT]["chi_ok"]], D[TT]["p_cos_theta_prime_tilde"][D[TT]["chi_ok"]], a.fr)
    w_r_m = weights(D[TT]["p_cos_theta_prime"][both[TT]], D[TT]["p_cos_theta_prime_tilde"][both[TT]], a.fr)

    ref = json.load(open(os.path.join(repo_paths.RESULTS, "osde_h170", "summary.json")))["levels"]
    R["auc_hh"] = {"parton": ref["parton"]["auc_hh_vs_tt"], "parton_trig": ref["parton_trig"]["auc_hh_vs_tt"],
                   "gen_chi2": ref["gen"]["auc_hh_vs_tt"],
                   "reco_chi2_all": auc(sc["hh"], sc["train"]),
                   "reco_chi2_on_matched_events": auc(yc_hh_m, yc_tt_m),
                   "reco_matched": auc(y_hh, y_tt),
                   "reco_matched_vs_test_tt": auc(y_hh, y_test),
                   "reco_matched_vs_pu0_tt": auc(y_hh, y_pu0)}
    R["auc_ttR"] = {"reco_chi2_all": auc(yc_tt_all, yc_tt_all, w_r_all, None),
                    "reco_chi2_on_matched_events": auc(yc_tt_m, yc_tt_m, w_r_m, None),
                    "reco_matched": auc(y_tt, y_tt, w_r, None)}
    # one-dimensional cos theta alone (the variable where HH(170) differs), for reference
    ct = lambda d, pre, s: d[f"{pre}_cos_theta"][s]
    a1 = auc(ct(D[HH], "m", ok[HH]), ct(D[TT], "m", ok[TT]))
    R["auc_hh_cos_theta_alone"] = {"reco_matched": max(a1, 1 - a1)}
    a1 = auc(ct(D[HH], "c", D[HH]["chi_ok"]), ct(D[TT], "c", D[TT]["chi_ok"]))
    R["auc_hh_cos_theta_alone"]["reco_chi2_all"] = max(a1, 1 - a1)

    # --- corner picture with the matched jets: angular score x Breit-Wigner mass score ---
    ym_tt = bw_score(D[TT], ok[TT], D[TT], ok[TT], "m")
    ym_hh = bw_score(D[TT], ok[TT], D[HH], ok[HH], "m")
    qo = {"tt": quantile(y_tt, y_tt), "hh": quantile(y_tt, y_hh)}
    qm = {"tt": quantile(ym_tt, ym_tt), "hh": quantile(ym_tt, ym_hh)}
    corner = {}
    for c in (0.2, 0.3):
        ft = float(((qo["tt"] < c) & (qm["tt"] < c)).mean())
        fh = float(((qo["hh"] < c) & (qm["hh"] < c)).mean())
        acc = ft
        corner[str(c)] = {"tt": ft, "hh": fh, "enrichment": fh / ft,
                          "err": float(np.sqrt(fh * len(qo["hh"])) / len(qo["hh"]) / ft),
                          "angles_alone_same_tt": float((qo["hh"] < acc).mean() / acc),
                          "masses_alone_same_tt": float((qm["hh"] < acc).mean() / acc)}
    R["corner_matched"] = corner
    R["corr_qOmega_qm_tt"] = float(np.corrcoef(qo["tt"], qm["tt"])[0, 1])
    R["auc_hh_masses_alone_matched"] = auc(ym_hh, ym_tt)
    R["auc_hh_product_matched"] = auc(qo["hh"] * qm["hh"], qo["tt"] * qm["tt"])
    R["n"] = {"tt_fit": int(ok[TT].sum()), "hh": int(ok[HH].sum()), "tt_test": int(ok[TT_TEST].sum()),
              "tt_pu0": int(ok[PU0].sum()), "n_coefficients": int(np.sum(np.abs(fm.c) > 0))}
    json.dump(R, open(os.path.join(OUT, f"results{a.tag}.json"), "w"), indent=1)
    print(json.dumps(R, indent=1))

    # --- quick-look plots ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for axi, (ytt, yhh, wr, title) in zip(ax[:2], [(yc_tt_all, sc["hh"], w_r_all, r"reco, $\chi^2$ jets (paper)"),
                                                    (y_tt, y_hh, w_r, r"reco, $\Delta R$-matched jets")]):
        hi = np.percentile(np.r_[ytt, yhh], 99.5)
        lo = np.percentile(np.r_[ytt, yhh], 0.2)
        e = np.linspace(lo, hi, 50)
        axi.hist(ytt, e, density=True, histtype="stepfilled", color="#B3C2DF", ec="#003594", label=r"$t\bar t$")
        axi.hist(ytt, e, density=True, weights=wr, histtype="step", color="0.3", ls="--", label=rf"$t\bar t_R$ ($F_R={a.fr:g}$)")
        axi.hist(yhh, e, density=True, histtype="step", color="#C8900A", lw=1.8, label="HH(170)")
        axi.set_title(f"{title}\nAUC(HH) = {auc(yhh, ytt):.3f}")
        axi.set_xlabel(r"$y = f_{t\bar t}(\mathbf{x})$")
        axi.legend(frameon=False)
    levels = ["parton", "parton_trig", "gen_chi2", "reco_chi2_all", "reco_chi2_on_matched_events", "reco_matched"]
    labels = ["parton", "parton\n+trig", r"gen $\chi^2$", r"reco $\chi^2$", "reco $\\chi^2$\n(matched evts)", "reco\nmatched"]
    ax[2].bar(range(len(levels)), [R["auc_hh"][k] for k in levels], color=["0.6"] * 4 + ["#C8900A", "#003594"])
    ax[2].set_xticks(range(len(levels)), labels, fontsize=8)
    ax[2].set_ylim(0.5, 0.7)
    ax[2].set_ylabel("AUC, HH(170) vs $t\\bar t$")
    for i, k in enumerate(levels):
        ax[2].text(i, R["auc_hh"][k] + 0.003, f"{R['auc_hh'][k]:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"summary{a.tag}.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(5, 4.3))
    edges = np.linspace(0, 1, 11)
    Ht = np.histogram2d(qo["tt"], qm["tt"], [edges, edges])[0] / len(qo["tt"])
    Hh = np.histogram2d(qo["hh"], qm["hh"], [edges, edges])[0] / len(qo["hh"])
    Rr = np.where(Ht > 0, Hh / np.maximum(Ht, 1e-12), np.nan)
    im = ax.pcolormesh(edges, edges, np.log2(Rr).T, cmap="PuOr_r", vmin=-1.6, vmax=1.6)
    cb = fig.colorbar(im, ax=ax, ticks=np.log2([1 / 3, 1 / 2, 1, 2, 3]))
    cb.ax.set_yticklabels(["1/3", "1/2", "1", "2", "3"])
    cb.set_label("HH / $t\\bar t$ density")
    ax.plot([0, 0.2, 0.2], [0.2, 0.2, 0], "k--")
    ax.text(0.03, 0.08, f"x{corner['0.2']['enrichment']:.1f}")
    ax.set_xlabel(r"$q_\Omega$ (angles)")
    ax.set_ylabel(r"$q_m$ (masses)")
    ax.set_title("HH(170), reco, $\\Delta R$-matched jets")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"corner_matched{a.tag}.png"), dpi=130)


if __name__ == "__main__":
    main()
