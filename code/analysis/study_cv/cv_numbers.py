#!/usr/bin/env python3
"""
Every number the paper quotes from f_tt̄, recomputed with one estimator (study_cv/cv_scores.py: |ĉ|/σ > 2.5 chosen by
5-fold cross-validation on tt̄ alone, tt̄ scored by 5-fold cross-fitting, signals by the same fold models).

Per level (parton_trig, gen, reco = chi2 jets, matched = ΔR-matched jets):
  AUC of HH(170) and hh(125) against tt̄ (low y = anomalous) with bootstrap errors, and the all-terms cross-fit;
  fractions with y <= 0; the HH/tt̄ ratio in deciles of the tt̄ quantile q (Fig. 7) and the KL divergence it carries;
matched level: chi2 assignment on the same events; the corner of Fig. 5 (angles x Breit-Wigner masses).
Writes results/analysis/study_cv/numbers.json.  Run: .venv/bin/python code/analysis/study_cv/cv_numbers.py
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path[:0] = [CODE, HERE]
from kl import kl_score, quantile  # noqa: E402
import repo_paths  # noqa: E402

OUT = repo_paths.results_dir("study_cv")       # cv_scores.py writes its scores here too

LEVELS = ("parton_trig", "gen", "reco", "matched")
MASSES = ("m_w_had", "m_t_lep", "m_t_had")


def auc(sig, bkg, nboot=200, seed=3):
    lab = np.r_[np.zeros(len(bkg)), np.ones(len(sig))]
    a = roc_auc_score(lab, -np.r_[bkg, sig])
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(nboot):
        b, s = bkg[rng.integers(0, len(bkg), len(bkg))], sig[rng.integers(0, len(sig), len(sig))]
        reps.append(roc_auc_score(lab, -np.r_[b, s]))
    return round(float(a), 4), round(float(np.std(reps)), 4)


def deciles(y_ref, y_sig):
    q = quantile(y_ref, y_sig)
    h, _ = np.histogram(q, np.linspace(0, 1, 11))
    r = h / len(q) / 0.1
    err = np.sqrt(h) / len(q) / 0.1
    return [round(float(x), 3) for x in r], [round(float(x), 3) for x in err]


def bw(ref, d):
    """Product of Cauchy factors centred on the tt̄ median with half-width = half the tt̄ IQR (large = tt̄-like)."""
    y = np.ones(len(d[MASSES[0]]))
    for m in MASSES:
        x = ref[m][np.isfinite(ref[m])]
        M, (q1, q3) = np.median(x), np.percentile(x, [25, 75])
        W = 0.5 * (q3 - q1)
        y = y * W / np.pi / ((d[m] - M) ** 2 + W ** 2)
    return np.nan_to_num(y, nan=0.0)


def main():
    out = {}
    summ = json.load(open(os.path.join(OUT, "cv_summary.json")))
    for lv in LEVELS:
        z = np.load(os.path.join(OUT, f"scores_{lv}.npz"))
        tt = z["tt_cv"]
        r = {"n_tt": int(len(tt)), "kept_terms_full_fit": summ[lv]["kept_full"], "kept_per_fold": summ[lv]["kept_per_fold"]}
        for s in ("HH170", "hh125"):
            if f"{s}_cv" not in z:
                continue
            ys = z[f"{s}_cv"]
            r[s] = {"n": int(len(ys)), "auc": auc(ys, tt), "auc_all_terms": auc(z[f"{s}_all_cv"], z["tt_all_cv"]),
                    "frac_y_le_0": round(float((ys <= 0).mean()), 4)}
            r[s]["deciles"], r[s]["deciles_err"] = deciles(tt, ys)
            r[s]["kl"] = [round(x, 4) for x in kl_score(tt, ys)]
            r[s]["kl_all_terms"] = [round(x, 4) for x in kl_score(z["tt_all_cv"], z[f"{s}_all_cv"], nboot=50)]
        r["tt_frac_y_le_0"] = round(float((tt <= 0).mean()), 4)
        r["tt_frac_y_le_0_all_terms"] = round(float((z["tt_all_cv"] <= 0).mean()), 4)
        if "tt_indep_full" in z:
            r["tt_indep_frac_y_le_0"] = round(float((z["tt_indep_full"] <= 0).mean()), 4)
        if lv == "matched":
            r["chi2_same_events"] = {"auc": auc(z["HH170_chi2_cv"], z["tt_chi2_cv"]),
                                     "auc_all_terms": auc(z["HH170_chi2_all_cv"], z["tt_chi2_all_cv"]),
                                     "n_tt": int(len(z["tt_chi2_cv"])), "n_HH": int(len(z["HH170_chi2_cv"]))}
            # the corner (Fig. 5): angular and mass scores as tt̄ quantiles
            ref = {m: z[f"tt_{m}"] for m in MASSES}
            hh = {m: z[f"HH170_{m}"] for m in MASSES}
            ym_t, ym_h = bw(ref, ref), bw(ref, hh)
            qo = {"tt": quantile(tt, tt), "hh": quantile(tt, z["HH170_cv"])}
            qm = {"tt": quantile(ym_t, ym_t), "hh": quantile(ym_t, ym_h)}
            c = 0.2
            ft = float(((qo["tt"] < c) & (qm["tt"] < c)).mean())
            k = int(((qo["hh"] < c) & (qm["hh"] < c)).sum())
            n = len(qo["hh"])
            enr = lambda sel: [round(float(sel.sum() / n / ft), 3), round(float(np.sqrt(sel.sum()) / n / ft), 3)]  # noqa: E731
            edges = np.linspace(0, 1, 6)
            Ht = np.histogram2d(qo["tt"], qm["tt"], [edges, edges])[0] / len(qo["tt"])
            Hh = np.histogram2d(qo["hh"], qm["hh"], [edges, edges])[0] / n
            r["corner"] = {"cut": c, "tt_frac": round(ft, 4), "hh_frac": round(k / n, 4),
                           "enrichment": enr((qo["hh"] < c) & (qm["hh"] < c)),
                           "masses_alone_same_tt": enr(qm["hh"] < ft), "angles_alone_same_tt": enr(qo["hh"] < ft),
                           "corr_q_tt": round(float(np.corrcoef(qo["tt"], qm["tt"])[0, 1]), 3),
                           "bin_ratios_5x5": np.round(np.where(Ht > 0, Hh / np.maximum(Ht, 1e-12), np.nan), 2).tolist(),
                           "auc_masses": auc(ym_h, ym_t), "auc_angles_x_masses": auc(np.log(np.maximum(qo['hh'], 1e-9)) + np.log(np.maximum(qm['hh'], 1e-9)),
                                                                                  np.log(np.maximum(qo['tt'], 1e-9)) + np.log(np.maximum(qm['tt'], 1e-9)))}
        out[lv] = r
        print(lv, json.dumps({k: v for k, v in r.items() if k not in ("corner",)}, default=str)[:900], flush=True)
    # derived: how much of the parton-to-reco gap the truth-matched jets recover (HH170)
    p, rc, m, ch = (out["parton_trig"]["HH170"]["auc"][0], out["reco"]["HH170"]["auc"][0], out["matched"]["HH170"]["auc"][0],
                    out["matched"]["chi2_same_events"]["auc"][0])
    out["derived"] = {"gap_parton_to_reco_HH": round(p - rc, 4), "recovered_by_matching": round((m - rc) / (p - rc), 3),
                      "of_which_assignment": round((m - ch) / (p - rc), 3), "of_which_selection": round((ch - rc) / (p - rc), 3),
                      "kl_fraction_reco": round(out["reco"]["HH170"]["kl"][0] / out["parton_trig"]["HH170"]["kl"][0], 3),
                      "kl_fraction_matched": round(out["matched"]["HH170"]["kl"][0] / out["parton_trig"]["HH170"]["kl"][0], 3),
                      "kl_fraction_gen": round(out["gen"]["HH170"]["kl"][0] / out["parton_trig"]["HH170"]["kl"][0], 3)}
    print("derived", out["derived"])
    json.dump(out, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
