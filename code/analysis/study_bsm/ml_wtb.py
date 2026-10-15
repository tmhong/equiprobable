#!/usr/bin/env python3
"""
Machine-learning comparisons for the right-handed-W benchmark of wtb_study.py.

Signal (as in wtb_reweight.py / wtb_study.py): the independent tt̄ sample (run_seed12345) reweighted event by
event with wtb_reweight.weights(cos θ'_lep, cos θ~'_had, F_R) of the PARTON-level angles.  At gen and reco the
events carry the parton weights of the same events (the levels are aligned event by event).  Background: the
same events with weight 1.  F_R = 0.05, 0.1, 0.3 at parton, gen and reco.

1. Unsupervised: the RealNVP normalizing flow of ml_compare.py (8 affine couplings; ml_compare.train_flow:
   AdamW, batch 512, at most 40 epochs, early stopping on the validation loss).  It is trained on the ten raw
   angles of the tt̄ FIT sample (run_seed1001: all 100k events at parton, reco_ok & finite at gen/reco) with
   ml_compare's 90% / 10% training / validation split (same RNG stream, so at parton level it is ml_compare's
   flow).  Score = -ln p_flow(x).  The flow never sees the signal.
2. Supervised reference.  It KNOWS THE SIGNAL WEIGHTS.  ml_compare's BDT (HistGradientBoostingClassifier with
   the same settings and ml_compare's "angles" inputs, with cos/sin of the azimuths) is trained on the first
   half of the independent sample.  Class 1 is the weighted copy (sample_weight w(F_R), normalised to the
   class-0 sum) and class 0 is the same events with weight 1.  Early stopping uses 10% of the training
   EVENTS, each with both copies (see train_bdt).  It is tested on the second half.  The BDT
   learns E[w | x], the likelihood ratio, so it measures what is reachable when the signal is known, not what
   an anomaly detector can do.
   Variant: the same BDT trained on the reweighted fit sample (100k / 41k / 24k events) and tested on the
   whole independent sample.  Oracle: the score w itself (parton truth, the optimum at parton level).
3. f_tt̄ (OSDE): 'angles, >10 sigma' and 'angles, >10 sigma + BW' are copied from wtb_study.json.  The
   angles-only score is also recomputed here, as in wtb_study.py: results/analysis/osde/f_ttbar_<level>.npz, salvage.surprisal,
   and the first 30k fit-sample events as the reference.  This checks the event alignment and evaluates the
   score on the BDT test half.
Metrics (as in wtb_reweight.py): weighted AUC is roc_auc_score on [unweighted copy, weighted copy] with
sample_weight; the signal efficiency at tt̄ efficiency 1% and 10% is wtb_reweight.eff_at.  Errors are
bootstrap errors over the test events.  Paired bootstrap errors are given for AUC(f_tt̄) - AUC(flow) and
AUC(BDT) - AUC(flow).  For the BDT, the spread from retraining on bootstrap resamples of its training half
is also given.  The flow is trained once (fixed seed).
Checks:
  - F_R = 0 (w = 1, the SM point as in wtb_study) gives AUC 0.500 for every score.
  - n_eff and the recomputed f_tt̄ AUC equal wtb_study.json.
  - corr(parton cos θ', level cos θ') confirms the alignment.
  - BDTs trained with event-shuffled weights (8 permutations) give a mean AUC ~ 0.5.
  - The flow NLL on the training / validation / independent tt̄ samples shows any overfitting.
  - The flow AUC for hh (run_seed2001) vs tt̄ at parton should match ml_compare (0.654).
Output: results/analysis/study_bsm/ml_wtb.json and a printed table.
Run: OMP_NUM_THREADS=3 .venv/bin/python code/analysis/study_bsm/ml_wtb.py      (~10 min; reading the 100k fit sample dominates)
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")             # shared machine: at most 3 cores

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import repo_paths  # noqa: E402
from plotting import make_plots  # noqa: E402
from analysis.ml_compare import SEED, angle_features, train_flow  # noqa: E402   (train_flow builds ml_compare.RealNVP)
from wtb_reweight import F_SM, eff_at, weights  # noqa: E402
from analysis.study_salvage.salvage import surprisal  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))      # ml_compare sets 8 when imported

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
FR_VALUES = [0.05, 0.1, 0.3]
OSDE_REF = ["angles, >10 sigma", "angles, >10 sigma + BW"]
N_REF = 30000              # wtb_study.py: the first 30k fit-sample events set the surprisal offset y0
N_RETRAIN = 8              # BDT retrainings (bootstrap resamples of the training half; shuffled-weight control)
# the BDT of ml_compare.main ("BDT [angles]"), same settings
BDT_SETTINGS = dict(max_iter=500, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
                    validation_fraction=0.1, random_state=SEED, class_weight="balanced")
SUPERVISED_NOTE = ("The BDT is a supervised reference that KNOWS THE SIGNAL: it is trained with the signal weights "
                   "w(F_R) (weighted vs unweighted copies of the same tt̄ events), so it approximates the likelihood "
                   "ratio E[w|x]. The flow and f_tt̄ are trained on SM tt̄ only and never see the signal.")


def select(e, lvl):
    """The ten angles of the events kept at this level and the mask, as in wtb_study.py: all angles finite
    and, at gen/reco, reco_ok."""
    ok = np.ones(len(e), bool) if lvl == "parton" else np.asarray(e.reco_ok, bool)
    ang = {a: np.asarray(e[a], float) for a in ANGLE_NAMES}
    ok &= np.all([np.isfinite(v) for v in ang.values()], 0)
    return {a: v[ok] for a, v in ang.items()}, ok


def wauc(s, w):
    """Weighted AUC as in wtb_reweight.py: background = the events with weight 1, signal = the same events with w."""
    n = len(s)
    return float(roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[s, s], sample_weight=np.r_[np.ones(n), w]))


def metrics(s, w, nboot=200):
    """AUC and signal efficiency at tt̄ efficiency 1% / 10% (high score = signal-like), with bootstrap errors
    (resampling events; each event keeps both of its copies)."""
    s, w = np.asarray(s, float), np.asarray(w, float)
    out = {"auc": wauc(s, w), "eff1": eff_at(s, w, s, 0.01), "eff10": eff_at(s, w, s, 0.10)}
    if nboot:
        rng = np.random.default_rng(SEED)
        b = []
        for _ in range(nboot):
            i = rng.integers(0, len(s), len(s))
            b.append((wauc(s[i], w[i]), eff_at(s[i], w[i], s[i], 0.01)))
        out["auc_err"], out["eff1_err"] = (float(x) for x in np.std(b, axis=0))
    return out


def paired_delta(s_a, s_b, w, nboot=200):
    """AUC(s_a) - AUC(s_b) on the same events, with a paired bootstrap error."""
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(nboot):
        i = rng.integers(0, len(w), len(w))
        d.append(wauc(s_a[i], w[i]) - wauc(s_b[i], w[i]))
    return {"delta": wauc(s_a, w) - wauc(s_b, w), "err": float(np.std(d))}


def n_eff(w):
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def ml_compare_split(n_tt, n_hh):
    """ml_compare.main's split with the same RNG stream: 90% / 10% of the tt̄ fit sample for training /
    validation, and its hh test set (the last 40% of the hh permutation)."""
    rng = np.random.default_rng(SEED)
    perm, tperm = rng.permutation(n_hh), rng.permutation(n_tt)
    return tperm[:int(0.9 * n_tt)], tperm[int(0.9 * n_tt):], perm[int(0.6 * n_hh):]


def fit_flow(x_fit, n_hh, epochs):
    """ml_compare's RealNVP on the raw angles of the fit sample, standardised with the training mean / std
    (exactly as in ml_compare.main).  Returns the scorer -ln p, training info and ml_compare's hh test set."""
    tr, va, hh_te = ml_compare_split(len(x_fit), n_hh)
    mu, sd = x_fit[tr].mean(0), x_fit[tr].std(0) + 1e-8
    t0 = time.time()
    flow = train_flow((x_fit[tr] - mu) / sd, (x_fit[va] - mu) / sd, epochs=epochs)

    def nll(x):
        """-ln p_flow in the standardised coordinates; this differs from -ln p(x) by a constant, so the ranking is the same."""
        with torch.no_grad():
            return -flow.log_prob(torch.as_tensor((x - mu) / sd, dtype=torch.float32)).double().numpy()

    info = {"n_train": int(len(tr)), "n_val": int(len(va)), "max_epochs": epochs, "seconds": round(time.time() - t0, 1),
            "nll_train": float(nll(x_fit[tr]).mean()), "nll_val": float(nll(x_fit[va]).mean())}
    return nll, info, hh_te


def train_bdt(x, w):
    """Class 1 = the events with sample_weight w (normalised to the class-0 sum); class 0 = the same events
    with weight 1.  Settings are ml_compare's.  The early-stopping validation set is 10% of the EVENTS, each
    with both of its copies.  sklearn's own split is by rows, so the opposite-label twin of most validation
    rows would sit in the training set.  That biases the validation loss upward and stopped the boosting
    after 10 trees (test AUC up to 0.015 lower at gen/reco)."""
    n = len(x)
    wn = w * n / np.sum(w)
    perm = np.random.default_rng(SEED).permutation(n)
    nv = int(BDT_SETTINGS["validation_fraction"] * n)

    def copies(i):
        return np.vstack([x[i], x[i]]), np.r_[np.zeros(len(i)), np.ones(len(i))], np.r_[np.ones(len(i)), wn[i]]
    (xt, yt, st), (xv, yv, sv) = copies(perm[nv:]), copies(perm[:nv])
    clf = HistGradientBoostingClassifier(**BDT_SETTINGS)
    clf.fit(xt, yt, sample_weight=st, X_val=xv, y_val=yv, sample_weight_val=sv)
    return clf


def proba(clf, x):
    return clf.predict_proba(x)[:, 1]


def cell(d):
    return f"{d['auc']:.3f} {100 * d['eff1']:4.1f}%"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", default=os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed1001"))
    p.add_argument("--test", default=os.path.join(repo_paths.REPO, "samples/ttbar_bblvqq/run_seed12345"))
    p.add_argument("--hh", default=os.path.join(repo_paths.REPO, "samples/hh_bblvqq/run_seed2001"))
    p.add_argument("--nmax-fit", type=int, default=None, help="events read from the fit sample (default: all, as ml_compare)")
    p.add_argument("--epochs", type=int, default=40, help="max flow epochs (ml_compare: 40 with early stopping)")
    p.add_argument("--levels", default="parton,gen,reco")
    p.add_argument("--out", default=os.path.join(repo_paths.results_dir("study_bsm"), "ml_wtb.json"))
    args = p.parse_args()
    levels = args.levels.split(",")
    load_levels = ["parton"] + [lv for lv in levels if lv != "parton"]

    t0 = time.time()
    fit, _ = make_plots.load_sample("fit", args.fit, load_levels, args.nmax_fit, OPTS)
    ind, _ = make_plots.load_sample("indep", args.test, load_levels, None, OPTS)
    hh, _ = make_plots.load_sample("hh", args.hh, ["parton"], None, OPTS)
    print(f"[ml_wtb] samples read in {time.time() - t0:.0f} s", flush=True)
    # parton cos θ' (lepton) and cos θ~' (down-type quark) of every event: the reweighting variables
    cpar = {k: (np.asarray(e["parton"].cos_theta_prime, float), np.asarray(e["parton"].cos_theta_prime_tilde, float))
            for k, e in (("fit", fit), ("ind", ind))}
    ang_hh, _ = select(hh["parton"], "parton")
    n_hh = len(ang_hh["Y"])
    del hh
    study = json.load(open(repo_paths.results("study_bsm", "wtb_study.json")))["levels"]
    mlc_path = repo_paths.results("ml", "results_parton.json")
    mlc_flow = (json.load(open(mlc_path))["results"].get("normalizing flow (RealNVP, 8 couplings)", {}).get("auc_test")
                if os.path.exists(mlc_path) else None)

    out = {"description": __doc__.strip().split("\n\n")[0], "note": SUPERVISED_NOTE, "F_SM": F_SM,
           "settings": {"seed": SEED, "flow": "ml_compare.RealNVP via ml_compare.train_flow, raw ten angles, standardised",
                        "bdt": {k: v for k, v in BDT_SETTINGS.items()}, "bdt_inputs": "ml_compare angle_features(periodic=True)",
                        "bdt_validation": "early stopping on 10% of the training events (both copies of each), not "
                                          "sklearn's row split (which separates the two copies of an event)",
                        "threads": int(os.environ["OMP_NUM_THREADS"]), "fit_sample": args.fit, "test_sample": args.test},
           "levels": {}}
    for lvl in levels:
        tl = time.time()
        ang_f, ok_f = select(fit[lvl], lvl)
        ang, ok = select(ind[lvl], lvl)
        cl, ch = cpar["ind"][0][ok], cpar["ind"][1][ok]            # parton angles of the same events
        cl_f, ch_f = cpar["fit"][0][ok_f], cpar["fit"][1][ok_f]
        n = len(ang["Y"])
        ntr = n // 2                                                # BDT: first half trains, second half tests
        print(f"\n[ml_wtb] {lvl}: fit sample {len(ang_f['Y'])} events, independent {n} (BDT train {ntr} / test {n - ntr})",
              flush=True)

        # 1. the flow (unsupervised, SM tt̄ only)
        nll, finfo, hh_te = fit_flow(angle_features(ang_f, False), n_hh, args.epochs)
        s_flow = nll(angle_features(ang, False))
        finfo["nll_indep"] = float(s_flow.mean())
        print(f"[ml_wtb] {lvl}: flow trained in {finfo['seconds']} s; mean -ln p train {finfo['nll_train']:.3f} "
              f"val {finfo['nll_val']:.3f} independent {finfo['nll_indep']:.3f}", flush=True)
        # f_tt̄ angles >10 sigma, recomputed as in wtb_study.py (alignment check; BDT test half)
        f_all = osde.load_density(repo_paths.results("osde", f"f_ttbar_{lvl}.npz"))
        f_10 = f_all.truncated(f_all.significance() > 10)
        n_ref = int(ok_f[:N_REF].sum())
        s_osde = surprisal(f_10(ang), f_10({a: v[:n_ref] for a, v in ang_f.items()}))

        xb, xb_f = angle_features(ang, True), angle_features(ang_f, True)
        res = {"n": n, "n_fit": int(len(ang_f["Y"])), "flow_training": finfo, "F_R": {}, "checks": {}}
        if lvl == "parton":
            s_hh = nll(angle_features(ang_hh, False))
            res["checks"]["hh vs tt AUC of the flow (unweighted)"] = {
                "hh test set of ml_compare (40%)": float(roc_auc_score(np.r_[np.zeros(n), np.ones(len(hh_te))],
                                                                       np.r_[s_flow, s_hh[hh_te]])),
                "all hh": float(roc_auc_score(np.r_[np.zeros(n), np.ones(n_hh)], np.r_[s_flow, s_hh])),
                "ml_compare (results_parton.json)": mlc_flow, "n_hh": n_hh}

        bdt_scores = {}
        for fr in FR_VALUES:
            w, w_f = weights(cl, ch, fr), weights(cl_f, ch_f, fr)
            st = study[lvl]["score"][str(fr)]
            bdt = train_bdt(xb[:ntr], w[:ntr])
            s_bdt = proba(bdt, xb[ntr:])
            bdt_scores[fr] = s_bdt
            bdt_f = train_bdt(xb_f, w_f)
            s_bdt_f = proba(bdt_f, xb)
            cut1 = np.quantile(s_bdt, 0.99)
            # training-sample spread of the BDT: retrain on bootstrap resamples of the training half
            rs = np.random.default_rng(SEED + 1)
            retrain = [wauc(proba(train_bdt(xb[i], w[i]), xb[ntr:]), w[ntr:])
                       for i in (rs.integers(0, ntr, ntr) for _ in range(N_RETRAIN))]
            e = {"n_eff": n_eff(w), "n_eff_wtb_study": st["n_eff"],
                 "flow": metrics(s_flow, w),
                 "bdt": {**metrics(s_bdt, w[ntr:]), "n_train": ntr, "n_test": n - ntr, "n_eff_test": n_eff(w[ntr:]),
                         "n_train_used_for_early_stopping": int(BDT_SETTINGS["validation_fraction"] * ntr),
                         "auc_train": wauc(proba(bdt, xb[:ntr]), w[:ntr]), "n_iter": int(bdt.n_iter_),
                         "n_unique_scores": int(len(np.unique(s_bdt))), "bkg_eff_at_1pct_cut": float(np.mean(s_bdt > cut1)),
                         "auc_retrain_sd": float(np.std(retrain)), "auc_retrain_mean": float(np.mean(retrain)),
                         "knows_signal": True},
                 "paired": {"auc(f_tt angles >10 sigma) - auc(flow), full sample": paired_delta(s_osde, s_flow, w),
                            "auc(bdt) - auc(flow), test half": paired_delta(s_bdt, s_flow[ntr:], w[ntr:])},
                 "osde": {k: {"auc": st[k]["auc"], "eff1": st[k]["eff_at_bkg_1pct"], "eff10": st[k]["eff_at_bkg_10pct"]}
                          for k in OSDE_REF},
                 "osde_recomputed": {"angles, >10 sigma": metrics(s_osde, w, nboot=0)},
                 "test_half": {"flow": metrics(s_flow[ntr:], w[ntr:]),
                               "osde angles, >10 sigma": metrics(s_osde[ntr:], w[ntr:])},
                 "bdt_fit_sample": {**metrics(s_bdt_f, w), "n_train": int(len(xb_f)), "n_eff_train": n_eff(w_f),
                                    "n_iter": int(bdt_f.n_iter_), "knows_signal": True},
                 "oracle_w": metrics(w, w)}
            res["F_R"][str(fr)] = e
            print(f"[ml_wtb] {lvl} F_R={fr}: n_eff {e['n_eff']:.0f}  f_tt>10σ {cell(e['osde'][OSDE_REF[0]])}  "
                  f"+BW {cell(e['osde'][OSDE_REF[1]])}  flow {cell(e['flow'])} ± {e['flow']['auc_err']:.3f}  "
                  f"BDT {cell(e['bdt'])} ± {e['bdt']['auc_err']:.3f} (train AUC {e['bdt']['auc_train']:.3f}, "
                  f"{e['bdt']['n_iter']} it., retrain sd {e['bdt']['auc_retrain_sd']:.3f})  BDT(fit-trained) "
                  f"{cell(e['bdt_fit_sample'])}  oracle {cell(e['oracle_w'])}  " + "  ".join(
                      f"{k}: {v['delta']:+.4f} ± {v['err']:.4f}" for k, v in e["paired"].items()), flush=True)

        # ---- checks
        one = np.ones(n)
        bdt0 = train_bdt(xb[:ntr], one[:ntr])
        null = {"flow": wauc(s_flow, one), "flow (test half)": wauc(s_flow[ntr:], one[ntr:]),
                "f_tt angles >10 sigma (recomputed)": wauc(s_osde, one),
                "BDT trained at F_R=0 (w=1)": wauc(proba(bdt0, xb[ntr:]), one[ntr:]),
                **{f"BDT trained at F_R={fr}": wauc(s, one[ntr:]) for fr, s in bdt_scores.items()},
                "flow eff at 1% (must be 0.01)": eff_at(s_flow, one, s_flow, 0.01)}
        res["checks"]["F_R=0 (w=1, SM; AUC must be 0.5)"] = null
        w0 = weights(cl, ch, 0.0)
        res["checks"]["F_R=0 via weights() (moves the regulator F_R=0.001 to 0)"] = {
            "flow": wauc(s_flow, w0), "f_tt angles >10 sigma": wauc(s_osde, w0), "max |w-1|": float(np.max(np.abs(w0 - 1)))}
        w3 = weights(cl, ch, 0.3)
        sh = [wauc(proba(train_bdt(xb[:ntr], w3[:ntr][np.random.default_rng(SEED + k).permutation(ntr)]), xb[ntr:]), w3[ntr:])
              for k in range(N_RETRAIN)]
        res["checks"][f"BDT trained with event-shuffled weights (F_R=0.3, {N_RETRAIN} permutations), tested with the true ones"] = {
            "auc_mean": float(np.mean(sh)), "auc_sd": float(np.std(sh)), "aucs": [float(a) for a in sh]}
        res["checks"]["corr(parton cos theta', this level)"] = float(np.corrcoef(cl, ang["cos_theta_prime"])[0, 1])
        res["checks"]["corr(parton cos theta~', this level)"] = float(np.corrcoef(ch, ang["cos_theta_prime_tilde"])[0, 1])
        res["checks"]["max |n_eff - n_eff(wtb_study)|"] = max(abs(v["n_eff"] - v["n_eff_wtb_study"]) for v in res["F_R"].values())
        res["checks"]["max |AUC f_tt>10 sigma recomputed - wtb_study|"] = max(
            abs(v["osde_recomputed"]["angles, >10 sigma"]["auc"] - v["osde"]["angles, >10 sigma"]["auc"]) for v in res["F_R"].values())
        res["seconds"] = round(time.time() - tl, 1)
        out["levels"][lvl] = res
        print(f"[ml_wtb] {lvl} checks: " + json.dumps(res["checks"], indent=None), flush=True)
        with open(args.out, "w") as fh:                  # write after every level
            json.dump(out, fh, indent=1)

    # ---- table
    print("\n" + SUPERVISED_NOTE)
    print("AUC (weighted; signal = reweighted tt̄) and signal efficiency at tt̄ efficiency 1%.  f_tt̄ and flow: full "
          "independent sample; BDT: second half (trained on the first half); BDT(fit): trained on the reweighted fit "
          "sample, tested on the full independent sample; oracle = w itself (parton truth).")
    hdr = (f"{'level':7s} {'F_R':>5s} {'n_eff':>6s} | {'f_tt angles':>13s} | {'f_tt ang+BW':>13s} | {'flow':>13s} | "
           f"{'BDT (knows w)':>13s} | {'BDT(fit)':>13s} | {'oracle w':>13s}")
    print(hdr + "\n" + "-" * len(hdr))
    for lvl, res in out["levels"].items():
        for fr, e in res["F_R"].items():
            print(f"{lvl:7s} {fr:>5s} {e['n_eff']:6.0f} | {cell(e['osde'][OSDE_REF[0]]):>13s} | {cell(e['osde'][OSDE_REF[1]]):>13s} | "
                  f"{cell(e['flow']):>13s} | {cell(e['bdt']):>13s} | {cell(e['bdt_fit_sample']):>13s} | {cell(e['oracle_w']):>13s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
