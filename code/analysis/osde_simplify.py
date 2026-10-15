#!/usr/bin/env python3
"""
How many of the 14400 coefficients matter?  Simplify f_tt̄ by dropping insignificant terms.

Criterion: keep a term iff |ĉ|/σ(ĉ) > t, always keeping the constant.
  * t = sqrt(2) is the Kronmal-Tarter rule: dropping a term costs |c|^2, keeping it costs its
    variance σ^2; with |ĉ|^2 - σ^2 an unbiased estimate of |c|^2, keeping pays iff |ĉ|^2 > 2σ^2.
    This minimizes the mean integrated squared error (MISE) of f.
  * The threshold is checked with 5-fold least-squares cross-validation on the training sample:
    ISE(f̂) - ∫f^2 = Σ_kept |ĉ|^2 - 2 Re Σ_kept ĉ conj(ĉ_heldout)  (orthonormal basis, no integrals).
Conjugate partners have equal |ĉ| and σ, so they are kept together and f stays real (checked).

Outputs in --osde (default results/analysis/osde): simplify_<level>.png (CV curve), f_ttbar_<level>_simplified.npz,
terms_<level>.txt (kept terms with physics labels), simplify_summary.json.
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
CODE = os.path.dirname(HERE)          # code/
sys.path.insert(0, CODE)

import repo_paths  # noqa: E402
from analysis.osde_fit import efficiency_curve  # noqa: E402
from lib import osde, style  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

THRESHOLDS = [0, 0.5, 1.0, osde.KT_THRESHOLD, 2, 2.5, 3, 4, 5, 7, 10, 15, 20]


def angles(sample, level, tag):
    z = np.load(os.path.join(sample, f"angles_{tag}.npz"))
    return {a: z[f"{level}/{a}"] for a in ANGLE_NAMES}


def evaluate(f, X, loo_train=True):
    """Scores on each sample (leave-one-out on the training sample)."""
    out = {}
    for name, Xs in X.items():
        out[name] = f.loo(Xs) if (name == "train" and loo_train) else f(Xs)
    return out


def stats(scores):
    tt = scores.get("test", scores["train"])
    s = lambda v: float(np.mean(-np.log(v[v > 0])))  # noqa: E731
    return {"auc": efficiency_curve(tt, scores["hh"])[2],
            "frac_negative": {k: float(np.mean(v <= 0)) for k, v in scores.items()},
            "mean_s": {k: s(v) for k, v in scores.items()}}


def plot_cv(res, lvl, folds, path):
    """Cross-validated ISE vs number of kept coefficients (labels: threshold in sigma)."""
    style.use()
    cv, cv_err, cv_kept = np.array(res["cv"]), np.array(res["cv_err"]), np.array(res["cv_kept"])
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    rel = cv - cv[0]
    ax.errorbar(cv_kept, rel, yerr=cv_err, fmt="o-", color="#1f4e9c", ms=6, lw=2)
    for t, kp, r in zip(THRESHOLDS, cv_kept, rel):
        if t in (0, 1.0, 2.5, 5, 20):                       # label a few thresholds only (no overlaps)
            ax.annotate(f"{t:.2g}$\\sigma$", (kp, r), textcoords="offset points", xytext=(6, 8), fontsize=17)
    kt = THRESHOLDS.index(osde.KT_THRESHOLD)
    ax.plot(cv_kept[kt], rel[kt], "s", color="#d1495b", ms=13, mfc="none", mew=2, label=r"Kronmal-Tarter ($\sqrt{2}\sigma$)")
    best = int(np.argmin(cv))
    ax.plot(cv_kept[best], rel[best], "*", color="#2a9d8f", ms=20, label=f"CV minimum ({THRESHOLDS[best]:.2g}$\\sigma$)")
    ax.set_xscale("log")
    ax.set_xlabel("coefficients kept (labels: threshold in $\\sigma$)")
    ax.set_ylabel("CV ISE $-$ ISE(all terms)")
    ax.set_title(f"{lvl.replace('_trig', '+trig')} level: {folds}-fold cross-validation")
    ax.legend(frameon=False, fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--test")
    p.add_argument("--hh", required=True)
    p.add_argument("--osde", default=repo_paths.results("osde"), help="osde_fit.py output directory")
    p.add_argument("--levels", default="parton,parton_trig,gen,reco")
    p.add_argument("--tag", default="down_truth", help="angle cache tag (<had_fermion>_<jet_assignment>)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--replot", action="store_true", help="only redraw the CV figures from simplify_summary.json")
    args = p.parse_args()
    if args.replot:
        S = json.load(open(os.path.join(args.osde, "simplify_summary.json")))
        for lvl, res in S["levels"].items():
            plot_cv(res, lvl, args.folds, os.path.join(args.osde, f"simplify_{lvl}.png"))
        return

    out = {"thresholds": THRESHOLDS, "levels": {}}
    for lvl in args.levels.split(","):
        f = osde.load_density(os.path.join(args.osde, f"f_ttbar_{lvl}.npz"))
        X = {"train": angles(args.train, lvl, args.tag), "hh": angles(args.hh, lvl, args.tag)}
        if args.test:
            X["test"] = angles(args.test, lvl, args.tag)

        # cross-validation of the threshold
        s1, s2, counts = f.fold_sums(X["train"], args.folds)
        cv, cv_err, cv_kept = osde.cv_threshold_scan(s1, s2, counts, THRESHOLDS)
        t_cv = THRESHOLDS[int(np.argmin(cv))]

        sig = f.significance()
        sig.flat[0] = np.inf
        partner = f.conjugate_partner()
        res = {"n_total": int(sig.size), "cv": cv.tolist(), "cv_err": cv_err.tolist(), "cv_kept": cv_kept.tolist(),
               "t_cv": t_cv, "variants": {}}
        for name, t in (("all", -1), ("kronmal_tarter", osde.KT_THRESHOLD), ("cv_best", t_cv), ("5sigma", 5.0)):
            keep = sig > t
            assert np.array_equal(keep.ravel(), keep.ravel()[partner]), "kept set not closed under conjugation"
            g = f.truncated(keep)
            sc = evaluate(g, X)
            res["variants"][name] = {"threshold": t, "kept": int(keep.sum()), **stats(sc)}
            if name == "cv_best":                          # the saved simplified f = CV-optimal truncation
                g.save(os.path.join(args.osde, f"f_ttbar_{lvl}_simplified.npz"), criterion=f"5-fold CV, t={t_cv}")
                order = np.argsort(-sig.ravel())
                with open(os.path.join(args.osde, f"terms_{lvl}.txt"), "w") as fh:
                    fh.write(f"# {lvl}: {int(keep.sum())} of {sig.size} terms kept (|c|/sigma > {t_cv}, 5-fold CV)\n")
                    fh.write("# rank  |c|/sigma     Re c        Im c       term  |  physics\n")
                    for r, i in enumerate(order[: int(keep.sum())]):
                        c = f.c.ravel()[i]
                        term, tag = osde.describe(f, i)
                        fh.write(f"{r:5d} {sig.ravel()[i]:9.1f} {c.real:+.3e} {c.imag:+.3e}  {term}  |  {tag}\n")
                res["top_terms"] = [dict(zip(("term", "physics"), osde.describe(f, i)),
                                         signif=float(min(sig.ravel()[i], 1e9))) for i in order[1:13]]
                # what kinds of physics survive
                cats = {}
                for i in order[1: int(keep.sum())]:
                    for tag in osde.describe(f, i)[1].split("; "):
                        key = tag.split(" (")[0]
                        cats[key] = cats.get(key, 0) + 1
                res["categories"] = cats
        cats5 = {}
        for i in np.where(sig.ravel() > 5)[0]:
            if i == 0:
                continue
            key = " + ".join(sorted({t.split(" (")[0] for t in osde.describe(f, i)[1].split("; ")}))
            cats5[key] = cats5.get(key, 0) + 1
        res["categories_5sigma"] = dict(sorted(cats5.items(), key=lambda kv: -kv[1]))
        res["max_imag_pull_5sigma"] = float(np.max(np.abs(f.c.ravel()[sig.ravel() > 5].imag)
                                                   / f.c_err.ravel()[sig.ravel() > 5]))
        out["levels"][lvl] = res

        plot_cv(res, lvl, args.folds, os.path.join(args.osde, f"simplify_{lvl}.png"))

        v = res["variants"]
        print(f"[simplify] {lvl}: kept  all {v['all']['kept']}, KT {v['kronmal_tarter']['kept']}, "
              f"CV-best (t={t_cv:.2g}) {v['cv_best']['kept']}, 5σ {v['5sigma']['kept']}")
        for name, r in v.items():
            fn = r["frac_negative"]
            print(f"      {name:15s} kept {r['kept']:5d}  AUC {r['auc']:.3f}  f<=0 test {100 * fn.get('test', 0):.1f}% "
                  f"hh {100 * fn['hh']:.1f}%  <s> train(LOO)/test {r['mean_s']['train']:.2f}/"
                  f"{r['mean_s'].get('test', float('nan')):.2f}")
    path = os.path.join(args.osde, "simplify_summary.json")
    if os.path.exists(path):                       # merge with levels done earlier
        old = json.load(open(path))
        old["levels"].update(out["levels"])
        out["levels"] = old["levels"]
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
