#!/usr/bin/env python3
"""
Build f_tt̄(x) on the ten angles with OSDE and apply it to tt̄ and hh samples.

  1. angles for every sample/level (cached in <sample>/angles_<options>.npz)
  2. fit the 14400 coefficients on the tt̄ training sample, per level
  3. evaluate f on the fit ("train") sample (leave-one-out, i.e. without each event's own
     contribution to the coefficients), on a separately generated tt̄ sample ("test")
     and on hh
  4. plots: 1D projections of f vs data (as in the paper), distributions of
     y = f(x) and s = -ln y for tt̄ vs hh, hh-vs-tt̄ efficiency curve
  5. <out>/summary.json with the numbers (used by code/docs/make_slides.py); --out defaults to results/analysis/osde

Example:
  python code/analysis/osde_fit.py --train samples/ttbar_bblvqq/run_seed1001 \\
                                   --test  samples/ttbar_bblvqq/run_seed12345 \\
                                   --hh    samples/hh_bblvqq/run_seed2001
"""
import argparse
import json
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)          # code/
sys.path.insert(0, CODE)

import repo_paths  # noqa: E402
from plotting import make_plots  # noqa: E402
from lib import angles, osde, style  # noqa: E402

LEVELS = ["parton", "parton_trig", "gen", "reco"]
COLORS = {"train": "#1f4e9c", "test": "#2a9d8f", "hh": "#d1495b"}
# "train": the tt̄ sample the coefficients are extracted from; "test": a separately generated
# tt̄ sample that f is applied to (never used in the fit); "hh": the hh sample
LABELS = {"train": r"$t\bar t$ fit sample", "test": r"$t\bar t$ independent sample", "hh": r"$hh$"}
ANGLE_TEX = {"Y": r"$Y$", "cos_Theta": r"$\cos\Theta$", "cos_theta": r"$\cos\theta$", "phi": r"$\phi$",
             "cos_theta_prime": r"$\cos\theta'$", "phi_prime": r"$\phi'$",
             "cos_theta_tilde": r"$\cos\tilde\theta$", "phi_tilde": r"$\tilde\phi$",
             "cos_theta_prime_tilde": r"$\cos\tilde\theta'$", "phi_prime_tilde": r"$\tilde\phi'$"}


def get_angles(sample, levels, opts, rebuild=False):
    """{level: {angle: array}} for events with a tt̄ system; cached next to the sample."""
    tag = f"{opts['had_fermion']}_{opts['jet_assignment']}"
    cache = os.path.join(sample, f"angles_{tag}.npz")
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        if all(f"{l}/Y" in z for l in levels):
            print(f"[osde] angles from cache {cache}")
            return {l: {a: z[f"{l}/{a}"] for a in angles.ANGLE_NAMES} for l in levels}
    events, _ = make_plots.load_sample(os.path.basename(os.path.normpath(sample)), sample, levels, None, opts)
    out, flat = {}, {}
    for lvl in levels:
        ev = events[lvl]
        ok = np.all([np.isfinite(np.asarray(ev[a])) for a in angles.ANGLE_NAMES], axis=0)
        if lvl not in ("parton", "parton_trig"):          # gen/reco: events with a reconstructed tt̄
            ok &= np.asarray(ev.reco_ok)
        out[lvl] = {a: np.asarray(ev[a], float)[ok] for a in angles.ANGLE_NAMES}
        flat.update({f"{lvl}/{a}": v for a, v in out[lvl].items()})
        flat[f"{lvl}/n_all"] = np.array(len(ev))
    np.savez(cache, **flat)
    print(f"[osde] cached angles in {cache}")
    return out


def domain(a):
    if a == "Y":
        return (-3.0, 3.0)
    return (-np.pi, np.pi) if a.startswith("phi") else (-1.0, 1.0)


PROJ_GROUPS = [("", ["Y", "cos_Theta", "cos_theta", "phi", "cos_theta_prime", "phi_prime"], (2, 3)),
               ("_had", ["cos_theta_tilde", "phi_tilde", "cos_theta_prime_tilde", "phi_prime_tilde"], (2, 2))]


def plot_projections(f, data, level, path, nbins=20):
    """Normalized histograms of each sample and the projection of f, for all ten angles.
    Two figures (legible on a slide): <path> = production + leptonic top (2x3),
    <path stem>_had.png = hadronic top (2x2).  Returns {angle: (chi2, ndf)} vs the fit sample."""
    style.use()
    chi2, curves = {}, {}
    for a in angles.ANGLE_NAMES:
        lo, hi = domain(a)
        edges = np.linspace(lo, hi, nbins + 1)
        hists = {}
        for name, X in data.items():
            v = X[a]
            h, _ = np.histogram(v, edges)
            norm = len(v) * np.diff(edges)
            hists[name] = (h / norm, np.sqrt(h) / norm)
        fine = np.linspace(lo, hi, nbins * 8 + 1)
        proj = f.projection(a, 0.5 * (fine[1:] + fine[:-1])).reshape(nbins, 8).mean(axis=1)
        dens, err = hists["train"]
        good = err > 0
        chi2[a] = float(np.sum(((dens - proj)[good] / err[good]) ** 2)), int(good.sum())
        x = np.linspace(lo, hi, 300)
        curves[a] = (edges, hists, x, f.projection(a, x))
    stem, ext = os.path.splitext(path)
    for suffix, names, (nr, nc) in PROJ_GROUPS:
        fig, axs = plt.subplots(nr, nc, figsize=((4.6 if nc == 3 else 6.9) * nc, 3.9 * nr))
        for ax, a in zip(np.ravel(axs), names):
            edges, hists, x, fx = curves[a]
            cen = 0.5 * (edges[1:] + edges[:-1])
            for name, (dens, err) in hists.items():
                ax.errorbar(cen, dens, yerr=err, fmt="o", ms=4, color=COLORS[name], elinewidth=1,
                            label=LABELS[name])
            ax.plot(x, fx, color="k", lw=2, label="OSDE $f_{t\\bar t}$")
            ax.set_xlabel(ANGLE_TEX[a])
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            c2, nd = chi2[a]
            ax.set_title(f"$\\chi^2$/ndf = {c2:.0f}/{nd}", fontsize=16, loc="right")
        np.ravel(axs)[0].legend(fontsize=16, frameon=True, framealpha=0.85, loc="best")
        fig.suptitle(f"OSDE projections, {level.replace('_trig', '+trig')} level"
                     + (" (hadronic top)" if suffix else " (production, leptonic top)"))
        fig.tight_layout()
        fig.savefig(stem + suffix + ext, dpi=110)
        plt.close(fig)
    return chi2


def plot_scores(scores, level, path):
    """y = f(x): linear axis (negatives visible); log axis with a y <= 0 bin; surprisal of the shifted
    y' = y + y0 (y0 > |min y| so that all y' > 0); ROC."""
    style.use()
    # two wide figures (fill a 16:9 slide): <path> = y (linear, log); <stem>_s.png = surprisal, ROC
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6.0))
    fig2, (a2, a3) = plt.subplots(1, 2, figsize=(15, 6.0))
    axs4 = [a0, a1, a2, a3]
    # linear y, including f <= 0 (Gibbs overshoot of the truncated series)
    allv = np.concatenate(list(scores.values()))
    lo_lin, hi_lin = np.percentile(allv, 0.05), np.percentile(allv, 99.95)
    lo_lin = min(lo_lin, -0.05 * hi_lin)
    expo = int(np.floor(np.log10(hi_lin)))
    unit = 10.0 ** expo                                  # plot y in units of 10^expo (no offset text)
    lbins = np.linspace(lo_lin, hi_lin, 80) / unit
    ax = axs4[0]
    ax.axvspan(lbins[0], 0, color="0.88", zorder=0)
    ax.axvline(0, color="0.3", lw=0.8)
    for name, v in scores.items():
        ax.hist(np.clip(v / unit, lbins[0], lbins[-1]), lbins, histtype="step", density=True, color=COLORS[name],
                lw=1.5, label=f"{LABELS[name]}: {100 * np.mean(v <= 0):.1f}% $\\leq0$")
    ax.set_yscale("log")
    ax.set_xlabel(rf"$y = f_{{t\bar t}}(\mathbf{{x}})$  [$10^{{{expo}}}$]  (shaded: $y<0$)")
    ax.set_ylabel(r"$(1/N)\,dN/dy$")
    ax.legend(fontsize=16, frameon=True, framealpha=0.85, loc="upper right")
    axs = axs4[1:]
    # log axis for y > 0, plus one extra bin on the left holding all events with y <= 0
    pos = {k: v[v > 0] for k, v in scores.items()}
    lo = min(np.percentile(v, 0.5) for v in pos.values())
    hi = max(np.percentile(v, 99.9) for v in pos.values())
    nb = 40
    ebins = np.logspace(np.log10(lo), np.log10(hi), nb + 1)
    step = ebins[1] / ebins[0]
    neg_lo, neg_hi = lo / step ** 3, lo / step      # the "y <= 0" bin, separated by a gap
    ax = axs[0]
    ax.axvspan(neg_lo, neg_hi, color="0.88", zorder=0)
    for name, v in scores.items():
        w = np.full(len(v), 1.0 / len(v))
        h, _ = np.histogram(np.clip(v, lo, hi * 0.999999), ebins, weights=w)
        ax.stairs(h, ebins, color=COLORS[name], lw=1.5, label=f"{LABELS[name]}")
        ax.stairs([np.mean(v <= 0)], [neg_lo, neg_hi], color=COLORS[name], lw=1.5)
    ax.text(np.sqrt(neg_lo * neg_hi), 0.98, r"$y\leq0$", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$y = f_{t\bar t}(\mathbf{x})$ (log; left bin: $y\leq0$)")
    ax.set_ylabel("fraction of events per bin")
    ax.legend(fontsize=16, frameon=True, framealpha=0.85, loc="lower center")
    # surprisal of the shifted density y' = y + y0, with y0 common to all samples so that y' > 0
    y0 = 1.05 * max(0.0, -min(float(v.min()) for v in scores.values()))
    shifted = {k: v + y0 for k, v in scores.items()}
    s_all = np.concatenate([-np.log(v) for v in shifted.values()])
    sb = np.linspace(np.percentile(s_all, 0.1), np.percentile(s_all, 99.9), 50)
    for name, v in shifted.items():
        s = -np.log(v)
        axs[1].hist(np.clip(s, sb[0], sb[-1]), sb, histtype="step", density=True, color=COLORS[name], lw=1.5,
                    label=f"{LABELS[name]}: $\\langle s'\\rangle$ = {np.mean(s):.2f}")
    axs[1].set_xlabel(r"surprisal $s' = -\ln y'$, $y' = y + y_0$")
    axs[1].set_ylabel("normalized")
    axs[1].set_title(f"$y_0$ = {y0:.1e} (all $y'>0$)", fontsize=19)
    axs[1].legend(fontsize=16, frameon=False, loc="upper left")
    # efficiency curve: call an event anomalous if f < threshold (f <= 0 is most anomalous)
    tt = scores.get("test", scores["train"])
    eff_tt, eff_hh, auc = efficiency_curve(tt, scores["hh"])
    axs[2].plot(eff_tt, eff_hh, color="k", lw=1.5, label=f"AUC = {auc:.3f}")
    axs[2].plot([0, 1], [0, 1], ls=":", color="0.5")
    axs[2].set_xlabel(r"$t\bar t$ (independent sample) efficiency")
    axs[2].set_ylabel(r"$hh$ efficiency")
    axs[2].set_title(r"ROC (scan over $y_\mathrm{cut}$; $f<y_\mathrm{cut}$ selected)", fontsize=16)
    axs[2].legend(fontsize=16, frameon=False, loc="lower right")
    for fg in (fig, fig2):
        fg.tight_layout()
    stem, ext = os.path.splitext(path)
    fig.savefig(path, dpi=110)
    fig2.savefig(stem + "_s" + ext, dpi=110)
    plt.close(fig)
    plt.close(fig2)
    return auc


def efficiency_curve(f_bkg, f_sig):
    """Efficiencies of f < cut for tt̄ (bkg) and hh (sig), and the AUC (prob. f_hh < f_tt)."""
    cuts = np.unique(np.concatenate([f_bkg, f_sig, [np.inf]]))
    eb = np.searchsorted(np.sort(f_bkg), cuts, side="left") / len(f_bkg)
    es = np.searchsorted(np.sort(f_sig), cuts, side="left") / len(f_sig)
    auc = float(np.trapezoid(es, eb))
    return eb, es, auc


def ks_distance(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a)
                               - np.searchsorted(b, x, side="right") / len(b))))


def truncation_study(f, X, ks=(0, 2, 3, 5)):
    """Keep only coefficients with |c|/σ > k (k = 0: all, the paper's choice); report the
    number kept, AUC(hh vs tt̄ test) and the overfitting measure <s>_test - <s>_train."""
    c0 = f.c.copy()
    sig = np.abs(c0) / np.where(f.c_err > 0, f.c_err, np.inf)
    sig.flat[0] = np.inf                      # always keep the constant term
    tt = "test" if "test" in X else "train"
    out = []
    for k in ks:
        f.c = np.where(sig > k, c0, 0)
        sc = {n: f(X[n]) for n in set(("train", tt, "hh"))}
        mean_s = {n: float(np.mean(-np.log(v[v > 0]))) for n, v in sc.items()}
        out.append({"k": k, "kept": int(np.sum(sig > k)), "auc": efficiency_curve(sc[tt], sc["hh"])[2],
                    "ds_test_train": mean_s[tt] - mean_s["train"]})
    f.c = c0
    return out


def y_scale_scan(Y, n_hermite, scales=(1.0, 0.9, 0.8, 0.7)):
    """chi2 of the Hermite expansion of the Y marginal vs the histogram, for several scales
    (plus s = std(Y)); the paper uses s = 1."""
    from lib.osde import hermite_functions
    edges = np.linspace(-3, 3, 25)
    h, _ = np.histogram(Y, edges)
    dens, err = h / len(Y) / np.diff(edges), np.sqrt(h) / len(Y) / np.diff(edges)
    good = err > 0
    out = []
    for sc in list(scales) + [round(float(np.std(Y)), 2)]:
        c = hermite_functions(Y, n_hermite, sc).mean(axis=0)
        fine = np.linspace(-3, 3, 24 * 8 + 1)
        proj = (hermite_functions(0.5 * (fine[1:] + fine[:-1]), n_hermite, sc) @ c).reshape(24, 8).mean(axis=1)
        out.append({"scale": sc, "chi2": float(np.sum(((dens - proj)[good] / err[good]) ** 2)),
                    "ndf": int(good.sum())})
    return out


def make_density(args, lvl):
    """The paper's basis, or the extended block basis with the orders found by osde_order_scan.py."""
    if args.basis == "paper":
        return osde.TTbarDensity(args.n_legendre, args.n_hermite, args.y_scale)
    if args.basis == "sh":
        return osde.SHDensity(args.n_legendre, args.n_hermite, args.y_scale, args.sh_l1, args.sh_l2)
    r = json.load(open(args.orders))[lvl]["_required"]
    cap = args.max_order
    phip = r["min_j1_j2_for_phi_prime"]
    J1 = min(cap, max(1, r["j1max"], phip))
    J2 = min(cap, max(2, r["j2max"], phip))
    nl_hi = min(cap + 1, max(args.n_legendre, r["n_legendre"]))
    nh_hi = min(cap + 1, max(args.n_hermite, r["n_hermite"]))
    f = osde.ExtendedDensity(J1, J2, args.n_legendre, args.n_hermite, nl_hi, nh_hi, r.get("y_scale", args.y_scale))
    print(f"[osde] {lvl}: extended basis J1={J1} J2={J2} Legendre {nl_hi} Hermite {nh_hi} (scale {f.y_scale}): "
          f"{f.n_terms} terms "
          + ", ".join(f"{b[0]} {n}" for b, n in zip(f.blocks, f.sizes)))
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="tt̄ sample used to fit the coefficients")
    p.add_argument("--test", help="independent tt̄ sample")
    p.add_argument("--hh", required=True, help="hh sample")
    p.add_argument("--levels", default=",".join(LEVELS))
    p.add_argument("--config", default=os.path.join(CODE, "plotting", "plot_config.py"))
    p.add_argument("--n-legendre", type=int, default=5)
    p.add_argument("--n-hermite", type=int, default=5)
    p.add_argument("--y-scale", type=float, default=1.0, help="Hermite scale s (paper: 1)")
    p.add_argument("--basis", choices=["paper", "extended", "sh"], default="paper",
                   help="paper: j1<=1, j2<=2 product basis; extended: + high-order blocks (orders from --orders); "
                        "sh: spherical-harmonic products Y_l1m1(θ,φ) Y_l2m2(θ',φ') per top")
    p.add_argument("--sh-l1", type=int, default=1, help="l1max for --basis sh")
    p.add_argument("--sh-l2", type=int, default=2, help="l2max for --basis sh")
    p.add_argument("--orders", default=repo_paths.results("osde", "order_scan.json"),
                   help="osde_order_scan.py output (used with --basis extended)")
    p.add_argument("--max-order", type=int, default=16, help="cap on the extended orders")
    p.add_argument("--out", default=repo_paths.results("osde"))
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()

    cfg = runpy.run_path(args.config)
    opts = {"had_fermion": "down", "jet_assignment": "truth", **cfg.get("ANGLES", {})}
    levels = args.levels.split(",")
    os.makedirs(args.out, exist_ok=True)

    samples = {"train": args.train, "hh": args.hh}
    if args.test:
        samples["test"] = args.test
    data = {name: get_angles(path, levels, opts, args.rebuild_cache) for name, path in samples.items()}

    summary = {"options": opts, "basis": args.basis, "n_legendre": args.n_legendre, "n_hermite": args.n_hermite,
               "y_scale": args.y_scale, "samples": samples, "levels": {}}
    for lvl in levels:
        X = {name: d[lvl] for name, d in data.items()}
        f = make_density(args, lvl).fit(X["train"])
        f.save(os.path.join(args.out, f"f_ttbar_{lvl}.npz"), level=lvl, train=args.train)
        signif = np.abs(f.c) / np.where(f.c_err > 0, f.c_err, np.inf)
        scores, imag = {}, 0.0
        for name, Xs in X.items():
            scores[name], im = f(Xs, imag=True)
            if name == "train":
                # leave-one-out: remove each fit event's own contribution (self-inclusion bias)
                naive = scores[name]
                scores[name] = f.loo(Xs)
                selfinc = {"frac_negative_naive": float(np.mean(naive <= 0)),
                           "mean_s_naive": float(np.mean(-np.log(naive[naive > 0]))),
                           "self_term_over_f_median": float(np.median(np.abs(naive - scores[name]) * (f.n_fit - 1)
                                                                      / f.n_fit / np.abs(naive)))}
            imag = max(imag, float(np.max(np.abs(im)) / np.max(np.abs(scores[name]))))
        np.savez(os.path.join(args.out, f"scores_{lvl}.npz"), **scores)
        chi2 = plot_projections(f, {k: X[k] for k in ("train", "test", "hh") if k in X}, lvl,
                                os.path.join(args.out, f"projections_{lvl}.png"))
        # projections of hh alone are not meaningful vs f_tt̄; plot tt̄ only as well
        plot_projections(f, {k: X[k] for k in ("train", "test") if k in X}, lvl,
                         os.path.join(args.out, f"projections_ttbar_{lvl}.png"))
        auc = plot_scores(scores, lvl, os.path.join(args.out, f"scores_{lvl}.png"))
        s = summary["levels"][lvl] = {
            "n": {k: int(len(v["Y"])) for k, v in X.items()},
            "n_coefficients": int(f.c.size),
            "basis_params": ({"J1": f.J1, "J2": f.J2, "n_legendre_hi": f.nl_hi, "n_hermite_hi": f.nh_hi,
                              "y_scale": f.y_scale,
                              "blocks": {b[0]: int(n) for b, n in zip(f.blocks, f.sizes)}}
                             if hasattr(f, "blocks") else
                             {"l1max": f.l1max, "l2max": f.l2max} if getattr(f, "sh", False) else {"J1": 1, "J2": 2}),
            "n_significant_3sigma": int(np.sum(signif > 3)),
            "n_significant_5sigma": int(np.sum(signif > 5)),
            "integral": float(f.integral()),
            "max_rel_imag": imag,
            "frac_negative": {k: float(np.mean(v <= 0)) for k, v in scores.items()},
            "mean_surprisal": {k: float(np.mean(-np.log(v[v > 0]))) for k, v in scores.items()},
            "median_f": {k: float(np.median(v)) for k, v in scores.items()},
            "auc_hh_vs_tt": auc,
            "self_inclusion": {**selfinc, **({"ks_naive_vs_test": ks_distance(naive, scores["test"]),
                                              "ks_loo_vs_test": ks_distance(scores["train"], scores["test"])}
                                             if "test" in scores else {})},
            "proj_chi2": {a: list(v) for a, v in chi2.items()},
        }
        s["truncation"] = truncation_study(f, X)
        s["y_scale_scan"] = y_scale_scan(X["train"]["Y"], args.n_hermite)
        print(f"[osde] {lvl}: N_train={s['n']['train']}  significant(>3σ)={s['n_significant_3sigma']}"
              f"/{s['n_coefficients']}  ∫f={s['integral']:.3f}  f<=0: "
              + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["frac_negative"].items())
              + f"  AUC(hh vs tt)={auc:.3f}")
    # merge with levels fitted earlier (so a single level can be (re)fitted with --levels)
    path = os.path.join(args.out, "summary.json")
    if os.path.exists(path):
        old = json.load(open(path))
        old["levels"].update(summary["levels"])
        summary["levels"] = old["levels"]
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[osde] outputs in {args.out}")


if __name__ == "__main__":
    main()
