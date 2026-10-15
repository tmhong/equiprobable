#!/usr/bin/env python3
"""
Reco-level analysis of one pileup case, identical for every <mu> so the cases are comparable.

  .venv/bin/python code/analysis/pileup/analyze_pileup.py --mu 50 --tt-fit samples/.../pu50_fit --tt-test samples/.../pu50_test \
        --hh samples/.../pu50_hh --out results/pileup/results_pu50.json

Does, at reco level (events passing 1 lepton, >=4 jets, >=1 b-jet and the chi2 reconstruction):
  validation   reco efficiency, jet and b-jet multiplicities, fraction of events with all four jets correct,
               neutrino p_z and MET resolution, resonance-mass widths (tt̄ and hh)
  f_tt̄        the paper-basis OSDE density fitted on the reco angles of the fit sample; all terms and the
               terms with |c|/sigma > 10, 5, 3; saved as results/pileup/f_ttbar_reco_pu<mu>.npz
  masses       one Cauchy factor per mass (M, W = median, half-IQR of the fit sample) for m_W,had, m_t,lep, m_t,had
  scores       surprisal of angles, masses, and angles + masses on the independent tt̄ and on hh:
               AUC (bootstrap error) and signal efficiency at tt̄ efficiency 1% and 10%
  gen check    the existing gen-level f (results/osde/f_ttbar_gen.npz) + masses on the gen angles of the same events:
               gen level is pileup-independent, so this must agree between the <mu> cases
Outputs: the JSON (schema used by code/docs/make_note.py), per-event scores results/pileup/scores_pu<mu>.npz, figures in
results/pileup/figures/.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, ROOT)
import repo_paths  # noqa: E402
from plotting import make_plots  # noqa: E402
from analysis.study_salvage.salvage import surprisal  # noqa: E402
from analysis.study_salvage.simplest import cauchy_surprisal  # noqa: E402
from lib import osde, style, summary  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402

OUT = repo_paths.results_dir("pileup")       # results/pileup: densities, scores, figures
OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
MASSES = ["m_w_had", "m_t_lep", "m_t_had"]


def load(path, nmax=None):
    ev, xsec = make_plots.load_sample(os.path.basename(path), path, ["parton", "gen", "reco"], nmax, OPTS)
    P, G, R = ev["parton"], ev["gen"], ev["reco"]
    out = {"n_all": len(R), "xsec": xsec}
    for lvl, E in (("gen", G), ("reco", R)):
        ok = np.asarray(E.reco_ok)
        ang = {a: np.asarray(E[a], float) for a in ANGLE_NAMES}
        ok &= np.all([np.isfinite(v) for v in ang.values()], 0)
        m = {"m_w_had": np.asarray(E.w_had.mass, float), "m_t_lep": np.asarray(E.t_lep.mass, float),
             "m_t_had": np.asarray(E.t_had.mass, float)}
        out[lvl] = {"ok": ok, "ang": {a: v[ok] for a, v in ang.items()}, "m": {k: v[ok] for k, v in m.items()}}
    ok = np.asarray(R.reco_ok)
    out["val"] = {
        "reco_eff": float(np.mean(ok)), "gen_eff": float(np.mean(np.asarray(G.reco_ok))),
        "njets_mean_all": float(np.mean(np.asarray(R.njets))), "njets_mean_reco_ok": float(np.mean(np.asarray(R.njets)[ok])),
        "nbjets_mean_all": float(np.mean(np.asarray(R.nbjets))),
        "nu_pz_half_width": float(summary.central((np.asarray(R.nu.pz) - np.asarray(P.nu.pz))[ok])[1]),
        "met_res_half_width": float(summary.central((np.asarray(R.met) - np.asarray(P.nu.pt))[ok])[1]),
    }
    try:
        corr = summary.correct_assignment(R, P)
        out["val"]["correct_jets_frac"] = float(corr.sum() / max(ok.sum(), 1))
    except Exception:
        out["val"]["correct_jets_frac"] = None
    return out


def auc_err(lab, score, n=100, seed=0):
    rng = np.random.default_rng(seed)
    a = roc_auc_score(lab, score)
    bs = [roc_auc_score(lab[i], score[i]) for i in (rng.integers(0, len(lab), len(lab)) for _ in range(n))]
    return float(a), float(np.std(bs))


def eff_at(s_sig, s_bkg, bkg_eff):
    return float(np.mean(s_sig > np.quantile(s_bkg, 1 - bkg_eff)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mu", required=True)
    p.add_argument("--tt-fit", required=True)
    p.add_argument("--tt-test", required=True)
    p.add_argument("--hh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fig-dir", default=os.path.join(OUT, "figures"))
    p.add_argument("--max-fit", type=int, default=None)
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    style.use()

    fit, tt, hh = load(args.tt_fit, args.max_fit), load(args.tt_test), load(args.hh)
    res = {"mu": args.mu, "samples": {"tt_fit": args.tt_fit, "tt_test": args.tt_test, "hh": args.hh},
           "n_tt_fit": int(fit["reco"]["ok"].sum()), "n_tt_test": int(tt["reco"]["ok"].sum()), "n_hh": int(hh["reco"]["ok"].sum()),
           "n_generated": {"tt_fit": fit["n_all"], "tt_test": tt["n_all"], "hh": hh["n_all"]},
           "reco_eff_tt": tt["val"]["reco_eff"], "reco_eff_hh": hh["val"]["reco_eff"], "gen_eff_tt": tt["val"]["gen_eff"],
           "njets_mean_tt": tt["val"]["njets_mean_all"], "njets_mean_hh": hh["val"]["njets_mean_all"],
           "nbjets_mean_tt": tt["val"]["nbjets_mean_all"], "correct_jets_frac_tt": tt["val"]["correct_jets_frac"],
           "nu_pz_half_width": tt["val"]["nu_pz_half_width"], "met_res_half_width": tt["val"]["met_res_half_width"]}
    # mass factors from the fit sample
    cauchy = {}
    for k in MASSES:
        x = fit["reco"]["m"][k]; x = x[np.isfinite(x)]
        M = float(np.median(x)); q1, q3 = np.percentile(x, [25, 75])
        cauchy[k] = (M, float(0.5 * (q3 - q1)))
        res[f"M_{k}"], res[f"W_{k}"] = cauchy[k]
        xh = hh["reco"]["m"][k]; xh = xh[np.isfinite(xh)]
        q1h, q3h = np.percentile(xh, [25, 75])
        res[f"W_{k}_hh"] = float(0.5 * (q3h - q1h))

    def s_mass(d):
        return sum(cauchy_surprisal(d["m"][k], *cauchy[k]) for k in MASSES)

    # the reco-level density
    f = osde.TTbarDensity().fit(fit["reco"]["ang"])
    f.save(os.path.join(OUT, f"f_ttbar_reco_pu{args.mu}.npz"), mu=args.mu, sample=args.tt_fit)
    sig = f.significance()
    variants = {"all": f, "10sigma": f.truncated(sig > 10), "5sigma": f.truncated(sig > 5), "3sigma": f.truncated(sig > 3)}
    res["n_terms"] = {k: int(np.count_nonzero(v.c)) for k, v in variants.items()}
    fref = {k: v(fit["reco"]["ang"]) for k, v in variants.items()}
    lab = np.r_[np.zeros(res["n_tt_test"]), np.ones(res["n_hh"])]
    scores = {}
    sm_tt, sm_hh = s_mass(tt["reco"]), s_mass(hh["reco"])
    scores["masses"] = (sm_tt, sm_hh)
    for k, v in variants.items():
        st, sh = surprisal(v(tt["reco"]["ang"]), fref[k]), surprisal(v(hh["reco"]["ang"]), fref[k])
        scores[f"angles_{k}"] = (st, sh)
        scores[f"angles_{k}_masses"] = (st + sm_tt, sh + sm_hh)
    for name, (st, sh) in scores.items():
        a, e = auc_err(lab, np.r_[st, sh])
        res[f"auc_{name}"], res[f"err_{name}"] = a, e
        res[f"eff1_{name}"], res[f"eff10_{name}"] = eff_at(sh, st, 0.01), eff_at(sh, st, 0.10)
    # aliases used by the note
    res["auc_angles_all"], res["auc_angles_10sigma"] = res["auc_angles_all"], res["auc_angles_10sigma"]
    res["auc_angles_masses"] = res["auc_angles_10sigma_masses"]
    # gen-level check with the existing gen density (pileup independent)
    fg = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", "f_ttbar_gen.npz"))
    fg = fg.truncated(fg.significance() > 10)
    gc = {}
    for k in MASSES:
        x = fit["gen"]["m"][k]; x = x[np.isfinite(x)]
        q1, q3 = np.percentile(x, [25, 75]); gc[k] = (float(np.median(x)), float(0.5 * (q3 - q1)))
    sg = lambda d: sum(cauchy_surprisal(d["m"][k], *gc[k]) for k in MASSES)  # noqa: E731
    fr = fg(fit["gen"]["ang"])
    st = surprisal(fg(tt["gen"]["ang"]), fr) + sg(tt["gen"]); sh = surprisal(fg(hh["gen"]["ang"]), fr) + sg(hh["gen"])
    labg = np.r_[np.zeros(len(st)), np.ones(len(sh))]
    res["gen_check_auc_angles_masses"], res["gen_check_err"] = auc_err(labg, np.r_[st, sh])
    res["gen_check_n"] = [int(len(st)), int(len(sh))]
    np.savez(os.path.join(OUT, f"scores_pu{args.mu}.npz"), **{f"{n}_tt": v[0] for n, v in scores.items()},
             **{f"{n}_hh": v[1] for n, v in scores.items()})
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))

    # figures: masses, multiplicities, scores
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, k, xl, rng in zip(axs, MASSES, [r"$m_{W,\mathrm{had}}$ [GeV]", r"$m_{t,\mathrm{lep}}$ [GeV]", r"$m_{t,\mathrm{had}}$ [GeV]"],
                              [(0, 200), (50, 400), (50, 400)]):
        bins = np.linspace(*rng, 51)
        ax.hist(tt["reco"]["m"][k], bins, density=True, histtype="step", lw=2.4, color="#1f4e9c", label=r"$t\bar t$")
        ax.hist(hh["reco"]["m"][k], bins, density=True, histtype="step", lw=2.4, color="#d1495b", label=r"$hh$")
        x = np.linspace(*rng, 2000); M, W = cauchy[k]; y = 1 / ((x - M) ** 2 + W ** 2); y /= np.trapezoid(y, x)
        ax.plot(x, y, "--", color="0.2", lw=1.8, label=rf"Cauchy $M$={M:.0f}, $W$={W:.0f}")
        ax.set_yscale("log"); ax.set_ylim(1e-5, 30 * ax.get_ylim()[1]); ax.set_xlabel(xl); ax.legend(fontsize=13, loc="upper center")
    axs[0].set_ylabel("unit area")
    fig.suptitle(rf"reco level, $\langle\mu\rangle={args.mu}$: resonance masses")
    fig.tight_layout(); fig.savefig(os.path.join(args.fig_dir, f"masses_pu{args.mu}.png"), dpi=110); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    for name, col in (("masses", "#2a9d8f"), ("angles_10sigma_masses", "#1f4e9c")):
        st, sh = scores[name]
        lo, hi = np.percentile(np.r_[st, sh], [0.5, 99.5]); bins = np.linspace(lo, hi, 41)
        ax = axs[0] if name == "masses" else axs[1]
        ax.hist(st, bins, density=True, histtype="step", lw=2.4, color="#1f4e9c", label=r"$t\bar t$ (independent)")
        ax.hist(sh, bins, density=True, histtype="step", lw=2.4, color="#d1495b", label=r"$hh$")
        ax.set_xlabel("surprisal: " + ("masses" if name == "masses" else r"angles ($>10\sigma$) + masses"))
        ax.set_title(rf"AUC {res['auc_' + name]:.3f}", fontsize=17); ax.legend(fontsize=14)
    axs[0].set_ylabel("unit area")
    fig.suptitle(rf"reco level, $\langle\mu\rangle={args.mu}$: scores")
    fig.tight_layout(); fig.savefig(os.path.join(args.fig_dir, f"scores_pu{args.mu}.png"), dpi=110); plt.close(fig)
    print(f"[pileup] wrote {args.out}")


if __name__ == "__main__":
    main()
