#!/usr/bin/env python3
"""
Compile the per-<mu> results (results/analysis/pileup/results_pu<mu>.json from analyze_pileup.py) into results/analysis/pileup/results.json
in the schema used by code/docs/make_note.py and code/docs/make_slides.py, plus the summary figure
results/analysis/pileup/figures/summary_vs_mu.png (AUC and efficiencies versus <mu>) and a short LaTeX summary paragraph.
The figure paths in results.json are relative to results/ (pileup/figures/...).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # code/
import repo_paths  # noqa: E402
from lib import style  # noqa: E402

OUT = repo_paths.results_dir("pileup")      # results/analysis/pileup
ROOT = repo_paths.RESULTS                   # figure paths in results.json are relative to results/

MUS = ["0", "50", "200"]
FIELDS = ["n_tt_fit", "n_tt_test", "n_hh", "reco_eff_tt", "reco_eff_hh", "gen_eff_tt", "njets_mean_tt", "njets_mean_hh",
          "nbjets_mean_tt", "correct_jets_frac_tt", "nu_pz_half_width", "met_res_half_width",
          "W_m_w_had", "W_m_t_lep", "W_m_t_had", "W_m_w_had_hh", "W_m_t_lep_hh", "W_m_t_had_hh",
          "auc_angles_all", "err_angles_all", "auc_angles_10sigma", "err_angles_10sigma", "auc_masses", "err_masses",
          "auc_angles_masses", "err_angles_10sigma_masses", "eff1_masses", "eff10_masses",
          "eff1_angles_10sigma_masses", "eff10_angles_10sigma_masses", "gen_check_auc_angles_masses", "gen_check_err", "n_terms"]


def main():
    per = {}
    for mu in MUS:
        p = os.path.join(OUT, f"results_pu{mu}.json")
        if os.path.exists(p):
            per[mu] = json.load(open(p))
    if not per:
        sys.exit("no results_pu<mu>.json found")
    out = {"mu": {mu: {k: r.get(k) for k in FIELDS} for mu, r in per.items()}, "figures": [], "summary_tex": ""}
    for mu in per:
        out["mu"][mu]["err_angles_masses"] = per[mu].get("err_angles_10sigma_masses")
    mus = sorted(per, key=float)
    x = [float(m) for m in mus]
    style.use()
    # Sized for a 16:9 slide (~150 x 65 mm, see lib/style.py): 15.5 x 6 in is scaled by ~0.38, so the
    # 16-20 pt text below lands at ~6-7.6 pt.  One unit per panel; distinct markers as well as colours.
    BLUE, RED, TEAL, ORANGE = "#1f4e9c", "#d1495b", "#2a9d8f", "#e9a03b"
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 6.0))
    kw = dict(ms=9, lw=2.2)
    for key, err, lab, col, mk in (("auc_masses", "err_masses", "masses (BW)", TEAL, "s"),
                                   ("auc_angles_masses", "err_angles_10sigma_masses", r"angles ($>10\sigma$) + BW", BLUE, "o"),
                                   ("auc_angles_10sigma", "err_angles_10sigma", r"angles ($>10\sigma$)", RED, "^"),
                                   ("auc_angles_all", "err_angles_all", "angles, all terms", ORANGE, "v")):
        axs[0].errorbar(x, [per[m][key] for m in mus], [per[m][err] for m in mus], marker=mk, capsize=4, color=col, label=lab, **kw)
    axs[0].axhline(0.5, color="0.6", lw=1, zorder=0)
    axs[0].set_ylabel(r"AUC ($hh$ vs $t\bar t$)")
    axs[0].set_title("separation")
    for key, lab, col, mk in (("reco_eff_tt", r"$t\bar t$ selected", BLUE, "o"), ("reco_eff_hh", r"$hh$ selected", RED, "s"),
                              ("correct_jets_frac_tt", r"$t\bar t$, 4 $\chi^2$ jets right", TEAL, "^")):
        axs[1].plot(x, [100 * per[m][key] for m in mus], marker=mk, color=col, label=lab, **kw)
    axs[1].set_ylabel("per cent of events")
    # the mean jet multiplicity that drives the selection, as a row of labels at the same <mu> (not a second scale)
    top = axs[1].secondary_xaxis("top")
    top.set_xticks(x, [f"{per[m]['njets_mean_tt']:.1f}" for m in mus])
    top.set_xlabel(r"mean jets / $t\bar t$ event", fontsize=17)
    for key, lab, col, mk in (("nu_pz_half_width", r"$\nu$ $p_z$", RED, "o"), ("met_res_half_width", r"MET $-$ $p_T^\nu$", TEAL, "s")):
        axs[2].plot(x, [per[m][key] for m in mus], marker=mk, color=col, label=lab, **kw)
    axs[2].set_ylabel("68% half-width [GeV]")
    axs[2].set_title(r"resolution ($t\bar t$, selected)")
    pad = 0.12 * (x[-1] - x[0]) or 10.0  # (a single <mu> would give a zero span)
    for ax in axs:
        ax.set_xlabel(r"mean pileup $\langle\mu\rangle$"); ax.set_xticks(x)
        ax.set_xlim(x[0] - pad, x[-1] + pad)
        y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + 0.45 * (y1 - y0))  # headroom: legends sit above the data
        ax.legend(fontsize=16, frameon=False, loc="upper left" if ax is not axs[0] else "upper right", handlelength=1.6)
    axs[1].set_ylim(0, None); axs[2].set_ylim(0, None)
    fig.suptitle(r"Reco level versus pileup (Delphes ATLAS pileup card; same hard events at every $\langle\mu\rangle$)")
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    figp = os.path.join(OUT, "figures", "summary_vs_mu.png")
    fig.savefig(figp, dpi=110)
    out["figures"] = [[os.path.relpath(figp, ROOT), r"AUC, selection efficiencies and resolutions at reco level versus $\langle\mu\rangle$."]]
    for mu in mus:
        out["figures"].append([os.path.relpath(os.path.join(OUT, "figures", f"masses_pu{mu}.png"), ROOT),
                               rf"Resonance masses at reco level for $\langle\mu\rangle={mu}$ ($t\bar t$ and $hh$, with the two-number Cauchy model)."])
    lo, hi = per[mus[0]], per[mus[-1]]
    out["summary_tex"] = (
        rf"From $\langle\mu\rangle={mus[0]}$ to {mus[-1]} the reco selection efficiency of $t\bar t$ goes from {100 * lo['reco_eff_tt']:.0f}\% to "
        rf"{100 * hi['reco_eff_tt']:.0f}\% (pileup jets pass the $\geq4$-jet cut; the card has no pileup-jet rejection), the mean jet "
        rf"multiplicity from {lo['njets_mean_tt']:.1f} to {hi['njets_mean_tt']:.1f}, the fraction of events with all four $\chi^2$ jets correct "
        rf"from {100 * lo['correct_jets_frac_tt']:.0f}\% to {100 * hi['correct_jets_frac_tt']:.1f}\%, and the neutrino $p_z$ half-width from "
        rf"{lo['nu_pz_half_width']:.0f} to {hi['nu_pz_half_width']:.0f}~GeV.  The $hh$ separation of the masses goes from AUC "
        rf"{lo['auc_masses']:.3f} to {hi['auc_masses']:.3f} and that of angles + masses from {lo['auc_angles_masses']:.3f} to "
        rf"{hi['auc_angles_masses']:.3f}; the gen-level check is {', '.join(f"{per[m]['gen_check_auc_angles_masses']:.3f}" for m in mus)} "
        r"at the three pileup levels, as it must be for paired events that do not see pileup at gen level.")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({m: {k: out["mu"][m][k] for k in ("reco_eff_tt", "njets_mean_tt", "correct_jets_frac_tt", "auc_masses", "auc_angles_masses", "gen_check_auc_angles_masses")}
                      for m in mus}, indent=1))
    print(f"[pileup] wrote {os.path.join(OUT, 'results.json')} and {figp}")


if __name__ == "__main__":
    main()
