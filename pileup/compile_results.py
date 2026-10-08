#!/usr/bin/env python3
"""
Compile the per-<mu> results (pileup/results_pu<mu>.json from analyze_pileup.py) into pileup/results.json
in the schema used by note/make_note.py and make_slides.py, plus the summary figure pileup/figures/summary_vs_mu.png
(AUC and efficiencies versus <mu>) and a short LaTeX summary paragraph.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from plotting import style  # noqa: E402

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
        p = os.path.join(HERE, f"results_pu{mu}.json")
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
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
    for key, lab, col, mk in (("auc_masses", "masses (BW)", "#2a9d8f", "s"), ("auc_angles_masses", r"angles ($>10\sigma$) + BW", "#1f4e9c", "o"),
                              ("auc_angles_10sigma", r"angles ($>10\sigma$)", "#d1495b", "^"), ("auc_angles_all", "angles, all terms", "#e9a03b", "v")):
        err = {"auc_masses": "err_masses", "auc_angles_masses": "err_angles_10sigma_masses", "auc_angles_10sigma": "err_angles_10sigma",
               "auc_angles_all": "err_angles_all"}[key]
        axs[0].errorbar(x, [per[m][key] for m in mus], [per[m][err] for m in mus], marker=mk, ms=9, lw=2.2, capsize=4, color=col, label=lab)
    axs[0].axhline(0.5, color="0.6", lw=1); axs[0].set_ylabel(r"AUC ($hh$ vs $t\bar t$, reco)"); axs[0].legend(fontsize=13, frameon=False)
    for key, lab, col in (("reco_eff_tt", r"reco selection, $t\bar t$", "#1f4e9c"), ("reco_eff_hh", r"reco selection, $hh$", "#d1495b"),
                          ("correct_jets_frac_tt", r"all 4 $\chi^2$ jets correct, $t\bar t$", "#2a9d8f")):
        axs[1].plot(x, [100 * per[m][key] for m in mus], "o-", ms=9, lw=2.2, color=col, label=lab)
    axs[1].set_ylabel("per cent"); axs[1].legend(fontsize=13, frameon=False)
    for key, lab, col in (("njets_mean_tt", r"jets / event, $t\bar t$", "#1f4e9c"), ("nu_pz_half_width", r"$\nu$ $p_z$ 68\% half-width [10 GeV]", "#d1495b"),
                          ("met_res_half_width", r"MET resolution [10 GeV]", "#2a9d8f")):
        sc = 0.1 if "width" in key else 1.0
        axs[2].plot(x, [sc * per[m][key] for m in mus], "o-", ms=9, lw=2.2, color=col, label=lab)
    axs[2].legend(fontsize=13, frameon=False)
    for ax in axs:
        ax.set_xlabel(r"mean pileup $\langle\mu\rangle$"); ax.set_xticks(x)
    fig.suptitle(r"Reco level versus pileup (Delphes ATLAS pileup card, same hard events at every $\langle\mu\rangle$)")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    figp = os.path.join(HERE, "figures", "summary_vs_mu.png")
    fig.savefig(figp, dpi=110)
    out["figures"] = [[os.path.relpath(figp, ROOT), r"AUC, selection efficiencies and resolutions at reco level versus $\langle\mu\rangle$."]]
    for mu in mus:
        out["figures"].append([os.path.relpath(os.path.join(HERE, "figures", f"masses_pu{mu}.png"), ROOT),
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
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({m: {k: out["mu"][m][k] for k in ("reco_eff_tt", "njets_mean_tt", "correct_jets_frac_tt", "auc_masses", "auc_angles_masses", "gen_check_auc_angles_masses")}
                      for m in mus}, indent=1))
    print(f"[pileup] wrote {os.path.join(HERE, 'results.json')} and {figp}")


if __name__ == "__main__":
    main()
