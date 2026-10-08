#!/usr/bin/env python3
"""
Make Beamer slides summarizing the ttbar -> bblvqq sample production and validation.

All numbers on the slides are recomputed from the sample, and the figures are taken
from the plots directory (run make_plots.py first, or pass --make-plots).
Edit build_slides() below to add, remove or reorder slides.

Examples:
  python make_slides.py --sample samples/ttbar_bblvqq/run_seed12345
  python make_slides.py --sample samples/ttbar_bblvqq/run_seed12345 --make-plots --author "A. Person"
  python make_slides.py --sample ... --no-compile          # only write slides/slides.tex
"""
import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_plots  # noqa: E402
from plotting import summary  # noqa: E402


# ============================================================================
# Slide content.  Each entry is one frame; helpers are defined further below.
#   S    = numbers from plotting/summary.py   (S["reco"]["hel_lep"]["F0"], ...)
#   meta = the sample's metadata.json         (cross section, seed, ...)
# ============================================================================
def build_slides(S, meta):
    xs = meta.get("cross_section_pb")
    slides = [
        section("Introduction"),
        bullets("Goal", [
            r"Generate $pp\to t\bar{t}\to b\bar{b}\,W^+W^-\to b\bar{b}\,\ell\nu\,q\bar{q}'$ "
            r"at parton, particle (gen) and detector (reco) level",
            r"Validate the samples and the ten angles $\mathbf{x}=\{Y,\Theta,\boldsymbol{\theta},"
            r"\tilde{\boldsymbol{\theta}}\}$ used for the $t\bar{t}$ density $f_{t\bar{t}}(\mathbf{x})$",
            r"Everything is scripted: \texttt{generate\_samples.py}, \texttt{make\_plots.py}, "
            r"\texttt{make\_slides.py}",
        ]),
        section(r"$t\bar t$ samples and validation", "ttbar samples and validation"),
        bullets("Sample generation", [
            r"MadGraph5\_aMC@NLO 3.6.7 (LO, \texttt{sm}, 4-flavour proton) "
            r"$\to$ Pythia8 $\to$ Delphes 3.5.1, ATLAS card, \textbf{no pileup}",
            r"Decay chains (spin correlations kept), no generator-level cuts: "
            r"$pp\to t\bar t$, $t\to bW^+(\to\ell^+\nu)$, $\bar t\to\bar bW^-(\to q\bar q')$, plus the charge conjugate",
            rf"$\sqrt{{s}}={fmt(meta.get('sqrts_GeV', 0) / 1000, 3)}$ TeV, "
            rf"$\ell={lepton_tex(meta.get('leptons', '?'))}$, "
            rf"{meta.get('nevents', S['reco']['n'])} events, seed {meta.get('seed', '?')}",
            rf"$\sigma = {fmt(xs, 4)}$ pb $\approx \sigma_{{t\bar{{t}}}}^\mathrm{{LO}}\times"
            r"2\,\mathcal{B}(W\to\ell\nu)\,\mathcal{B}(W\to q\bar{q}')$ (with $\ell=e,\mu$)",
        ]),
        table("Event levels",
              ["", "parton", r"parton + $\ell$ trig.", "gen (particle)", "reco"], [
            ["source", "MG5 LHE", "MG5 LHE", "Pythia8", "Delphes ATLAS"],
            ["selection", "none", r"$\ell$ cuts only", r"$1\ell,\geq4j,\geq1b,\chi^2$", r"$1\ell,\geq4j,\geq1b,\chi^2$"],
            ["events kept", r"100\%", rf"{100 * S['parton']['trigger_eff']:.1f}\%",
             rf"{100 * S['gen']['n_reco'] / S['gen']['n']:.1f}\%", rf"{100 * S['reco']['n_reco'] / S['reco']['n']:.1f}\%"],
            [r"$E_T^\mathrm{miss}$", r"$p_T^\nu$", r"$p_T^\nu$", r"$\sum$ all $\nu$", r"$-\sum$ calo towers"],
        ], note=r"Parton = MG5 partons with \emph{no} cuts; parton + $\ell$ trigger adds only the lepton cuts "
                r"($p_T>25$ GeV, $|\eta|<2.47$ ($e$), $2.5$ ($\mu$)). Reco and gen $E_T^\mathrm{miss}$ agree for muon and "
                rf"electron events (reco$-$gen along the lepton: ${S['met_check']['mu']:+.1f}$ vs "
                rf"${S['met_check']['e']:+.1f}$ GeV), so the calorimeter MET does not lose muons."),
        table("Object definitions (gen and reco)", ["", "particle level (gen)", "reco (Delphes ATLAS)"], [
            ["jets", r"GenJet anti-$k_t$ 0.4", r"Jet anti-$k_t$ 0.4"],
            ["", r"\multicolumn{2}{c}{$p_T>25$ GeV, $|\eta|<2.5$}"],
            ["b-jets", r"$\Delta R(j,b)<0.4$, $p_T^b>5$ GeV", "BTag bit 0"],
            ["leptons", r"status-1 $e/\mu$ from $W/Z/h$ (or such a $\tau$)", "Electron, Muon"],
            ["", r"\multicolumn{2}{c}{$p_T>25$ GeV, $|\eta|<2.47\ (e)$, $2.5\ (\mu)$}"],
            ["overlap", r"drop jets with $\Delta R(j,\ell)<0.4$", "UniqueObjectFinder"],
        ], note=r"Preselection: exactly 1 lepton, $\geq 4$ jets, $\geq 1$ b-jet, then the $\chi^2$ $t\bar t$ reconstruction."),
        table("Cutflow", ["selection", "gen", "", "reco", ""],
              cutflow_rows(S), note=r"Fractions relative to all generated events."),
        figures(r"Leptons and $E_T^\mathrm{miss}$", ["lep_pt", "lep_eta", "met", "mtw"]),
        figures("Jets", ["njets", "nbjets", "jet1_pt", "bjet_pt"]),
        bullets(r"$t\bar{t}$ reconstruction", [
            r"$\chi^2=\left(\frac{m_{jj}-m_W}{\sigma_W}\right)^2+\left(\frac{m_{jjb}-m_t}{\sigma_t}\right)^2"
            r"+\left(\frac{m_{\ell\nu b}-m_t}{\sigma_t}\right)^2$ over assignments of the 6 leading jets",
            r"b slots hold $\min(n_b,2)$ b-tagged jets",
            r"$p_z^\nu$ from the $m_W$ constraint; both solutions enter the $\chi^2$, so the "
            r"$m_t$ term resolves the two-fold ambiguity",
            rf"All four jets correct in {pct(S['gen'], 'n_correct', 'n_reco')} (gen) and "
            rf"{pct(S['reco'], 'n_correct', 'n_reco')} (reco) of reconstructed events",
        ]),
        figures(r"Reconstructed masses", ["top_had_mass", "top_lep_mass", "w_had_mass", "ttbar_mass"]),
        table(r"Mass resolution (median $\pm$ 68\% half-width, GeV)",
              ["", r"$m_{t,\mathrm{had}}$, all", r"$m_{t,\mathrm{had}}$, correct jets",
               r"$m_{W,\mathrm{had}}$, all", r"$m_{W,\mathrm{had}}$, correct jets"],
              [[lvl] + [pm(S[lvl][k]) for k in ("m_thad", "m_thad_correct", "m_whad", "m_whad_correct")]
               for lvl in ("gen", "reco")],
              note=r"The broad inclusive peaks come from wrong jet choices (combinatorics) "
                   r"and quarks that are out of acceptance, not from the jet response."),
        section("The ten angles"),
        wide_figure("Definition of the ten angles", os.path.join(HERE, "figures", "angle_diagram_slide.pdf")),
        bullets("The ten angles (as in the paper)", [
            r"$Y$: $t\bar{t}$ rapidity (lab); $\Theta$: polar angle of $t_\mathrm{lep}$ in the $t\bar{t}$ frame",
            r"$(\theta,\phi)$: $W$ in the top rest frame; $z=\hat{p}_t^\mathrm{lab}$, "
            r"$y=\hat{z}_\mathrm{beam}\times z$, $x=y\times z$",
            r"$(\theta',\phi')$: fermion in the $W$ rest frame (boost from top frame); "
            r"$z'=\hat{p}_W$, $y'=z\times z'$, $x'=y'\times z'$, i.e.\ $D(\phi,\theta,0)$",
            r"Leptonic top: fermion $=\ell$. Hadronic top ($\sim$): fermion = down-type quark",
            r"gen/reco: down-type jet chosen by truth-matching to the LHE quarks "
            r"(assumes up/down tagging, as in the paper)",
            r"Check: building momenta from known angles and recovering them agrees to $10^{-14}$",
        ]),
        figures(r"Production angles $Y$, $\cos\Theta$", ["angle_prod_Y", "angle_prod_cosTheta"]),
        figures(r"Leptonic top: $\theta,\phi,\theta',\phi'$",
                ["angle_lep_costheta", "angle_lep_phi", "angle_lep_costheta_prime", "angle_lep_phi_prime"]),
        figures(r"Hadronic top: $\tilde\theta,\tilde\phi,\tilde\theta',\tilde\phi'$",
                ["angle_had_costheta", "angle_had_phi", "angle_had_costheta_prime", "angle_had_phi_prime"]),
        table(r"$W$ helicity fractions from $\cos\theta'$ (moments)",
              ["", r"$F_0$", r"$F_L$", r"$F_R$"], helicity_rows(S),
              note=r"SM (LO): $F_0=m_t^2/(m_t^2+2m_W^2)\approx0.70$, $F_L\approx0.30$, $F_R\approx0$. "
                   r"Parton level reproduces it; gen/reco are distorted by acceptance and "
                   r"combinatorics (no unfolding)."),
        table(r"Angle resolution vs parton level (correlation / median $|\Delta|$)",
              [r"angle", "gen, all", "gen, correct jets", "reco, all", "reco, correct jets"],
              angle_rows(S)),
        table(r"Neutrino $p_z$ resolution", ["", r"median $\Delta p_z$ [GeV]", r"68\% half-width [GeV]"],
              [[lvl, fmt(S[lvl]["nu_pz"][0], 2), fmt(S[lvl]["nu_pz"][1], 3)] for lvl in ("gen", "reco")],
              note=r"$\Delta p_z = p_z^{\nu,\mathrm{reco}}-p_z^{\nu,\mathrm{parton}}$, events with a "
                   r"reconstructed $t\bar{t}$ system."),
    ]
    slides += nu_pz_slides()
    slides += lep_had_slides()
    return slides


def lep_had_slides():
    """Leptonic vs hadronic decay angles (lep_had_compare.py)."""
    if not os.path.exists(os.path.join(PLOTS_DIR, "lep_vs_had_parton_decay.png")):
        return []
    r = json.load(open(os.path.join(PLOTS_DIR, "lep_vs_had.json")))
    key = "cos_theta_prime|cos_theta_prime_tilde"
    pp = ", ".join(f"{v['p']:.2f}" for k, v in r["parton"].items() if k != "n")
    return [wide_figure(r"Leptonic vs hadronic top, parton level: $W$ decay angles",
                        os.path.join(PLOTS_DIR, "lep_vs_had_parton_decay.png"),
                        note=rf"SM: the down-type quark plays the role of the charged lepton. Parton level agrees "
                             rf"(KS $p$ = {pp} for $\cos\theta$, $\phi$, $\cos\theta'$, $\phi'$)."),
            wide_figure(r"Leptonic vs hadronic top, reco level with correct jets",
                        os.path.join(PLOTS_DIR, "lep_vs_had_reco_correct_jets_decay.png"),
                        note=rf"Even with perfect flavour tagging and correct jets they differ ($\cos\theta'$ KS distance "
                             rf"{r['reco, correct jets'][key]['ks']:.3f}): different acceptance and resolution on the two "
                             r"sides (lepton cuts + $\nu$ vs jet cuts and response).")]


def nu_pz_slides():
    """Δp_z distributions from nu_pz_plot.py (plots/nu_pz_{a,b,c}.png + nu_pz.json)."""
    if not os.path.exists(os.path.join(PLOTS_DIR, "nu_pz_a.png")):
        return []
    r = json.load(open(os.path.join(PLOTS_DIR, "nu_pz.json")))
    ch = r["choice"]
    return [figures(r"Neutrino $\Delta p_z$", ["nu_pz_a", "nu_pz_b"], note=(
                rf"68\% half-width: gen {r['gen']['half_width_68']:.0f} GeV, reco {r['reco']['half_width_68']:.0f} GeV; "
                rf"reco with all jets correct {r['reco_all jets correctly assigned']['half_width_68']:.0f} GeV.")),
            figures(r"Neutrino $\Delta p_z$: which $m_W$ solution?", ["nu_pz_c"], note=(
                rf"No real solution in {100 * ch['frac_complex']:.0f}\% of events. With two, the top-mass $\chi^2$ picks "
                rf"the one closer to the truth in {100 * ch['frac_chi2_picks_closer']:.0f}\% of events vs "
                rf"{100 * ch['frac_smaller_picks_closer']:.0f}\% for the smaller $|p_z|$."))]


def hh_whad_slides():
    """Hadronic-W mass in hh, split into on-shell W and W* (hh_plots.py)."""
    path = os.path.join(HERE, "plots_hh", "w_had_mass_split.png")
    if not os.path.exists(path):
        return []
    r = json.load(open(os.path.join(HERE, "plots_hh", "w_had_mass_split.json")))
    return [wide_figure(r"$hh$: hadronic $W$ mass, on-shell $W$ vs $W^*$", path, note=(
        rf"Hadronic boson is the $W^*$ in {100 * (1 - r['frac_had_onshell']):.0f}\% of events "
        rf"(true median {r['offshell_parton_median']:.0f} GeV), but the $t\bar t$-hypothesis $\chi^2$ picks jet "
        rf"pairs near $m_W$ (median {r['offshell_gen_median']:.0f} / {r['offshell_reco_median']:.0f} GeV at "
        r"gen / reco): the reconstruction makes $hh$ look like $t\bar t$."))]


def score_slides(title, d, lvl):
    """The two score figures of osde_fit.plot_scores: y (linear, log) and surprisal + ROC."""
    return [wide_figure(f"{title}: $y$", os.path.join(d, f"scores_{lvl}.png"),
                        note=r"Left: $y=f_{t\bar t}(\mathbf{x})$, linear (shaded: $y<0$). Right: log scale; the left "
                             r"bin holds all events with $y\le0$. $t\bar t$: independent sample and fit sample "
                             r"(leave-one-out)."),
            wide_figure(f"{title}: surprisal and ROC", os.path.join(d, f"scores_{lvl}_s.png"),
                        note=r"Left: $s'=-\ln(y+y_0)$, shifted so every event has $y'>0$. Right: ROC from a scan over "
                             r"$y_\mathrm{cut}$ (events with $f<y_\mathrm{cut}$ called $hh$-like).")]


def build_hh_osde_slides(H, O, tests):
    """Slides for the hh sample and for f_tt̄ from OSDE (H: hh numbers, O: osde/summary.json)."""
    lv = O["levels"]
    slides = [
        section(r"$hh$ samples", "hh samples"),
        bullets(r"$hh\to b\bar{b}\,WW^*\to b\bar{b}\,\ell\nu\,q\bar{q}'$ sample", [
            r"MadGraph5 built-in \texttt{heft} model (LO, $m_t\to\infty$), same Pythia8 + Delphes ATLAS chain",
            r"$gg\to hh$, $h\to b\bar b$, $h\to\ell\nu q\bar q'$ (+ charge conjugate), generated as one process",
            r"$h\to WW^*$ has one off-shell $W$, so $h\to\ell\nu q\bar{q}'$ is a 4-body decay",
            rf"{H['meta'].get('nevents', '?')} events, seed {H['meta'].get('seed', '?')}; "
            rf"$\sigma\times\mathcal{{B}} = {1000 * H['meta']['cross_section_pb']:.3g}$ fb "
            r"(LO HEFT with MG defaults; \emph{not} an SM prediction, so plots use unit area)",
            r"For $t\bar t$-hypothesis quantities, $hh$ uses the same $\chi^2$; at parton level the b pairing "
            r"is the one closest to $m_t$",
            r"Prompt leptons: status-1 $e/\mu$ from $W$, $Z$, $h$, or a $\tau$ from one of these "
            r"(in $h\to WW^*$ the lepton can come directly from the $h$)",
        ]),
        table(r"Cutflow: $t\bar{t}$ vs $hh$ (reco)", ["selection", r"$t\bar t$", "", r"$hh$", ""],
              [[a[0], str(a[1]), rf"{100 * a[1] / H['tt_cutflow'][0][1]:.1f}\%",
                str(b[1]), rf"{100 * b[1] / H['hh_cutflow'][0][1]:.1f}\%"]
               for a, b in zip(H["tt_cutflow"], H["hh_cutflow"])],
              note=r"$hh$ leptons are softer (from $W^*$) and less isolated, so the efficiency is lower."),
        wide_figure(r"$hh$ sample: parton, gen and reco level", os.path.join(HERE, "plots_hh", "hh_overview.png")),
        *hh_whad_slides(),

        section(r"$f_{t\bar t}$ by OSDE", "f_ttbar by OSDE"),
        bullets(r"Basis functions (as in the paper)", [
            r"$M^{j_1j_2}_{m'm}(\phi,\theta,\phi',\theta')=\frac{\sqrt{2j_1+1}\sqrt{2j_2+1}}{4\pi}"
            r"D^{j_1}_{m'm}(\phi,\theta,0)\,D^{j_2}_{m0}(\phi',\theta',0)$, $j_1\le1$, $j_2\le2$: 24 per top",
            r"$p_l(\cos\Theta)=\sqrt{\tfrac{2l+1}{2}}P_l$, $h_n(Y)=(2^n n!\sqrt{\pi})^{-1/2}H_n(Y)e^{-Y^2/2}$, "
            rf"$l<{O['n_legendre']}$, $n<{O['n_hermite']}$",
            rf"$B=M_{{\mathbf{{j}}}}M_{{\tilde{{\mathbf{{j}}}}}}\,p_l\,h_n$: $24\times24\times"
            rf"{O['n_legendre']}\times{O['n_hermite']}={lv[next(iter(lv))]['n_coefficients']}$ coefficients",
            r"OSDE: $\hat{c}=\frac{1}{N}\sum_k B^*(\mathbf{x}_k)$, with standard error; "
            r"$f_{t\bar t}=\sum\hat{c}\,B$ (real, since the set is closed under conjugation)",
            r"Hermite \emph{functions}: bare $H_n$ are orthogonal only with weight $e^{-Y^2}$",
            r"\textbf{Three separate fits}: $f^\mathrm{parton}_{t\bar t}$ from parton-level $t\bar t$ angles, applied "
            r"to parton-level $t\bar t$ and $hh$; likewise $f^\mathrm{gen}$ (gen on gen) and $f^\mathrm{reco}$ "
            r"(reco on reco)",
        ]),
        bullets("Checks of the basis and of OSDE", [
            rf"Unit tests: {tests}",
            r"Wigner $d^j_{m'm}$ agrees with closed forms and with \texttt{sympy} to $10^{-12}$",
            r"Gram matrices of $M$ (exact quadrature), $p_l$, $h_n$ equal the identity to $10^{-12}$",
            r"Toy: $2\times10^5$ events from a density with known coefficients; all 14400 recovered "
            r"with pulls of unit width; projections and $\int f=1$ reproduced",
            r"Fitted $f$: imaginary part $<10^{-10}$ of the real part",
            r"Extended basis: Wigner $d$ stable to $j=24$ (Jacobi form); blocks disjoint and orthonormal; "
            r"reduces exactly to the paper basis; leave-one-out equals brute-force refits",
        ]),
    ]
    for lvl in ("parton", "parton_trig", "gen", "reco"):
        if lvl in lv:
            s = lv[lvl]
            c2 = s["proj_chi2"]
            worst = max(c2, key=lambda a: c2[a][0] / max(c2[a][1], 1))
            slides.append(wide_figure(
                rf"$f^\mathrm{{{lvl}}}_{{t\bar t}}$ (fit on {lvl}-level $t\bar t$, {s['n']['train']} events): "
                rf"projections",
                os.path.join(O["_dir"], f"projections_ttbar_{lvl}.png")))
            slides.append(wide_figure(
                rf"$f^\mathrm{{{lvl}}}_{{t\bar t}}$ projections, hadronic top",
                os.path.join(O["_dir"], f"projections_ttbar_{lvl}_had.png"),
                note=rf"Black: $f_{{t\bar t}}$ integrated over the other nine angles. "
                     rf"Largest $\chi^2$/ndf vs training data: {ANGLE_TEX[worst]} "
                     rf"({c2[worst][0]:.0f}/{c2[worst][1]})."))
            slides += score_slides(rf"$f^\mathrm{{{lvl}}}_{{t\bar t}}$ applied to {lvl}-level $t\bar t$ and $hh$",
                                   O["_dir"], lvl)
    rows = []
    for lvl in ("parton", "parton_trig", "gen", "reco"):
        if lvl in lv:
            s = lv[lvl]
            fn = s["frac_negative"]
            rows.append([lvl, str(s["n"]["train"]), f"{s['n_significant_3sigma']}",
                         f"{s['integral']:.3f}",
                         rf"{100 * fn.get('test', fn['train']):.1f}\% / {100 * fn['hh']:.1f}\%",
                         f"{s['mean_surprisal'].get('test', s['mean_surprisal']['train']):.2f} / "
                         f"{s['mean_surprisal']['hh']:.2f}",
                         f"{s['auc_hh_vs_tt']:.3f}"])
    slides.append(table(r"$f_{t\bar t}$ summary",
                        ["level", r"$N_\mathrm{train}$", r"$|\hat c|>3\sigma$", r"$\int f$",
                         r"$f\le0$: $t\bar t$/$hh$", r"$\langle s\rangle$: $t\bar t$/$hh$", "AUC"], rows,
                        note=r"$t\bar t$ numbers are from the independent test sample. "
                             r"$f\le0$ comes from the truncated series (Gibbs overshoot), as noted in the paper. "
                             r"AUC = probability that an $hh$ event has lower $f_{t\bar t}$ than a $t\bar t$ event."))
    rows = []
    for lvl in ("parton", "parton_trig", "gen", "reco"):
        if lvl in lv and "self_inclusion" in lv[lvl]:
            si, fn, ms = lv[lvl]["self_inclusion"], lv[lvl]["frac_negative"], lv[lvl]["mean_surprisal"]
            rows.append([lvl, rf"{100 * si['self_term_over_f_median']:.0f}\%",
                         rf"{100 * si['frac_negative_naive']:.1f}\% / {100 * fn['train']:.1f}\% / "
                         rf"{100 * fn['test']:.1f}\%",
                         f"{si['mean_s_naive']:.2f} / {ms['train']:.2f} / {ms['test']:.2f}",
                         f"{si['ks_naive_vs_test']:.3f} / {si['ks_loo_vs_test']:.3f}"])
    if rows:
        slides.append(table(r"Fit sample vs independent sample: self-inclusion bias",
                            ["level", r"self-term / $f$", r"$f\le0$: fit / fit LOO / indep.",
                             r"$\langle s\rangle$: fit / fit LOO / indep.", "KS vs indep.: fit / LOO"], rows,
                            note=r"Each fit event adds $\frac1N\sum_j|B_j(\mathbf{x}_k)|^2$ to its own $f$. "
                                 r"Removing it (leave-one-out, LOO) makes the fit sample agree with the "
                                 r"independent $t\bar t$ sample: the difference was entirely this bias, "
                                 r"largest at reco (smallest $N$). Plots use LOO for the fit sample.", fit=True))
    rows = []
    for lvl in ("parton", "parton_trig", "gen", "reco"):
        if lvl in lv:
            rows.append([lvl] + [rf"{t['kept']} / {t['auc']:.3f} / ${t['ds_test_train']:+.3f}$"
                                 for t in lv[lvl]["truncation"]])
    ks = [t["k"] for t in lv[next(iter(lv))]["truncation"]]
    slides.append(table(r"Diagnostics: statistical noise of the 14400 coefficients",
                        ["level"] + [("all (paper)" if k == 0 else rf"$|\hat c|>{k}\sigma$") for k in ks], rows,
                        note=r"Entries: coefficients kept / AUC($hh$ vs $t\bar t$ test) / "
                             r"$\langle s\rangle_\mathrm{test}-\langle s\rangle_\mathrm{train}$ (overfitting). "
                             r"Dropping insignificant terms removes the overfitting, but the gen/reco AUC stays "
                             r"$\approx0.5$, so the loss of separation is not caused by noise.", fit=True))
    yrows = []
    for lvl in ("parton", "parton_trig", "gen", "reco"):
        if lvl in lv:
            yrows.append([lvl] + [f"{y['chi2']:.0f}/{y['ndf']}" for y in lv[lvl]["y_scale_scan"]])
    first = lv[next(iter(lv))]["y_scale_scan"]
    slides.append(table(r"Diagnostics: Hermite scale for $Y$ ($\chi^2$ of the $Y$ projection, train)",
                        ["level"] + [f"$s={y['scale']}$" for y in first[:-1]] + [r"$s=\mathrm{std}(Y)$"], yrows,
                        note=r"The paper uses $s=1$ (the default here; it is a free parameter of the fit). "
                             r"Gen/reco $Y$ is narrower (std $\approx0.8$), so $s=1$ fits poorly there and $f$ "
                             r"overshoots below zero at large $|Y|$."))
    return slides


def term_tex(term):
    """'lep[j1=0,j2=1,m'=0,m=0] x P2(cosΘ)' -> LaTeX."""
    import re
    def m_fn(mt):
        side, j1, j2, mp, m = mt.groups()
        sym = r"M" if side == "lep" else r"\tilde M"
        return rf"${sym}^{{{j1}{j2}}}_{{{mp},{m}}}$"
    t = re.sub(r"(lep|had)\[j1=(-?\d),j2=(-?\d),m'=(-?\d),m=(-?\d)\]", m_fn, term)
    t = re.sub(r"P(\d)\(cosΘ\)", r"$P_{\1}(\\cos\\Theta)$", t)
    t = re.sub(r"H(\d)\(Y\)", r"$h_{\1}(Y)$", t)
    return t.replace(" x ", r" $\times$ ")


def extended_slides(odir, edir):
    """Why the paper basis misses gen/reco, how many orders are needed, and the extended basis."""
    def load(p):
        return json.load(open(p)) if os.path.exists(p) else None
    tr = load(os.path.join(odir, "truncation_gen.json"))
    scan = load(os.path.join(odir, "order_scan.json"))
    comp = load(os.path.join(edir, "projection_chi2_comparison.json"))
    esum = load(os.path.join(edir, "summary.json"))
    esimp = load(os.path.join(edir, "simplify_summary.json"))
    if not (tr and scan and comp and esum and esimp):
        print("[slides] extended-basis outputs missing; skipping that section")
        return []
    slides = [section("Higher orders for gen and reco", "Higher orders for gen and reco"),
              bullets(r"Why $f^\mathrm{gen}$, $f^\mathrm{reco}$ miss the data with the paper basis", [
                  r"$j_1\le1$: at most \emph{linear} in $\cos\theta$; $j_2\le2$: at most \emph{quadratic} in "
                  r"$\cos\theta'$; $|m|\le1$: only $\cos\phi'$, $\sin\phi'$",
                  r"Complete for the physical (parton) decay distribution, but the acceptance cuts create sharp "
                  r"edges ($\cos\theta\to1$, $\cos\theta'\to-1$, $\phi'\approx0$) that need higher orders",
                  rf"Code check: the projection of $f$ equals a direct 1-D expansion at the same order "
                  rf"(max difference {100 * max(v['max_rel_diff_f_vs_same_order_1d'] for v in tr.values()):.1f}\%)",
                  r"At gen level, same-order $\chi^2$: "
                  + ", ".join(f"{v['chi2_same_order']:.0f}" for v in tr.values())
                  + r" (25 bins); higher order: " + ", ".join(f"{v['chi2_high_order']:.0f}" for v in tr.values()),
              ]),
              wide_figure(r"Paper basis at gen level: the order, not the fit, limits $f$",
                          os.path.join(odir, "truncation_gen.png"))]
    names = ["cos_theta", "phi", "cos_theta_prime", "phi_prime", "cos_theta_tilde", "phi_tilde",
             "cos_theta_prime_tilde", "phi_prime_tilde", "cos_Theta", "Y"]
    rows = [[ANGLE_TEX[a]] + [str(scan[l][a]) for l in ("parton", "parton_trig", "gen", "reco")] for a in names]
    req = [scan[l]["_required"] for l in ("parton", "parton_trig", "gen", "reco")]
    rows.append([r"\textbf{needs}"] + [rf"$j_1\ge{r['j1max']}, j_2\ge{max(r['j2max'], r['min_j1_j2_for_phi_prime'])}$"
                                        for r in req])
    slides.append(table(r"How many orders each angle needs (1-D cross-validated order)",
                        ["angle"] + [LVL_TEX[l] for l in ("parton", "parton_trig", "gen", "reco")], rows,
                        note=r"Order = Legendre degree ($\cos$), harmonic ($\phi$), Hermite order ($Y$, scale = std$(Y)$). "
                             r"Parton level confirms the paper: $j_1\le1$, $j_2\le2$ (only $\cos\Theta$, $Y$ need more). "
                             r"A full product at gen/reco orders would need $>10^{6}$ terms per top pair.", fit=True))
    bp = {l: esum["levels"][l]["basis_params"] for l in ("parton", "parton_trig", "gen", "reco")}
    slides.append(bullets(r"Extended (block) basis", [
        r"Union of disjoint subsets of the full orthonormal product basis $\Rightarrow$ still orthonormal, "
        r"OSDE unchanged",
        r"A: the paper's full product (all correlations at physics order); "
        r"L, H: high-order single-top blocks (all $(\theta,\phi)$ terms, all $m'=0$ terms) "
        r"with the other top constant; P, Y: high-order $\cos\Theta$ and $Y$",
        r"Orders from the scan, capped at 16: " + "; ".join(
            rf"{l}: $j_1\le{bp[l]['J1']}$, $j_2\le{bp[l]['J2']}$, {bp[l]['n_legendre_hi']} Legendre, "
            rf"{bp[l]['n_hermite_hi']} Hermite ($s={bp[l]['y_scale']}$)" for l in bp),
        r"Terms: " + ", ".join(f"{l} {esum['levels'][l]['n_coefficients']}" for l in bp)
        + r"; then the same cross-validated truncation",
        r"Trade-off: correlations \emph{between} the two tops stay at the physics order",
    ]))
    rows = []
    for l in ("parton", "parton_trig", "gen", "reco"):
        for key, lab in (("paper", "paper"), ("paper_simplified", "paper, CV"), ("extended", "extended"),
                         ("extended_simplified", "extended, CV")):
            c = comp.get(f"{key}/{l}")
            if c:
                rows.append([l if key == "paper" else "", lab] + [f"{c[a][0] / c[a][1]:.1f}" for a in names[::-1]])
    slides.append(table(r"Projection $\chi^2$/ndf vs the $t\bar t$ fit sample",
                        ["level", "basis"] + [ANGLE_TEX[a] for a in names[::-1]], rows,
                        note=r"Extended basis (after CV truncation): $\chi^2$/ndf $\approx$ 0.4--2 for every "
                             r"angle at every level; the paper basis is good only at parton level (apart from "
                             r"$Y$, $\cos\Theta$).", fit=True))
    for l in ("gen", "reco"):
        slides.append(wide_figure(rf"Extended $f^\mathrm{{{l}}}_{{t\bar t}}$ (CV-truncated): projections",
                                  os.path.join(edir, f"projections_ttbar_simplified_{l}.png")))
        slides.append(wide_figure(rf"Extended $f^\mathrm{{{l}}}_{{t\bar t}}$ (CV-truncated): hadronic top",
                                  os.path.join(edir, f"projections_ttbar_simplified_{l}_had.png")))
    for l in ("reco",):
        slides += score_slides(rf"Extended $f^\mathrm{{{l}}}_{{t\bar t}}$ applied to {l}-level $t\bar t$ and $hh$",
                               edir, l)
    rows = []
    for l in ("parton", "parton_trig", "gen", "reco"):
        v = esimp["levels"][l]["variants"]
        for key, lab in (("all", "all terms"), ("cv_best", "CV best"), ("5sigma", r"$5\sigma$")):
            r = v[key]
            fn = r["frac_negative"]
            rows.append([l if key == "all" else "", lab, str(r["kept"]), f"{r['auc']:.3f}",
                         rf"{100 * fn.get('test', fn['train']):.1f}\%", rf"{100 * fn['hh']:.1f}\%"])
    slides.append(table(r"Extended basis: performance", ["level", "terms", "number", r"AUC ($hh$ vs $t\bar t$)",
                                                          r"$f\le0$ ($t\bar t$ indep.)", r"$f\le0$ ($hh$)"], rows,
                        note=r"A much better description of $t\bar t$ hardly changes the $hh$ separation at gen/reco: "
                             r"the limitation is the $t\bar t$-hypothesis reconstruction, not the density. "
                             r"Without truncation the many high-order terms add noise ($f\le0$ up to 13\%).",
                        fit=True))
    return slides


LVL_TEX = {"parton": "parton", "parton_trig": "parton+trig", "gen": "gen", "reco": "reco"}


def sh_slides(sdir):
    """Spherical-harmonic product basis vs the paper's M functions (basis_compare.py)."""
    p = os.path.join(sdir, "basis_comparison.json")
    if not os.path.exists(p):
        return []
    R = json.load(open(p))
    lv = [l for l in ("parton", "parton_trig", "gen", "reco") if f"paper (M functions)|{l}|CV" in R]
    slides = [section("Spherical-harmonic basis", "Spherical-harmonic basis"),
              bullets(r"Spherical-harmonic products instead of the $M$ functions", [
                  r"Per top: $Y_{l_1}^{m_1}(\theta,\phi)\,Y_{l_2}^{m_2}(\theta',\phi')$, $l_1\le1$, $l_2\le2$: "
                  r"36 functions (vs 24 $M$); with the same $p_l(\cos\Theta)$, $h_n(Y)$: 32\,400 coefficients",
                  r"Orthonormal and closed under conjugation (tests); same OSDE, leave-one-out and CV truncation",
                  r"Physics difference: $M$ uses spin-weighted $D^{j}_{m'm}(\phi,\theta,0)$; e.g.\ "
                  r"$d^1_{01}\propto\sin\theta$ is not a finite sum of $Y_{l0}$, so in principle the SM decay density, "
                  r"finite in $M$, needs infinitely many $Y_{lm}$ products (tested below at 100k events)",
                  r"Figure of merit: held-out ISE $=\sum|\hat c|^2-2\langle\hat f\rangle_\mathrm{indep}$ "
                  r"($=\int(\hat f-f)^2-\int f^2$; lower = closer to the true density), on the independent $t\bar t$ sample",
              ]),
              wide_figure(r"Paper ($M$) vs spherical harmonics vs extended", os.path.join(sdir, "basis_comparison.png"))]
    rows = []
    for l in lv:
        for lab, key in (("paper ($M$)", "paper (M functions)"), ("sph. harmonics", "spherical harmonics"),
                         ("extended", "extended")):
            for tr in ("all", "CV"):
                r = R.get(f"{key}|{l}|{tr}")
                if not r:
                    continue
                d = (rf"${1e6 * r['delta_ise_vs_paper']:+.2f}\pm{1e6 * r['delta_ise_vs_paper_err']:.2f}$"
                     if "delta_ise_vs_paper" in r and not key.startswith("paper") else "--")
                rows.append([LVL_TEX[l] if (key.startswith("paper") and tr == "all") else "",
                             f"{lab}, {'all' if tr == 'all' else 'CV'}", str(r["terms"]), d,
                             f"{r['worst_chi2ndf']:.1f}", rf"{100 * r['frac_negative_indep']:.1f}\%",
                             f"{r['auc']:.3f}" if r["auc"] else "--"])
    slides.append(table(r"Basis comparison (independent $t\bar t$ sample)",
                        ["level", "basis", "terms", r"$\Delta$ISE vs $M$ [$10^{-6}$]", r"worst $\chi^2$/ndf",
                         r"$f\le0$", "AUC"], rows,
                        note=r"$\Delta$ISE $<0$: closer to the true density than the paper's $M$ basis (paired, same "
                             r"events). Worst $\chi^2$/ndf of the 1-D projections excludes $Y$.", fit=True))

    def dlt(key, l):
        r = R.get(f"{key}|{l}|CV", {})
        return r.get("delta_ise_vs_paper", float("nan")) * 1e6, r.get("delta_ise_vs_paper_err", float("nan")) * 1e6
    sh_all = [R[f"spherical harmonics|{l}|all"]["delta_ise_vs_paper"] / R[f"spherical harmonics|{l}|all"]["delta_ise_vs_paper_err"]
              for l in lv if "delta_ise_vs_paper" in R.get(f"spherical harmonics|{l}|all", {})]
    items = [
        rf"All terms kept: spherical harmonics are worse than $M$ at every level ({min(sh_all):.0f}--{max(sh_all):.0f} "
        r"$\sigma$ in $\Delta$ISE): 32\,400 terms, most of them noise",
        r"After CV truncation: " + "; ".join(
            rf"{LVL_TEX[l]} $\Delta$ISE(SH) $={dlt('spherical harmonics', l)[0]:+.2f}\pm{dlt('spherical harmonics', l)[1]:.2f}$"
            for l in lv) + r" ($10^{-6}$): spherical harmonics and $M$ describe $t\bar t$ about equally well",
        rf"Spherical harmonics fit the trigger-sculpted $\phi'$ better (parton+trig worst $\chi^2$/ndf "
        rf"{R['spherical harmonics|parton_trig|CV']['worst_chi2ndf']:.1f} vs {R['paper (M functions)|parton_trig|CV']['worst_chi2ndf']:.1f}): "
        r"$l_2=2$ allows a second harmonic in $\phi'$, which $M$ ties to $j_1\le1$" if "spherical harmonics|parton_trig|CV" in R else
        r"",
        r"The extended basis is best at gen/reco: " + "; ".join(
            rf"{LVL_TEX[l]} $\Delta$ISE $={dlt('extended', l)[0]:+.2f}\pm{dlt('extended', l)[1]:.2f}$" for l in lv),
        r"The $hh$ AUC hardly depends on the basis",
    ]
    slides.append(bullets("Basis comparison: conclusions", [i for i in items if i]))
    return slides


def ml_slides(mdir, explainer=()):
    """Is the OSDE approach helping?  Unsupervised and supervised ML on the same inputs (ml_compare.py)."""
    out = []
    res = {}
    for l in ("parton", "parton_trig"):
        p = os.path.join(mdir, f"results_{l}.json")
        if os.path.exists(p):
            res[l] = json.load(open(p))
    if not res:
        return []
    out.append(section("Is the OSDE approach helping? (ML)", "Is the OSDE approach helping? (ML)"))
    r0 = res[next(iter(res))]
    out.append(bullets("Setup: same events, same inputs", [
        rf"Test set for every method: the independent $t\bar t$ sample ({r0['n_test_tt']}) and a held-out 40\% of "
        rf"$hh$ ({r0['n_test_hh']}); training: $t\bar t$ fit sample + 50\% of $hh$ (10\% each for validation)",
        r"\textbf{Unsupervised} (trained on $t\bar t$ only, like $f_{t\bar t}$): Gaussian-mixture density, "
        r"autoencoder, normalizing flow (RealNVP) on the same ten angles",
        r"\textbf{Supervised} ($t\bar t$ vs $hh$ labels; an upper bound): BDT, MLP, FT-Transformer on (a) our angles, "
        r"(b) angles + parton momenta; particle transformer (pairwise $m$, $\Delta R$) on momenta",
        r"\textbf{Does $y$ help ML?} the same classifiers with $y=f_{t\bar t}$ and $s'$ added (leave-one-out $y$ for "
        r"$t\bar t$ training events)",
        r"Controls: shuffled labels and $t\bar t$-vs-$t\bar t$ must give AUC $\approx0.5$; train vs test AUC",
    ]))
    out += list(explainer)
    for l, r in res.items():
        for key, what in (("a", "unsupervised vs supervised, angles"), ("b", "does $y$ help?"),
                          ("c", "angles + momenta")):
            p = os.path.join(mdir, f"roc_{l}_{key}.png")
            if os.path.exists(p):
                out.append(wide_figure(rf"ROC, {LVL_TEX[l]} level: {what}", p))
    for l, r in res.items():
        rr = r["results"]
        rows = [[v["group"].replace("+", " + "), k.split(" [")[0].replace("_", " ").replace("f tt", r"$f_{t\bar t}$"),
                 rf"${v['auc_test']:.3f}\pm{v['auc_err']:.3f}$",
                 f"{v['auc_train']:.3f}" if "auc_train" in v and v["auc_train"] == v["auc_train"] else "--"]
                for k, v in rr.items()]
        out.append(table(rf"AUC ($hh$ vs $t\bar t$), {LVL_TEX[l]} level", ["inputs", "method", "AUC (test)", "AUC (train)"],
                         rows, fit=True))
        unsup = {k: v for k, v in rr.items() if v["group"] == "unsupervised"}
        osde_best = max((v["auc_test"], k) for k, v in unsup.items() if k.startswith("OSDE"))
        other_best = max((v["auc_test"], k) for k, v in unsup.items() if not k.startswith("OSDE"))
        sup = {k: v for k, v in rr.items() if v["group"] == "angles"}
        supy = {k: v for k, v in rr.items() if v["group"] == "angles+y"}
        best_sup = max(v["auc_test"] for v in sup.values())
        best_supy = max(v["auc_test"] for v in supy.values())
        gain = [supy[k.replace("[angles]", "[angles+y]")]["auc_test"] - v["auc_test"] for k, v in sup.items()
                if k.replace("[angles]", "[angles+y]") in supy]
        mom = max(v["auc_test"] for v in rr.values() if v["group"].startswith("angles+momenta"))
        out.append(bullets(rf"Answer, {LVL_TEX[l]} level", [
            rf"As an anomaly detector (no $hh$ used): OSDE $f_{{t\bar t}}$ AUC {osde_best[0]:.3f} vs "
            rf"{other_best[0]:.3f} for the best other unsupervised method ({other_best[1].split(' (')[0]})"
            + "".join(rf"; paired difference {v['delta']:+.3f} $\pm$ {v['err']:.3f}" for k, v in r.get("paired", {}).items()
                      if k.startswith(osde_best[1]) and other_best[1] in k),
            rf"Knowing $hh$ (supervised) on the same angles: AUC {best_sup:.3f}, the upper bound for these inputs",
            rf"Adding $y$ to the supervised classifiers: AUC change {min(gain):+.3f} to {max(gain):+.3f} "
            rf"(best {best_supy:.3f}), within the $\pm${r['results'][next(iter(sup))]['auc_err']:.3f} uncertainty: "
            r"$y$ carries no information the classifier does not already learn from the angles",
            rf"Momenta (which contain $m_{{b\bar b}}=m_h$ etc.) make the task trivial: AUC {mom:.4f}",
            rf"Controls: shuffled labels {rr['control: BDT, shuffled labels']['auc_test']:.3f}, $t\bar t$ vs $t\bar t$ "
            rf"{rr['control: BDT, tt (fit) vs tt (independent)']['auc_test']:.3f} (expected 0.5)",
        ]))
    return out


def washout_slides(wdir):
    """Why the parton-level hh/tt̄ separation washes out at gen level (study_washout/)."""
    m, d = os.path.join(wdir, "mechanism", "figures"), os.path.join(wdir, "decomposition")
    if not os.path.exists(os.path.join(d, "decomposition_chain.png")):
        return []
    return [
        section(r"Why the separation washes out at gen level", "Washout parton to gen"),
        bullets(r"Short answer", [
            r"At parton level the $hh$ ``tops'' are not tops: each is the $W$ from one Higgs plus the $b$ from the other "
            r"($m_{Wb}$ median 135 GeV), and one $W$ is off-shell ($m_{W^*}$ median 32 GeV)",
            r"$\Rightarrow$ $hh$ has $\cos\theta$, $\cos\tilde\theta$ peaked at $-1$, isotropic $\cos\Theta$ "
            r"(no $t$-channel), and no $V\!-\!A$ zero in $\cos\theta'$",
            r"At gen level the $t\bar t$-hypothesis reconstruction rebuilds $m_W$ and $m_t$ from whatever jets fit: "
            r"the $\chi^2$ picks the true jets in 8\% of $hh$ (34\% of $t\bar t$); the $m_W$-constrained $\nu$ puts "
            r"the leptonic $W^*$ on shell",
            r"Ranked losses in $f_{t\bar t}$ AUC: wrong jets in the $\chi^2$ ($-0.06$) $>$ FSR ($-0.03$) "
            r"$\approx$ $\nu$ from $m_W$ ($-0.03$); selection and ISR recoil cost nothing",
            r"The information is \textbf{not} gone: a BDT on hypothesis-free gen observables ($m_{bb}$, "
            r"$\Delta R_{bb}$, MET, $m_T$, \ldots) gives AUC 0.87 on the same events",
        ]),
        wide_figure(r"Parton level: what separates $hh$ from $t\bar t$", os.path.join(m, "fig1_parton_signatures.png")),
        wide_figure(r"Step by step from parton to gen: all selected events", os.path.join(d, "slide_chain_main.png"),
                    note=r"Green: parton-level $f_{t\bar t}$ applied unchanged; orange: $f_{t\bar t}$ refit at each step; blue: "
                         r"supervised BDT on the ten angles. Biggest $f_{t\bar t}$ loss: the $\chi^2$ jet choice."),
        wide_figure(r"Step by step: events with correctly matched jets", os.path.join(d, "slide_chain_matched.png"),
                    note=r"When all four partons match distinct jets, jets cost nothing; the $m_W$-constrained $\nu$ "
                         r"(choice of $p_z$ solution) costs 0.05--0.08."),
        wide_figure(r"Which angles lose the separation, and where", os.path.join(d, "slide_angles.png"),
                    note=r"$\cos\Theta$ and $\cos\tilde\theta$ die at the $\chi^2$ jet choice; $\cos\theta'$ dies with "
                         r"the $m_W$-constrained neutrino; ISR recoil mainly costs $\phi$, $\tilde\phi$, $Y$."),
        wide_figure(r"The $\chi^2$ builds $W$ and top masses that $hh$ does not have",
                    os.path.join(m, "fig2_chi2_mass_pull.png")),
        wide_figure(r"$hh$ angles become $t\bar t$-like", os.path.join(m, "fig3_angle_morphing.png"),
                    note=r"``true jets + true $\nu$'' is a small, biased subset (532 $hh$ events)."),
        wide_figure(r"Gen level still separates, with hypothesis-free observables", os.path.join(m, "fig5_hypothesis_free.png")),
        wide_figure(r"Summary of the chain (supervised BDT)", os.path.join(m, "fig4_auc_chain.png")),
        bullets(r"Implications", [
            r"The angles are the right variables for $t\bar t$, but at gen/reco they are computed under the "
            r"$t\bar t$ hypothesis, which removes exactly what makes $hh$ different",
            r"$f_{t\bar t}$ is unsupervised: it flags only where $t\bar t$ is rare; the fitted reconstruction moves $hh$ "
            r"into the $t\bar t$-dense region",
            r"Options: (i) add the $\chi^2$ value and the masses ($m_{bb}$, $m_{jj}$, $m_{jjb}$) as extra dimensions "
            r"of $f_{t\bar t}$; (ii) use $hh$-hypothesis angles; (iii) loosen the $m_W$ constraint for the $\nu$",
            r"Caveats: matched $hh$ subsets are small (500 events); FSR step uses last-copy partons (slight overestimate)",
        ]),
    ]


def salvage_slides(sdir):
    """Momentum in the density method: the resonance masses (study_salvage/)."""
    p, q = os.path.join(sdir, "simplest.json"), os.path.join(sdir, "salvage.json")
    if not os.path.exists(p):
        return []
    r, rf_ = json.load(open(p)), (json.load(open(q)) if os.path.exists(q) else {})
    P, G = r["parton"], r["gen"]
    return [
        section(r"Adding momentum to $y$: the resonance masses", "Momentum in y"),
        bullets(r"Motivation: what the ten angles leave out", [
            r"6 final-state particles = 18 momentum components = 10 paper variables ($Y$, $\Theta$, 8 decay angles) "
            r"+ 4 \textbf{resonance masses} ($m_{W,\ell}, m_{W,h}, m_{t,\ell}, m_{t,h}$) + $m_{t\bar t}$, "
            r"$\vec p_T^{\,t\bar t}$, overall azimuth",
            r"For $t\bar t$ the masses carry no information: each is a Breit--Wigner fixed by $M$ and $\Gamma$, and "
            r"(narrow width) $|\mathcal M|^2$ factorizes: angles $\times\prod_k \mathrm{BW}(m_k)$. So the paper could "
            r"drop them",
            r"$hh$ is \emph{not} on that shell: one $W$ is off shell (median 32 GeV) and its ``tops'' sit at 135 GeV. "
            r"Its difference lives in exactly the dimensions we dropped",
        ]),
        wide_figure(r"The dropped dimensions, parton level", os.path.join(sdir, "masses_parton.png"),
                    note=r"$t\bar t$ (blue) is the Breit--Wigner (dashed; MadGraph cuts it at $\pm15\Gamma$). $hh$ (red) "
                         r"is spread over 100 GeV."),
        bullets(r"The extension: one factor per mass, no new coefficients", [
            r"$y_{\rm ext}(\Omega, m) = y(\Omega)\times\prod_{k=1}^{4}\mathrm{BW}_k(m_k)$ \quad (the $t\bar t$ "
            r"density on the full phase space)",
            r"In surprisal (what $t$ is built from) the factors simply add: "
            r"$s_{\rm ext} = s(\Omega) + \sum_k \ln\!\big[(m_k^2-M_k^2)^2+M_k^2\Gamma_k^2\big] + \text{const}$",
            r"Parton level: $M$, $\Gamma$ from the PDG. Nothing is fitted",
            r"Gen/reco level: resolution replaces $\Gamma$; use the same shape with $M$, $W$ = median and half "
            r"interquartile range of $t\bar t$: two numbers per mass",
            r"A fitted OSDE correction to this shape (Legendre in $u=\frac12+\frac1\pi\arctan\frac{m-M}{W}$) "
            + (rf"changes the gen AUC by only {rf_['gen']['auc_masses'] - G['masses_only']:+.3f}" if rf_ else "is not needed"),
        ]),
        wide_figure(r"The same dimensions at gen level ($\chi^2$ reconstruction)", os.path.join(sdir, "masses_gen.png"),
                    note=r"Resolution widens the $t\bar t$ shell (dashed: two-number model); $hh$ is still wider, mainly in "
                         r"the tails. $m_{W,\ell}$ is missing: the $\nu$ solution sets it to $m_W$ by construction."),
        table(r"Result: AUC ($hh$ vs $t\bar t$), unsupervised", ["", "parton", r"gen ($\chi^2$)"], [
            [r"angles only, $y(\Omega)$ (paper)", f"{P['angles']:.3f}", f"{G['angles']:.3f}"],
            [r"$+\,m_{W,\ell}$", f"{P['angles + ... + m_w_lep']:.3f}", "(fixed by $\\nu$)"],
            [r"$+\,m_{W,h}$", f"{P['angles + ... + m_w_had']:.3f}", f"{G['angles + ... + m_w_had']:.3f}"],
            [r"$+\,m_{t,\ell}$", f"{P['angles + ... + m_t_lep']:.3f}", f"{G['angles + ... + m_t_lep']:.3f}"],
            [r"$+\,m_{t,h}$ (all masses)", f"{P['angles + ... + m_t_had']:.3f}", f"{G['angles + ... + m_t_had']:.3f}"],
        ], note=rf"\textbf{{Gen level with the masses ({G['angles + ... + m_t_had']:.2f}) beats the parton-level angles "
                rf"({P['angles']:.2f})}}: the parton-level performance is regained. Rows add one mass at a time. "
                r"Test: half of 40k $t\bar t$ (not used to set $M$, $W$; angles scored leave-one-out), 20k / 4.4k $hh$."),
        bullets(r"Momentum in $y$: summary and next steps", [
            r"Physics: $hh$ differs from $t\bar t$ mostly \emph{off} the $t\bar t$ mass shell; the angles alone cannot see it",
            r"Simplest fix: multiply $y$ by one Breit--Wigner per resonance (2 numbers each at gen). Nothing else changes; "
            r"$t$ is built from the summed surprisal as before",
            rf"Parton: {P['angles']:.2f} $\to$ {P['angles + ... + m_t_had']:.2f}; gen: {G['angles']:.2f} $\to$ "
            rf"{G['angles + ... + m_t_had']:.2f}",
            r"Next: solve the $\nu$ with the $m_{t,\ell}$ constraint (frees $m_{W,\ell}$ at gen); add $m_{t\bar t}$; "
            r"reco level",
        ]),
    ]


def ml_explainer_slides():
    """Plain-language description of the ML methods used in the comparison."""
    return [
        bullets(r"The ML methods in plain words: unsupervised (see only $t\bar t$)", [
            r"\textbf{Same job as $f_{t\bar t}$}: learn the $t\bar t$ density from $t\bar t$ events; low density "
            r"$\Rightarrow$ not $t\bar t$",
            r"\textbf{Gaussian mixture}: $f_{t\bar t}$ = a sum of 32 Gaussian blobs in the 10-D angle space; blob "
            r"positions and widths fitted to $t\bar t$",
            r"\textbf{Autoencoder}: a network squeezes the 10 angles through a few numbers and rebuilds them; trained on "
            r"$t\bar t$, so events it rebuilds badly are unusual. Not a density",
            r"\textbf{Normalizing flow}: learn a smooth, invertible map $z=T(x)$ that turns the $t\bar t$ distribution into "
            r"a plain Gaussian. Then $f_{t\bar t}(x)=\mathcal N(T(x))\,|\det\partial T/\partial x|$",
            r"The mass map used later ($u=\frac12+\frac1\pi\arctan\frac{m-M}{W}$, density $g_u\,du/dm$) is exactly a "
            r"1-D flow; the normalizing flow learns the same thing in 10-D. RealNVP stacks 8 simple steps, each shifting and rescaling half of the "
            r"variables by amounts computed from the other half, so the Jacobian stays easy",
        ]),
        bullets(r"The ML methods in plain words: supervised (see $t\bar t$ \emph{and} $hh$)", [
            r"\textbf{Different job}: learn $P(hh\,|\,x)$ from labelled examples. The best possible is the ratio "
            r"$f_{hh}/f_{t\bar t}$; $f_{t\bar t}$ alone assumes nothing about $hh$ (as if $f_{hh}$ were flat)",
            r"\textbf{BDT}: a few hundred small decision trees (``if $\cos\theta<-0.6$ and $|Y|>1$\ldots''), each "
            r"fixing the mistakes of the previous ones",
            r"\textbf{MLP}: a neural network; layers of weighted sums with a simple non-linearity in between",
            r"\textbf{FT-Transformer}: each input variable becomes a vector (``token''); attention lets every variable "
            r"look at every other one, so it learns which combinations matter",
            r"\textbf{Particle Transformer}: the tokens are the six particles' four-momenta; attention is told each pair's "
            r"invariant mass and $\Delta R$, so it sees $m_{b\bar b}=m_h$ directly",
            r"Why they win: they know what $hh$ looks like. Our goal is different: a $t\bar t$-only (unsupervised) "
            r"$y$ that flags anything non-$t\bar t$",
        ]),
    ]


def discriminator_slides(odir):
    """The paper's y and t figure (paper_style_plots.py) and the y0 sensitivity."""
    slides = [section(r"Discriminator $t$ (paper style)", "Discriminator t (paper style)")]
    slides.append(bullets(r"Definition of $t$ (as in the paper)", [
        r"$y'=y-y_0$, $s'=-\ln y'$, $t=(s'_1-s')/(s'_1-s'_0)\in[0,1]$; $t\to0$: less $t\bar t$-like",
        r"$y_0=2.5\times\min y$: the paper figure's axis starts at $\approx2.5\times\min y$ and its $t$ is populated "
        r"from 0, which needs $y_0$ well below $\min y$",
        r"$s'_0$, $s'_1$ = 0.5\%, 99.5\% quantiles of $s'$ ($t\bar t$ + $hh$); events beyond go into the first/last "
        r"bin, so $t$ fills [0, 1] with no empty bins",
        r"$t\bar t$: fit sample with leave-one-out $y$; both samples at unit area; ratio $hh/(hh+t\bar t)$ below",
    ]))
    for l in ("parton", "parton_trig", "gen", "reco"):
        p = os.path.join(odir, f"tt_hh_dNdy_dNdt_{l}.pdf")
        if os.path.exists(p):
            slides.append(wide_figure(rf"$y=f^\mathrm{{{l}}}_{{t\bar t}}$ and $t$, {l} level (paper basis)", p))
    p = os.path.join(odir, "t_vs_y0_parton.png")
    if os.path.exists(p):
        slides.append(wide_figure(r"Shape of $t$ depends on $y_0$", p, note=(
            r"$s'_1$ is set by the most negative event, so $y_0$ just below $\min y$ squeezes $t$ toward 1. "
            r"The paper figure (axis from $-1.5\times10^{-5}$, $t$ populated from 0) implies "
            r"$y_0\approx2.5\times\min y$, used here. The ROC/AUC do not depend on $y_0$ (monotonic).")))
    return slides


def coefficient_slides(C, odir):
    """How many of the 14400 coefficients matter (osde_simplify.py)."""
    lv = C["levels"]
    lvls = [l for l in ("parton", "parton_trig", "gen", "reco") if l in lv]
    v = {l: lv[l]["variants"] for l in lvls}
    slides = [
        section("How many coefficients matter?"),
        bullets(r"Criterion for dropping coefficients", [
            r"Keep a term iff $|\hat c|/\sigma(\hat c)>t$ (constant always kept). Conjugate partners have "
            r"equal $|\hat c|$ and $\sigma$, so they are kept together and $f$ stays real (checked)",
            r"Kronmal--Tarter ($t=\sqrt2$): dropping a term costs $|c|^2$, keeping it costs $\sigma^2$; "
            r"keep iff $|\hat c|^2>2\sigma^2$ (minimizes the integrated squared error per term)",
            r"With 14400 mostly-null terms, $\sqrt2\sigma$ still admits $\sim$16\% of the pure-noise terms",
            r"Choose $t$ by 5-fold cross-validation: for an orthonormal basis, "
            r"$\mathrm{ISE}-\!\int\! f^2=\sum_\mathrm{kept}|\hat c|^2-2\,\mathrm{Re}\sum_\mathrm{kept}"
            r"\hat c\,\hat c^{*}_\mathrm{held\text{-}out}$ (no integrals needed)",
            rf"CV minimum at $t={lv[lvls[0]]['t_cv']:g}$: "
            + ", ".join(f"{v[l]['cv_best']['kept']} ({l})" for l in lvls)
            + r" of 14400 terms; flat out to $5\sigma$: "
            + ", ".join(f"{v[l]['5sigma']['kept']}" for l in lvls),
        ]),
        *[figures(rf"Cross-validated error vs number of coefficients, {LVL_TEX[l]}",
                  [os.path.relpath(os.path.join(odir, f"simplify_{l}"), HERE)]) for l in lvls],
    ]
    rows = []
    names = [("all", "all (paper)"), ("kronmal_tarter", r"$\sqrt2\sigma$ (K--T)"),
             ("cv_best", "CV best"), ("5sigma", r"$5\sigma$")]
    for l in lvls:
        for key, lab in names:
            r = v[l][key]
            fn = r["frac_negative"]
            rows.append([l if key == "all" else "", lab, str(r["kept"]), f"{r['auc']:.3f}",
                         rf"{100 * fn.get('test', fn['train']):.1f}\%", rf"{100 * fn['hh']:.1f}\%",
                         f"{r['mean_s']['train']:.2f} / {r['mean_s'].get('test', float('nan')):.2f}"])
    slides.append(table(r"Simplified $f_{t\bar t}$: performance",
                        ["level", "terms kept", "number", "AUC", r"$f\le0$ ($t\bar t$ indep.)",
                         r"$f\le0$ ($hh$)", r"$\langle s\rangle$ fit LOO / indep."], rows,
                        note=r"Dropping insignificant terms cuts the negative-$f$ (Gibbs) fraction and keeps the "
                             r"$hh$ separation: the simplified $f$ is as good, with $\sim$1--3\% of the terms.",
                        fit=True))
    P = lv["parton"] if "parton" in lv else lv[lvls[0]]
    rows = [[str(i + 1), term_tex(t["term"]), f"{t['signif']:.0f}", t["physics"].replace("tt", r"$t\bar t$")]
            for i, t in enumerate(P["top_terms"][:8])]
    slides.append(table(r"Leading terms at parton level (after the constant)",
                        [r"\#", "term", r"$|\hat c|/\sigma$", "physics"], rows,
                        note=r"$M$: leptonic top, $\tilde M$: hadronic top; "
                             r"$j_1=0$: $W$ helicity, $j_1=1$: top spin. Production ($P_l$), $W$ helicity and $Y$ "
                             r"dominate.", fit=True))
    cats = {l: lv[l].get("categories_5sigma", {}) for l in lvls}
    keys = sorted({k for c in cats.values() for k in c}, key=lambda k: -max(c.get(k, 0) for c in cats.values()))
    rows = [[k.replace("tt ", r"$t\bar t$ ").replace("top pol. x W helicity", r"top pol.\,$\times$\,$W$ hel.")]
            + [str(cats[l].get(k, "--")) for l in lvls] for k in keys[:10]]
    rows.append([r"\textbf{total}"] + [rf"\textbf{{{sum(cats[l].values())}}}" for l in lvls])
    slides.append(table(r"What the significant ($>5\sigma$) terms describe",
                        ["physics content of the term"] + lvls, rows,
                        note=rf"Parton: production, $W$ helicity and $t\bar t$ spin correlations; imaginary parts "
                             rf"consistent with 0 (max {lv[lvls[0]].get('max_imag_pull_5sigma', 0):.1f}$\sigma$), as "
                             rf"expected without CP violation. Reco: spin correlations no longer resolved; "
                             rf"acceptance-induced ``polarization'' terms appear instead.", fit=True))
    return slides


def acceptance_slides(adir):
    """Acceptance plots from acceptance_plots.py (parton-level angles before/after cuts)."""
    slides = [section("Acceptance"), bullets("Acceptance: how the selection sculpts the angles", [
        r"Same (parton-level) angle values for every curve, so shape changes are acceptance only; "
        r"dotted = angles measured at reco level (adds resolution and jet assignment)",
        r"$\cos\theta'\to-1$, $\phi'\approx0$: the lepton is emitted against the top's flight "
        r"direction and fails $p_T>25$ GeV (already at gen level, so not isolation)",
        r"$|Y|\gtrsim2$: $|\eta|<2.5$ acceptance. $\cos\theta,\cos\tilde\theta\to+1$: "
        r"the b quark goes backward and is too soft for the 25 GeV jet cut",
        r"$\cos\Theta$ and the $\phi$ angles: nearly flat efficiency",
        r"$hh$: the $t\bar t$-hypothesis $\chi^2$ turns flat parton $\cos\Theta$ into "
        r"$t\bar t$-like peaks at $\pm1$ (dotted): the reconstruction, more than the acceptance, "
        r"makes $hh$ look like $t\bar t$",
    ])]
    for label in ("tt", "hh"):
        for kind, title in (("shapes", "no cuts vs after cuts"), ("cutflow", "efficiency vs angle, cut by cut")):
            name = r"$t\bar t$" if label == "tt" else r"$hh$"
            for suffix, part in (("", "production, leptonic top"), ("_had", "hadronic top")):
                path = os.path.join(adir, f"acceptance_{label}_{kind}{suffix}.png")
                if os.path.exists(path):
                    slides.append(wide_figure(f"Acceptance, {name}: {title} ({part})", path))
    return slides


def closing_slide(S, O=None):
    """Summary slides; numbers are read from the analysis outputs when they exist."""
    def load(*parts):
        p = os.path.join(HERE, *parts)
        return json.load(open(p)) if os.path.exists(p) else None

    nu, wh = load("plots", "nu_pz.json"), load("plots_hh", "w_had_mass_split.json")
    C = load("osde", "simplify_summary.json")
    items = [
        r"$t\bar t$ samples (parton, gen, reco; ATLAS card, no pileup) produced and validated; "
        r"$hh$ (HEFT) likewise",
        r"Ten angles as in the paper (diagram); parton level reproduces the SM $W$ helicity fractions "
        rf"($F_0={S['parton']['hel_lep']['F0']:.3f}\pm{S['parton']['hel_lep']['eF0']:.3f}$)",
        r"Reco angles track parton angles when the jets are assigned correctly; combinatorics dominate the smearing",
    ]
    if nu:
        items.append(rf"$\nu$ $p_z$: {nu['gen']['half_width_68']:.0f} / {nu['reco']['half_width_68']:.0f} GeV "
                     r"(gen / reco, 68\% half-width); the top-mass choice of the solution is not better than "
                     rf"smaller $|p_z|$ ({100 * nu['choice']['frac_chi2_picks_closer']:.0f}\% vs "
                     rf"{100 * nu['choice']['frac_smaller_picks_closer']:.0f}\% closer to truth)")
    slides = [bullets("Summary (1/2): samples and angles", items)]
    if O:
        lv = O["levels"]
        auc = ", ".join(f"{l} {lv[l]['auc_hh_vs_tt']:.2f}" for l in ("parton", "parton_trig", "gen", "reco") if l in lv)
        items = [rf"$f_{{t\bar t}}$ from 14400 OSDE coefficients; AUC($hh$ vs independent $t\bar t$): {auc}"]
        if wh:
            items.append(r"Separation is lost at gen/reco because the $t\bar t$-hypothesis $\chi^2$ makes $hh$ "
                         r"look like $t\bar t$: e.g.\ a hadronic $W^*$ (true median "
                         rf"{wh['offshell_parton_median']:.0f} GeV) is reconstructed near $m_W$; acceptance "
                         r"sculpts $\cos\theta'$, $\phi'$, $Y$")
        items.append(r"Fit-sample vs independent-sample differences are entirely the self-inclusion bias "
                     r"(removed by leave-one-out)")
        if C:
            cl = C["levels"]
            items.append(r"Only $\sim$50--350 coefficients matter: CV keeps "
                         + ", ".join(f"{cl[l]['variants']['cv_best']['kept']}" for l in ("parton", "parton_trig", "gen", "reco")
                                     if l in cl)
                         + r" (parton, gen, reco); simplified $f$ has fewer negative values and the same AUC; "
                           r"parton level resolves $t\bar t$ spin correlations")
        E = load("osde_ext", "simplify_summary.json")
        if E:
            el = E["levels"]
            items.append(r"Paper basis has the right order at parton level, but gen/reco acceptance needs "
                         r"$j_{1,2}$ up to 13--16: the extended block basis ("
                         + ", ".join(f"{el[l]['variants']['cv_best']['kept']}" for l in ("parton", "parton_trig", "gen", "reco"))
                         + r" terms after CV) describes every angle ($\chi^2$/ndf $\approx$ 0.3--1.6; gen $Y$ "
                           r"still imperfect); the $hh$ AUC is unchanged")
        items.append(r"Paper-style $t$: its shape depends on the choice of $y_0$ (AUC does not)")
        items.append(r"Next: equiprobable-contour binning; truncated/extended $f$ as default?; $y_0$ convention; "
                     r"8-angle set; an $f_{hh}$ for A--B testing; pileup")
        chunks = [items[i:i + 4] for i in range(0, len(items), 4)]
        for k, chunk in enumerate(chunks):
            slides.append(bullets(f"Summary ({k + 2}/{len(chunks) + 1}): $f_{{t\\bar t}}$ and $hh$", chunk))
        slides[0] = slides[0].replace("Summary (1/2)", f"Summary (1/{len(chunks) + 1})")
    return "\n".join(slides)


# ============================================================================
# Helpers that turn content into Beamer frames
# ============================================================================
PLOTS_DIR = None   # set in main(); figures() looks plots up here


def fmt(x, sig=3):
    return "--" if x is None else f"{x:.{sig}g}"


def lepton_tex(leps):
    return ", ".join({"e": "e", "mu": r"\mu", "ta": r"\tau"}.get(x, x) for x in leps.split(","))


def pm(v):
    return rf"${v[0]:.0f}\pm{v[1]:.0f}$"


def pct(s, num, den):
    return rf"{100 * s[num] / s[den]:.0f}\%"


def code(text):
    return ("\\begin{semiverbatim}\\scriptsize\n" + text.replace("\\", "\\\\")
            .replace("{", "\\{").replace("}", "\\}") + "\n\\end{semiverbatim}")


def bullets(title, items):
    body = ["\\begin{itemize}"]
    for it in items:
        body.append(it if it.startswith("\\begin{semiverbatim}") else f"  \\item {it}")
    body.append("\\end{itemize}")
    return frame(title, "\n".join(body), fragile=any("semiverbatim" in i for i in items))


def figures(title, names, note=None):
    """Plots from the plots directory, at most two per slide (so they stay legible); more names
    become several slides titled (1/n), (2/n), ..."""
    chunks = [names[i:i + 2] for i in range(0, len(names), 2)]
    out = []
    for k, chunk in enumerate(chunks):
        t = title if len(chunks) == 1 else f"{title} ({k + 1}/{len(chunks)})"
        w = 0.6 if len(chunk) == 1 else 0.48
        h = 0.74 - (0.06 if note else 0)
        cells = []
        for name in chunk:
            path = os.path.join(HERE, name + ".png") if "/" in name else os.path.join(PLOTS_DIR, name + ".png")
            if os.path.exists(path):
                cells.append(rf"\includegraphics[width={w}\textwidth,height={h}\textheight,keepaspectratio]"
                             rf"{{{os.path.relpath(path, OUT_DIR)}}}")
            else:
                print(f"[slides] warning: missing plot {path}")
                cells.append(rf"\fbox{{\parbox{{{w}\textwidth}}{{\centering\tiny missing: "
                             rf"\texttt{{{name.replace('_', chr(92) + '_')}}}}}}}")
        body = "\\centering\n" + r" \hspace{0.02\textwidth} ".join(cells)
        if note and k == len(chunks) - 1:
            body += f"\n\n\\vspace{{2pt}}{{\\footnotesize {note}}}"
        out.append(frame(t, body))
    return "\n".join(out)


def wide_figure(title, path, note=None):
    """One full-width image (e.g. the 2x5 projection panels)."""
    if not os.path.exists(path):
        print(f"[slides] warning: missing {path}")
        return frame(title, rf"\centering missing: \texttt{{{os.path.basename(path).replace('_', chr(92) + '_')}}}")
    # leave room for the note: the longer the note, the shorter the figure
    h = 0.76 if not note else (0.64 if len(note) < 220 else 0.56)
    body = (rf"\centering\includegraphics[width=\textwidth,height={h}\textheight,keepaspectratio]"
            rf"{{{os.path.relpath(path, OUT_DIR)}}}")
    if note:
        body += f"\n\n\\vspace{{2pt}}{{\\footnotesize {note}}}"
    return frame(title, body)


def table(title, header, rows, note=None, fit=False, max_rows=12):
    """fit=True shrinks a table that is wider than the text (never enlarges it).  Tables longer
    than max_rows continue on further slides, split where a row's first cell starts a new group."""
    if len(rows) > max_rows:
        groups, cur = [], []
        for r in rows:
            if cur and r[0] not in ("", None):
                groups.append(cur)
                cur = []
            cur.append(r)
        groups.append(cur)
        pages, page = [], []
        for g in groups:
            if page and len(page) + len(g) > max_rows:
                pages.append(page)
                page = []
            page += g
        pages.append(page)
        if len(pages) > 1:
            return "\n".join(table(f"{title} ({k + 1}/{len(pages)})", header, p, note if k == len(pages) - 1 else None,
                                    fit, max_rows=10 ** 6) for k, p in enumerate(pages))
    cols = "l" + "c" * (len(header) - 1)
    lines = [r"\centering\small", (r"\begin{adjustbox}{max width=\textwidth}" if fit else "")
             + rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if len(r) == 2 and r[1].startswith(r"\multicolumn"):
            lines.append(r[0] + " & " + r[1] + r" \\")
        else:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}" + (r"\end{adjustbox}" if fit else "")]
    if note:
        lines.append(rf"\par\vspace{{6pt}}{{\footnotesize {note}}}")
    return frame(title, "\n".join(lines))


def frame(title, body, fragile=False):
    # display name of the parton + lepton-trigger level (the code name contains an underscore);
    # file names ("..._parton_trig.png") are preceded by "_" and are left alone
    title = re.sub(r"(?<![_/\w])parton_trig(?![\w.])", "parton+trig", title)
    body = re.sub(r"(?<![_/\w])parton_trig(?![\w.])", "parton+trig", body)
    opt = "[fragile]" if fragile else ""
    return f"\\begin{{frame}}{opt}{{{title}}}\n{body}\n\\end{{frame}}\n"


def cutflow_rows(S):
    g, r = S["gen"]["cutflow"], S["reco"]["cutflow"]
    n_g, n_r = g[0][1], r[0][1]
    return [[gn, str(gv), rf"{100 * gv / n_g:.1f}\%", str(rv), rf"{100 * rv / n_r:.1f}\%"]
            for (gn, gv), (_, rv) in zip(g, r)]


def helicity_rows(S):
    rows = []
    for lvl in S["levels"]:
        for key, lab in (("hel_lep", r"$\ell$"), ("hel_had", r"down-type $q$")):
            h = S[lvl][key]
            rows.append([f"{lvl}, {lab}"] + [rf"${h[k]:.3f}\pm{h['e' + k]:.3f}$" for k in ("F0", "FL", "FR")])
    return rows


ANGLE_TEX = {"Y": "$Y$", "cos_Theta": r"$\cos\Theta$", "cos_theta": r"$\cos\theta$", "phi": r"$\phi$",
             "cos_theta_prime": r"$\cos\theta'$", "phi_prime": r"$\phi'$",
             "cos_theta_tilde": r"$\cos\tilde\theta$", "phi_tilde": r"$\tilde\phi$",
             "cos_theta_prime_tilde": r"$\cos\tilde\theta'$", "phi_prime_tilde": r"$\tilde\phi'$"}


def angle_rows(S):
    rows = []
    for a, tex in ANGLE_TEX.items():
        row = [tex]
        for lvl in ("gen", "reco"):
            for k in ("angles_all", "angles_correct"):
                c, d = S[lvl][k][a]
                row.append(f"{c:.2f} / {d:.2f}")
        rows.append(row)
    return rows


PREAMBLE = r"""\documentclass[aspectratio=169,11pt]{beamer}
% ---- University of Pittsburgh colors (Pitt Royal #003594, Pitt Gold #FFB81C) ----
\usetheme{default}
\usefonttheme{professionalfonts}
\definecolor{pittroyal}{HTML}{003594}
\definecolor{pittgold}{HTML}{FFB81C}
\definecolor{pittgray}{HTML}{6B7280}
\setbeamercolor{structure}{fg=pittroyal}
\setbeamercolor{normal text}{fg=black!90}
\setbeamercolor{frametitle}{bg=pittroyal, fg=white}
\setbeamercolor{pittgoldbar}{bg=pittgold}
\setbeamercolor{title}{fg=white, bg=pittroyal}
\setbeamercolor{item}{fg=pittroyal}
\setbeamercolor{section in toc}{fg=pittroyal}
\setbeamercolor{alerted text}{fg=pittroyal}
\setbeamerfont{frametitle}{size=\large, series=\bfseries}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize items}[circle]
\setbeamertemplate{section in toc}[sections numbered]
\setbeamertemplate{frametitle}{%
  \nointerlineskip
  \begin{beamercolorbox}[wd=\paperwidth, ht=2.6ex, dp=1.2ex, leftskip=0.8em]{frametitle}
    \usebeamerfont{frametitle}\insertframetitle
  \end{beamercolorbox}%
  \nointerlineskip
  \begin{beamercolorbox}[wd=\paperwidth, ht=2.5pt, dp=0pt]{pittgoldbar}\end{beamercolorbox}%
  \vspace{2pt}}
\setbeamertemplate{footline}{%
  \leavevmode\hbox{%
  \begin{beamercolorbox}[wd=.35\paperwidth, ht=2.4ex, dp=1ex, leftskip=0.8em]{frametitle}%
    \tiny\insertshortauthor\ \textbar\ University of Pittsburgh\end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.55\paperwidth, ht=2.4ex, dp=1ex, center]{frametitle}%
    \tiny\insertshorttitle\end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.10\paperwidth, ht=2.4ex, dp=1ex, right, rightskip=0.8em]{frametitle}%
    \tiny\color{pittgold}\insertframenumber/\inserttotalframenumber\end{beamercolorbox}}}
\setbeamertemplate{title page}{%
  \vspace*{0.6cm}
  \begin{beamercolorbox}[wd=\textwidth, sep=12pt, center, rounded=false]{title}
    {\Large\bfseries\inserttitle\par}
  \end{beamercolorbox}
  {\color{pittgold}\rule{\textwidth}{3pt}}\par\vspace{0.9cm}
  \centering{\large\color{pittroyal}\insertauthor\par}\vspace{0.25cm}
  {\small\insertinstitute\par}\vspace{0.25cm}{\small\color{pittgray}\insertdate\par}}
% outline at the start of each section, current section highlighted
\AtBeginSection[]{\ifnum\value{section}>1\begin{frame}{Outline}\tableofcontents[currentsection]\end{frame}\fi}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
"""


def run_tests():
    """Run the unit tests and return e.g. '7 passed' for the slides."""
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider", os.path.join(HERE, "tests")],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
    last = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "no output"
    print(f"[slides] tests: {last}")
    return last.split(" in ")[0].strip("= ")


SHORT_TITLE = r"$t\bar t$ and $hh\to b\bar b\,\ell\nu\,q\bar q'$: samples, angles, OSDE"


def section(name, plain=None):
    """Start a new section (it appears in the outline); `plain` is the PDF-bookmark text."""
    if plain:
        return f"\\section{{\\texorpdfstring{{{name}}}{{{plain}}}}}\n"
    return f"\\section{{{name}}}\n"


def write_tex(slides, title, author, path):
    with open(path, "w") as f:
        f.write(PREAMBLE)
        f.write(f"\\title[{SHORT_TITLE}]{{{title}}}\n\\author{{{author}}}\n\\date{{\\today}}\n")
        f.write("\\institute{PITT PACC and Department of Physics and Astronomy, University of Pittsburgh}\n")
        f.write("\\begin{document}\n\\begin{frame}[plain]\\titlepage\\end{frame}\n")
        f.write("\\begin{frame}{Outline}\\tableofcontents\\end{frame}\n\n")
        f.write("\n".join(slides))
        f.write("\\end{document}\n")


OUT_DIR = None


def main():
    global PLOTS_DIR, OUT_DIR
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True, help="sample directory from generate_samples.py")
    p.add_argument("--plots", default=os.path.join(HERE, "plots"))
    p.add_argument("--config", default=os.path.join(HERE, "plot_config.py"))
    p.add_argument("--out", default=os.path.join(HERE, "slides"))
    p.add_argument("--title", default=r"$t\bar{t}$ and $hh\to b\bar{b}\,\ell\nu\,q\bar{q}'$: samples, "
                                      r"the ten angles, and $f_{t\bar t}$ by OSDE")
    p.add_argument("--author", default="Tae Min Hong")
    p.add_argument("--make-plots", action="store_true", help="run make_plots.py first")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--hh", default=None, help="hh sample directory (adds the hh slides)")
    p.add_argument("--osde", default=None, help="osde_fit.py output directory (adds the OSDE slides)")
    p.add_argument("--acceptance", default=None, help="acceptance_plots.py output dir (adds slides)")
    p.add_argument("--no-tests", action="store_true", help="don't run the unit tests for the slides")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    PLOTS_DIR, OUT_DIR = os.path.abspath(args.plots), os.path.abspath(args.out)
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.make_plots:
        subprocess.run([sys.executable, os.path.join(HERE, "make_plots.py"), "--sample",
                        f"nominal={args.sample}", "--out", PLOTS_DIR, "--config", args.config], check=True)

    cfg = runpy.run_path(args.config)
    angle_opts = {"had_fermion": "down", "jet_assignment": "truth", **cfg.get("ANGLES", {})}
    events, xsec = make_plots.load_sample("sample", args.sample, ["parton", "gen", "reco"],
                                          args.max_events, angle_opts)
    if not {"parton", "gen", "reco"} <= set(events):
        sys.exit("make_slides.py needs a sample with both parton_events.lhe.gz and delphes_events.root")
    meta_path = os.path.join(args.sample, "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.setdefault("cross_section_pb", xsec)

    S = summary.summarize(events)
    tex = os.path.join(OUT_DIR, "slides.tex")
    slides = build_slides(S, meta)
    O = None
    if args.hh and args.osde:
        H = {"meta": json.load(open(os.path.join(args.hh, "metadata.json")))}
        hh_events, _ = make_plots.load_sample("hh", args.hh, ["reco"], args.max_events, angle_opts)
        H["hh_cutflow"] = summary.cutflow(hh_events["reco"])
        H["tt_cutflow"] = S["reco"]["cutflow"]
        O = json.load(open(os.path.join(args.osde, "summary.json")))
        O["_dir"] = os.path.abspath(args.osde)
        slides += build_hh_osde_slides(H, O, run_tests() if not args.no_tests else "not run")
        simp = os.path.join(args.osde, "simplify_summary.json")
        if os.path.exists(simp):
            slides += coefficient_slides(json.load(open(simp)), os.path.abspath(args.osde))
        slides += extended_slides(os.path.abspath(args.osde), os.path.join(HERE, "osde_ext"))
        slides += sh_slides(os.path.join(HERE, "osde_sh"))
        slides += ml_slides(os.path.join(HERE, "ml"), ml_explainer_slides())
        slides += washout_slides(os.path.join(HERE, "study_washout"))
        slides += salvage_slides(os.path.join(HERE, "study_salvage"))
        slides += discriminator_slides(os.path.abspath(args.osde))
    elif args.hh or args.osde:
        sys.exit("--hh and --osde go together")
    if args.acceptance:
        slides += acceptance_slides(os.path.abspath(args.acceptance))
    slides.append(section("Summary"))
    slides.append(closing_slide(S, O))
    write_tex(slides, args.title, args.author, tex)
    print(f"[slides] wrote {tex}")

    if args.no_compile:
        return
    if not shutil.which("latexmk"):
        sys.exit("[slides] latexmk not found; compile slides.tex yourself (pdflatex)")
    res = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "slides.tex"],
                         cwd=OUT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print("\n".join(res.stdout.splitlines()[-30:]))
        sys.exit(f"[slides] LaTeX failed, see {os.path.join(OUT_DIR, 'slides.log')}")
    print(f"[slides] wrote {os.path.join(OUT_DIR, 'slides.pdf')}")


if __name__ == "__main__":
    main()
