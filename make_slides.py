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
            r"Decay chains keep spin correlations; no generator-level cuts:",
            code(r"""generate    p p > t t~, (t > b w+, w+ > lep+ nul), (t~ > b~ w-, w- > q q)
add process p p > t t~, (t > b w+, w+ > q q), (t~ > b~ w-, w- > lep- nul~)"""),
            rf"$\sqrt{{s}}={fmt(meta.get('sqrts_GeV', 0) / 1000, 3)}$ TeV, "
            rf"$\ell={lepton_tex(meta.get('leptons', '?'))}$, "
            rf"{meta.get('nevents', S['reco']['n'])} events, seed {meta.get('seed', '?')}",
            rf"$\sigma = {fmt(xs, 4)}$ pb $\approx \sigma_{{t\bar{{t}}}}^\mathrm{{LO}}\times"
            r"2\,\mathcal{B}(W\to\ell\nu)\,\mathcal{B}(W\to q\bar{q}')$ (with $\ell=e,\mu$)",
        ]),
        table("Object definitions and levels",
              ["", "parton", r"parton + $\ell$ trigger", "particle level (gen)", "reco (Delphes ATLAS)"], [
            ["source", "MG5 LHE, before shower", "MG5 LHE, before shower", "Pythia8 final state",
             "Delphes ATLAS card"],
            ["leptons", r"LHE $e/\mu$, no cut", r"$p_T>25$, $|\eta|<2.47\,(e), 2.5\,(\mu)$",
             r"status-1 $e/\mu$ from $W/Z/h$ ($\tau$)", "Electron, Muon (same cuts)"],
            ["jets", r"LHE $q,\bar q'$ (no cut)", r"LHE $q,\bar q'$ (no cut)",
             r"GenJet anti-$k_t$ 0.4, $p_T>25$, $|\eta|<2.5$", "Jet (same cuts)"],
            ["b", r"LHE $b,\bar b$", r"LHE $b,\bar b$", r"$\Delta R(j,b)<0.4$, $p_T^b>5$", "BTag bit 0"],
            ["overlap", "--", "--", r"drop jets $\Delta R(j,\ell)<0.4$", "UniqueObjectFinder"],
            [r"$E_T^\mathrm{miss}$", r"$p_T^\nu$ (LHE)", r"$p_T^\nu$ (LHE)", r"GenMissingET: $\sum$ all $\nu$",
             r"MissingET: $-\sum$ calo towers"],
            ["selection", "none", r"lepton only", r"$1\ell$, $\geq4j$, $\geq1b$, $\chi^2$", r"$1\ell$, $\geq4j$, $\geq1b$, $\chi^2$"],
            ["events kept", r"100\%", rf"{100 * S['parton']['trigger_eff']:.1f}\%",
             rf"{100 * S['gen']['n_reco'] / S['gen']['n']:.1f}\%", rf"{100 * S['reco']['n_reco'] / S['reco']['n']:.1f}\%"],
        ], note=r"Parton level means the MG5 partons with \emph{no} cuts; ``parton + $\ell$ trigger'' adds only the "
                r"offline single-lepton cuts. Reco $E_T^\mathrm{miss}$ is the negative vector sum of the calorimeter "
                r"towers; checked: muon and electron events show the same reco$-$gen $E_T^\mathrm{miss}$ along the "
                rf"lepton (${S['met_check']['mu']:+.1f}$ vs ${S['met_check']['e']:+.1f}$ GeV), so muons are not missed.",
           fit=True),
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
        wide_figure("Definition of the ten angles", os.path.join(HERE, "figures", "angle_diagram.pdf")),
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
    return slides


def nu_pz_slides():
    """Δp_z distributions from nu_pz_plot.py (plots/nu_pz.png + .json)."""
    path = os.path.join(PLOTS_DIR, "nu_pz.png")
    if not os.path.exists(path):
        return []
    r = json.load(open(os.path.join(PLOTS_DIR, "nu_pz.json")))
    ch = r["choice"]
    return [wide_figure(r"Neutrino $\Delta p_z$", path, note=(
        rf"Correct jets: {r['reco_all jets correctly assigned']['half_width_68']:.0f} GeV vs "
        rf"{r['reco']['half_width_68']:.0f} GeV for all events (68\% half-width). "
        rf"No real solution in {100 * ch['frac_complex']:.0f}\% of events. With two solutions, the "
        rf"top-mass $\chi^2$ picks the one closer to the truth in {100 * ch['frac_chi2_picks_closer']:.0f}\% "
        rf"of events vs {100 * ch['frac_smaller_picks_closer']:.0f}\% for smaller $|p_z|$."))]


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


def build_hh_osde_slides(H, O, tests):
    """Slides for the hh sample and for f_tt̄ from OSDE (H: hh numbers, O: osde/summary.json)."""
    lv = O["levels"]
    slides = [
        section(r"$hh$ samples", "hh samples"),
        bullets(r"$hh\to b\bar{b}\,WW^*\to b\bar{b}\,\ell\nu\,q\bar{q}'$ sample", [
            r"MadGraph5 built-in \texttt{heft} model (LO, $m_t\to\infty$), same Pythia8 + Delphes ATLAS chain",
            code(r"""generate    g g > h h, (h > b b~), (h > lep+ nul q q)
add process g g > h h, (h > b b~), (h > lep- nul~ q q)"""),
            r"$h\to WW^*$ has one off-shell $W$, so $h\to\ell\nu q\bar{q}'$ is a 4-body decay",
            rf"{H['meta'].get('nevents', '?')} events, seed {H['meta'].get('seed', '?')}; "
            rf"$\sigma\times\mathcal{{B}} = {1000 * H['meta']['cross_section_pb']:.3g}$ fb "
            r"(LO HEFT with MG defaults; \emph{not} an SM prediction, so plots use unit area)",
            r"For tt-hypothesis quantities, $hh$ uses the same $\chi^2$; at parton level the b pairing "
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
    for lvl in ("parton", "gen", "reco"):
        if lvl in lv:
            s = lv[lvl]
            c2 = s["proj_chi2"]
            worst = max(c2, key=lambda a: c2[a][0] / max(c2[a][1], 1))
            slides.append(wide_figure(
                rf"$f^\mathrm{{{lvl}}}_{{t\bar t}}$ (fit on {lvl}-level $t\bar t$, {s['n']['train']} events): "
                rf"projections",
                os.path.join(O["_dir"], f"projections_ttbar_{lvl}.png"),
                note=rf"Black: $f_{{t\bar t}}$ integrated over the other nine angles. "
                     rf"Largest $\chi^2$/ndf vs training data: {ANGLE_TEX[worst]} "
                     rf"({c2[worst][0]:.0f}/{c2[worst][1]})."))
            slides.append(wide_figure(
                rf"$f^\mathrm{{{lvl}}}_{{t\bar t}}$ applied to {lvl}-level $t\bar t$ (independent) and $hh$",
                os.path.join(O["_dir"], f"scores_{lvl}.png"),
                note=r"Left: $y=f_{t\bar t}(\mathbf{x})$; middle: surprisal $s=-\ln y$; right: "
                     r"$hh$ vs $t\bar t$ efficiency when calling $f<y_\mathrm{cut}$ anomalous."))
    rows = []
    for lvl in ("parton", "gen", "reco"):
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
    for lvl in ("parton", "gen", "reco"):
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
    for lvl in ("parton", "gen", "reco"):
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
    for lvl in ("parton", "gen", "reco"):
        if lvl in lv:
            yrows.append([lvl] + [f"{y['chi2']:.0f}/{y['ndf']}" for y in lv[lvl]["y_scale_scan"]])
    first = lv[next(iter(lv))]["y_scale_scan"]
    slides.append(table(r"Diagnostics: Hermite scale for $Y$ ($\chi^2$ of the $Y$ projection, train)",
                        ["level"] + [f"$s={y['scale']}$" for y in first[:-1]] + [r"$s=\mathrm{std}(Y)$"], yrows,
                        note=r"The paper uses $s=1$ (default here); \texttt{osde\_fit.py --y-scale} changes it. "
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
    rows = [[ANGLE_TEX[a]] + [str(scan[l][a]) for l in ("parton", "gen", "reco")] for a in names]
    req = [scan[l]["_required"] for l in ("parton", "gen", "reco")]
    rows.append([r"\textbf{needs}"] + [rf"$j_1\ge{r['j1max']}, j_2\ge{max(r['j2max'], r['min_j1_j2_for_phi_prime'])}$"
                                        for r in req])
    slides.append(table(r"How many orders each angle needs (1-D cross-validated order)",
                        ["angle", "parton", "gen", "reco"], rows,
                        note=r"Order = Legendre degree ($\cos$), harmonic ($\phi$), Hermite order ($Y$, scale = std$(Y)$). "
                             r"Parton level confirms the paper: $j_1\le1$, $j_2\le2$ (only $\cos\Theta$, $Y$ need more). "
                             r"A full product at gen/reco orders would need $>10^{6}$ terms per top pair.", fit=True))
    bp = {l: esum["levels"][l]["basis_params"] for l in ("parton", "gen", "reco")}
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
    for l in ("parton", "gen", "reco"):
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
    rows = []
    for l in ("parton", "gen", "reco"):
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
    for l in ("reco",):
        slides.append(wide_figure(rf"Extended $f^\mathrm{{{l}}}_{{t\bar t}}$ applied to {l}-level $t\bar t$ and $hh$",
                                  os.path.join(edir, f"scores_{l}.png")))
    return slides


def discriminator_slides(odir):
    """The paper's y and t figure (paper_style_plots.py) and the y0 sensitivity."""
    slides = [section(r"Discriminator $t$ (paper style)", "Discriminator t (paper style)")]
    for l in ("parton", "gen", "reco"):
        p = os.path.join(odir, f"tt_hh_dNdy_dNdt_{l}.pdf")
        if os.path.exists(p):
            slides.append(wide_figure(rf"$y=f^\mathrm{{{l}}}_{{t\bar t}}$ and $t$, {l} level (paper basis)", p,
                                      note=r"$y'=y-y_0$, $y_0=2.5\times\min y$ (inferred from the paper figure); $s'=-\ln y'$; "
                                           r"$t=(s'_1-s')/(s'_1-s'_0)$. $t\bar t$: fit sample (leave-one-out); "
                                           r"unit area."))
    p = os.path.join(odir, "t_vs_y0_parton.png")
    if os.path.exists(p):
        slides.append(wide_figure(r"Shape of $t$ depends on $y_0$", p, note=(
            r"$s'_1$ is set by the most negative event, so $y_0$ just below $\min y$ squeezes $t$ toward 1. "
            r"The paper figure (axis from $-1.5\times10^{-5}$, $t$ populated from 0) implies "
            r"$y_0\approx2.5\times\min y$, used here; please confirm. The ROC/AUC do not depend on $y_0$ (monotonic).")))
    return slides


def coefficient_slides(C, odir):
    """How many of the 14400 coefficients matter (osde_simplify.py)."""
    lv = C["levels"]
    lvls = [l for l in ("parton", "gen", "reco") if l in lv]
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
        figures(r"Cross-validated error vs number of coefficients",
                [os.path.relpath(os.path.join(odir, f"simplify_{l}"), HERE) for l in lvls]),
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
            path = os.path.join(adir, f"acceptance_{label}_{kind}.png")
            if os.path.exists(path):
                name = r"$t\bar t$" if label == "tt" else r"$hh$"
                slides.append(wide_figure(f"Acceptance, {name}: {title}", path))
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
        auc = ", ".join(f"{l} {lv[l]['auc_hh_vs_tt']:.2f}" for l in ("parton", "gen", "reco") if l in lv)
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
                         + ", ".join(f"{cl[l]['variants']['cv_best']['kept']}" for l in ("parton", "gen", "reco")
                                     if l in cl)
                         + r" (parton, gen, reco); simplified $f$ has fewer negative values and the same AUC; "
                           r"parton level resolves $t\bar t$ spin correlations")
        E = load("osde_ext", "simplify_summary.json")
        if E:
            el = E["levels"]
            items.append(r"Paper basis has the right order at parton level, but gen/reco acceptance needs "
                         r"$j_{1,2}$ up to 13--16: the extended block basis ("
                         + ", ".join(f"{el[l]['variants']['cv_best']['kept']}" for l in ("parton", "gen", "reco"))
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
    """Plots from the plots directory: one row for 1-3 plots, 2x2 for 4, rows of 3 beyond."""
    n = len(names)
    ncols = {1: 1, 2: 2, 3: 3, 4: 2}.get(n, 3)
    w = {1: 0.6, 2: 0.47, 3: 0.30}[ncols]
    h = (0.72 if n <= 3 else 0.41) - (0.04 if note else 0)
    cells = []
    for name in names:
        path = os.path.join(HERE, name + ".png") if "/" in name else os.path.join(PLOTS_DIR, name + ".png")
        if os.path.exists(path):
            cells.append(rf"\includegraphics[width={w}\textwidth,height={h}\textheight,keepaspectratio]"
                         rf"{{{os.path.relpath(path, OUT_DIR)}}}")
        else:
            print(f"[slides] warning: missing plot {path}")
            cells.append(rf"\fbox{{\parbox{{{w}\textwidth}}{{\centering\tiny missing: "
                         rf"\texttt{{{name.replace('_', chr(92) + '_')}}}}}}}")
    rows = [r" \hspace{0.03\textwidth} ".join(cells[i:i + ncols]) for i in range(0, n, ncols)]
    body = "\\centering\n" + " \\\\[2pt]\n".join(rows)
    if note:
        body += f"\n\n\\vspace{{2pt}}{{\\footnotesize {note}}}"
    return frame(title, body)


def wide_figure(title, path, note=None):
    """One full-width image (e.g. the 2x5 projection panels)."""
    if not os.path.exists(path):
        print(f"[slides] warning: missing {path}")
        return frame(title, rf"\centering missing: \texttt{{{os.path.basename(path).replace('_', chr(92) + '_')}}}")
    body = (rf"\centering\includegraphics[width=\textwidth,height=0.74\textheight,keepaspectratio]"
            rf"{{{os.path.relpath(path, OUT_DIR)}}}")
    if note:
        body += f"\n\n\\vspace{{2pt}}{{\\footnotesize {note}}}"
    return frame(title, body)


def table(title, header, rows, note=None, fit=False):
    """fit=True shrinks a table that is wider than the text (never enlarges it)."""
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
\AtBeginSection[]{\begin{frame}{Outline}\tableofcontents[currentsection]\end{frame}}
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
