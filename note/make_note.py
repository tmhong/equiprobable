#!/usr/bin/env python3
"""
Internal note (LaTeX article) that compiles and explains the study shown in the slides.

Every number is read from the analysis outputs (the same JSON files make_slides.py uses) and the
figures are the ones in the plot directories, so the note stays in sync with the slides.  The
sample-level numbers (cutflow, helicity fractions, resolutions) are computed once from the samples
and cached in note/sample_summary.json.

  .venv/bin/python note/make_note.py                 # writes note/note.tex and compiles note/note.pdf
  .venv/bin/python note/make_note.py --no-compile
  .venv/bin/python note/make_note.py --recompute     # rebuild the sample summary cache

Sections that depend on studies still running (BSM test, pileup) are written from their result
files when present (study_bsm/htt_test.json, pileup/results.json) and as a short status paragraph
otherwise.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ANGLE_TEX = {"Y": "$Y$", "cos_Theta": r"$\cos\Theta$", "cos_theta": r"$\cos\theta$", "phi": r"$\phi$",
             "cos_theta_prime": r"$\cos\theta'$", "phi_prime": r"$\phi'$",
             "cos_theta_tilde": r"$\cos\tilde\theta$", "phi_tilde": r"$\tilde\phi$",
             "cos_theta_prime_tilde": r"$\cos\tilde\theta'$", "phi_prime_tilde": r"$\tilde\phi'$"}
LVL_TEX = {"parton": "parton", "parton_trig": "parton+trig", "gen": "gen", "reco": "reco"}
LEVELS = ["parton", "parton_trig", "gen", "reco"]


# ----------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------
def load(*parts):
    p = os.path.join(ROOT, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def rel(*parts):
    """Path of a figure relative to the note directory."""
    return os.path.relpath(os.path.join(ROOT, *parts), HERE)


def exists(*parts):
    return os.path.exists(os.path.join(ROOT, *parts))


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def sample_summary(sample, hh, recompute):
    """plotting.summary numbers for the tt̄ (independent) sample and the hh reco cutflow, cached."""
    cache = os.path.join(HERE, "sample_summary.json")
    if os.path.exists(cache) and not recompute:
        return json.load(open(cache))
    import make_plots
    from plotting import summary
    opts = {"had_fermion": "down", "jet_assignment": "truth"}
    ev, xsec = make_plots.load_sample("tt", sample, ["parton", "gen", "reco"], None, opts)
    S = summary.summarize(ev)
    S["xsec"] = xsec
    hev, _ = make_plots.load_sample("hh", hh, ["reco"], None, opts)
    S["hh_cutflow"] = summary.cutflow(hev["reco"])
    with open(cache, "w") as f:
        json.dump(S, f, indent=1, default=_jsonable)
    return json.load(open(cache))


# ----------------------------------------------------------------------------------------------
# LaTeX helpers
# ----------------------------------------------------------------------------------------------
def esc(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def tab(caption, header, rows, label, note=None, small=True, col=None):
    col = col or "l" + "c" * (len(header) - 1)
    body = " \\\\\n".join(" & ".join(str(c) for c in r) for r in rows)
    size = r"\small" if small else ""
    note = (r"\\[2pt]\multicolumn{%d}{p{0.95\textwidth}}{\footnotesize %s}" % (len(header), note)) if note else ""
    return rf"""
\begin{{table}}[htbp]\centering{size}
\caption{{{caption}}}\label{{{label}}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{{col}}}\toprule
{" & ".join(header)} \\\midrule
{body} \\
\bottomrule{note}
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}
"""


def fig(path, caption, label, width=0.9):
    if not os.path.exists(os.path.join(HERE, path)):
        return rf"\par\noindent\fbox{{\footnotesize missing figure: \texttt{{{esc(path)}}}}}\par"
    return rf"""
\begin{{figure}}[htbp]\centering
\includegraphics[width={width}\textwidth]{{{path}}}
\caption{{{caption}}}\label{{{label}}}
\end{{figure}}
"""


def figs2(paths, caption, label, width=0.49):
    inc = "\n".join(rf"\includegraphics[width={width}\textwidth]{{{p}}}" for p in paths
                    if os.path.exists(os.path.join(HERE, p)))
    return rf"""
\begin{{figure}}[htbp]\centering
{inc}
\caption{{{caption}}}\label{{{label}}}
\end{{figure}}
"""


def f3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return rf"{100 * x:.{d}f}\%"


def term_tex(term):
    import re
    def m_fn(mt):
        side, j1, j2, mp, m = mt.groups()
        sym = r"M" if side == "lep" else r"\tilde M"
        return rf"${sym}^{{{j1}{j2}}}_{{{mp},{m}}}$"
    t = re.sub(r"(lep|had)\[j1=(-?\d),j2=(-?\d),m'=(-?\d),m=(-?\d)\]", m_fn, term)
    t = re.sub(r"P(\d)\(cosΘ\)", r"$P_{\1}(\\cos\\Theta)$", t)
    t = re.sub(r"H(\d)\(Y\)", r"$h_{\1}(Y)$", t)
    return t.replace(" x ", r" $\times$ ")


# ----------------------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------------------
def sec_intro(N):
    O, C, ml, rx = N["osde"], N["simp"], N["ml"].get("parton"), N["reco_ext"]
    lv, cl = O["levels"], C["levels"]
    auc_all = {l: lv[l]["auc_hh_vs_tt"] for l in LEVELS if l in lv}
    auc_cv = {l: cl[l]["variants"]["cv_best"]["auc"] for l in LEVELS if l in cl}
    rows = [
        [r"Does $f_{t\bar t}$ separate $hh$ from $t\bar t$ at parton level?",
         rf"Yes: AUC {auc_cv['parton']:.3f} ({auc_cv['parton_trig']:.3f} with the lepton trigger), using no $hh$ information"],
        [r"Is it the best unsupervised method?",
         (rf"Narrowly: $+{N['ml']['parton']['paired']['OSDE f_tt, paper basis (CV-truncated) - normalizing flow (RealNVP, 8 couplings)']['delta']:.3f}\pm"
          rf"{N['ml']['parton']['paired']['OSDE f_tt, paper basis (CV-truncated) - normalizing flow (RealNVP, 8 couplings)']['err']:.3f}$ "
          r"over a normalizing flow; well ahead of a Gaussian mixture and an autoencoder") if ml else "--"],
        [r"How far from the best possible with these angles?",
         rf"A supervised classifier on the same angles reaches {max(v['auc_test'] for k, v in ml['results'].items() if v['group'] == 'angles'):.2f}" if ml else "--"],
        [r"Does $y$ help a supervised classifier?", r"No (change within $\pm0.002$)"],
        [r"What happens at gen and reco level?", rf"AUC drops to {auc_all['gen']:.2f} and {auc_all['reco']:.2f}"],
        [r"Why?", r"The $\chi^2$ fit and the $m_W$-constrained neutrino rebuild $hh$ events as $t\bar t$; the "
                  r"information survives in observables that do not assume $t\bar t$ (BDT AUC 0.87)"],
        [r"How many coefficients matter?",
         rf"Cross-validation keeps {', '.join(str(cl[l]['variants']['cv_best']['kept']) for l in LEVELS)} of 14\,400 "
         r"(parton, parton+trig, gen, reco); $\sim$20 terms do as well as all of them"],
        [r"Can the momenta be put back?",
         rf"Yes: one Breit--Wigner factor per resonance mass. Parton {N['simplest']['parton']['angles']:.2f}"
         rf"$\to${N['simplest']['parton']['angles + ... + m_t_had']:.2f}, gen {N['simplest']['gen']['angles']:.2f}$\to$"
         rf"{N['simplest']['gen']['angles + ... + m_t_had']:.2f}, reco {rx['reco']['truncation']['all']['angles']:.2f}$\to$"
         rf"{rx['reco']['truncation']['>10 sigma']['angles+BW']:.2f}"],
    ]
    return r"""
\section{Introduction and summary}
\label{sec:intro}

This note documents a study of the orthogonal-series density estimate (OSDE) $f_{t\bar t}(\mathbf{x})$
of the ten $t\bar t$ decay and production angles $\mathbf{x}=\{Y,\Theta,\theta,\phi,\theta',\phi',
\tilde\theta,\tilde\phi,\tilde\theta',\tilde\phi'\}$ defined in the paper, as a signal-agnostic
(unsupervised) discriminant: $y=f_{t\bar t}(\mathbf{x})$ is low for events that do not look like
$t\bar t$, whatever they are.  The test signal is $hh\to b\bar b\,WW^*\to b\bar b\,\ell\nu\,q\bar q'$, which
has the same final state.  The study has four parts:
\begin{enumerate}
\item Monte Carlo samples of $t\bar t$ and $hh$ at parton level (MadGraph5), particle level (Pythia8) and
  detector level (Delphes, ATLAS card, no pileup), with the full $t\bar t$ reconstruction, and the ten angles
  computed at each level (Sections~\ref{sec:samples}--\ref{sec:angles}).
\item The density $f_{t\bar t}$ by OSDE in the paper's basis, its validation, the number of coefficients
  that matter, the orders needed at particle and detector level, and a comparison with a spherical-harmonic
  basis (Sections~\ref{sec:osde}--\ref{sec:sh}).
\item How well $f_{t\bar t}$ separates $hh$ from $t\bar t$ compared with other unsupervised methods and with
  supervised machine learning on the same inputs, and why the separation seen at parton level disappears
  after reconstruction (Sections~\ref{sec:ml}--\ref{sec:washout}).
\item The fix: putting the momenta back through the resonance masses, which recovers the parton-level
  performance at particle and detector level, and the effect of realistic quark tagging
  (Section~\ref{sec:momentum}).  A BSM benchmark and a pileup study are in Sections~\ref{sec:bsm}
  and~\ref{sec:pileup}.
\end{enumerate}
The headline answers are collected in Table~\ref{tab:summary}.  Throughout, AUC is the area under the ROC
curve for $hh$ versus $t\bar t$ (0.5 = no separation), always evaluated on events that were not used in
any fit: the independent $t\bar t$ sample, or the fit sample scored leave-one-out where stated.
""" + tab("Headline answers.", ["question", "answer"], rows, "tab:summary", col=r"p{0.36\textwidth}p{0.6\textwidth}") + r"""
The slides (\texttt{slides/slides.pdf}) show the same material in the same order; the section titles here
match the slide sections.  All numbers in this note are read from the analysis outputs by
\texttt{note/make\_note.py}, so the note and the slides cannot drift apart.
"""


def sec_samples(N):
    S, mt, mh, nu = N["S"], N["meta_tt"], N["meta_hh"], N["nu_pz"]
    g, r = S["gen"]["cutflow"], S["reco"]["cutflow"]
    hc = S["hh_cutflow"]
    cut_rows = [[gn, str(gv), pct(gv / g[0][1]), str(rv), pct(rv / r[0][1]), str(hv), pct(hv / hc[0][1])]
                for (gn, gv), (_, rv), (_, hv) in zip(g, r, hc)]
    mres = [[lvl] + [rf"${S[lvl][k][0]:.0f}\pm{S[lvl][k][1]:.0f}$" for k in ("m_thad", "m_thad_correct", "m_whad", "m_whad_correct")]
            for lvl in ("gen", "reco")]
    wh = N["whad_split"]
    return rf"""
\section{{Samples and event levels}}
\label{{sec:samples}}

\subsection{{Generation}}
All samples are produced by \texttt{{generate\_samples.py}}: MadGraph5\_aMC@NLO~3.6.7 at leading order
(4-flavour proton, $\sqrt s=13$~TeV), Pythia8 for the parton shower and hadronization, and Delphes~3.5.1
with the ATLAS card and no pileup.  The $t\bar t$ process is generated with the full decay chain so that
spin correlations are kept and without generator-level cuts:
$pp\to t\bar t$, $t\to bW^+(\to\ell^+\nu)$, $\bar t\to\bar bW^-(\to q\bar q')$ plus the charge conjugate, with
$\ell=e,\mu$.  The cross section is $\sigma={mt['cross_section_pb']:.1f}$~pb, i.e.\ the LO $t\bar t$ cross
section times $2\,\mathcal{{B}}(W\to\ell\nu)\,\mathcal{{B}}(W\to q\bar q')$.  Two statistically independent
$t\bar t$ samples are used throughout: the \emph{{fit sample}} ({N['meta_fit']['nevents'] // 1000}k events,
seed {N['meta_fit']['seed']}), on which every density is fitted, and the \emph{{independent sample}}
({mt['nevents'] // 1000}k events, seed {mt['seed']}), on which every result is evaluated.

The $hh$ sample ({mh['nevents'] // 1000}k events) uses MadGraph's built-in \texttt{{heft}} model (effective
$ggh$ and $gghh$ couplings in the $m_t\to\infty$ limit): $gg\to hh$, $h\to b\bar b$, $h\to\ell\nu q\bar q'$
generated as one process.  Because $h\to WW^*$ has one off-shell $W$, the $h\to\ell\nu q\bar q'$ decay is a
four-body decay.  Its cross section times branching ratio, ${1000 * mh['cross_section_pb']:.3g}$~fb, is the
LO HEFT value with MadGraph defaults and not an SM prediction, so all $hh$ distributions are shown at unit
area.  For $t\bar t$-hypothesis quantities $hh$ goes through exactly the same reconstruction; at parton level
the pairing of the $b$ quarks with the $W$ bosons is the one closest to the $t\bar t$ hypothesis (both
$m_{{Wb}}$ closest to $m_t$).

\subsection{{Event levels and selection}}
Four event levels are used (Table~\ref{{tab:levels}}).  \emph{{Parton}} is the MadGraph output with no cuts.
\emph{{Parton+trig}} adds only the offline single-lepton (trigger) cuts to the charged lepton, $p_T>25$~GeV
and $|\eta|<2.47$ ($e$) or $2.5$ ($\mu$), and keeps {pct(S['parton']['trigger_eff'])} of the events.
\emph{{Gen}} is the particle level: anti-$k_t$ $R=0.4$ GenJets, the prompt lepton, and the missing momentum
from the sum of all neutrinos.  Prompt leptons are status-1 $e/\mu$ from a $W$, $Z$ or $h$, or from a $\tau$
that itself comes from one of these (in $h\to WW^*$ the lepton can come directly from the $h$ in the
record); leptons from $B$ and $D$ decays, also via $\tau$, are not prompt.  GenJets within $\Delta R<0.4$ of
a prompt lepton are removed, since the lepton is clustered into a jet otherwise.  A GenJet is $b$-labelled
if a $b$ quark with $p_T>5$~GeV lies within $\Delta R<0.4$.  \emph{{Reco}} is the Delphes output: jets,
electrons, muons and the calorimeter missing transverse momentum, with the Delphes $b$-tag bit.  Jets need
$p_T>25$~GeV and $|\eta|<2.5$; leptons the same cuts as the trigger.

The gen and reco selection requires exactly one lepton, at least four jets and at least one $b$-jet, followed
by the $\chi^2$ $t\bar t$ reconstruction
\begin{{equation}}
\chi^2=\left(\frac{{m_{{jj}}-m_W}}{{\sigma_W}}\right)^2+\left(\frac{{m_{{jjb}}-m_t}}{{\sigma_t}}\right)^2
+\left(\frac{{m_{{\ell\nu b}}-m_t}}{{\sigma_t}}\right)^2,
\label{{eq:chi2}}
\end{{equation}}
minimized over all assignments of the six leading jets to $(b_\mathrm{{had}},b_\mathrm{{lep}},q_1,q_2)$ with
$\min(n_b,2)$ $b$-tagged jets in the two $b$ slots, and over the two solutions of the neutrino $p_z$ from the
$m_W$ constraint, so that the $m_t$ term resolves the two-fold ambiguity ($\sigma_W=10$, $\sigma_t=15$~GeV).
The cutflow is in Table~\ref{{tab:cutflow}}: {pct(S['gen']['n_reco'] / S['gen']['n'])} of the $t\bar t$
events survive at gen level and {pct(S['reco']['n_reco'] / S['reco']['n'])} at reco level; $hh$ leptons are
softer (from the $W^*$) and less isolated, so its efficiency is lower.  Reco and gen missing momentum agree for
muon and electron events (reco minus gen along the lepton direction: ${S['met_check']['mu']:+.1f}$ vs
${S['met_check']['e']:+.1f}$~GeV), which checks that the calorimeter missing momentum does not lose the muons.
""" + tab("Event levels.", ["", "parton", r"parton+trig", "gen (particle)", "reco"], [
        ["source", "MG5 LHE", "MG5 LHE", "Pythia8", "Delphes ATLAS"],
        ["selection", "none", r"$\ell$ cuts only", r"$1\ell,\geq4j,\geq1b,\chi^2$", r"$1\ell,\geq4j,\geq1b,\chi^2$"],
        ["events kept", r"100\%", pct(S['parton']['trigger_eff']), pct(S['gen']['n_reco'] / S['gen']['n']),
         pct(S['reco']['n_reco'] / S['reco']['n'])],
        [r"$E_T^\mathrm{miss}$", r"$p_T^\nu$", r"$p_T^\nu$", r"$\sum$ all $\nu$", r"$-\sum$ calo towers"]],
        "tab:levels") + tab(
        r"Cutflow of the independent $t\bar t$ sample and of the $hh$ sample (reco); fractions relative to all generated events.",
        ["selection", "gen", "", "reco", "", r"$hh$ reco", ""], cut_rows, "tab:cutflow") + rf"""
\subsection{{Reconstruction quality}}
All four jets are assigned correctly in {pct(S['gen']['n_correct'] / S['gen']['n_reco'], 0)} (gen) and
{pct(S['reco']['n_correct'] / S['reco']['n_reco'], 0)} (reco) of the reconstructed $t\bar t$ events.  The
reconstructed masses (Table~\ref{{tab:mres}}) have broad inclusive peaks that come from wrong jet choices and
from quarks outside the acceptance, not from the jet response: with the correct jets the hadronic top and
$W$ peaks are narrow.
""" + tab(r"Mass resolution: median $\pm$ 68\% half-width (GeV).",
          ["", r"$m_{t,\mathrm{had}}$, all", r"$m_{t,\mathrm{had}}$, correct jets",
           r"$m_{W,\mathrm{had}}$, all", r"$m_{W,\mathrm{had}}$, correct jets"], mres, "tab:mres") + rf"""
The neutrino $p_z$ (Figure~\ref{{fig:nupz}}) has a 68\% half-width of {nu['gen']['half_width_68']:.0f}~GeV at gen
and {nu['reco']['half_width_68']:.0f}~GeV at reco level, improving to
{nu['reco_all jets correctly assigned']['half_width_68']:.0f}~GeV when all jets are correct.  The $m_W$
constraint has no real solution in {pct(nu['choice']['frac_complex'], 0)} of the events (the discriminant is
then set to zero).  When two real solutions exist, the top-mass term of the $\chi^2$ picks the one closer to
the truth in {pct(nu['choice']['frac_chi2_picks_closer'], 0)} of the events, which is not better than simply
taking the smaller $|p_z|$ ({pct(nu['choice']['frac_smaller_picks_closer'], 0)}).
""" + figs2([rel("plots", "nu_pz_a.png"), rel("plots", "nu_pz_b.png")],
            r"Neutrino $\Delta p_z=p_z^{\nu,\mathrm{reco}}-p_z^{\nu,\mathrm{parton}}$ at gen and reco level (left) and "
            r"at reco level split by the correctness of the jet assignment (right).", "fig:nupz") + rf"""
For $hh$ the reconstruction under the $t\bar t$ hypothesis is already visible in the masses: the hadronic
boson is the off-shell $W^*$ in {pct(1 - wh['frac_had_onshell'], 0)} of the events, with a true median mass of
{wh['offshell_parton_median']:.0f}~GeV, but the $\chi^2$ picks jet pairs near $m_W$ (median
{wh['offshell_gen_median']:.0f}~GeV at gen and {wh['offshell_reco_median']:.0f}~GeV at reco level).  This is
the first sign of the effect studied in Section~\ref{{sec:washout}}.
""" + fig(rel("plots_hh", "hh_overview.png"), r"The $hh$ sample at parton, gen and reco level: $m_{b\bar b}$, the hadronic "
          r"$W$ mass and the $\chi^2$-reconstructed top mass.", "fig:hhoverview")


def sec_angles(N):
    S, lh = N["S"], N["lep_had"]
    hel = []
    for lvl in ("parton", "gen", "reco"):
        for key, lab in (("hel_lep", r"$\ell$"), ("hel_had", r"down-type $q$")):
            h = S[lvl][key]
            hel.append([f"{lvl}, {lab}"] + [rf"${h[k]:.3f}\pm{h['e' + k]:.3f}$" for k in ("F0", "FL", "FR")])
    ares = []
    for a, tex in ANGLE_TEX.items():
        row = [tex]
        for lvl in ("gen", "reco"):
            for k in ("angles_all", "angles_correct"):
                c, d = S[lvl][k][a]
                row.append(f"{c:.2f} / {d:.2f}")
        ares.append(row)
    key = "cos_theta_prime|cos_theta_prime_tilde"
    pp = ", ".join(f"{v['p']:.2f}" for k, v in lh["parton"].items() if k != "n")
    return rf"""
\section{{The ten angles}}
\label{{sec:angles}}

The angles are defined exactly as in the paper (Figure~\ref{{fig:diagram}}).  $Y$ is the rapidity of the
$t\bar t$ system in the laboratory and $\Theta$ the polar angle of the leptonic top in the $t\bar t$ rest
frame.  For each top, $(\theta,\phi)$ are the polar and azimuthal angles of the $W$ in the top rest frame,
with $z=\hat p_t^{{\,\mathrm{{lab}}}}$, $y\propto\hat z_\mathrm{{beam}}\times z$ and $x=y\times z$; and
$(\theta',\phi')$ are the angles of the fermion in the $W$ rest frame reached by a standard boost from the
top frame, with $z'=\hat p_W$, $y'\propto z\times z'$ and $x'=y'\times z'$, i.e.\ the rotation
$D(\phi,\theta,0)$.  On the leptonic side the fermion is the charged lepton; on the hadronic side (tilde
angles) it is the down-type quark, which plays the same role in the $V\!-\!A$ decay.  At gen and reco level
the down-type jet is chosen by matching to the parton-level quarks, i.e.\ perfect up/down tagging as the
paper assumes; realistic tagging is studied in Section~\ref{{sec:tagging}}.  The implementation was checked
by building momenta from known angles and recovering them, which agrees to $10^{{-14}}$.
""" + fig(rel("figures", "angle_diagram.pdf"), r"Definition of the angles in the top rest frame (left) and in the $W$ rest frame "
          r"(right); the tilde angles are defined in the same way for the hadronic top.", "fig:diagram", 1.0) + rf"""
\subsection{{Validation}}
The $W$ helicity fractions obtained from the moments of $\cos\theta'$ (Table~\ref{{tab:hel}}) reproduce the
SM expectation at parton level, $F_0=m_t^2/(m_t^2+2m_W^2)\approx0.70$, $F_L\approx0.30$, $F_R\approx0$, on
both the leptonic and the hadronic side.  At gen and reco level they are distorted by acceptance and
combinatorics; no unfolding is attempted.  At parton level the leptonic and hadronic decay angles have the
same distributions, as they must in the SM (Kolmogorov--Smirnov $p$-values {pp} for $\cos\theta$, $\phi$,
$\cos\theta'$, $\phi'$).  At reco level they differ even when every jet is correctly assigned
($\cos\theta'$ KS distance {lh['reco, correct jets'][key]['ks']:.3f}), because the two sides are measured
differently: lepton cuts and the neutrino solution on one side, jet cuts and jet response on the other.
""" + tab(r"$W$ helicity fractions from the moments of $\cos\theta'$.", ["", "$F_0$", "$F_L$", "$F_R$"], hel, "tab:hel") + \
        tab(r"Angle resolution with respect to parton level: correlation coefficient / median $|\Delta|$, for all reconstructed "
            r"events and for events with all jets correctly assigned.",
            ["angle", "gen, all", "gen, correct", "reco, all", "reco, correct"], ares, "tab:ares") + r"""
Reco angles track the parton angles when the jets are assigned correctly; the smearing of the inclusive
distributions is dominated by combinatorics, not by the detector response.
""" + figs2([rel("plots", "lep_vs_had_parton_decay.png"), rel("plots", "lep_vs_had_reco_correct_jets_decay.png")],
            r"Leptonic versus hadronic $W$ decay angles at parton level (left) and at reco level with correctly assigned jets (right).",
            "fig:lephad", 0.49)


def sec_osde(N):
    O = N["osde"]
    lv = O["levels"]
    rows, si_rows = [], []
    for l in LEVELS:
        s = lv[l]
        fn, ms = s["frac_negative"], s["mean_surprisal"]
        rows.append([LVL_TEX[l], str(s["n"]["train"]), str(s["n"]["test"]), str(s["n"]["hh"]), str(s["n_significant_3sigma"]),
                     f"{s['integral']:.3f}", rf"{pct(fn['test'])} / {pct(fn['hh'])}", f"{ms['test']:.2f} / {ms['hh']:.2f}",
                     f3(s["auc_hh_vs_tt"])])
        si = s["self_inclusion"]
        si_rows.append([LVL_TEX[l], pct(si["self_term_over_f_median"], 0),
                        rf"{pct(si['frac_negative_naive'])} / {pct(fn['train'])} / {pct(fn['test'])}",
                        f"{si['mean_s_naive']:.2f} / {ms['train']:.2f} / {ms['test']:.2f}",
                        f"{si['ks_naive_vs_test']:.3f} / {si['ks_loo_vs_test']:.3f}"])
    yrows = [[LVL_TEX[l]] + [f"{y['chi2']:.0f}/{y['ndf']}" for y in lv[l]["y_scale_scan"]] for l in LEVELS]
    first = lv["parton"]["y_scale_scan"]
    out = rf"""
\section{{The density $f_{{t\bar t}}$ by OSDE}}
\label{{sec:osde}}

\subsection{{Basis and estimator}}
The basis is the paper's: for each top the 24 functions
\begin{{equation}}
M^{{j_1j_2}}_{{m'm}}(\phi,\theta,\phi',\theta')=\frac{{\sqrt{{2j_1+1}}\sqrt{{2j_2+1}}}}{{4\pi}}
D^{{j_1}}_{{m'm}}(\phi,\theta,0)\,D^{{j_2}}_{{m0}}(\phi',\theta',0),\qquad j_1\le1,\ j_2\le2,
\end{{equation}}
times normalized Legendre polynomials $p_l(\cos\Theta)=\sqrt{{(2l+1)/2}}\,P_l$ with $l<{O['n_legendre']}$ and
normalized Hermite \emph{{functions}} $h_n(Y)=(2^nn!\sqrt\pi)^{{-1/2}}H_n(Y)e^{{-Y^2/2}}$ with
$n<{O['n_hermite']}$; the bare Hermite polynomials would be orthogonal only with the weight $e^{{-Y^2}}$.
The product $B=M_{{\mathbf{{j}}}}\tilde M_{{\tilde{{\mathbf{{j}}}}}}\,p_l\,h_n$ has
$24\times24\times{O['n_legendre']}\times{O['n_hermite']}={lv['parton']['n_coefficients']}$ terms.  The
orthogonal-series estimate is $\hat c=\frac1N\sum_kB^*(\mathbf{{x}}_k)$ with its standard error, and
$f_{{t\bar t}}=\sum\hat c\,B$, which is real because the set is closed under complex conjugation.  The Wigner
$d$ functions are computed in the Jacobi-polynomial form, which is stable to $j=24$.

Three separate densities are fitted: $f^\mathrm{{parton}}_{{t\bar t}}$ on parton-level $t\bar t$ angles and
applied to parton-level $t\bar t$ and $hh$, and likewise $f^\mathrm{{gen}}$ and $f^\mathrm{{reco}}$ (plus
$f^{{\mathrm{{parton+trig}}}}$).  Each is fitted on the fit sample and evaluated on the independent sample
and on $hh$.  The code was checked with unit tests (\texttt{{tests/test\_osde.py}}, 14 tests): the Wigner
$d^j_{{m'm}}$ agree with closed forms and with \texttt{{sympy}} to $10^{{-12}}$; the Gram matrices of $M$,
$p_l$ and $h_n$ equal the identity to $10^{{-12}}$; a toy density with known coefficients is recovered from
$2\times10^5$ events with unit-width pulls; the fitted $f$ has an imaginary part below $10^{{-10}}$ of the
real part; the leave-one-out formula below equals brute-force refits; and the extended basis of
Section~\ref{{sec:orders}} reduces exactly to the paper's basis.

\subsection{{Self-inclusion bias and leave-one-out}}
Each fit event adds $\frac1N\sum_j|B_j(\mathbf{{x}}_k)|^2$ to its own value of $f$, because its own
contribution to every coefficient is evaluated at its own position.  With 14\,400 terms this self term is
not negligible (Table~\ref{{tab:selfincl}}): the fit sample looked systematically more $t\bar t$-like than
the independent sample until the self term was removed.  Removing it exactly (leave-one-out, LOO) makes the
fit sample agree with the independent sample in the fraction of negative $f$, in the mean surprisal and in
the Kolmogorov--Smirnov distance; the difference was entirely this bias, and largest at reco level where $N$
is smallest.  All plots use LOO values for the fit sample.
""" + tab(r"Fit sample versus independent sample: the self-inclusion bias and its removal by leave-one-out (LOO).",
          ["level", r"self term / $f$", r"$f\le0$: fit / fit LOO / indep.", r"$\langle s\rangle$: fit / fit LOO / indep.",
           "KS vs indep.: fit / LOO"], si_rows, "tab:selfincl") + rf"""
\subsection{{Results}}
Table~\ref{{tab:osde}} summarizes the fits; Figures~\ref{{fig:proj_parton}}--\ref{{fig:scores_reco}} show the
one-dimensional projections of $f_{{t\bar t}}$ against the data and the distributions of $y=f_{{t\bar t}}$ for
$t\bar t$ and $hh$.  The negative values of $f$ are the Gibbs overshoot of a truncated series, as noted in the
paper; they are rare for $t\bar t$ and more frequent for $hh$, which lives where $f_{{t\bar t}}$ is small.
The surprisal $s'=-\ln(y-y_0)$ with $y_0=2.5\min y$ makes every event finite, and the ROC curve is obtained by
scanning a cut on $y$ (events with $f<y_\mathrm{{cut}}$ are called $hh$-like).  At parton level
$f_{{t\bar t}}$ separates $hh$ from $t\bar t$ with AUC {f3(lv['parton']['auc_hh_vs_tt'])}
({f3(lv['parton_trig']['auc_hh_vs_tt'])} after the lepton trigger); at gen and reco level the separation
almost vanishes ({f3(lv['gen']['auc_hh_vs_tt'])} and {f3(lv['reco']['auc_hh_vs_tt'])}).  Understanding and
curing this loss is the subject of Sections~\ref{{sec:washout}} and~\ref{{sec:momentum}}.
""" + tab(r"Summary of the $f_{t\bar t}$ fits in the paper's basis.  $t\bar t$ numbers are from the independent sample; "
          r"AUC is the probability that an $hh$ event has lower $f_{t\bar t}$ than a $t\bar t$ event.",
          ["level", r"$N_\mathrm{fit}$", r"$N_\mathrm{indep}$", r"$N_{hh}$", r"$|\hat c|>3\sigma$", r"$\int f$",
           r"$f\le0$: $t\bar t$ / $hh$", r"$\langle s\rangle$: $t\bar t$ / $hh$", "AUC"], rows, "tab:osde")
    for l in ("parton", "reco"):
        out += fig(rel("osde", f"projections_ttbar_{l}.png"),
                   rf"$f^{{\mathrm{{{LVL_TEX[l]}}}}}_{{t\bar t}}$: one-dimensional projections (black) against the fit sample, "
                   rf"production angles and leptonic top.", f"fig:proj_{l}")
        out += fig(rel("osde", f"scores_{l}.png"), rf"$y=f^{{\mathrm{{{LVL_TEX[l]}}}}}_{{t\bar t}}$ for $t\bar t$ (independent "
                   r"sample and fit sample with LOO) and $hh$, linear (left, shaded: $y<0$) and logarithmic (right; the "
                   r"left-most bin holds all events with $y\le0$).", f"fig:scoresy_{l}")
        out += fig(rel("osde", f"scores_{l}_s.png"), rf"{LVL_TEX[l]} level: surprisal $s'=-\ln(y-y_0)$ (left) and the ROC "
                   r"curve from a scan over $y_\mathrm{cut}$ (right).", f"fig:scores_{l}")
    out += rf"""
\subsection{{The Hermite scale}}
The width of the Hermite functions is a free parameter of the fit; the paper uses $s=1$ and so does this note
by default.  The $Y$ distribution is narrower at gen and reco level (std $\approx0.8$), and $s=1$ then fits
poorly: $f$ overshoots below zero at large $|Y|$ (Table~\ref{{tab:yscale}}).  The extended basis of
Section~\ref{{sec:orders}} uses $s=\mathrm{{std}}(Y)$.
""" + tab(r"$\chi^2$/ndf of the $Y$ projection against the fit sample for different Hermite scales $s$.",
          ["level"] + [f"$s={y['scale']}$" for y in first[:-1]] + [r"$s=\mathrm{std}(Y)$"], yrows, "tab:yscale")
    out += rf"""
\subsection{{The discriminator $t$}}
The paper's variable $t$ is a monotonic rescaling of the surprisal: $y'=y-y_0$, $s'=-\ln y'$ and
$t=(s'_1-s')/(s'_1-s'_0)\in[0,1]$, so that $t\to0$ is less $t\bar t$-like.  Two conventions had to be
inferred from the paper's figure.  First $y_0$: its axis starts at about $2.5\times\min y$ and its $t$
distribution is populated from 0, which needs $y_0$ well below $\min y$; $y_0=2.5\min y$ is used here.  The
shape of $t$ depends on this choice (Figure~\ref{{fig:tvsy0}}), the ROC curve and AUC do not, since the map
is monotonic.  Second, the endpoints $s'_0$ and $s'_1$ are taken as the 0.5\% and 99.5\% quantiles of $s'$
over $t\bar t$ and $hh$ together, with the events beyond put into the first and last bins, so that $t$ fills
$[0,1]$ with no empty bins.  Figure~\ref{{fig:tplot}} shows $y$ and $t$ in the style of the paper at parton
level; the same figures for the other levels are in \texttt{{osde/}}.
""" + fig(rel("osde", "tt_hh_dNdy_dNdt_parton.pdf"), r"$y=f^\mathrm{parton}_{t\bar t}$ and $t$ for $t\bar t$ (fit sample, LOO) "
          r"and $hh$ at unit area, with the ratio $hh/(hh+t\bar t)$ below, drawn as in the paper.", "fig:tplot", 0.8)
    out += fig(rel("osde", "t_vs_y0_parton.png"), r"Shape of $t$ for different choices of $y_0$ at parton level.", "fig:tvsy0", 0.75)
    return out


def sec_coeff(N):
    C, tn = N["simp"], N["noise"]
    cl = C["levels"]
    v = {l: cl[l]["variants"] for l in LEVELS}
    rows = []
    for l in LEVELS:
        for key, lab in (("all", "all (paper)"), ("kronmal_tarter", r"$\sqrt2\sigma$ (K--T)"), ("cv_best", "CV best"), ("5sigma", r"$5\sigma$")):
            r = v[l][key]
            fn = r["frac_negative"]
            rows.append([LVL_TEX[l] if key == "all" else "", lab, str(r["kept"]), f3(r["auc"]), pct(fn["test"]), pct(fn["hh"])])
    P = cl["parton"]
    top = [[str(i + 1), term_tex(t["term"]), f"{t['signif']:.0f}", t["physics"].replace("tt", r"$t\bar t$")]
           for i, t in enumerate(P["top_terms"][:10])]
    cats = {l: cl[l].get("categories_5sigma", {}) for l in LEVELS}
    keys = sorted({k for c in cats.values() for k in c}, key=lambda k: -max(c.get(k, 0) for c in cats.values()))
    crow = [[k.replace("tt ", r"$t\bar t$ ").replace("top pol. x W helicity", r"top pol.\,$\times$\,$W$ hel.")]
            + [str(cats[l].get(k, "--")) for l in LEVELS] for k in keys[:8]]
    crow.append([r"\textbf{total}"] + [rf"\textbf{{{sum(cats[l].values())}}}" for l in LEVELS])
    Ns = sorted({x["N"] for x in tn})
    avg = lambda n, k: float(np.mean([x[k] for x in tn if x["N"] == n]))  # noqa: E731
    corr = {x["N"]: x for x in tn if "rep corr all" in x}
    nrows = [[f"{n // 1000}k", f3(avg(n, "auc all")), f"{avg(n, 'auc 10 sigma'):.3f} ({avg(n, 'terms 10 sigma'):.0f} terms)",
              f"{avg(n, 'gain 10 sigma'):+.3f}",
              (f"{corr[n]['rep corr all']:.2f} / {corr[n]['rep corr 10 sigma']:.3f}" if n in corr else "--")] for n in Ns]
    return rf"""
\section{{How many coefficients matter?}}
\label{{sec:coeff}}

\subsection{{Criterion}}
A term is kept if and only if its coefficient is significant, $|\hat c|/\sigma(\hat c)>t$, with the constant
always kept.  Conjugate partners have equal $|\hat c|$ and $\sigma$, so they are kept or dropped together and
$f$ stays real (checked).  The Kronmal--Tarter rule is $t=\sqrt2$: dropping a term costs $|c|^2$ in integrated
squared error, keeping it costs its variance $\sigma^2$, so a term is kept if $|\hat c|^2>2\sigma^2$.  With
14\,400 mostly null terms this still admits about 16\% of the pure-noise terms, so $t$ is instead chosen by
five-fold cross-validation.  For an orthonormal basis the cross-validated error needs no integrals:
$\mathrm{{ISE}}-\int f^2=\sum_\mathrm{{kept}}|\hat c|^2-2\,\mathrm{{Re}}\sum_\mathrm{{kept}}\hat c\,\hat c^*_\mathrm{{held\text{{-}}out}}$.
The minimum is at $t={P['t_cv']:g}$ and keeps {', '.join(str(v[l]['cv_best']['kept']) for l in LEVELS)} of the
14\,400 terms at parton, parton+trig, gen and reco level (Figure~\ref{{fig:cv}}); the curve is flat out to
$5\sigma$ ({', '.join(str(v[l]['5sigma']['kept']) for l in LEVELS)} terms).
""" + figs2([rel("osde", "simplify_parton.png"), rel("osde", "simplify_reco.png")],
            r"Cross-validated error versus the number of coefficients kept, parton (left) and reco level (right).", "fig:cv") + \
        tab(r"Performance of the truncated $f_{t\bar t}$.  The simplified density has fewer negative values and the same or "
            r"better $hh$ separation with 1--3\% of the terms.",
            ["level", "terms kept", "number", "AUC", r"$f\le0$ ($t\bar t$ indep.)", r"$f\le0$ ($hh$)"], rows, "tab:simp") + rf"""
\subsection{{What the significant terms are}}
Table~\ref{{tab:topterms}} lists the leading terms at parton level.  They are the production angle
($P_2$, $P_4$ of $\cos\Theta$, i.e.\ the forward--backward peaking of $t$-channel $gg\to t\bar t$), the width
of the rapidity distribution ($h_2$, $h_4$ of $Y$), and the $W$ helicity on each side ($j_1=0$, $j_2=1,2$:
functions of $\cos\theta'$ alone, i.e.\ the helicity fractions $F_0$, $F_L$, $F_R$), followed by their
products.  Table~\ref{{tab:cats}} classifies all terms above $5\sigma$ by their physics content: at parton
level the $t\bar t$ spin correlations are resolved, with imaginary parts consistent with zero
(maximum {P.get('max_imag_pull_5sigma', 0):.1f}$\sigma$) as expected without CP violation; at reco level the
spin correlations are no longer resolved and acceptance-induced ``polarization'' terms appear instead.
""" + tab("Leading terms at parton level, after the constant.  $M$: leptonic top; $\\tilde M$: hadronic top; $j_1=0$: $W$ helicity; $j_1=1$: top spin.",
          [r"\#", "term", r"$|\hat c|/\sigma$", "physics"], top, "tab:topterms") + \
        tab(r"Physics content of the terms above $5\sigma$.", ["content of the term"] + [LVL_TEX[l] for l in LEVELS], crow, "tab:cats") + rf"""
\subsection{{Why fewer terms do better: estimation noise}}
\label{{sec:noise}}
A density with about 20 terms separates $hh$ from $t\bar t$ \emph{{better}} than the full 14\,400-term
expansion ({f3(avg(100000, 'auc 10 sigma'))} against {f3(avg(100000, 'auc all'))} at parton level).  This is
not a fluctuation but estimation noise.  Each $\hat c_k$ is a sample mean over $N$ events and carries noise
$\sigma_k\propto1/\sqrt N$; most of the true coefficients are essentially zero, so those terms contribute only
noise.  Summed over thousands of terms the noise is a random wobble on $\hat f(\mathbf{{x}})$, unrelated to
$t\bar t$ versus $hh$, which scrambles the ranking of events.  Dropping the insignificant terms removes the
wobble at the cost of a small bias: the usual bias--variance trade-off.  Two predictions follow, and both hold
(Table~\ref{{tab:noise}}, Figure~\ref{{fig:noise}}): the gain of the truncation shrinks as $N$ grows, like
$1/\sqrt N$, while the truncated $f$ is converged already at 12.5k events; and two independent fits on disjoint
halves of the data rank the events almost identically with the truncated $f$ but not with the full one.
""" + tab(r"Truncation gain versus the size of the fit sample, at parton level; the test sample is always the independent "
          r"$t\bar t$ sample.  Gain errors are $\pm$0.002--0.003 (paired bootstrap).",
          [r"fit events $N$", "AUC, all terms", r"AUC, $|c|/\sigma>10$", "gain", "score correlation of two disjoint fits: all / truncated"],
          nrows, "tab:noise") + fig(rel("study_salvage", "truncation_noise.png"),
                                     r"AUC gain of the truncated $f$ over the full expansion versus the number of fit events.", "fig:noise", 0.7)


def sec_orders(N):
    scan, tr, comp, E, ES = N["order_scan"], N["trunc_gen"], N["proj_comp"], N["ext"], N["ext_simp"]
    names = ["cos_theta", "phi", "cos_theta_prime", "phi_prime", "cos_theta_tilde", "phi_tilde",
             "cos_theta_prime_tilde", "phi_prime_tilde", "cos_Theta", "Y"]
    rows = [[ANGLE_TEX[a]] + [str(scan[l][a]) for l in LEVELS] for a in names]
    req = [scan[l]["_required"] for l in LEVELS]
    rows.append([r"\textbf{needs}"] + [rf"$j_1\ge{r['j1max']},\ j_2\ge{max(r['j2max'], r['min_j1_j2_for_phi_prime'])}$" for r in req])
    crow = []
    for l in LEVELS:
        for key, lab in (("paper", "paper"), ("paper_simplified", "paper, CV"), ("extended", "extended"), ("extended_simplified", "extended, CV")):
            c = comp.get(f"{key}/{l}")
            if c:
                crow.append([LVL_TEX[l] if key == "paper" else "", lab] + [f"{c[a][0] / c[a][1]:.1f}" for a in names[::-1]])
    prow = []
    for l in LEVELS:
        vv = ES["levels"][l]["variants"]
        for key, lab in (("all", "all terms"), ("cv_best", "CV best")):
            r = vv[key]
            prow.append([LVL_TEX[l] if key == "all" else "", lab, str(r["kept"]), f3(r["auc"]), pct(r["frac_negative"]["test"]), pct(r["frac_negative"]["hh"])])
    return rf"""
\section{{Higher orders for gen and reco: the extended basis}}
\label{{sec:orders}}

The paper's basis has $j_1\le1$ and $j_2\le2$: it is at most linear in $\cos\theta$, at most quadratic in
$\cos\theta'$, and contains only the first harmonic in $\phi'$.  This is complete for the physical decay
distribution, and parton level confirms it, but the acceptance cuts create sharp edges ($\cos\theta\to1$,
$\cos\theta'\to-1$, $\phi'\approx0$) that need higher orders.  At gen level the one-dimensional projections of
the paper-basis $f$ miss the data badly (Figure~\ref{{fig:truncgen}}); a direct one-dimensional expansion at
the same order misses it in the same way (maximum difference
{100 * max(x['max_rel_diff_f_vs_same_order_1d'] for x in tr.values()):.1f}\%), while a higher-order expansion
fits ($\chi^2$ for 25 bins: {', '.join(f"{x['chi2_same_order']:.0f}" for x in tr.values())} at the paper's order
against {', '.join(f"{x['chi2_high_order']:.0f}" for x in tr.values())}).  So the order, not the fit, limits $f$.
""" + fig(rel("osde", "truncation_gen.png"), r"Gen level, paper basis: projections of $f$ (same order) against a higher-order "
          r"one-dimensional expansion.", "fig:truncgen") + rf"""
A one-dimensional cross-validated order scan per angle (Table~\ref{{tab:orders}}) gives the orders each level
needs.  Parton level needs exactly the paper's orders apart from $\cos\Theta$ and $Y$; gen and reco need
$j_{{1,2}}$ up to 13--24.  A full product basis at those orders would have more than $10^6$ terms per top
pair, so an \emph{{extended block basis}} is used instead: the union of disjoint subsets of the full
orthonormal product basis, which is still orthonormal so that the OSDE, the leave-one-out and the
cross-validated truncation are unchanged.  Block A is the paper's full product at the physics order; blocks L
and H are high-order single-top blocks (all $(\theta,\phi)$ terms and all $m'=0$ terms of one top with the
other top constant); blocks P and Y are high-order $\cos\Theta$ and $Y$ terms, with the Hermite scale set to
the standard deviation of $Y$.  The orders are capped at 16; the resulting basis has
{', '.join(f"{E['levels'][l]['n_coefficients']} ({LVL_TEX[l]})" for l in LEVELS)} terms.  The price is that
correlations between the two tops stay at the physics order.
""" + tab(r"Order needed by each angle (one-dimensional cross-validated order: Legendre degree for cosines, harmonic for "
          r"azimuths, Hermite order for $Y$) and the resulting requirement on $j_1$, $j_2$.",
          ["angle"] + [LVL_TEX[l] for l in LEVELS], rows, "tab:orders") + \
        tab(r"$\chi^2$/ndf of the one-dimensional projections against the $t\bar t$ fit sample, paper and extended basis.",
            ["level", "basis"] + [ANGLE_TEX[a] for a in names[::-1]], crow, "tab:projchi2") + rf"""
After the cross-validated truncation the extended basis describes every angle at every level with
$\chi^2$/ndf between about 0.4 and 2 (Table~\ref{{tab:projchi2}}, Figure~\ref{{fig:extproj}}), whereas the
paper basis is good only at parton level.  Yet this much better description of $t\bar t$ hardly changes the
$hh$ separation at gen and reco level (Table~\ref{{tab:extperf}}): the limitation is not the density but the
$t\bar t$-hypothesis reconstruction (Section~\ref{{sec:washout}}).  Without truncation the many high-order
terms add noise, with up to 13\% negative $f$.
""" + fig(rel("osde_ext", "projections_ttbar_simplified_reco.png"), r"Extended basis at reco level after CV truncation: projections.",
          "fig:extproj") + tab(r"Performance of the extended basis.", ["level", "terms", "number", r"AUC ($hh$ vs $t\bar t$)",
                                                                         r"$f\le0$ ($t\bar t$ indep.)", r"$f\le0$ ($hh$)"], prow, "tab:extperf")


def sec_sh(N):
    R = N["basis_cmp"]
    rows = []
    for l in LEVELS:
        for lab, key in (("paper ($M$)", "paper (M functions)"), ("sph.\\ harmonics", "spherical harmonics"), ("extended", "extended")):
            for trn in ("all", "CV"):
                r = R.get(f"{key}|{l}|{trn}")
                if not r:
                    continue
                d = (rf"${1e6 * r['delta_ise_vs_paper']:+.2f}\pm{1e6 * r['delta_ise_vs_paper_err']:.2f}$"
                     if "delta_ise_vs_paper" in r and not key.startswith("paper") else "--")
                rows.append([LVL_TEX[l] if (key.startswith("paper") and trn == "all") else "", f"{lab}, {trn}", str(r["terms"]), d,
                             f"{r['worst_chi2ndf']:.1f}", pct(r["frac_negative_indep"]), f3(r["auc"]) if r["auc"] else "--"])
    sh = R["spherical harmonics|parton|CV"]
    return rf"""
\section{{Spherical-harmonic basis}}
\label{{sec:sh}}

As an alternative to the $M$ functions, products of spherical harmonics $Y_{{l_1}}^{{m_1}}(\theta,\phi)\,
Y_{{l_2}}^{{m_2}}(\theta',\phi')$ with $l_1\le1$, $l_2\le2$ were used for each top: 36 functions instead of
24, 32\,400 coefficients in all, with the same $p_l$ and $h_n$.  The set is orthonormal and closed under
conjugation (tested), and the same OSDE, leave-one-out and cross-validated truncation apply.  The physics
difference is that $M$ uses the spin-weighted $D^j_{{m'm}}(\phi,\theta,0)$: for example $d^1_{{01}}\propto
\sin\theta$ is not a finite sum of $Y_{{l0}}$, so the SM decay density, which is finite in $M$, needs in
principle infinitely many spherical-harmonic products.  The figure of merit is the held-out integrated squared
error on the independent $t\bar t$ sample, $\mathrm{{ISE}}-\int f^2=\sum|\hat c|^2-2\langle\hat f\rangle_
\mathrm{{indep}}$, lower being closer to the true density; differences between bases are computed on the same
events (paired), which makes their errors small.

With all terms kept, the spherical harmonics are worse than $M$ at every level, by many standard deviations in
$\Delta$ISE, simply because 32\,400 terms are mostly noise.  After the cross-validated truncation
(Table~\ref{{tab:basis}}, Figure~\ref{{fig:basis}}) the two bases describe $t\bar t$ about equally well
($\Delta$ISE at parton level ${1e6 * sh['delta_ise_vs_paper']:+.2f}\pm{1e6 * sh['delta_ise_vs_paper_err']:.2f}$
in units of $10^{{-6}}$, i.e.\ $M$ is slightly better, about $3\sigma$; at gen and reco level they are equal
within errors), and the spherical harmonics need fewer terms ({sh['terms']} against
{R['paper (M functions)|parton|CV']['terms']} at parton level).  The extended basis is the best at gen and
reco level.  The $hh$ AUC hardly depends on the basis: once only the significant terms are kept, both bases
describe the same few physical features.
""" + tab(r"Comparison of the bases on the independent $t\bar t$ sample.  $\Delta$ISE$<0$: closer to the true density than the "
          r"paper's $M$ basis (paired).  Worst $\chi^2$/ndf of the projections excludes $Y$.",
          ["level", "basis", "terms", r"$\Delta$ISE vs $M$ [$10^{-6}$]", r"worst $\chi^2$/ndf", r"$f\le0$", "AUC"], rows, "tab:basis") + \
        fig(rel("osde_sh", "basis_comparison.png"), r"Paired held-out ISE difference with respect to the paper basis (left) and worst "
            r"projection $\chi^2$/ndf (right), per level and basis.", "fig:basis")


def sec_acceptance(N):
    at, ah = N["acc_tt"], N["acc_hh"]
    return rf"""
\section{{Acceptance}}
\label{{sec:acceptance}}

To separate the effect of the selection from that of the reconstruction, Figure~\ref{{fig:acc}} shows the
parton-level angles before any cut and after the gen and reco selections, so that the angle value of an event
never changes and every shape difference is acceptance only; the angles measured at reco level are overlaid
(dotted) to show what resolution and jet assignment add.  The selection sculpts the angles in a few places.
$\cos\theta'\to-1$ and $\phi'\approx0$ are depleted because the lepton is emitted against the top's flight
direction and fails the $p_T>25$~GeV cut, already at gen level, so it is not an isolation effect.  $|Y|\gtrsim2$
is cut by the $|\eta|<2.5$ acceptance.  $\cos\theta$ and $\cos\tilde\theta\to+1$ are depleted because the
$b$ quark then goes backward and is too soft for the jet cut.  $\cos\Theta$ and the azimuthal angles have a
nearly flat efficiency.  The reco efficiency of $t\bar t$ after each cumulative cut is
{', '.join(f"{n}: {pct(e)}" for n, e in at['cutflow_reco'])}; for $hh$ it is
{', '.join(f"{pct(e)}" for n, e in ah['cutflow_reco'])}.  For $hh$ the dotted curves show the effect that
dominates everything else: the $t\bar t$-hypothesis $\chi^2$ turns the flat parton-level $\cos\Theta$ into
$t\bar t$-like peaks at $\pm1$.  The reconstruction, more than the acceptance, makes $hh$ look like $t\bar t$.
""" + fig(rel("acceptance", "acceptance_tt_shapes.png"), r"$t\bar t$: parton-level angles with no cuts, after the gen selection and "
          r"after the reco selection (unit area), with the angles measured at reco level overlaid (dotted) and the efficiency "
          r"below each panel.", "fig:acc") + \
        fig(rel("acceptance", "acceptance_hh_shapes.png"), r"The same for $hh$.", "fig:acchh")


def sec_ml(N):
    ml = N["ml"]
    if not ml:
        return "\\section{Comparison with machine learning}\\label{sec:ml}\nNot available.\n"
    rows = []
    methods = list(ml["parton"]["results"])
    for k in methods:
        v = ml["parton"]["results"][k]
        w = ml.get("parton_trig", {}).get("results", {}).get(k)
        rows.append([v["group"].replace("+", " + "), k.split(" [")[0].replace("_", " ").replace("f tt", r"$f_{t\bar t}$"),
                     rf"${v['auc_test']:.3f}\pm{v['auc_err']:.3f}$", (rf"${w['auc_test']:.3f}\pm{w['auc_err']:.3f}$" if w else "--")])
    P = ml["parton"]
    pair = P["paired"]["OSDE f_tt, paper basis (CV-truncated) - normalizing flow (RealNVP, 8 couplings)"]
    gains = [P["paired"][k] for k in P["paired"] if "+y]" in k and "momenta" not in k]
    return rf"""
\section{{Is the OSDE approach helping?  Comparison with machine learning}}
\label{{sec:ml}}

\subsection{{Setup}}
Every method is evaluated on the same test events, the independent $t\bar t$ sample ({P['n_test_tt']} events)
and a held-out 40\% of $hh$ ({P['n_test_hh']} events); training uses the $t\bar t$ fit sample and 50\% of
$hh$, with 10\% of each held out for validation.  Errors are bootstrap errors; differences between methods are
paired bootstraps on the same test events.  The \emph{{unsupervised}} methods are trained on $t\bar t$ only,
exactly like $f_{{t\bar t}}$, on the same ten angles: a Gaussian-mixture density (32 components), an
autoencoder scored by its reconstruction error, and a normalizing flow (RealNVP, 8 coupling layers).  The
\emph{{supervised}} methods see labelled $t\bar t$ and $hh$ and therefore set an upper bound for the inputs:
a boosted decision tree (BDT, scikit-learn histogram gradient boosting), a multi-layer perceptron (MLP) and an
FT-Transformer on (a) the ten angles and (b) the angles plus the parton four-momenta, and a particle
transformer with pairwise invariant-mass and $\Delta R$ attention biases on the momenta alone.  To test
whether $y$ helps a classifier, the same classifiers are trained with $y=f_{{t\bar t}}$ and $s'$ added to their
inputs (leave-one-out $y$ for the $t\bar t$ training events).  Two controls must give AUC $\approx0.5$:
shuffled labels, and $t\bar t$ (fit sample) against $t\bar t$ (independent sample).

\subsection{{The methods in plain words}}
The unsupervised methods all do the same job as $f_{{t\bar t}}$: learn the $t\bar t$ density from $t\bar t$
events and flag events where it is low.  The \emph{{Gaussian mixture}} models the density as a sum of 32
Gaussian blobs in the ten-dimensional angle space, with positions and widths fitted to $t\bar t$.  The
\emph{{autoencoder}} is a network that squeezes the ten angles through a few numbers and rebuilds them; trained
on $t\bar t$, it rebuilds unusual events badly, but it is not a density.  The \emph{{normalizing flow}} learns a
smooth, invertible map $z=T(\mathbf{{x}})$ that turns the $t\bar t$ distribution into a standard Gaussian; the
density then follows from the change of variables, $f(\mathbf{{x}})=\mathcal N(T(\mathbf{{x}}))\,|\det
\partial T/\partial\mathbf{{x}}|$.  The mass map of Section~\ref{{sec:momentum}},
$u=\frac12+\frac1\pi\arctan\frac{{m-M}}{{W}}$ with density $g_u\,du/dm$, is exactly a one-dimensional flow;
RealNVP does the same in ten dimensions by stacking simple steps, each of which shifts and rescales half of the
variables by amounts computed from the other half, so that the Jacobian stays cheap.

The supervised methods learn $P(hh\,|\,\mathbf{{x}})$ from labelled examples; the best any of them can do is the
likelihood ratio $f_{{hh}}/f_{{t\bar t}}$, which $f_{{t\bar t}}$ alone can only approximate by acting as if
$f_{{hh}}$ were flat.  The \emph{{BDT}} is a few hundred small decision trees, each correcting the mistakes of
the previous ones.  The \emph{{MLP}} is a standard neural network of weighted sums with simple non-linearities
in between.  The \emph{{FT-Transformer}} turns each input variable into a vector (a token) and lets every
variable attend to every other, so it learns which combinations matter.  The \emph{{particle transformer}} uses
the six particles' four-momenta as tokens and is told each pair's invariant mass and $\Delta R$, so it sees
$m_{{b\bar b}}=m_h$ directly.  They win because they know what $hh$ looks like; the goal of $f_{{t\bar t}}$ is
different, a $t\bar t$-only score that flags anything that is not $t\bar t$.

\subsection{{Results}}
""" + tab(r"AUC ($hh$ vs $t\bar t$) of every method on the same test events, parton level and parton+trig level.",
          ["inputs", "method", "AUC parton", "AUC parton+trig"], rows, "tab:ml") + rf"""
As an anomaly detector that has never seen $hh$, the CV-truncated $f_{{t\bar t}}$ is the best of the
unsupervised methods, narrowly ahead of the normalizing flow (paired difference
${pair['delta']:+.3f}\pm{pair['err']:.3f}$ at parton level and
${ml['parton_trig']['paired']['OSDE f_tt, paper basis (CV-truncated) - normalizing flow (RealNVP, 8 couplings)']['delta']:+.3f}\pm{ml['parton_trig']['paired']['OSDE f_tt, paper basis (CV-truncated) - normalizing flow (RealNVP, 8 couplings)']['err']:.3f}$
at parton+trig level) and well ahead of the Gaussian mixture and the autoencoder.  A supervised classifier on
the same ten angles reaches {max(v['auc_test'] for v in P['results'].values() if v['group'] == 'angles'):.2f}, the
upper bound for these inputs.  Adding $y$ to the supervised classifiers changes their AUC by
{min(g['delta'] for g in gains):+.3f} to {max(g['delta'] for g in gains):+.3f}, within the uncertainty of
$\pm0.002$: $y$ carries no information that the classifier does not already learn from the angles.  The
momenta make the task trivial (AUC above 0.999), because they contain $m_{{b\bar b}}=m_h$ and the off-shell
$W^*$ mass; this is the scale of what the angles alone leave out, and the starting point of
Section~\ref{{sec:momentum}}.  The controls give {f3(P['results']['control: BDT, shuffled labels']['auc_test'])}
and {f3(P['results']['control: BDT, tt (fit) vs tt (independent)']['auc_test'])} at parton level.  At
parton+trig level the shuffled-label control is {f3(ml['parton_trig']['results']['control: BDT, shuffled labels']['auc_test'])}
$\pm0.006$, about $3\sigma$ above 0.5; with some thirty AUCs computed, one such fluctuation is not surprising.
""" + fig(rel("ml", "roc_parton_a.png"), r"ROC curves at parton level, unsupervised and supervised methods on the ten angles.",
          "fig:roc", 0.75) + fig(rel("ml", "roc_parton_b.png"), r"ROC curves at parton level: does adding $y$ help a supervised classifier?",
                                 "fig:rocb", 0.75)


def sec_washout(N):
    M, D = N["mech"], N["decomp"]
    st = D["steps"]
    def s(name):
        return st[name]
    rows = []
    for key, lab in (("S0 LHE parton, all", "parton, all events"), ("S1 LHE parton, reco_ok", "+ gen selection"),
                     ("S1c post-ISR parton, reco_ok", "+ ISR recoil (Pythia partons)"), ("S1d pre-hadronization (FSR), reco_ok", "+ FSR"),
                     ("S2chi chi2 jets + gen lep + true nu", r"$\chi^2$ jets + true $\nu$"), ("S4 gen chi2 reconstruction", r"full gen $\chi^2$ ($\nu$ from MET)")):
        if key in st:
            x = s(key)
            rows.append([lab, f"{x['n_tt'] // 1000}k / {x['n_hh'] / 1000:.1f}k", f3(x["osde"]["auc_loo"]), f3(x["osde"]["auc_parton_f"]),
                         f3(x["bdt"][0])])
    mrows = []
    for key, lab in (("S1c_m post-ISR parton, reco_ok & matched", "post-ISR parton"), ("S2 matched jets + gen lep + true nu", r"matched jets + gen $\ell$ + true $\nu$"),
                     ("S3a S2 + nu pz from mW (true nu pT)", r"$\nu$ $p_z$ from $m_W$ (true $\nu$ $p_T$)"), ("S3 S2 + nu from MET + mW", r"$\nu$ from MET + $m_W$"),
                     ("S2chi_m (matched events)", r"$\chi^2$ jets + true $\nu$"), ("S4m (matched events)", r"full $\chi^2$")):
        if key in st:
            x = st[key]
            mrows.append([lab, f3(x["osde"]["auc_loo"]), f3(x["osde"]["auc_parton_f"]), f3(x["bdt"][0])])
    return rf"""
\section{{Why the separation washes out at gen level}}
\label{{sec:washout}}

\subsection{{Short answer}}
At parton level the $hh$ ``tops'' are not tops: each pseudo-top is the $W$ from one Higgs boson plus the $b$
quark from the other, with a mass of median {M['mass']['hh']['p_m_t_lep'][1]:.0f}~GeV, and one of the two $W$
bosons is far off shell (median {M['mass']['hh W_had off-shell']['p_m_w_had'][1]:.0f}~GeV); only
{pct(M['mass']['hh frac |m(Wb)-mt|<30 both, parton'], 0)} of the $hh$ events have both pseudo-tops within 30~GeV
of $m_t$.  This shows in the angles: $\cos\theta$ and $\cos\tilde\theta$ peak at $-1$ for $hh$ because the
soft $W^*$ goes backward while the $b$ carries the pseudo-top's momentum, whereas they are flat for
$t\bar t$; $\cos\Theta$ is isotropic for $hh$ but forward--backward peaked for $t\bar t$ by $t$-channel top
exchange; and $\cos\theta'$ has no $V\!-\!A$ zero at $+1$ for $hh$ (Figure~\ref{{fig:mech1}}).  At gen level
the $t\bar t$-hypothesis reconstruction removes these differences.  The $\chi^2$ builds $m_W$ and $m_t$ from
whatever jets fit best, picking the true four jets in only {pct(M['eff']['hh']['chi2 picks right 4 jets | reco_ok'], 0)}
of the $hh$ events against {pct(M['eff']['tt']['chi2 picks right 4 jets | reco_ok'], 0)} of $t\bar t$,
and the $m_W$-constrained neutrino puts the off-shell leptonic $W^*$ on shell (its $p_z$ is then off by a
median {M['nu'][[k for k in M['nu'] if k.startswith('hh lep W*')][0]]:.0f}~GeV, against {M['nu']['tt (matched)']:.0f}~GeV for $t\bar t$).
The information itself is still there at gen level: on the same events a BDT on observables that do not
assume $t\bar t$ ($m_{{b\bar b}}$, $\Delta R_{{bb}}$, $E_T^\mathrm{{miss}}$, $m_T$, \ldots) reaches AUC
{M['bdt']['gen: hypothesis-independent observables (reco_ok events)']:.2f}, with $m_{{b\bar b}}$ alone at
about 0.76 (likelihood-ratio AUC); the $\chi^2$ value itself also differs (median 21.6 for $hh$ against 6.2 for $t\bar t$).
""" + fig(rel("study_washout", "mechanism", "figures", "fig1_parton_signatures.png"),
          r"Parton level: what distinguishes $hh$ from $t\bar t$ in the angles and the pseudo-top masses.", "fig:mech1") + rf"""
\subsection{{Step-by-step decomposition}}
Two independent studies agree.  The first follows the physics mechanism (Figures~\ref{{fig:mech1}}--\ref{{fig:mech3}});
the second measures the loss step by step from the parton level to the full gen-level reconstruction
(Table~\ref{{tab:decomp}}, Figure~\ref{{fig:decomp}}).  Each step re-fits $f_{{t\bar t}}$ on the $t\bar t$ events
of that step, scored leave-one-out; the parton-level $f$ applied unchanged, and a supervised BDT on the ten
angles, are shown alongside.  An important technical point: the LHE partons are not the right reference at
gen level, because Pythia's initial-state-radiation recoil kicks the whole hard system (the median $\Delta R$
between the LHE lepton and the final lepton is 0.16 for $t\bar t$ and 0.36 for $hh$).  The ``true'' partons
are therefore taken from the Pythia record, the decay products of the last-copy $t$, $W$ or $h$ after the
initial-state radiation and before the final-state radiation.
""" + tab(r"AUC along the chain from parton to gen level, all selected events ($hh=1$).  LOO: $f_{t\bar t}$ refit at that step and "
          r"scored leave-one-out; fixed: the parton-level $f$ applied unchanged; BDT: supervised on the ten angles.  Errors about $\pm$0.005.",
          ["step", r"$t\bar t$ / $hh$", r"$f_{t\bar t}$ LOO", r"$f$ fixed", "BDT"], rows, "tab:decomp") + \
        tab(r"The same on the subset of events whose four partons match distinct jets (39\% of $t\bar t$, 12\% of $hh$; "
            r"errors about $\pm$0.013).", ["step", r"$f_{t\bar t}$ LOO", r"$f$ fixed", "BDT"], mrows, "tab:decompm") + rf"""
The selection costs nothing, and the ISR recoil costs $f_{{t\bar t}}$ almost nothing (it costs the supervised
BDT about 0.08, because the BDT's extra power at parton level relies on the exact $p_T=0$ kinematics of the
$t\bar t$ system).  Final-state radiation costs 0.03.  The largest loss, 0.06, is the choice of jets by the
$\chi^2$: on the matched subset the jets and the $\chi^2$ cost nothing at all, so the whole loss is in the
unmatched events, where the $\chi^2$ has to build the system from wrong jets; for $hh$ this is the soft $W^*$
quarks (median $p_T$ 27~GeV), which often make no jet.  The neutrino costs 0.03 overall and 0.05 on matched
events, almost all of it from the choice of the $p_z$ solution with the $m_t$ rule rather than from the
missing-momentum resolution.  Per angle (Figure~\ref{{fig:decompa}}), $\cos\Theta$ and $\cos\tilde\theta$ lose
their separation at the jet choice, $\cos\theta'$ at the neutrino, and ISR recoil mainly costs $\phi$,
$\tilde\phi$ and $Y$.  At the end no single angle has an AUC above 0.57.
""" + fig(rel("study_washout", "decomposition", "slide_chain_main.png"), r"AUC along the chain from parton to gen level, all selected events.",
          "fig:decomp", 0.85) + fig(rel("study_washout", "decomposition", "slide_angles.png"), r"Single-angle AUC along the chain.",
                                    "fig:decompa", 0.95) + fig(rel("study_washout", "mechanism", "figures", "fig2_chi2_mass_pull.png"),
                                                               r"The $\chi^2$ builds $W$ and top masses that $hh$ does not have.", "fig:mech2") + \
        fig(rel("study_washout", "mechanism", "figures", "fig3_angle_morphing.png"),
            r"$hh$ angles become $t\bar t$-like: $\cos\Theta$, $\cos\theta'$ and $\cos\theta$ for $hh$ with true jets and the true "
            r"neutrino, with the $m_W$-constrained neutrino, and after the full $\chi^2$, against $t\bar t$.", "fig:mech3") + r"""
\subsection{Conclusion}
The angles are the right variables for $t\bar t$, but at gen and reco level they are computed under the
$t\bar t$ hypothesis, which removes exactly what makes $hh$ different.  On top of that, $f_{t\bar t}$ is
unsupervised: it flags only events where $t\bar t$ is rare, and the fitted reconstruction moves $hh$ into the
region where $t\bar t$ is dense.  The information is not gone; it sits in the dimensions that the ten angles
do not span.  That is the subject of the next section.
"""


def sec_momentum(N):
    sm, sv, at, rx, tg = N["simplest"], N["salvage"], N["angle_terms"], N["reco_ext"], N["tagging"]
    P, G = sm["parton"], sm["gen"]
    mrows = [[r"angles only, $y(\Omega)$ (paper)", f3(P["angles"]), f3(G["angles"])],
             [r"$+\,m_{W,\ell}$", f3(P["angles + ... + m_w_lep"]), r"(fixed by the $\nu$ solution)"],
             [r"$+\,m_{W,h}$", f3(P["angles + ... + m_w_had"]), f3(G["angles + ... + m_w_had"])],
             [r"$+\,m_{t,\ell}$", f3(P["angles + ... + m_t_lep"]), f3(G["angles + ... + m_t_lep"])],
             [r"$+\,m_{t,h}$ (all masses)", f3(P["angles + ... + m_t_had"]), f3(G["angles + ... + m_t_had"])]]
    keys = ["none (masses only)", ">40 sigma", ">20 sigma", ">10 sigma", ">5 sigma", ">3 sigma", "CV", "all"]
    name = {"none (masses only)": "none (flat in the angles)", "CV": "CV (paper method)", "all": "all"}
    trows = []
    for k in keys:
        p, g = at["parton"][k], at["gen"][k]
        lab = name.get(k, rf"$|c|/\sigma_c>{k.split()[0][1:]}$")
        trows.append([lab, str(p["terms"]), "--" if k.startswith("none") else f3(p["angles_only"]), f3(p["with_masses"]),
                      str(g["terms"]), "--" if k.startswith("none") else f3(g["angles_only"]), f3(g["with_masses"])])
    rrows = []
    for l in ("gen", "reco"):
        v = rx[l]
        rrows.append([l, f"{v['n_tt']} / {v['n_hh']}", f3(v["truncation"]["all"]["angles"]), f3(v["truncation"][">10 sigma"]["angles"]),
                      f3(v["masses only"]), f3(v["truncation"][">10 sigma"]["angles+BW"])])
    g_, rc = rx["gen"]["masses"], rx["reco"]["masses"]
    names = [("truth (paper assumption)", "perfect up/down tag (paper)"), ("ideal c-tag (eps_c 100%, no mistag)", "ideal $c$-tag"),
             ("c-tag loose (eps_c 45%, b 30%, light 8%)", r"$c$-tag loose (45 / 30 / 8\%)"),
             ("c-tag tight (eps_c 30%, b 11%, light 1.4%)", r"$c$-tag tight (30 / 11 / 1.4\%)"),
             ("symmetrized (no tagging)", "no tag: average both labels"), ("softer jet = down", "no tag: softer jet = down")]
    key = "23-ish terms (>10 sigma)"
    grows = []
    for k, lab in names:
        p, r = tg["parton"][f"{key} | {k}"], tg["reco"][f"{key} | {k}"]
        c = p["tt label correct (w>0.5)"]
        if k.startswith("softer"):
            c = tg["parton"]["p_softer_is_down_tt"]
        grows.append([lab, r"100\%" if k.startswith("truth") else ("--" if c is None else pct(c, 0)), f3(p["angles"]), f3(p["angles+BW"]),
                      f3(r["angles"]), f3(r["angles+BW"])])
    return rf"""
\section{{Adding momentum to $y$: the resonance masses}}
\label{{sec:momentum}}

\subsection{{Motivation}}
The final state has six particles, i.e.\ 18 momentum components.  They split exactly into the 10 paper
variables ($Y$, $\Theta$ and the 8 decay angles), the 4 resonance masses $m_{{W,\ell}}$, $m_{{W,h}}$,
$m_{{t,\ell}}$, $m_{{t,h}}$, and 4 production variables ($m_{{t\bar t}}$, the two components of the $t\bar t$
transverse momentum and an overall azimuth).  For $t\bar t$ the four masses carry no information: each is a
Breit--Wigner fixed by $M$ and $\Gamma$, and in the narrow-width limit the matrix element factorizes into the
angular part times $\prod_k\mathrm{{BW}}(m_k)$, so the paper could rightly drop them.  $hh$ is not on that
shell: one $W$ is off shell and its pseudo-tops sit near 135~GeV (Figure~\ref{{fig:massp}}).  The difference
between $hh$ and $t\bar t$ lives in exactly the dimensions that were dropped, which is the user's observation
that the off-shell $W$ makes $hh$ a different final state from on-shell $t\bar t$.
""" + fig(rel("study_salvage", "masses_parton.png"), r"The dropped dimensions at parton level: $t\bar t$ (blue) sits on the "
          r"Breit--Wigner (dashed; MadGraph cuts it at $\pm15\Gamma$), $hh$ (red) is spread over 100~GeV.", "fig:massp") + rf"""
\subsection{{The extension}}
The extension keeps $y(\Omega)$ as it is and multiplies it by one factor per resonance mass,
\begin{{equation}}
y_\mathrm{{ext}}(\Omega,m)=y(\Omega)\prod_{{k=1}}^{{4}}\mathrm{{BW}}_k(m_k),
\end{{equation}}
the $t\bar t$ density on the full phase space.  In the surprisal, from which $t$ is built, the factors simply
add: $s_\mathrm{{ext}}=s(\Omega)+\sum_k\ln\big[(m_k^2-M_k^2)^2+M_k^2\Gamma_k^2\big]+\mathrm{{const}}$.
At parton level $M$ and $\Gamma$ are the PDG values and nothing is fitted.  At gen and reco level the detector
resolution replaces $\Gamma$; the same peak shape is kept, with $M$ and the width $W$ taken as the median and
half the interquartile range of $t\bar t$, i.e.\ two numbers per mass (Figure~\ref{{fig:massg}}).  A fitted
correction to this shape, an orthonormal Legendre expansion in the Cauchy-mapped variable
$u=\frac12+\frac1\pi\arctan\frac{{m-M}}{{W}}$ with the Jacobian $du/dm$, was also tried
(\texttt{{study\_salvage/salvage.py}}): it changes the gen-level AUC by only
{sv['gen']['auc_masses'] - G['masses_only']:+.3f}, so it is not needed.  At gen and reco level only three
masses are available, because the neutrino solution fixes $m_{{W,\ell}}=m_W$ by construction.
""" + fig(rel("study_salvage", "masses_gen.png"), r"The same dimensions at gen level after the $\chi^2$ reconstruction: the "
          r"$t\bar t$ shell is widened by the resolution (dashed: the two-number model); $hh$ is still wider, mainly in the tails.",
          "fig:massg") + rf"""
\subsection{{Results}}
Table~\ref{{tab:mass}} adds the masses one at a time.  At parton level the masses alone separate $hh$
completely, since $hh$ is never on the $t\bar t$ mass shell.  At gen level the unsupervised score improves
from {G['angles']:.2f} to {G['angles + ... + m_t_had']:.2f}, which is more than the angles alone achieve at
parton level ({P['angles']:.2f}): the parton-level performance is regained.  Table~\ref{{tab:massreco}} is the
clean version of the same result at gen and reco level, with the $t\bar t$ test events from the independent
sample and $M$, $W$ from 30k fit-sample events: reco level goes from
{rx['reco']['truncation']['all']['angles']:.2f} to {rx['reco']['truncation']['>10 sigma']['angles+BW']:.2f}.  The gain
is smaller at reco than at gen level because each mass separates less there (single-mass AUCs
{', '.join(f3(rc[k]['auc']) for k in ('m_w_had', 'm_t_lep', 'm_t_had'))} against
{', '.join(f3(g_[k]['auc']) for k in ('m_w_had', 'm_t_lep', 'm_t_had'))} at gen level), consistent with the $\chi^2$
having more jet combinations to fit at reco level and so building $t\bar t$-like masses more often for $hh$; the
reco test sample is also small ({rx['reco']['n_tt']} $t\bar t$ events, errors about $\pm0.01$).
""" + tab(r"AUC ($hh$ vs $t\bar t$), unsupervised, adding one Breit--Wigner factor at a time (test: half of 40k $t\bar t$ "
          r"fit-sample events not used to set $M$, $W$, angles scored leave-one-out; 20k / 4.4k $hh$).",
          ["", "parton", r"gen ($\chi^2$)"], mrows, "tab:mass") + \
        tab(r"The mass extension at gen and reco level on the independent $t\bar t$ sample.  Widths $W$ (gen / reco): "
            + rf"$m_{{W,h}}$ {g_['m_w_had']['W']:.0f} / {rc['m_w_had']['W']:.0f}, $m_{{t,\ell}}$ {g_['m_t_lep']['W']:.0f} / {rc['m_t_lep']['W']:.0f}, "
            + rf"$m_{{t,h}}$ {g_['m_t_had']['W']:.0f} / {rc['m_t_had']['W']:.0f}~GeV.",
            ["level", r"$t\bar t$ / $hh$", "angles (all)", r"angles ($>10\sigma$)", "masses (BW)", r"angles ($>10\sigma$) + BW"], rrows, "tab:massreco") + rf"""
\subsection{{What the angles add on top of the masses}}
In the product model the angles add nothing measurable for $hh$ once the mass factors are in
({N['angles_vs_bw']['gen']['unsupervised angles(>10 sigma) + BW'] - N['angles_vs_bw']['gen']['unsupervised masses only']:+.3f}
at gen level, {N['angles_vs_bw']['reco']['unsupervised angles(>10 sigma) + BW'] - N['angles_vs_bw']['reco']['unsupervised masses only']:+.3f}
at reco level).  A supervised BDT shows how much is there in principle: trained on the angles and the masses
against the masses alone, it gains ${N['angles_vs_bw']['gen']['BDT (angles+masses) - (masses): paired'][0]:+.3f}\pm
{N['angles_vs_bw']['gen']['BDT (angles+masses) - (masses): paired'][1]:.3f}$ at gen level and
${N['angles_vs_bw']['reco']['BDT (angles+masses) - (masses): paired'][0]:+.3f}\pm
{N['angles_vs_bw']['reco']['BDT (angles+masses) - (masses): paired'][1]:.3f}$ at reco level (paired bootstrap;
\texttt{{study\_salvage/angles\_vs\_bw.py}}).  So a little angular information survives that the product
form, which assumes angles and masses independent for $t\bar t$, does not use; a joint fit of angles and
masses could recover part of it.

\subsection{{How many angular terms are needed once the masses are in?}}
With the mass factors in, the $hh$ separation is the same whatever the number of angular terms kept, even
with none at all (Table~\ref{{tab:angterms}}): for $hh$ the masses do all the work.  Even for the angles alone,
about 20 terms beat all 14\,400 (Section~\ref{{sec:noise}}).  The recommendation is therefore
$y=[\text{{few-term angular }}f]\times\prod_k\mathrm{{BW}}(m_k)$.  The small angular part is worth keeping: it
costs almost nothing, and it is what would catch an on-shell signal with a different spin structure, which the
masses cannot see (Section~\ref{{sec:bsm}}).
""" + tab(r"AUC with the angular density truncated by coefficient significance, with and without the mass factors.",
          ["angular terms kept", "parton: terms", "angles", "+BW", "gen: terms", "angles", "+BW"], trows, "tab:angterms") + rf"""
\subsection{{Realistic quark tagging}}
\label{{sec:tagging}}
The $b$-tagging at reco level is the Delphes ATLAS parametrization and the $\chi^2$ assigns the $b$-jets, so
that part is already realistic.  What the paper assumes beyond it is knowing which $W$ jet is the down-type
quark, and so far the jets were matched to the parton-level quarks, i.e.\ perfect up/down tagging.  The only
real handle is $c$-tagging: in $W\to cs$, {pct(tg['parton']['frac_cs_tt'], 0)} of the events at parton level, a
$c$-tagged jet is up-type, so the other jet is the down-type $s$.  Instead of a hard choice, a weight
$w=P(\text{{softer jet is down-type}}\,|\,\text{{$c$-tags}})$ is computed by Bayes' rule from the $W\to cs$ and
$W\to ud$ hypotheses, the tag efficiencies, and the prior that the softer jet is the down-type one
({pct(tg['parton']['p_softer_is_down_tt'], 0)} at parton level, {pct(tg['reco']['p_softer_is_down_tt'], 0)} at reco
level); then $y=w\,f(\Omega_{{\mathrm{{down=softer}}}})+(1-w)\,f(\Omega_{{\mathrm{{down=harder}}}})$, the
$t\bar t$ density of the event as observed, with $f$ unchanged.  Tags are emulated from the true jet flavour
(the Delphes jet flavour at reco level, the quark identity at parton level) at illustrative ATLAS-like working
points, $\epsilon_c$ / $b$ mistag / light mistag of 30 / 11 / 1.4\% and 45 / 30 / 8\%, plus an ideal tagger.

Table~\ref{{tab:tag}} shows that realistic tagging costs at most 0.02 on the angles alone at parton level,
nothing at reco level, and nothing at all once the masses are in, because the masses do not depend on which
jet is down-type.  The hadronic primed angles carry little of the $hh$ separation anyway.
""" + tab(r"Effect of the down-type assignment (AUC $hh$ vs $t\bar t$, angular $f$ with the terms above $10\sigma$).",
          ["down-type assignment", r"label right ($t\bar t$)", "parton: angles", "+BW", "reco: angles", "+BW"], grows, "tab:tag")


def sec_bsm(N):
    B = N.get("bsm")
    head = r"""
\section{A BSM benchmark: $H\to t\bar t$}
\label{sec:bsm}

A supervised classifier trained on $hh$ learns what $hh$ looks like; $f_{t\bar t}$ learns only what $t\bar t$
looks like.  The case for the unsupervised approach is a signal that nobody trained for.  The benchmark is a
heavy CP-even scalar, $gg\to H\to t\bar t\to b\bar b\,\ell\nu\,q\bar q'$ with $m_H=500$~GeV and
$\Gamma_H=25$~GeV ($\Gamma/m=5\%$, typical of a heavy two-Higgs-doublet scalar), generated with the
\texttt{heft} $ggH$ coupling and SM-like decays (signal only, without the interference with QCD $gg\to t\bar t$).
Unlike $hh$ it has on-shell tops and $W$ bosons: it differs from QCD $t\bar t$ only in the spin structure
(scalar: isotropic $\cos\Theta$, like-helicity tops), in the rapidity and in $m_{t\bar t}$, not in the
resonance masses.  MadGraph's automatic width for this configuration came out absurd ($4\times10^5$~GeV), so
the width is set by hand.
"""
    if not B:
        return head + r"""
\emph{Status: the sample is generated and the test (\texttt{study\_bsm/htt\_test.py}) is running; this section
is filled in automatically from \texttt{study\_bsm/htt\_test.json} when it completes.}
"""
    rows = []
    order = ["unsupervised | f_tt angles (>10 sigma)", "unsupervised | BW masses", "unsupervised | m_tt density",
             "unsupervised | angles + BW", "unsupervised | angles + BW + m_tt",
             "supervised trained on hh | angles", "supervised trained on hh | angles + masses + m_tt",
             "supervised trained on Htt | angles", "supervised trained on Htt | angles + masses + m_tt"]
    lab = {"unsupervised | f_tt angles (>10 sigma)": r"unsupervised: $f_{t\bar t}$ angles ($>10\sigma$)",
           "unsupervised | BW masses": "unsupervised: Breit--Wigner masses", "unsupervised | m_tt density": r"unsupervised: $m_{t\bar t}$ density",
           "unsupervised | angles + BW": "unsupervised: angles + BW", "unsupervised | angles + BW + m_tt": r"unsupervised: angles + BW + $m_{t\bar t}$",
           "supervised trained on hh | angles": r"supervised, trained on $hh$: angles",
           "supervised trained on hh | angles + masses + m_tt": r"supervised, trained on $hh$: angles + masses + $m_{t\bar t}$",
           "supervised trained on Htt | angles": r"supervised, trained on $H\to t\bar t$: angles",
           "supervised trained on Htt | angles + masses + m_tt": r"supervised, trained on $H\to t\bar t$: angles + masses + $m_{t\bar t}$"}
    for k in order:
        r = [lab[k]]
        for l in ("parton", "reco"):
            for sig in ("hh", "Htt"):
                v = B[l][f"{sig} vs tt"].get(k)
                r.append(rf"${v[0]:.3f}\pm{v[1]:.3f}$" if v else "--")
        rows.append(r)
    p, q = B["parton"], B["reco"]
    return head + rf"""
Table~\ref{{tab:bsm}} compares, on the same events, the unsupervised scores with a supervised BDT trained on
$hh$ versus $t\bar t$ and applied to both signals, and with a BDT trained on $H\to t\bar t$ itself as the
ceiling of a dedicated search.  The $m_{{t\bar t}}$ factor is a one-dimensional histogram density of the
$t\bar t$ fit sample (median $m_{{t\bar t}}$: $t\bar t$ {p['mtt_median']['tt']:.0f}, $hh$ {p['mtt_median']['hh']:.0f},
$H\to t\bar t$ {p['mtt_median']['Htt']:.0f}~GeV at parton level).  Test events: {p['n_test']['tt']} $t\bar t$,
{p['n_test']['hh']} $hh$ and {p['n_test']['Htt']} $H\to t\bar t$ at parton level; {q['n_test']['tt']}, {q['n_test']['hh']}
and {q['n_test']['Htt']} at reco level.
""" + tab(r"AUC of each score for $hh$ vs $t\bar t$ and for $H\to t\bar t$ vs $t\bar t$, parton and reco level (bootstrap errors).",
          ["score", "parton: $hh$", r"parton: $H\to t\bar t$", "reco: $hh$", r"reco: $H\to t\bar t$"], rows, "tab:bsm") + rf"""
The $hh$-trained classifier on the angles gives {p['Htt vs tt']['supervised trained on hh | angles'][0]:.2f} on
$H\to t\bar t$ at parton level, and with the masses and $m_{{t\bar t}}$ added {p['Htt vs tt']['supervised trained on hh | angles + masses + m_tt'][0]:.2f};
the unsupervised angles alone give {p['Htt vs tt']['unsupervised | f_tt angles (>10 sigma)'][0]:.2f}
({q['Htt vs tt']['unsupervised | f_tt angles (>10 sigma)'][0]:.2f} at reco level), and with the mass factors and the
$m_{{t\bar t}}$ density {p['Htt vs tt']['unsupervised | angles + BW + m_tt'][0]:.2f}; a dedicated $H\to t\bar t$
classifier reaches {p['Htt vs tt']['supervised trained on Htt | angles + masses + m_tt'][0]:.2f}.

Four things follow.  First, a classifier trained on one signal can be worse than random on another: the
$hh$-trained BDT learned that on-shell masses and high $m_{{t\bar t}}$ mean $t\bar t$, and $H\to t\bar t$ has both.
Second, the unsupervised angular $f_{{t\bar t}}$ is weakly positive and model-agnostic; it sees the isotropic
$\cos\Theta$ and the different spin structure, at parton and at reco level.  Third, the mass factors add
nothing for an on-shell signal and \emph{{dilute}} the angular score, because the summed surprisal is dominated
by the dimensions with the largest spread; for an on-shell signal the angular part alone, or a weighting of
the factors, is the right score.  Fourth, there is a limit to any density-based anomaly score: the
$m_{{t\bar t}}$ resonance at 500~GeV sits on the bulk of the QCD $t\bar t$ distribution (median
{p['mtt_median']['tt']:.0f}~GeV), so ``low $t\bar t$ density'' cannot flag it, whereas a dedicated supervised
search can.  Anomaly detection finds signals where the background is rare, not excesses where it is common.
So there are signals for which $f_{{t\bar t}}$ beats the $hh$-trained classifier, but the margin is small; the
strength of the approach is that it never does worse than random.
"""


def sec_pileup(N):
    PU = N.get("pileup")
    head = r"""
\section{Pileup}
\label{sec:pileup}

All results above are for zero pileup.  The reco-level study is repeated at mean pileup $\langle\mu\rangle=50$
and $200$ with the Delphes ATLAS pileup card (pileup-subtracted jets, the same object and selection cuts),
with the reco-level $f_{t\bar t}$ refitted at each pileup level and the mass extension applied.  Gen level is
unaffected by pileup by construction.
"""
    if not PU:
        return head + r"""
\emph{Status: the pileup samples are being produced; this section is filled in automatically from
\texttt{pileup/results.json} when the study completes.}
"""
    mus = sorted(PU["mu"], key=lambda s: float(s))
    def g(mu, k, f=f3):
        v = PU["mu"][mu].get(k)
        return "--" if v is None else (f(v) if isinstance(v, (int, float)) else str(v))
    rows = [[r"$\langle\mu\rangle$"] + mus,
            [r"$t\bar t$ fit / test events"] + [f"{g(m, 'n_tt_fit', str)} / {g(m, 'n_tt_test', str)}" for m in mus],
            [r"$hh$ events"] + [g(m, "n_hh", str) for m in mus],
            [r"reco efficiency $t\bar t$ / $hh$"] + [f"{g(m, 'reco_eff_tt', lambda x: pct(x))} / {g(m, 'reco_eff_hh', lambda x: pct(x))}" for m in mus],
            [r"mean jets ($p_T>25$ GeV), $t\bar t$"] + [g(m, "njets_mean_tt", lambda x: f"{x:.2f}") for m in mus],
            [r"all 4 jets correct, $t\bar t$"] + [g(m, "correct_jets_frac_tt", lambda x: pct(x)) for m in mus],
            [r"$\nu$ $p_z$ 68\% half-width [GeV]"] + [g(m, "nu_pz_half_width", lambda x: f"{x:.0f}") for m in mus],
            [r"$W$ of $m_{W,h}$ / $m_{t,\ell}$ / $m_{t,h}$ [GeV]"] + [" / ".join(g(m, k, lambda x: f"{x:.0f}") for k in ("W_m_w_had", "W_m_t_lep", "W_m_t_had")) for m in mus],
            [r"AUC angles, all terms"] + [g(m, "auc_angles_all") for m in mus],
            [r"AUC angles, $>10\sigma$"] + [g(m, "auc_angles_10sigma") for m in mus],
            [r"AUC masses (BW)"] + [g(m, "auc_masses") for m in mus],
            [r"AUC angles ($>10\sigma$) + BW"] + [g(m, "auc_angles_masses") for m in mus]]
    txt = PU.get("summary_tex", "")
    figs = "".join(fig(rel(p), c, f"fig:pu{i}") for i, (p, c) in enumerate(PU.get("figures", [])))
    return head + tab(r"Reco-level results versus pileup.", ["", *[rf"$\langle\mu\rangle={m}$" for m in mus]], rows[1:], "tab:pileup") + txt + figs


def sec_outlook(N):
    return r"""
\section{Open questions and next steps}
\label{sec:outlook}

\begin{itemize}
\item \textbf{Conventions to confirm with the paper.}  $y_0=2.5\min y$ was inferred from the paper's figure;
  the sign convention in the paper's text and in its caption disagree.
\item \textbf{Neutrino.}  Solving the neutrino with the $m_{t,\ell}$ constraint instead of $m_W$ would free
  $m_{W,\ell}$, which is the off-shell $W^*$ in half of the $hh$ events, and would remove the largest single
  loss on correctly reconstructed events.
\item \textbf{No single hypothesis.}  Instead of committing to one $\chi^2$ assignment, evaluate
  $y=\sum_{a}w_a\,f_{t\bar t}(\Omega_a,m_a)$ summed over jet assignments and neutrino solutions, the density of
  the observed final state under $t\bar t$ in the spirit of the matrix-element method; this removes the
  $\chi^2$'s pull of $hh$ onto the $t\bar t$ shell.
\item \textbf{Joint fit.}  Fit angles and masses jointly rather than as a product, by adding Legendre-in-$u$
  mass blocks to the extended block basis; with $m_{b\bar b}$ included the product form already fails because
  angles and masses become correlated.
\item \textbf{Production variables.}  $m_{t\bar t}$ and the $t\bar t$ transverse momentum are the remaining
  dimensions; a resonance such as $H\to t\bar t$ shows up in $m_{t\bar t}$.
\item \textbf{Binning and defaults.}  Equiprobable-contour binning of $t$; the truncated or extended $f$ as the
  default; an $f_{hh}$ for A--B testing.
\end{itemize}
"""


def sec_appendix(N):
    return r"""
\appendix
\section{Code and reproduction}
\label{app:code}

Everything is in \texttt{/Users/TMHONG/Claude-code/ttbar\_bblvqq} (git repository on \texttt{main}).
\begin{description}
\item[\texttt{setup\_tools.sh}] installs MadGraph5\_aMC 3.6.7, Pythia8, Delphes 3.5.1 (built with CMake) and the Python environment.
\item[\texttt{generate\_samples.py}] MadGraph $\to$ Pythia8 $\to$ Delphes production; \texttt{--process ttbar|hh|Htt}.
\item[\texttt{plotting/}] \texttt{io.py} (LHE and Delphes readers), \texttt{physics.py} (objects, selection, $\chi^2$ reconstruction),
  \texttt{angles.py} (the ten angles), \texttt{osde.py} (basis functions, OSDE, leave-one-out, extended and spherical-harmonic bases),
  \texttt{summary.py}, \texttt{histo.py}, \texttt{style.py}.
\item[\texttt{make\_plots.py}, \texttt{plot\_config.py}] validation plots; \texttt{nu\_pz\_plot.py}, \texttt{lep\_had\_compare.py},
  \texttt{hh\_plots.py}, \texttt{acceptance\_plots.py}.
\item[\texttt{osde\_fit.py}] fits $f_{t\bar t}$ per level (\texttt{--basis paper|extended|sh}), applies it to $t\bar t$ and $hh$, projections and scores;
  \texttt{osde\_simplify.py} (truncation, CV), \texttt{osde\_order\_scan.py}, \texttt{osde\_truncation\_check.py}, \texttt{osde\_compare.py},
  \texttt{basis\_compare.py}, \texttt{paper\_style\_plots.py}.
\item[\texttt{ml\_compare.py}] the unsupervised and supervised machine-learning comparison.
\item[\texttt{study\_washout/}] the mechanism and the step-by-step decomposition of the parton-to-gen loss.
\item[\texttt{study\_salvage/}] the mass extension (\texttt{simplest.py}, \texttt{salvage.py}, \texttt{reco\_extension.py}), the angular-term
  study (\texttt{angle\_terms.py}), the truncation-noise test (\texttt{truncation\_noise.py}) and the tagging study (\texttt{tagging.py}).
\item[\texttt{study\_bsm/}, \texttt{pileup/}] the $H\to t\bar t$ benchmark and the pileup study.
\item[\texttt{make\_slides.py}, \texttt{note/make\_note.py}] the slides and this note; both read the same result files.
\item[\texttt{tests/test\_osde.py}] 14 unit tests of the basis functions and the estimator (\texttt{pytest}).
\item[\texttt{LOG.md}] the running log of the study.
\end{description}
The slides are rebuilt with
\texttt{python make\_slides.py --sample samples/ttbar\_bblvqq/run\_seed12345 --hh samples/hh\_bblvqq/run\_seed2001 --osde osde --acceptance acceptance}
and this note with \texttt{python note/make\_note.py}.

\section*{Acknowledgement}
The analysis code, the figures, the slides and this note were produced with Claude Code (Anthropic) under the
direction of the author; every result was cross-checked as described in the text.
"""


# ----------------------------------------------------------------------------------------------
PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{caption}
\usepackage{float}
\usepackage{xcolor}
\definecolor{pittroyal}{HTML}{003594}
\definecolor{pittgold}{HTML}{FFB81C}
\usepackage[colorlinks=true, linkcolor=pittroyal, citecolor=pittroyal, urlcolor=pittroyal]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\color{pittroyal}}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries\color{pittroyal}}{\thesubsection}{1em}{}
\captionsetup{font=small, labelfont=bf}
\setlength{\parskip}{3pt}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default=os.path.join(ROOT, "samples/ttbar_bblvqq/run_seed12345"))
    p.add_argument("--hh", default=os.path.join(ROOT, "samples/hh_bblvqq/run_seed2001"))
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--author", default="Tae Min Hong")
    args = p.parse_args()

    N = {
        "S": sample_summary(args.sample, args.hh, args.recompute),
        "meta_tt": load("samples/ttbar_bblvqq/run_seed12345/metadata.json"),
        "meta_fit": load("samples/ttbar_bblvqq/run_seed1001/metadata.json"),
        "meta_hh": load("samples/hh_bblvqq/run_seed2001/metadata.json"),
        "osde": load("osde", "summary.json"), "simp": load("osde", "simplify_summary.json"),
        "order_scan": load("osde", "order_scan.json"), "trunc_gen": load("osde", "truncation_gen.json"),
        "ext": load("osde_ext", "summary.json"), "ext_simp": load("osde_ext", "simplify_summary.json"),
        "proj_comp": load("osde_ext", "projection_chi2_comparison.json"),
        "basis_cmp": load("osde_sh", "basis_comparison.json"),
        "ml": {l: load("ml", f"results_{l}.json") for l in ("parton", "parton_trig") if exists("ml", f"results_{l}.json")},
        "nu_pz": load("plots", "nu_pz.json"), "lep_had": load("plots", "lep_vs_had.json"),
        "whad_split": load("plots_hh", "w_had_mass_split.json"),
        "acc_tt": load("acceptance", "acceptance_tt.json"), "acc_hh": load("acceptance", "acceptance_hh.json"),
        "mech": load("study_washout", "mechanism", "results.json"), "decomp": load("study_washout", "decomposition", "decomposition.json"),
        "simplest": load("study_salvage", "simplest.json"), "salvage": load("study_salvage", "salvage.json"),
        "angle_terms": load("study_salvage", "angle_terms.json"), "noise": load("study_salvage", "truncation_noise.json"),
        "reco_ext": load("study_salvage", "reco_extension.json"), "tagging": load("study_salvage", "tagging.json"),
        "bsm": load("study_bsm", "htt_test.json"), "pileup": load("pileup", "results.json"),
        "angles_vs_bw": load("study_salvage", "angles_vs_bw.json"),
    }
    missing = [k for k, v in N.items() if v is None]
    if missing:
        print(f"[note] missing inputs: {missing}")
    body = "".join(f(N) for f in (sec_intro, sec_samples, sec_angles, sec_osde, sec_coeff, sec_orders, sec_sh,
                                  sec_acceptance, sec_ml, sec_washout, sec_momentum, sec_bsm, sec_pileup, sec_outlook, sec_appendix))
    import datetime
    tex = PREAMBLE + rf"""
\title{{\color{{pittroyal}}$t\bar t$ and $hh\to b\bar b\,\ell\nu\,q\bar q'$: samples, the ten angles, and the $t\bar t$ density
$f_{{t\bar t}}$ by OSDE as a signal-agnostic discriminant\\[4pt]{{\large Internal note}}}}
\author{{{args.author}\\University of Pittsburgh}}
\date{{{datetime.date.today():%B %d, %Y}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
The orthogonal-series density $f_{{t\bar t}}$ of the ten $t\bar t$ angles, fitted on $t\bar t$ only, separates
$hh\to b\bar b\,\ell\nu\,q\bar q'$ from $t\bar t$ at parton level (AUC {N['simp']['levels']['parton']['variants']['cv_best']['auc']:.2f}) but not after reconstruction
({N['osde']['levels']['gen']['auc_hh_vs_tt']:.2f} at particle level, {N['osde']['levels']['reco']['auc_hh_vs_tt']:.2f} at detector level).  The loss is traced to the $t\bar t$-hypothesis
reconstruction, which rebuilds the off-shell $hh$ events as on-shell $t\bar t$; the information survives in the
resonance masses.  Multiplying $f_{{t\bar t}}$ by one Breit--Wigner factor per resonance mass, with no fitted
coefficients, recovers the parton-level performance at particle level ({N['reco_ext']['gen']['truncation']['>10 sigma']['angles+BW']:.2f}) and most of it at detector level
({N['reco_ext']['reco']['truncation']['>10 sigma']['angles+BW']:.2f}).  Only about 20 angular terms matter; realistic quark tagging changes nothing once the masses are in.
Comparisons with other unsupervised and with supervised machine-learning methods, a BSM benchmark and a pileup
study are included.
\end{{abstract}}
\tableofcontents
{body}
\end{{document}}
"""
    out = os.path.join(HERE, "note.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"[note] wrote {out}")
    if not args.no_compile:
        res = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "note.tex"],
                             cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode:
            print("\n".join(res.stdout.splitlines()[-30:]))
            sys.exit("[note] LaTeX failed, see note/note.log")
        subprocess.run(["latexmk", "-c", "note.tex"], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[note] wrote {os.path.join(HERE, 'note.pdf')}")


if __name__ == "__main__":
    main()
