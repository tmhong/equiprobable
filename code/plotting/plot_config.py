"""
Plot definitions for make_plots.py.  Edit freely.

Each entry in PLOTS is a dict:
  name     output file name (and default x label)
  levels   which event levels to draw, any of "parton", "parton_trig" (parton events whose
           lepton passes the offline single-lepton cuts), "gen", "reco".
           All requested levels for all samples are overlaid on one canvas.
  expr     function ev -> values with one entry per event: a number, or
           several per event (jagged like ev.jets.pt, or an (nevents, k) array
           like np.stack([ev.b.pt, ev.bbar.pt], axis=1)).  None/NaN are dropped.
           For a level-dependent expression pass a dict {level: function}.
  bins     (nbins, low, high) or explicit list of edges
  select   optional: name of a boolean attribute (e.g. "presel", "reco_ok")
           or a function ev -> bool mask.  Can also be a {level: ...} dict.
           Events failing it are removed before expr is called, so expr can
           safely index e.g. ev.ljets[:, 1] after requiring >= 2 light jets.
  optional: xlabel, ylabel, logy, ratio (default True), ratio_range,
            overflow (default True), show_mean (default True), density, title

What is available on `ev` (see lib/physics.py):
  parton: t_lep, t_had (= W + b; for hh the pairing that best fits tt̄), w_lep, w_had, b, bbar, b_lep, b_had,
          lep, nu, quarks (pt-ordered), ttbar, lep_charge, lep_flavour
  gen & reco: jets, bjets, ljets, leptons, lep (leading), met, met_phi, mtw,
          njets, nbjets, nleptons, presel (1 lep, >=4 jets, >=1 b)
          and the chi2 ttbar reconstruction (NaN unless presel):
          reco_ok, reco_chi2, w_had, t_had, b_had, w_lep, t_lep, b_lep, nu, ttbar
  reco only: electrons, muons, jets.flavor (truth flavour of reco jets)
  all levels, the ten angles of the paper (NaN unless a tt̄ system exists):
          Y, cos_Theta, cos_theta, phi, cos_theta_prime, phi_prime,
          cos_theta_tilde, phi_tilde, cos_theta_prime_tilde, phi_prime_tilde
          and the W_had daughters q_down, q_up (gen/reco: also q1, q2 by pT)
  All four-vectors have .pt .eta .phi .mass .E .px .py .pz .rapidity,
  .deltaR(other), and can be added: (ev.jets[:, 0] + ev.jets[:, 1]).mass
"""
import awkward as ak
import numpy as np

# Override object cuts here if you like (defaults in lib/physics.py):
CUTS = {
    # "jet_pt": 30.0,
    # "reco_max_jets": 5,
}

# The ten angles of the paper (lib/angles.py):
#   had_fermion:    which W_had daughter defines the primed hadronic angles,
#                   "down" (paper: the analogue of the charged lepton) or "up"
#   jet_assignment: how gen/reco decide which W_had jet is the down-type quark:
#                   "truth" (match to the LHE quarks; the paper assumes up/down
#                   tagging), "softer" or "harder" (lower/higher-pT jet)
ANGLES = {"had_fermion": "down", "jet_assignment": "truth"}

ALL = ["parton", "parton_trig", "gen", "reco"]
GR = ["gen", "reco"]

PLOTS = [
    # ---------------- top / W masses: parton vs particle vs reco -------------
    dict(name="top_had_mass", levels=ALL, ratio=False, select={"gen": "reco_ok", "reco": "reco_ok"},
         expr=lambda ev: ev.t_had.mass, bins=(60, 50, 350),
         xlabel=r"$m_{t,\mathrm{had}}$ [GeV]"),
    dict(name="top_lep_mass", levels=ALL, ratio=False, select={"gen": "reco_ok", "reco": "reco_ok"},
         expr=lambda ev: ev.t_lep.mass, bins=(60, 50, 350),
         xlabel=r"$m_{t,\mathrm{lep}}$ [GeV]"),
    dict(name="w_had_mass", levels=ALL, ratio=False, select={"gen": "reco_ok", "reco": "reco_ok"},
         expr=lambda ev: ev.w_had.mass, bins=(60, 0, 200),
         xlabel=r"$m_{W,\mathrm{had}}$ [GeV]"),
    dict(name="top_had_pt", levels=ALL, select={"gen": "reco_ok", "reco": "reco_ok"},
         expr=lambda ev: ev.t_had.pt, bins=(40, 0, 600), logy=True,
         xlabel=r"$p_T^{t,\mathrm{had}}$ [GeV]"),
    dict(name="ttbar_mass", levels=ALL, select={"gen": "reco_ok", "reco": "reco_ok"},
         expr=lambda ev: ev.ttbar.mass, bins=(50, 300, 1500), logy=True,
         xlabel=r"$m_{t\bar{t}}$ [GeV]"),
    dict(name="reco_chi2", levels=GR, select="reco_ok",
         expr=lambda ev: ev.reco_chi2, bins=(50, 0, 50), logy=True, xlabel=r"$\chi^2_{\min}$"),

    # ---------------- leptons / MET -----------------------------------------
    dict(name="lep_pt", levels=ALL, expr=lambda ev: ev.lep.pt, bins=(50, 0, 300),
         xlabel=r"Leading lepton $p_T$ [GeV]"),
    dict(name="lep_eta", levels=ALL, expr=lambda ev: ev.lep.eta, bins=(50, -3, 3),
         xlabel=r"Leading lepton $\eta$"),
    dict(name="lep_flavour", levels=GR, select="presel",
         expr=lambda ev: ev.lep.flavour, bins=(4, 10, 14), xlabel="Lepton |PDG id| (11=e, 13=$\\mu$)",
         show_mean=False),
    dict(name="nleptons", levels=GR, expr=lambda ev: ev.nleptons, bins=(5, -0.5, 4.5),
         xlabel="Number of leptons", show_mean=False),
    dict(name="met", levels=GR + ["parton"],
         expr={"parton": lambda ev: ev.nu.pt, "gen": lambda ev: ev.met, "reco": lambda ev: ev.met},
         bins=(50, 0, 400), xlabel=r"$E_T^{\mathrm{miss}}$ (parton: $p_T^\nu$) [GeV]"),
    dict(name="mtw", levels=GR, select="presel", expr=lambda ev: ev.mtw, bins=(40, 0, 200),
         xlabel=r"$m_T^W$ [GeV]"),

    # ---------------- jets --------------------------------------------------
    dict(name="njets", levels=GR, expr=lambda ev: ev.njets, bins=(11, -0.5, 10.5),
         xlabel="Number of jets", show_mean=False),
    dict(name="nbjets", levels=GR, expr=lambda ev: ev.nbjets, bins=(5, -0.5, 4.5),
         xlabel="Number of b-jets", show_mean=False),
    dict(name="jet1_pt", levels=GR, expr=lambda ev: ev.jets[:, :1].pt, bins=(50, 0, 500),
         xlabel=r"Leading jet $p_T$ [GeV]"),
    dict(name="jet_pt_all", levels=GR, expr=lambda ev: ev.jets.pt, bins=(50, 0, 400), logy=True,
         xlabel=r"Jet $p_T$ (all jets) [GeV]"),
    dict(name="jet_eta_all", levels=GR, expr=lambda ev: ev.jets.eta, bins=(50, -2.5, 2.5),
         xlabel=r"Jet $\eta$ (all jets)"),
    dict(name="bjet_pt", levels=["parton"] + GR,
         expr={"parton": lambda ev: np.stack([ev.b.pt, ev.bbar.pt], axis=1),
               "gen": lambda ev: ev.bjets.pt, "reco": lambda ev: ev.bjets.pt},
         bins=(50, 0, 400), xlabel=r"b-quark / b-jet $p_T$ [GeV]"),
    dict(name="mbb", levels=ALL, ratio=False,
         select={"gen": lambda ev: ak.num(ev.bjets) >= 2, "reco": lambda ev: ak.num(ev.bjets) >= 2},
         expr={"parton": lambda ev: (ev.b + ev.bbar).mass,
               "gen": lambda ev: (ev.bjets[:, 0] + ev.bjets[:, 1]).mass,
               "reco": lambda ev: (ev.bjets[:, 0] + ev.bjets[:, 1]).mass},
         bins=(50, 0, 400), xlabel=r"$m_{b\bar b}$ (gen/reco: two leading b-jets) [GeV]"),
    dict(name="reco_jet_truth_flavour", levels=["reco"],
         expr=lambda ev: ev.jets.flavor, bins=(23, -0.5, 22.5),
         xlabel="Reco jet truth flavour (0/21=g, 1-4=light/c, 5=b)", show_mean=False),
    dict(name="mjj_leading_light", levels=GR, select=lambda ev: ak.num(ev.ljets) >= 2,
         expr=lambda ev: (ev.ljets[:, 0] + ev.ljets[:, 1]).mass, bins=(50, 0, 400),
         xlabel=r"$m_{jj}$ of two leading non-b jets [GeV]"),

    # ---------------- the ten angles of the paper ----------------------------
    *[dict(name=name, levels=ALL, select={"gen": "reco_ok", "reco": "reco_ok"},
           expr=(lambda ev, a=a: ev[a]), bins=b, xlabel=lab, ratio_range=(0.5, 1.5))
      for name, a, b, lab in [
          ("angle_prod_Y", "Y", (40, -3, 3), r"$Y$ ($t\bar{t}$ rapidity)"),
          ("angle_prod_cosTheta", "cos_Theta", (20, -1, 1), r"$\cos\Theta$ ($t_\mathrm{lep}$ in $t\bar{t}$ frame)"),
          ("angle_lep_costheta", "cos_theta", (20, -1, 1), r"$\cos\theta$ ($W$ in $t_\mathrm{lep}$ frame)"),
          ("angle_lep_phi", "phi", (20, -np.pi, np.pi), r"$\phi$ ($W$ in $t_\mathrm{lep}$ frame)"),
          ("angle_lep_costheta_prime", "cos_theta_prime", (20, -1, 1), r"$\cos\theta'$ ($\ell$ in $W$ frame)"),
          ("angle_lep_phi_prime", "phi_prime", (20, -np.pi, np.pi), r"$\phi'$ ($\ell$ in $W$ frame)"),
          ("angle_had_costheta", "cos_theta_tilde", (20, -1, 1), r"$\cos\tilde\theta$ ($W$ in $t_\mathrm{had}$ frame)"),
          ("angle_had_phi", "phi_tilde", (20, -np.pi, np.pi), r"$\tilde\phi$ ($W$ in $t_\mathrm{had}$ frame)"),
          ("angle_had_costheta_prime", "cos_theta_prime_tilde", (20, -1, 1), r"$\cos\tilde\theta'$ (down-type $q$ in $W$ frame)"),
          ("angle_had_phi_prime", "phi_prime_tilde", (20, -np.pi, np.pi), r"$\tilde\phi'$ (down-type $q$ in $W$ frame)"),
      ]],

    # ---------------- parton-level only -------------------------------------
    dict(name="parton_top_rapidity", levels=["parton"],
         expr=lambda ev: np.stack([ev.t_lep.rapidity, ev.t_had.rapidity], axis=1),
         bins=(50, -4, 4), xlabel="Top quark rapidity"),
    dict(name="parton_dR_qq", levels=["parton"],
         expr=lambda ev: ev.quarks[:, 0].deltaR(ev.quarks[:, 1]), bins=(50, 0, 5),
         xlabel=r"$\Delta R(q, q')$ from $W_\mathrm{had}$"),
    dict(name="parton_lep_charge", levels=["parton"], expr=lambda ev: ev.lep_charge,
         bins=(3, -1.5, 1.5), xlabel="Lepton charge", show_mean=False),
]
