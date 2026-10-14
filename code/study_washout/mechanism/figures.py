"""Figures for the washout mechanism study."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
# OUT_SUFFIX (e.g. "_h170"): signal data, results and figures in the sibling study_washout<SUFFIX>/mechanism (tt̄ data
# always from here); SIGNAL_LABEL: name of the signal in the figures.  Defaults: as before.
SUFFIX = os.environ.get("OUT_SUFFIX", "")
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "study_washout" + SUFFIX, "mechanism") if SUFFIX else HERE
SIG = os.environ.get("SIGNAL_LABEL", "hh")
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 17, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "legend.fontsize": 14, "lines.linewidth": 2.4,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
C_TT, C_HH, C_HH2 = "#1f4e9c", "#d1495b", "#e9a23b"
h = dict(np.load(os.path.join(OUT, "data_hh.npz"))); t = dict(np.load(os.path.join(HERE, "data_tt.npz")))
R = json.load(open(os.path.join(OUT, "results.json")))
for d in (h, t):
    d["all"] = np.ones(len(d["p_Y"]), bool); d["reco"] = d["g_reco_ok"].astype(bool); d["match"] = d["t_matched"].astype(bool)
off = h["p_m_w_had"] < h["p_m_w_lep"]
# one W off shell (h(125) -> WW*) or both on shell (m_H > 2 m_W): only the wording below depends on it
WSTAR = np.median(np.minimum(h["p_m_w_had"], h["p_m_w_lep"])) < 60

def hist(ax, x, e, **kw):
    x = x[np.isfinite(x)]
    c, _ = np.histogram(np.clip(x, e[0], e[-1] - 1e-9), e)
    ax.stairs(c / c.sum() / np.diff(e), e, **kw)

# ---------- Fig 1: parton-level signatures
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
e = np.linspace(-1, 1, 21)
A1 = R["auc1d"]["parton (LHE)"]
panels = [("cos_Theta", r"cos$\Theta$  (production)", f"gg→tt̄ peaks forward/backward;\n{SIG} pseudo-tops isotropic"),
          ("cos_theta", r"cos$\theta$  (W in leptonic 'top' frame)", f"{SIG}: {'W*' if WSTAR else 'W'} softer than the b\n→ W goes backward"),
          ("cos_theta_prime", r"cos$\theta'$  (lepton in W frame)", f"tt̄: V−A, no W helicity +1\n{SIG}: no such zero"),
          (None, r"m(Wb) of leptonic 'top' [GeV]", f"{SIG} pseudo-top is not a top")]
for ax, (a, xl, note) in zip(axs.flat, panels):
    if a is None:
        ee = np.linspace(50, 350, 31)
        hist(ax, t["p_m_t_lep"], ee, color=C_TT, label="tt̄")
        hist(ax, h["p_m_t_lep"], ee, color=C_HH, label=SIG)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 0.5)
    else:
        hist(ax, t[f"p_{a}"], e, color=C_TT, label="tt̄")
        hist(ax, h[f"p_{a}"], e, color=C_HH, label=f"{SIG} (all)")
        hist(ax, h[f"p_{a}"][~off], e, color=C_HH, ls=":", lw=2, label=f"{SIG}, leptonic W*" if WSTAR else f"{SIG}, m(ℓν) < m(qq̄′)")
        ax.set_ylim(0, None)
        ax.text(0.03, 0.04, f"1D AUC {A1[a]:.2f}", transform=ax.transAxes, fontsize=14)
    ax.set_xlabel(xl); ax.set_ylabel("density")
    ax.text(0.97, 0.96, note, transform=ax.transAxes, ha="right", va="top", fontsize=13)
axs[1, 1].legend(handles=axs[0, 0].get_legend_handles_labels()[0], labels=axs[0, 0].get_legend_handles_labels()[1], loc="upper left", fontsize=13, frameon=False)
fig.suptitle(f"Parton level: {SIG} is not tt̄-like in the decay and production angles", fontsize=18)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_parton_signatures.png"), dpi=130); plt.close(fig)

# ---------- Fig 2: chi2 pulls hh masses onto the tt̄ hypothesis
fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
e = np.linspace(0, 200, 41)
hist(axs[0], h["p_m_w_had"], e, color=C_HH, ls=":", label=f"{SIG} parton (true qq̄′)")
hist(axs[0], h["c_m_w_had"][h["reco"]], e, color=C_HH, label=f"{SIG} χ²-chosen jj")
hist(axs[0], t["c_m_w_had"][t["reco"]], e, color=C_TT, label="tt̄ χ²-chosen jj")
axs[0].axvline(80.4, color="0.4", lw=1, ls="--"); axs[0].set_xlabel(r"$m_{jj}$ (hadronic W) [GeV]"); axs[0].set_ylabel("density")
axs[0].legend(frameon=False, fontsize=13, loc="upper right")
e = np.linspace(50, 400, 36)
hist(axs[1], h["p_m_t_had"], e, color=C_HH, ls=":", label=f"{SIG} parton (W+b)")
hist(axs[1], h["c_m_t_had"][h["reco"]], e, color=C_HH, label=f"{SIG} χ²")
hist(axs[1], t["c_m_t_had"][t["reco"]], e, color=C_TT, label="tt̄ χ²")
axs[1].axvline(172.5, color="0.4", lw=1, ls="--"); axs[1].set_xlabel(r"$m_{jjb}$ (hadronic top) [GeV]")
eh, et = R["eff"]["hh"], R["eff"]["tt"]
axs[1].text(0.97, 0.62, f"χ² picks the true 4 jets:\n  {SIG} {100*eh['chi2 picks right 4 jets | reco_ok']:.0f}%   tt̄ {100*et['chi2 picks right 4 jets | reco_ok']:.0f}%",
            transform=axs[1].transAxes, ha="right", fontsize=13)
fig.suptitle(f"The χ² builds W and top masses that {SIG} does not have" if WSTAR else
             f"χ² masses: {SIG} has real W bosons; its m(Wb) is a broad bump near m_t", fontsize=18)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_chi2_mass_pull.png"), dpi=130); plt.close(fig)

# ---------- Fig 3: angles morph toward tt̄ step by step
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
e = np.linspace(-1, 1, 11)
for ax, a, xl in zip(axs, ["cos_Theta", "cos_theta_prime", "cos_theta"],
                     [r"cos$\Theta$", r"cos$\theta'$ (lepton)", r"cos$\theta$ (leptonic side)"]):
    hist(ax, t[f"c_{a}"][t["reco"]], e, color=C_TT, lw=3, label="tt̄, χ²")
    hist(ax, h[f"t_{a}"][h["match"]], e, color=C_HH2, ls="--", label=f"{SIG}, true jets + true ν")
    hist(ax, h[f"n_{a}"][h["match"]], e, color=C_HH, ls=":", label=f"{SIG}, true jets + m_W-ν")
    hist(ax, h[f"c_{a}"][h["reco"]], e, color=C_HH, label=f"{SIG}, χ²")
    ax.set_xlabel(xl); ax.set_ylim(0, None)
axs[0].set_ylabel("density"); axs[0].legend(frameon=False, fontsize=12, loc="upper center")
fig.suptitle(f"The m_W-constrained ν and the χ² make {SIG} angles tt̄-like" if WSTAR else
             f"The χ² reconstruction pushes {SIG} angles toward tt̄", fontsize=18)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_angle_morphing.png"), dpi=130); plt.close(fig)

# ---------- Fig 4: separation (supervised BDT AUC on the ten angles) step by step + hypothesis-free
B = R["bdt"]
rows = [("parton (LHE)", "parton (LHE)"), ("after shower recoil", "parton after shower recoil"),
        ("+ gen selection", "+ gen selection (reco_ok)"), ("true jets + true ν", "truth-matched jets + true ν"),
        ("true jets + m_W-ν", "truth jets + m_W-constrained ν"), ("χ² reconstruction", "χ² (gen)")]
fig, ax = plt.subplots(figsize=(11, 6))
y = np.arange(len(rows) + 2)[::-1]
vals = [B[k] for _, k in rows] + [B["gen: hypothesis-independent observables (reco_ok events)"], B["gen: χ² angles + χ² masses + chi2"]]
labs = [l for l, _ in rows] + ["gen, hypothesis-free obs.\n(m_bb, ΔR_bb, MET, m_T…)", "gen, χ² angles + masses"]
cols = [C_HH] * len(rows) + ["#2a9d8f", "#2a9d8f"]
ax.barh(y, np.array(vals) - 0.5, left=0.5, color=cols, height=0.6)
for yy, v in zip(y, vals):
    ax.text(v + 0.005, yy, f"{v:.2f}", va="center", fontsize=14)
ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0.5, 0.95)
ax.set_xlabel(f"AUC, {SIG} vs tt̄ (supervised BDT, 3-fold CV)")
ax.text(0.945, y[2] if vals[2] < 0.79 else y[5], f"for reference, unsupervised\nf_tt̄ AUC: parton {R['f_auc']['parton']:.2f}\ngen χ² {R['f_auc']['gen']:.2f}", ha="right", va="center", fontsize=13)
ax.set_title("Ten angles lose separation along the chain; gen level still separates", fontsize=17)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_auc_chain.png"), dpi=130); plt.close(fig)

from analyze_util import auc1d
# ---------- Fig 5: hypothesis-independent gen observables
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, v, e, xl in zip(axs, ["h_m_bb", "h_dR_bb", "h_met"],
                        [np.linspace(0, 400, 41), np.linspace(0, 5, 26), np.linspace(0, 300, 31)],
                        [r"$m_{bb}$ (2 b-jets) [GeV]", r"$\Delta R_{bb}$", r"$E_T^{miss}$ [GeV]"]):
    hist(ax, t[v][t["reco"]], e, color=C_TT, label="tt̄")
    hist(ax, h[v][h["reco"]], e, color=C_HH, label=SIG)
    ax.set_xlabel(xl); ax.set_ylim(0, None)
    ax.text(0.97, 0.95, f"1D AUC {auc1d(h[v][h['reco']], t[v][t['reco']], e[0], e[-1]):.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=14)
axs[0].set_ylabel("density"); axs[0].legend(frameon=False, loc="center right")
fig.suptitle(f"Gen level, same events: hypothesis-free observables still separate {SIG} from tt̄", fontsize=17)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_hypothesis_free.png"), dpi=130); plt.close(fig)
print("done")
