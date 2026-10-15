"""Summary figure of decomposition.json."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# results/analysis/study_washout/decomposition, or with OUT_SUFFIX (e.g. "_h170") the sibling results/analysis/study_washout<SUFFIX>/decomposition:
# read decomposition.json and write the figures there; SIGNAL_LABEL: legend name
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # code/
import repo_paths
SUFFIX = os.environ.get("OUT_SUFFIX", "")
OUT = repo_paths.results_dir("study_washout" + SUFFIX, "decomposition")
SIG = os.environ.get("SIGNAL_LABEL", "hh")
DJ = json.load(open(os.path.join(OUT, "decomposition.json")))
R = DJ["steps"]
FM = {s: 100 * DJ["info"][s]["matched_given_reco_ok"] for s in ("tt", "hh")}   # % of reco_ok events with all 4 partons matched
ANG = ["Y", "cos_Theta", "cos_theta", "phi", "cos_theta_prime", "phi_prime",
       "cos_theta_tilde", "phi_tilde", "cos_theta_prime_tilde", "phi_prime_tilde"]
ALAB = ["Y", "cosΘ", "cosθ", "φ", "cosθ′", "φ′", "cosθ̃", "φ̃", "cosθ̃′", "φ̃′"]
MAIN = [("S0 LHE parton, all", "S0\nLHE parton\nall"),
        ("S1 LHE parton, reco_ok", "S1\n+ gen\nselection"),
        ("S1c post-ISR parton, reco_ok", "S1c\n+ ISR\nrecoil"),
        ("S1d pre-hadronization (FSR), reco_ok", "S1d\n+ FSR\n(partons)"),
        ("S2chi chi2 jets + gen lep + true nu", "S2χ\nχ² jets\n+ true ν"),
        ("S4 gen chi2 reconstruction", "S4\nfull gen\n(MET ν)")]
SUB = [("S1c_m post-ISR parton, reco_ok & matched", "post-ISR\nparton"),
       ("R_lep post-ISR + gen lepton", "+gen ℓ\nonly"),
       ("R_b post-ISR + matched b jets", "+b jets\nonly"),
       ("R_q post-ISR + matched light jets", "+light\njets only"),
       ("S2 matched jets + gen lep + true nu", "S2 all\njets+ℓ,\ntrue ν"),
       ("S3a S2 + nu pz from mW (true nu pT)", "S3a\nν pz from\nm_W"),
       ("S3 S2 + nu from MET + mW", "S3\nν from\nMET"),
       ("S2chi_m (matched events)", "χ² jets\n+ true ν"),
       ("S4m (matched events)", "S4\nfull gen")]
C = {"bdt": "#2a78d6", "loo": "#eb6834", "pf": "#1baf7a"}
YTOP = 0.92 if max(R[k]["bdt"][0] for k, _ in MAIN + SUB) < 0.89 else 1.0   # legend headroom if the BDT runs high
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], width_ratios=[1, 1.35], hspace=0.45, wspace=0.18)


def chain(ax, steps, title):
    x = np.arange(len(steps))
    bdt = [R[k]["bdt"][0] for k, _ in steps]; be = [R[k]["bdt"][1] for k, _ in steps]
    loo = [R[k]["osde"]["auc_loo"] for k, _ in steps]; le = [R[k]["osde"]["err"] for k, _ in steps]
    pf = [R[k]["osde"]["auc_parton_f"] for k, _ in steps]
    ax.errorbar(x, bdt, be, color=C["bdt"], lw=2, marker="o", ms=8, capsize=3, label="supervised BDT (10 angles)")
    ax.errorbar(x, loo, le, color=C["loo"], lw=2, marker="s", ms=8, capsize=3, label="f_tt̄ fit at this step (LOO)")
    ax.plot(x, pf, color=C["pf"], lw=2, marker="^", ms=8, ls="--", label="f_tt̄ fit at parton level, applied here")
    ax.axhline(0.5, color="#888", lw=1, ls=":")
    ax.set_xticks(x, [l for _, l in steps], fontsize=11)
    ax.set_ylabel(f"AUC ({SIG} vs tt̄)")
    ax.set_ylim(0.46, YTOP)
    ax.grid(axis="y", color="#ddd", lw=0.8)
    ax.set_title(title, fontsize=14, loc="left")
    for i, k in enumerate(steps):
        ax.text(i, 0.472, f"{R[k[0]]['n_tt']/1000:.0f}k / {R[k[0]]['n_hh']/1000:.1f}k", ha="center", fontsize=9, color="#555")


ax1 = fig.add_subplot(gs[0, 0])
chain(ax1, MAIN, "(a) main chain (S1→S4 on gen reco_ok events)")
ax1.legend(loc="upper right", fontsize=11, frameon=False)
ax2 = fig.add_subplot(gs[0, 1])
chain(ax2, SUB, "(b) reco_ok events with all 4 partons matched to distinct GenJets\n"
                f"     ({FM['tt']:.0f}% of tt̄, {FM['hh']:.0f}% of {SIG})")

ax3 = fig.add_subplot(gs[1, :])
M = np.array([[R[k]["single"][a] for a in ANG] for k, _ in MAIN])
im = ax3.imshow(M.T, cmap="Blues", vmin=0.5, vmax=0.66, aspect="auto")
ax3.set_yticks(range(len(ANG)), ALAB)
ax3.set_xticks(range(len(MAIN)), [l.replace("\n", " ") for _, l in MAIN], fontsize=10)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax3.text(i, j, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=10,
                 color="white" if M[i, j] > 0.6 else "black")
ax3.set_title("(c) single-angle AUC (BDT on one angle) along the main chain", fontsize=14, loc="left")
cb = fig.colorbar(im, ax=ax3, fraction=0.025, pad=0.01); cb.set_label("AUC")
fig.text(0.01, 0.005, f"numbers at the bottom: n(tt̄) / n({SIG}) events.  Error bars: Hanley–McNeil (half-split test sizes).",
         fontsize=10, color="#555")
fig.savefig(os.path.join(OUT, "decomposition_summary.png"), dpi=130, bbox_inches="tight")
print("ok")


# ---- slide versions: one message per figure, large fonts
plt.rcParams.update({"font.size": 18})
for name, steps, title in (("slide_chain_main", MAIN, "selected events (main chain)"),
                           ("slide_chain_matched", SUB, f"events with all 4 partons matched to jets ({FM['tt']:.0f}% tt̄, "
                                                        f"{FM['hh']:.0f}% {SIG})")):
    f, ax = plt.subplots(figsize=(13, 6.6))
    chain(ax, steps, title)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_title(title, loc="left", fontsize=19)
    for t in ax.texts:
        t.set_fontsize(12)
    ax.legend(loc="upper right", fontsize=16, frameon=True, framealpha=0.9)
    f.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
f, ax = plt.subplots(figsize=(15, 7.4))
im = ax.imshow(M.T, cmap="Blues", vmin=0.5, vmax=0.66, aspect="auto")
ax.set_yticks(range(len(ANG)), ALAB)
ax.set_xticks(range(len(MAIN)), [l for _, l in MAIN], fontsize=15)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(i, j, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=15,
                color="white" if M[i, j] > 0.6 else "black")
f.colorbar(im, ax=ax, fraction=0.03, pad=0.01).set_label("single-angle AUC")
f.savefig(os.path.join(OUT, "slide_angles.png"), dpi=110, bbox_inches="tight")
print("slides ok")
