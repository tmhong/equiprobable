#!/usr/bin/env python3
"""The missing directions: resonance masses for tt̄ and hh, parton (top) and gen χ² (bottom),
with the one-curve-per-mass tt̄ model (Breit-Wigner at parton, Cauchy with tt̄ median/half-IQR at gen)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from plotting import style  # noqa: E402

tt = np.load(os.path.join(ROOT, "study_washout/mechanism/data_tt.npz"))
hh = np.load(os.path.join(ROOT, "study_washout/mechanism/data_hh.npz"))
COLS = [("m_w_had", r"$m_{W,\mathrm{had}}$ [GeV]", (0, 160), 80.4, 2.085),
        ("m_t_lep", r"$m_{t,\mathrm{lep}}$ [GeV]", (50, 350), 172.5, 1.42),
        ("m_t_had", r"$m_{t,\mathrm{had}}$ [GeV]", (50, 350), 172.5, 1.42)]
style.use()
for lvl, pre, mk, title in (("parton", "p_", None, "parton level: tt̄ sits on the mass shell (Breit–Wigner), hh does not"),
                            ("gen", "c_", "g_reco_ok", r"gen level ($\chi^2$): shell smeared by resolution; hh still wider")):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (v, xl, rng, M, G) in zip(axs, COLS):
        xt = tt[pre + v] if mk is None else tt[pre + v][tt[mk].astype(bool)]
        xh = hh[pre + v] if mk is None else hh[pre + v][hh[mk].astype(bool)]
        bins = np.linspace(*rng, 61)
        ax.hist(xt, bins, density=True, histtype="step", lw=2.6, color="#1f4e9c", label=r"$t\bar t$")
        ax.hist(xh, bins, density=True, histtype="step", lw=2.6, color="#d1495b", label=r"$hh$")
        x = np.linspace(*rng, 6000)
        if lvl == "parton":
            y = 1 / ((x ** 2 - M ** 2) ** 2 + M ** 2 * G ** 2) * x
            lab = r"Breit–Wigner ($M$, $\Gamma$ from PDG)"
        else:
            Mm = np.median(xt); q1, q3 = np.percentile(xt, [25, 75]); W = 0.5 * (q3 - q1)
            y = 1 / ((x - Mm) ** 2 + W ** 2)
            lab = rf"Cauchy: $M$={Mm:.0f}, $W$={W:.0f} GeV"
        frac = np.mean((xt > rng[0]) & (xt < rng[1]))
        y = y / np.trapezoid(y, x) * frac
        ax.plot(x, y, color="0.2", ls="--", lw=1.8, label=lab)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 300 * ax.get_ylim()[1])      # headroom on the log axis for the legend
        ax.set_xlim(*rng)
        ax.set_xlabel(xl)
        ax.legend(fontsize=13.5, frameon=True, framealpha=0.9, loc="upper center", ncol=1)
    axs[0].set_ylabel("unit area")
    fig.suptitle(title.replace("tt̄", r"$t\bar t$"))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"masses_{lvl}.png"), dpi=110)
print("ok")
