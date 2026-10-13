#!/usr/bin/env python3
"""
Fig. 6 (fig:negprob) with Delta-R-matched reco jets: HH(170)/tt̄ density in the plane of the tt̄ quantiles of the
angular score (f_tt̄, LOO for tt̄) and the Breit-Wigner mass score; corner and single-score enrichments.

Run: OMP_NUM_THREADS=3 .venv/bin/python study_truthmatch/fig6_corner_matched.py [--fr 0.3] [--bins 5]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "paper_figs"), os.path.join(ROOT, "study_bsm"), HERE]
import make_paper_figs as M  # noqa: E402
from analyze import bw_score, quantile  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402
from wtb_reweight import weights  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--fr", type=float, default=0.3)
p.add_argument("--bins", type=int, default=5)
p.add_argument("--corner", type=float, default=0.2)
a = p.parse_args()

TT, HH = (np.load(os.path.join(HERE, f"data_{s}.npz")) for s in ("run_seed1001", "run_m170_all"))
ok_t, ok_h = TT["ok"], HH["ok"]
f = osde.load_density(os.path.join(HERE, "osde_dRmatch", "f_ttbar_reco.npz"))
yo_t = f.loo({k: TT[f"m_{k}"][ok_t] for k in ANGLE_NAMES})
yo_h = f({k: HH[f"m_{k}"][ok_h] for k in ANGLE_NAMES})
ym_t = bw_score(TT, ok_t, TT, ok_t, "m")
ym_h = bw_score(TT, ok_t, HH, ok_h, "m")
qo = {"tt": quantile(yo_t, yo_t), "hh": quantile(yo_t, yo_h)}
qm = {"tt": quantile(ym_t, ym_t), "hh": quantile(ym_t, ym_h)}
w_r = weights(TT["p_cos_theta_prime"][ok_t], TT["p_cos_theta_prime_tilde"][ok_t], a.fr)

def corner_plot(path, edges, R, corner, enrich, mark=M.RATIO_COLOR):
    """As make_paper_figs.fig5_plot, with the corner box and its enrichment in the annotation red (not ink), so they
    do not read as data on the amber cells."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("npad", [M.NAVY, "#FFFFFF", M.AMBER])
    norm = LogNorm(vmin=1 / 3, vmax=3)
    P, T, B, CB, TOP = 2.2, 0.78, 0.62, 0.14, 0.95
    W = T + P + 0.22
    H = B + P + 0.22 + CB + TOP
    fig = M.plt.figure(figsize=(W, H))
    ax = M.place(fig, T, B, P, P, W, H)
    im = ax.pcolormesh(edges, edges, R.T, cmap=cmap, norm=norm, rasterized=True)
    ax.plot([0, corner, corner], [corner, corner, 0], color=mark, lw=1.6, ls=(0, (4, 2)))
    ax.text(0.5 * corner, 0.5 * corner, rf"$\times{enrich:.1f}$", ha="center", va="center", color=mark)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$q_\Omega\ \ (\mathrm{angles})$")
    ax.set_ylabel(r"$q_m$", rotation=0, ha="right", va="center", labelpad=6)
    M.finish(ax, no_values_y=False)
    cax = M.place(fig, T, B + P + 0.22, P, CB, W, H)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[1 / 3, 1 / 2, 1, 2, 3])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.set_xticklabels([r"$\frac{1}{3}$", r"$\frac{1}{2}$", "$1$", "$2$", "$3$"])
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.6)
    cb.ax.set_title(r"$HH\ /\ t\bar{t}\ \mathrm{density}$", pad=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    M.plt.close(fig)
    return W


c, n_h = a.corner, len(qo["hh"])
in_t = (qo["tt"] < c) & (qm["tt"] < c)
in_h = (qo["hh"] < c) & (qm["hh"] < c)
ft, fh = in_t.mean(), in_h.mean()
fr_ = np.sum(w_r * in_t) / w_r.sum()


def enr(sel_h, acc):
    k = sel_h.sum()
    return float(k / n_h / acc), float(np.sqrt(k) / n_h / acc)


res = {"n_tt": int(len(qo["tt"])), "n_hh": int(n_h), "corner": c,
       "corner_tt": float(ft), "corner_hh": float(fh), "corner_enrichment_hh": enr(in_h, ft),
       "corner_enrichment_ttR": float(fr_ / ft),
       "masses_alone_same_tt": enr(qm["hh"] < ft, ft), "angles_alone_same_tt": enr(qo["hh"] < ft, ft),
       "corr_q_tt": float(np.corrcoef(qo["tt"], qm["tt"])[0, 1])}
edges = np.linspace(0, 1, a.bins + 1)
Ht = np.histogram2d(qo["tt"], qm["tt"], [edges, edges])[0] / len(qo["tt"])
Hh = np.histogram2d(qo["hh"], qm["hh"], [edges, edges])[0] / n_h
R = np.where(Ht > 0, Hh / np.maximum(Ht, 1e-12), np.nan)
res["width_in"] = corner_plot(os.path.join(HERE, "fig6_corner_recomatched"), edges, R, c, res["corner_enrichment_hh"][0])
json.dump(res, open(os.path.join(HERE, "fig6_corner_recomatched.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
