#!/usr/bin/env python3
"""
Paper figures for reco level with Delta-R-matched jets (study T), in the style of paper_figs/make_paper_figs.py.

  fig2_osde_projections_recomatched   Fig. 2: four angles, tt̄ (matched reco) with the f_tt̄ projection; tt̄_R,
                                      HH(170), hh(125) overlaid (matched reco too)
  fig3_discriminator_recomatched      Fig. 3: y and υ for tt̄ (LOO), tt̄_R, HH(170), hh(125)
  figA_levels_hh170                   appendix: HH(170) vs tt̄ at every level (parton, parton+trigger, gen chi2,
                                      reco chi2, reco matched): ROC, and the HH/tt̄ density ratio vs the tt̄ quantile q

Run: OMP_NUM_THREADS=3 .venv/bin/python study_truthmatch/paper_figs_matched.py [--fr 0.3]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "paper_figs"), os.path.join(ROOT, "study_bsm")]
import make_paper_figs as M  # noqa: E402  (also sets the paper's rcParams)
from make_paper_figs import plt  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402
from wtb_reweight import weights  # noqa: E402

SAMPLES = {"tt": "run_seed1001", "hh": "run_m170_all", "h1": "run_seed2001"}
HH_LBL = r"$HH_{170}$"    # short legend label
HH_C = M.AMBER          # the amber of the toy figure (compare_y.py LINE_J), now that hh(125) is not drawn


CV = os.path.join(ROOT, "study_cv")   # the one estimator: |c|/sigma > 2.5 (5-fold CV on tt̄), tt̄ cross-fitted
T_CV = json.load(open(os.path.join(CV, "cv_summary.json")))["threshold"]


def load(fit_dir):
    D = {k: np.load(os.path.join(HERE, f"data_{s}.npz")) for k, s in SAMPLES.items()}
    x = {k: {a: D[k][f"m_{a}"][D[k]["ok"]] for a in ANGLE_NAMES} for k in D}
    f_all = osde.load_density(os.path.join(fit_dir, "f_ttbar_reco.npz"))
    f = f_all.truncated(f_all.significance() > T_CV)                 # the projections: full-sample truncated fit
    z = np.load(os.path.join(CV, "scores_matched.npz"))               # the scores: cross-fitted (study_cv)
    y = {"tt": z["tt_cv"], "hh": z["HH170_cv"], "h1": z["hh125_cv"]}
    assert len(y["tt"]) == len(x["tt"]["Y"]) and len(y["hh"]) == len(x["hh"]["Y"]) and len(y["h1"]) == len(x["h1"]["Y"])
    ok = D["tt"]["ok"]
    w_r = weights(D["tt"]["p_cos_theta_prime"][ok], D["tt"]["p_cos_theta_prime_tilde"][ok], A.fr)
    return D, x, f, y, w_r


def tuck_ylabel(ax, fig):
    """The upright y-axis title at the spine minus (outward tick + labelpad), as without tick values: a lone 0 at
    the bottom of the axis does not push the title (at mid-height) outward."""
    d_in = (M.TICK_LEN + 6.0) / 72.0
    w_in = ax.get_position().width * fig.get_figwidth()
    ax.yaxis.set_label_coords(-d_in / w_in, 0.5)


def fig2(x, f, y, w_r, path, with_h1=False):
    """As make_paper_figs.fig2, with the matched-reco samples."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("tt", ["#ffffff", M.NAVY])
    ups = M.npad_upsilon({"tt": y["tt"], "x": y["h1"]}, {"tt": y["tt"]})[0]["tt"]
    edges = np.linspace(-1, 1, 25)
    u_bins = {a: np.histogram(x["tt"][a], edges, weights=ups)[0] / np.maximum(np.histogram(x["tt"][a], edges)[0], 1)
              for a, _ in M.ANG2}
    lo_u = min(v.min() for v in u_bins.values()); hi_u = max(v.max() for v in u_bins.values())
    grid = np.linspace(-1, 1, 401)
    P = M.PANEL_DATA
    T, GX, B, GY, TOP = 0.68, 0.10, 0.55, 0.60, 0.08
    W = T + P + GX + T + P + 0.12
    H = B + P + GY + P + TOP
    fig = plt.figure(figsize=(W, H))
    for i, (a, sym) in enumerate(M.ANG2):
        r, c = divmod(i, 2)
        ax = M.place(fig, T + c * (P + GX + T), B + (1 - r) * (P + GY), P, P, W, H)
        h_tt, _ = np.histogram(x["tt"][a], edges, density=True)
        h_hh, _ = np.histogram(x["hh"][a], edges, density=True)
        h_h1 = np.histogram(x["h1"][a], edges, density=True)[0] if with_h1 else np.zeros(len(edges) - 1)
        cols = cmap(0.15 + 0.55 * np.clip((u_bins[a] - lo_u) / (hi_u - lo_u), 0, 1))
        ax.bar(0.5 * (edges[1:] + edges[:-1]), h_tt, width=np.diff(edges), color=cols, edgecolor=cols, lw=0.3)
        ax.stairs(h_tt, edges, color=M.NAVY, lw=1.0)
        ax.plot(grid, f.projection(a, grid), color=M.INK, lw=1.2)
        ax.stairs(h_hh, edges, color=HH_C, lw=1.3)
        if with_h1:
            ax.stairs(h_h1, edges, color=M.HH125_C, lw=1.3)
        top = max(h_tt.max(), h_hh.max(), h_h1.max())
        low = min(0.0, float(f.projection(a, grid).min()))             # show where the fit dips below zero
        ax.set_xlim(-1, 1)
        ax.set_ylim(low - 0.04 * top if low < 0 else 0, (1.75 if i == 1 else 1.08) * top)
        if low < 0:
            ax.axhline(0, color="0.6", lw=0.6, zorder=0)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel(rf"$x = {sym}$")
        M.upright_ylabel(ax, M.dndx("x"))
        M.finish(ax)
        from matplotlib.ticker import FixedLocator, FixedFormatter
        ax.yaxis.set_major_locator(FixedLocator([0.0]))                # only the zero is labelled
        ax.yaxis.set_major_formatter(FixedFormatter(["$0$"]))
        ax.tick_params(axis="y", labelleft=True)
        tuck_ylabel(ax, fig)
    h = [plt.Rectangle((0, 0), 1, 1, fc=cmap(0.45), ec=M.NAVY, lw=1.0), plt.Line2D([], [], color=M.INK, lw=1.2),
         plt.Line2D([], [], color=HH_C, lw=1.3)]
    labs = [M.TT_L, r"$f_{t\bar{t}}$", HH_LBL]
    if with_h1:
        h.append(plt.Line2D([], [], color=M.HH125_C, lw=1.3)); labs.append(M.H1_L)
    fig.axes[1].legend(h, labs, loc="upper right", ncol=1, handlelength=1.4, handletextpad=0.4,
                       labelspacing=0.2, borderaxespad=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return W


def fig3_plot(path, panels):
    """As make_paper_figs.fig3_plot: y and υ side by side, tt̄ filled with the white-to-navy gradient in υ,
    HH(170) as a line; no tt̄_R."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("tt", ["#ffffff", M.NAVY])
    P = M.PANEL_DATA
    T, GX, B, RT, TOP = 0.75, 0.12, 0.62, 0.15, 0.12
    W = T + P + GX + T + P + RT
    H = B + P + TOP
    fig = plt.figure(figsize=(W, H))
    for col, (v, edges, sym, xlab, shade_bins) in enumerate(panels):
        ax = M.place(fig, T + col * (P + GX + T), B, P, P, W, H)
        clip = lambda u: np.clip(u, edges[0], edges[-1] - 1e-9 * (edges[-1] - edges[0]))
        wd = np.diff(edges)
        h = {k: np.histogram(clip(v[k]), edges)[0] / len(v[k]) for k in ("tt", "hh")}
        colors = cmap(0.08 + 0.92 * shade_bins)
        for i in range(len(wd)):
            ax.fill_between([edges[i], edges[i + 1]], 0, h["tt"][i] / wd[i], color=colors[i], lw=0)
        ax.stairs(h["tt"] / wd, edges, color=M.NAVY, lw=1.0)
        ax.stairs(h["hh"] / wd, edges, color=HH_C, lw=1.3)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, 1.08 * max((hh / wd).max() for hh in h.values()))
        ax.set_xlabel(xlab)
        M.upright_ylabel(ax, M.dndx(sym))
        M.finish(ax)
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "$0$" if abs(v) < 1e-12 else ""))   # only the zero
        ax.tick_params(axis="y", labelleft=True)
        tuck_ylabel(ax, fig)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=cmap(0.45), ec=M.NAVY, lw=1.0), plt.Line2D([], [], color=HH_C, lw=1.3)]
    fig.axes[0].legend(hs, [M.TT_L, HH_LBL], loc="upper right", bbox_to_anchor=(1.03, 1.0), handlelength=1.3,
                       handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return W


def fig3(y, w_r, path, with_h1=False):
    """As make_paper_figs.fig3 (extra=True), with the matched-reco scores."""
    y_tt, y_hh, y_h1 = y["tt"], y["hh"], y["h1"]
    n_eff_r = w_r.sum() ** 2 / (w_r ** 2).sum()
    t, _ = M.npad_upsilon({"tt": y_tt, "x": y_h1}, {"tt": y_tt, "hh": y_hh, "x": y_h1})
    expo = int(np.floor(np.log10(np.percentile(y_tt, 99))))
    scale = 10.0 ** expo
    ylo = np.percentile(np.r_[y_tt, y_hh], 0.05)
    yhi = np.percentile(np.r_[y_tt, y_hh], 99.7)
    y_edges = np.linspace(ylo - 0.05 * (yhi - ylo), yhi, 41) / scale
    t_edges = np.linspace(0, 1, 34)
    order = np.argsort(y_tt)
    ups_of_y = lambda yv: np.interp(yv * scale, y_tt[order], t["tt"][order])
    lo, hi = np.percentile(t["tt"], [5, 60])
    shade = lambda u: np.clip((u - lo) / (hi - lo), 0, 1)
    yc, tc = 0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (t_edges[1:] + t_edges[:-1])
    v_y = {"tt": y_tt / scale, "hh": y_hh / scale, **({"h1": y_h1 / scale} if with_h1 else {})}
    v_t = {"tt": t["tt"], "hh": t["hh"], **({"h1": t["x"]} if with_h1 else {})}
    panels = [(v_y, y_edges, "y", rf"$y\ \ [10^{{{expo}}}]$", shade(ups_of_y(yc))),
              (v_t, t_edges, r"\upsilon", r"$\upsilon$", shade(tc))]
    return fig3_plot(path, panels), n_eff_r


def roc_and_ratio(y_tt, y_s, q_edges):
    """Low y = anomalous.  ROC: tt̄ and signal efficiencies of y < cut; ratio: signal / tt̄ density in bins of the
    tt̄ quantile q of y (tt̄ is uniform in q)."""
    s = np.sort(y_tt)
    q_s = np.searchsorted(s, y_s, side="right") / len(s)
    eff_t = np.linspace(0, 1, 201)
    eff_s = np.array([np.mean(q_s <= e) for e in eff_t])
    h = np.histogram(np.clip(q_s, 0, 1 - 1e-9), q_edges)[0] / len(q_s) / np.diff(q_edges)
    err = np.sqrt(np.histogram(np.clip(q_s, 0, 1 - 1e-9), q_edges)[0]) / len(q_s) / np.diff(q_edges)
    return eff_t, eff_s, h, err


def auc(y_tt, y_s):
    x = np.r_[y_s, y_tt]
    r = x.argsort().argsort() + 1.0
    return 1 - (r[:len(y_s)].sum() - len(y_s) * (len(y_s) + 1) / 2) / (len(y_s) * len(y_tt))


def fig_levels(y_matched, path):
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("tt", ["#ffffff", M.NAVY])
    lv = [("parton_trig", "parton + trigger", "-", M.INK),
          ("gen", r"generator + $\chi^2$", (0, (5, 2)), blues(0.7)),
          ("reco", r"reconstructed + $\chi^2$", "-", blues(0.45)),
          ("matched", r"reco. truth-matched", "-", HH_C)]      # ditto: reconstructed, truth-matched
    q_edges = np.linspace(0, 1, 11)
    curves = {}
    for key, *_ in lv:
        sc = np.load(os.path.join(CV, f"scores_{key}.npz"))           # cross-fitted, |c|/sigma > 2.5 at every level
        yt, ys = sc["tt_cv"], sc["HH170_cv"]
        curves[key] = (*roc_and_ratio(yt, ys, q_edges), auc(yt, ys), len(ys))
    P = 3.0                                                     # one panel (the ROC curves bunch up), legend inside;
    T, GX, B, RT, TOP = 0.95, 0.55, 0.62, 0.2, 0.12             # set at 0.8 of a column the text prints at ~9 pt
    W = T + P + RT
    H = B + P + TOP
    fig = plt.figure(figsize=(W, H))
    ax2 = M.place(fig, T, B, P, P, W, H)
    qc = 0.5 * (q_edges[1:] + q_edges[:-1])
    for key, lab, ls, col in lv:
        eff_t, eff_s, h, err, a, _ = curves[key]
        lw = 1.6 if key in ("matched", "parton", "parton_trig") else 1.2
        ax2.plot([], [], color=col, ls=ls, lw=lw, label=lab)
        # histograms in bins of 0.1 in q, drawn as smooth (shape-preserving) curves through the bin centres
        from scipy.interpolate import PchipInterpolator
        qq = np.linspace(qc[0], qc[-1], 200)
        if key in ("parton", "parton_trig"):        # the reference: as a histogram
            ax2.stairs(h, q_edges, color=col, ls=ls, lw=lw, baseline=None)
        else:
            ax2.plot(qq, PchipInterpolator(qc, h)(qq), color=col, ls=ls, lw=lw)
        if key == "matched":
            ax2.fill_between(qq, PchipInterpolator(qc, h - err)(qq), PchipInterpolator(qc, h + err)(qq),
                             color=col, alpha=0.2, lw=0)
    ax2.legend(loc="upper right", handlelength=2.0, labelspacing=0.2, borderaxespad=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 3.0)
    ax2.set_xlabel(r"$t\bar{t}$ quantile $q$ of $y_{t\bar{t}}$")
    ax2.set_ylabel(r"Normalized $H\!H/t\bar{t}$ ratio")
    for ax in (ax2,):
        M.finish(ax, no_values_y=False)
    from matplotlib.ticker import FixedLocator, MultipleLocator
    ax2.yaxis.set_major_locator(FixedLocator([0, 1, 2, 3]))        # values 0, 1, 2, 3 only
    ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax2.set_yticklabels(["$0$", "$1$", "$2$", "$3$"])
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return {k: {"auc": float(v[4]), "n_hh": int(v[5]), "ratio_lowest_q_bin": float(v[2][0])} for k, v in curves.items()}, W


def main():
    global A
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fr", type=float, default=0.3)
    p.add_argument("--fit", default=os.path.join(HERE, "osde_dRmatch"))
    A = p.parse_args()
    D, x, f, y, w_r = load(A.fit)
    col = 0.95 * 3.375
    incl = lambda w: f"width={w * (col / M.DATA_WIDTH_IN) / 3.375:.3f}\\linewidth"
    W2 = fig2(x, f, y, w_r, os.path.join(HERE, "fig2_osde_projections_recomatched"))
    W3, neff = fig3(y, w_r, os.path.join(HERE, "fig3_discriminator_recomatched"))
    lev, WA = fig_levels(y, os.path.join(HERE, "figA_levels_hh170"))
    info = {"n": {k: int(len(v["Y"])) for k, v in x.items()}, "ttR_n_eff": float(neff),
            "auc_hh170": float(auc(y["tt"], y["hh"])), "auc_hh125": float(auc(y["tt"], y["h1"])),
            "auc_ttR": None, "levels_hh170": lev,
            "include": {"fig2": incl(W2), "fig3": incl(W3), "figA": incl(WA)}}
    json.dump(info, open(os.path.join(HERE, "paper_figs_matched.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
