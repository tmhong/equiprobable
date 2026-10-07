#!/usr/bin/env python3
"""
The paper's Fig. "tt_hh_dNdy_dNdt": distributions of y = f_tt̄(x) and of the discriminator t for
tt̄ and hh, with the ratio N_hh / (N_hh + N_tt̄) below, in the paper's style.

  y' = y - y0,  y0 = 2.5 * (most negative y)  (inferred from the paper figure: its y axis starts at y0 and
               t is populated continuously from 0, which needs y0 well below min y)
  s' = -ln y',  t = (s'_1 - s') / (s'_1 - s'_0),  s'_0 / s'_1 = 0.5% / 99.5% quantiles of s' (events beyond
  go into the first / last bin; the paper's min / max leave empty bins)  ->  t in [0, 1] with no empty bins,
  t -> 0: less tt̄-like.  y0, s'_0, s'_1 are taken over tt̄ and hh together, per level.

tt̄: the fit sample with leave-one-out scores (unbiased, high statistics); hh: the hh sample.
Both are normalized to unit area, so the ratio compares shapes.

Example:
  python paper_style_plots.py --dir osde          # paper basis   -> osde/tt_hh_dNdy_dNdt_<level>.{pdf,png}
  python paper_style_plots.py --dir osde_ext      # extended basis
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, FuncFormatter  # noqa: E402

TT_DARK = "#0a2f8f"
HH = "#e8a33d"
RATIO = "#d62728"
CMAP = LinearSegmentedColormap.from_list("tt", ["#ffffff", TT_DARK])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 15, "axes.labelsize": 17,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 3,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "axes.linewidth": 1.0,
})


def transform(scores, y0_factor=2.5, edge_quantile=0.005):
    """y -> t as in the paper.  y0 = y0_factor x (most negative y).  s'_0, s'_1 are the
    edge_quantile and 1 - edge_quantile quantiles of s' over all samples (the paper's min / max
    correspond to edge_quantile = 0, which lets single extreme events stretch the range and leave
    empty bins); events beyond them go into the first / last bin (t clipped to [0, 1])."""
    y0 = y0_factor * min(0.0, min(float(v.min()) for v in scores.values()))
    sp = {k: -np.log(v - y0) for k, v in scores.items()}
    allsp = np.concatenate(list(sp.values()))
    s0, s1 = float(np.quantile(allsp, edge_quantile)), float(np.quantile(allsp, 1 - edge_quantile))
    return {k: np.clip((s1 - v) / (s1 - s0), 0, 1) for k, v in sp.items()}, (y0, s0, s1)


def merged_ratio(edges, h_hh, h_tt, n_hh, n_tt, min_count=25):
    """hh/(hh+tt) of unit-normalized histograms, merging sparse bins (both samples >= min_count)."""
    out_e, out_r, out_err = [edges[0]], [], []
    acc_hh = acc_tt = 0
    for i in range(len(h_hh)):
        acc_hh += h_hh[i]
        acc_tt += h_tt[i]
        last = i == len(h_hh) - 1
        if (acc_hh >= min_count and acc_tt >= min_count) or last:
            a, b = acc_hh / n_hh, acc_tt / n_tt
            r = a / (a + b) if a + b > 0 else np.nan
            ea, eb = np.sqrt(acc_hh) / n_hh, np.sqrt(acc_tt) / n_tt
            err = np.hypot(b * ea, a * eb) / (a + b) ** 2 if a + b > 0 else np.nan
            if last and not (acc_hh >= min_count and acc_tt >= min_count) and out_r:
                out_e[-1] = edges[i + 1]           # merge the sparse tail into the previous bin
            else:
                out_e.append(edges[i + 1])
                out_r.append(r)
                out_err.append(err)
            acc_hh = acc_tt = 0
    return np.array(out_e), np.array(out_r), np.array(out_err)


def gradient_fill(ax, h, edges, colors):
    for i in range(len(h)):
        ax.fill_between([edges[i], edges[i + 1]], 0, h[i], color=colors[i], lw=0, step=None)


def panel(ax, rax, v_tt, v_hh, edges, t_of_bin, xlabel, xlim, show_legend):
    h_tt, _ = np.histogram(np.clip(v_tt, edges[0], edges[-1] - 1e-9 * (edges[-1] - edges[0])), edges)
    h_hh, _ = np.histogram(np.clip(v_hh, edges[0], edges[-1] - 1e-9 * (edges[-1] - edges[0])), edges)
    w = np.diff(edges)
    d_tt, d_hh = h_tt / len(v_tt) / w, h_hh / len(v_hh) / w
    # color gradient linear in t, same mapping in both panels (white at low t, dark at high t)
    lo, hi = np.percentile(t_of_bin["tt_ref"], [5, 60])
    shade = np.clip((t_of_bin["bins"] - lo) / (hi - lo), 0, 1)
    gradient_fill(ax, d_tt, edges, CMAP(0.08 + 0.92 * shade))
    ax.stairs(d_tt, edges, color=TT_DARK, lw=1.8)
    ax.stairs(d_hh, edges, color=HH, lw=2.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.12 * max(d_tt.max(), d_hh.max()))
    ax.tick_params(labelbottom=False)
    if show_legend:
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=CMAP(0.55), ec=TT_DARK, lw=1.8),
                           plt.Line2D([], [], color=HH, lw=2.2)], labels=[r"$t\bar{t}$", r"$hh$"],
                  frameon=False, loc="upper right", fontsize=16)
    e, r, er = merged_ratio(edges, h_hh, h_tt, len(v_hh), len(v_tt))
    rax.stairs(r + er, e, baseline=r - er, fill=True, color=RATIO, alpha=0.2, lw=0)
    rax.stairs(r, e, color=RATIO, lw=1.8)
    rax.set_xlim(*xlim)
    rax.set_ylim(0, 1)
    rax.set_yticks([0, 0.5, 1])
    rax.set_xlabel(xlabel)
    clean = FuncFormatter(lambda v, _: f"{v:g}" if abs(v) > 1e-12 else "0")   # 0, 0.5, 1 (no trailing zeros)
    for a in (ax, rax):
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.xaxis.set_major_formatter(clean)
        a.yaxis.set_major_formatter(clean)


def make(scores, level, out_base, title=None, y0_factor=2.5, edge_quantile=0.005):
    tt, hh = scores["train"], scores["hh"]          # tt̄: fit sample, leave-one-out scores
    t, (y0, s0, s1) = transform({"tt": tt, "hh": hh}, y0_factor, edge_quantile)
    expo = int(np.floor(np.log10(np.percentile(tt, 99))))
    scale = 10.0 ** expo
    ylo = y0                                       # the left axis starts at the shift point, as in the paper
    yhi = np.percentile(np.concatenate([tt, hh]), 99.7)
    y_edges = np.linspace(ylo, yhi, 61) / scale
    t_edges = np.linspace(0, 1, 51)

    def t_of_y(yv):
        return np.clip((s1 + np.log(yv * scale - y0)) / (s1 - s0), 0, 1)

    fig = plt.figure(figsize=(9.6, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.1, 1], hspace=0.06, wspace=0.18)
    axl, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    rl, rr = fig.add_subplot(gs[1, 0], sharex=axl), fig.add_subplot(gs[1, 1], sharex=axr)
    yc = 0.5 * (y_edges[1:] + y_edges[:-1])
    panel(axl, rl, tt / scale, hh / scale, y_edges,
          {"tt_ref": t["tt"], "bins": np.where(yc * scale > y0, t_of_y(np.maximum(yc, y0 / scale + 1e-30)), 0)},
          rf"$y\ \ [10^{{{expo}}}]$", (y_edges[0], y_edges[-1]), False)
    tc = 0.5 * (t_edges[1:] + t_edges[:-1])
    panel(axr, rr, t["tt"], t["hh"], t_edges, {"tt_ref": t["tt"], "bins": tc},
          "$t = -\\ln y$\nrescaled, flipped, shifted", (0, 1), True)
    axl.set_ylabel("Norm.  events")
    rl.set_ylabel(r"$\dfrac{hh}{hh+t\bar{t}}$")
    if title:
        axl.text(0.04, 0.95, title, transform=axl.transAxes, ha="left", va="top", fontsize=13)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return y0, s0, s1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="osde", help="osde_fit.py output directory (scores_<level>.npz)")
    p.add_argument("--levels", default="parton,gen,reco")
    p.add_argument("--edge-quantile", type=float, default=0.005,
                   help="s'_0, s'_1 = this and 1 - this quantile of s' (0 = the paper's min / max)")
    p.add_argument("--y0-factor", type=float, default=2.5,
                   help="y0 = factor x (most negative y). 2.5 reproduces the paper figure (its y axis starts at "
                        "y0 ~ 2.5 x min y and t is populated continuously from 0); 1 is singular")
    args = p.parse_args()
    for lvl in args.levels.split(","):
        path = os.path.join(args.dir, f"scores_{lvl}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        y0, s0, s1 = make({k: z[k] for k in z.files}, lvl, os.path.join(args.dir, f"tt_hh_dNdy_dNdt_{lvl}"),
                          title=f"{lvl} level", y0_factor=args.y0_factor, edge_quantile=args.edge_quantile)
        print(f"{args.dir} {lvl}: y0={y0:.3e}  s'0={s0:.2f}  s'1={s1:.2f}")


if __name__ == "__main__":
    main()
