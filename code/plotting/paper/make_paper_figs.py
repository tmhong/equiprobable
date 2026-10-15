#!/usr/bin/env python3
"""
Figures 2 and 3 of the letter (Overleaf NPAD/main.tex), at one-column size (3.4 in), Pitt colours.

Fig. 2 (fig:OSDE): reco-level tt̄ (fit sample after the reco selection, 24k) with the one-dimensional projections of
  the OSDE fit f_tt̄ (results/osde/f_ttbar_reco.npz), for four of the ten angles, with the two BSM benchmarks overlaid (unit area):
    HH(170)  = samples/hh_bblvqq/run_m170_seed2170,
    tt̄R     = the same tt̄ events reweighted to a right-handed W helicity fraction F_R (code/study_bsm/wtb_reweight.py).
  Angles: cos Θ (production), cos θ (W in the top frame), cos θ' (lepton in the W frame), cos θ̃' (down-type quark in
  the hadronic W frame): HH(170) differs most in cos Θ and cos θ, tt̄R only in cos θ' and cos θ̃'.
Fig. 3 (fig:p): reco level, y = f_tt̄(x) and υ (called t before: rescaled, flipped, shifted -ln y, as paper_style_plots.py) for tt̄
  (fit sample, leave-one-out scores), tt̄R (the same events, weighted) and HH(170), with signal/(signal+tt̄) below.

Fig. 5 (fig:negprob): reco level, the two tt̄ scores υ_Ω (angles, f_tt̄) and υ_m (resonance masses, product of
  Cauchy factors) for tt̄, tt̄R and HH(170); the corner where both are small is where new physics sits.

Run: OMP_NUM_THREADS=3 .venv/bin/python code/paper_figs/make_paper_figs.py [--fr 0.3]
     -> results/paper_figs/fig2_*, fig3_*, fig5_* .pdf / .png
     (first run reads the samples, ~4 min, and caches the arrays in results/paper_figs/cache.npz)
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path[:0] = [ROOT, os.path.join(ROOT, "study_bsm")]
import repo_paths  # noqa: E402
OUT = repo_paths.results_dir("paper_figs", create=False)    # the figures and the cache (created by main)

# ---- the NPAD paper style, copied from ../paper/compare_y.py (Figs. 4-5 of the letter) ----------------------------
# one font size (14 pt) for all text, Computer Modern (all text is written as mathtext), square panels of fixed size in
# inches, outward ticks with minor ticks, no tick values on the density axes, (1/N) dN/d... titles upright on the
# left, ratios on a right-hand axis.  Figures are ~5.7-6 in wide like Fig. 4, so at 0.95 of a column the text
# prints at ~7.5 pt.
FONT = 14
NAVY, AMBER, GRAY, INK = "#003094", "#e3a43a", "#7f7f7f", "#0b0b0b"     # compare_y.py: COLOR_I, LINE_J, GRAY, INK
FILL_I, RATIO_COLOR = "#ccd5e9", "#d62728"
CYAN = "#17BECF"
HH125_C, HH170_C = AMBER, CYAN    # hh(125) amber as the hh of Fig. 5 (compare_y.py); HH(170) cyan
PANEL = 1.8                                                            # in, side of each square plot
TICK_LEN = 4.5
REF_WIDTH_IN = 5.66                                                    # Fig. 4 (compare_lny.pdf)
PANEL_DATA, DATA_WIDTH_IN = 2.05, 6.04       # Fig. 5 (tt_f_theta_phi_t.pdf): PANEL + PANEL_GROW, and its width
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": True,
    "font.size": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
    "axes.labelsize": FONT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "axes.labelpad": 2, "xtick.major.size": TICK_LEN, "ytick.major.size": TICK_LEN,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.top": False, "ytick.right": False,
    "legend.frameon": False, "savefig.dpi": 300,
})
TT_L = r"$t\bar{t}$"
HH_L, H1_L = r"$HH\ (170)$", r"$hh\ (125)$"


def ttr_label(fr):
    return rf"$t\bar{{t}}_R\ (F_R={fr:g})$"


def dndx(sym):
    """(1/N) dN/d<sym> as an upright title, as compare_y.py writes it."""
    sep = r"\," if sym.startswith(("\\cos", "\\ln")) else ""
    return rf"$\dfrac{{1}}{{N}}\dfrac{{\mathrm{{d}}N}}{{\mathrm{{d}}{sep}{sym}}}$"


def place(fig, x_in, y_in, w_in, h_in, W, H):
    """An axes at (x, y) inches from the lower-left corner, w x h inches, in a W x H inch figure."""
    return fig.add_axes([x_in / W, y_in / H, w_in / W, h_in / H])


def compact(ax_axis):
    """Tick values as Fig. 4 writes them: 0, 0.5, 1, -1 (no trailing zeros), in Computer Modern."""
    from matplotlib.ticker import FuncFormatter
    ax_axis.set_major_formatter(FuncFormatter(lambda v, _: f"${0 if abs(v) < 1e-12 else v:g}$"))


def finish(ax, no_values_y=True):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    compact(ax.xaxis)
    compact(ax.yaxis)
    if no_values_y:
        ax.tick_params(axis="y", labelleft=False)


def upright_ylabel(ax, text):
    ax.set_ylabel(text, rotation=0, ha="right", va="center", labelpad=6)


ANG2 = [("cos_Theta", r"\cos\Theta"), ("cos_theta", r"\cos\theta"),
        ("cos_theta_prime", r"\cos\theta'"), ("cos_theta_prime_tilde", r"\cos\tilde\theta'")]
FIT_SAMPLE, TT_INDEP = "samples/ttbar_bblvqq/run_seed1001", "samples/ttbar_bblvqq/run_seed12345"
HH170 = "samples/hh_bblvqq/run_m170_seed2170"
HH125 = "samples/hh_bblvqq/run_seed2001"
OPTS = {"had_fermion": "down", "jet_assignment": "truth"}


MASSES = ("m_w_had", "m_t_lep", "m_t_had")     # chi2-reconstructed masses (m_W,lep is fixed by the nu solution)
CACHE_VERSION = 4


def load_inputs(cache):
    if os.path.exists(cache):
        z = np.load(cache)
        if "version" in z.files and int(z["version"]) == CACHE_VERSION:
            return {k: z[k] for k in z.files}
    import make_plots
    from plotting.angles import ANGLE_NAMES
    out = {}
    fit, _ = make_plots.load_sample("fit", os.path.join(repo_paths.REPO, FIT_SAMPLE), ["parton", "reco"], None, OPTS)
    hh, _ = make_plots.load_sample("hh170", os.path.join(repo_paths.REPO, HH170), ["parton", "reco"], None, OPTS)
    for a in ANGLE_NAMES:
        out[f"tt_p_{a}"] = np.asarray(fit["parton"][a], float)
        out[f"hh_p_{a}"] = np.asarray(hh["parton"][a], float)
    out["tt_p_trig"] = np.asarray(fit["parton"].lep_trigger, bool)     # the parton_trig level of osde_fit.py
    out["hh_p_trig"] = np.asarray(hh["parton"].lep_trigger, bool)
    h1, _ = make_plots.load_sample("hh125", os.path.join(repo_paths.REPO, HH125), ["parton"], None, OPTS)
    for a in ANGLE_NAMES:
        out[f"h1_p_{a}"] = np.asarray(h1["parton"][a], float)
    out["h1_p_trig"] = np.asarray(h1["parton"].lep_trigger, bool)
    # reco selection of the fit sample, as osde_fit.py (reco_ok and all ten angles finite), and the parton
    # angles of the same events (the tt̄R weights)
    mass = {"m_w_had": lambda e: e.w_had.mass, "m_t_lep": lambda e: e.t_lep.mass, "m_t_had": lambda e: e.t_had.mass}
    for key, smp in (("tt", fit), ("hh", hh)):
        e = smp["reco"]
        ok = np.asarray(e.reco_ok, bool) & np.all([np.isfinite(np.asarray(e[a], float)) for a in ANGLE_NAMES], 0)
        out[f"{key}_reco_ok"] = ok
        for a in ANGLE_NAMES:
            out[f"{key}_r_{a}"] = np.asarray(e[a], float)[ok]
        for m in MASSES:
            out[f"{key}_r_{m}"] = np.asarray(mass[m](e), float)[ok]
    out["reco_ok"] = out["tt_reco_ok"]
    out["version"] = np.array(CACHE_VERSION)
    np.savez_compressed(cache, **out)
    return out


def fig2(D, fr, path, level="parton_trig"):
    """Four angles: tt̄ (filled, navy outline) with the f_tt̄ projection (ink), tt̄R (gray dashed), HH(170) (amber),
    hh(125) (cyan); unit area.  parton_trig: parton angles of the events that pass the lepton trigger."""
    from plotting import osde
    from plotting.angles import ANGLE_NAMES
    from wtb_reweight import weights
    f = osde.load_density(os.path.join(repo_paths.RESULTS, "osde", f"f_ttbar_{level}.npz"))

    def sel(key):
        ok = np.all([np.isfinite(D[f"{key}_p_{a}"]) for a in ANGLE_NAMES], 0)
        return ok & D[f"{key}_p_trig"] if level == "parton_trig" else ok
    if level == "reco":
        x = {k: {a: D[f"{k}_r_{a}"] for a in ANGLE_NAMES} for k in ("tt", "hh")}
        ok_tt, x["h1"] = D["reco_ok"], None
    else:
        x = {k: {a: D[f"{k}_p_{a}"][sel(k)] for a in ANGLE_NAMES} for k in ("tt", "hh", "h1")}
        ok_tt = sel("tt")
    w_r = weights(D["tt_p_cos_theta_prime"][ok_tt], D["tt_p_cos_theta_prime_tilde"][ok_tt], fr)   # parton angles
    # tt̄ bars shaded by the mean υ of their events, on Fig. 3's white-to-navy scale (as Fig. 5 shades its
    # projections), so the legend swatch is the same as Fig. 3's
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("tt", ["#ffffff", NAVY])
    sc = np.load(os.path.join(repo_paths.RESULTS, "osde", f"scores_{level}.npz"))
    assert len(sc["train"]) == len(x["tt"]["Y"]), "tt̄ scores and events are not aligned"
    ups = npad_upsilon({"tt": sc["train"], "x": sc["hh"]}, {"tt": sc["train"]})[0]["tt"]
    edges = np.linspace(-1, 1, 25)
    u_bins = {a: np.histogram(x["tt"][a], edges, weights=ups)[0] / np.maximum(np.histogram(x["tt"][a], edges)[0], 1)
              for a, _ in ANG2}
    lo_u = min(v.min() for v in u_bins.values()); hi_u = max(v.max() for v in u_bins.values())
    grid = np.linspace(-1, 1, 401)
    P = PANEL_DATA                                             # the panel side of Fig. 5 (compare_y.py, event files)
    T, GX, B, GY, TOP = 0.68, 0.10, 0.55, 0.60, 0.08           # inches: left titles, column gap, bottom, row gap, top
    W = T + P + GX + T + P + 0.12
    H = B + P + GY + P + TOP
    fig = plt.figure(figsize=(W, H))
    for i, (a, sym) in enumerate(ANG2):
        r, c = divmod(i, 2)
        ax = place(fig, T + c * (P + GX + T), B + (1 - r) * (P + GY), P, P, W, H)
        h_tt, _ = np.histogram(x["tt"][a], edges, density=True)
        h_r, _ = np.histogram(x["tt"][a], edges, weights=w_r, density=True)
        h_hh, _ = np.histogram(x["hh"][a], edges, density=True)
        cols = cmap(0.15 + 0.55 * np.clip((u_bins[a] - lo_u) / (hi_u - lo_u), 0, 1))   # light to mid navy
        ax.bar(0.5 * (edges[1:] + edges[:-1]), h_tt, width=np.diff(edges), color=cols, edgecolor=cols, lw=0.3)
        ax.stairs(h_tt, edges, color=NAVY, lw=1.0)
        ax.plot(grid, f.projection(a, grid), color=INK, lw=1.2)
        ax.stairs(h_r, edges, color=GRAY, lw=1.2, ls=(0, (4, 2)))
        ax.stairs(h_hh, edges, color=HH170_C, lw=1.3)
        top = max(h_tt.max(), h_r.max(), h_hh.max())
        if x["h1"] is not None:
            h_h1, _ = np.histogram(x["h1"][a], edges, density=True)
            ax.stairs(h_h1, edges, color=HH125_C, lw=1.3)
            top = max(top, h_h1.max())
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, (1.75 if i == 1 else 1.08) * top)      # cos θ panel: room for the legend at the top right
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel(rf"$x = {sym}$")                        # as Fig. 4: define x on each axis, then dN/dx
        upright_ylabel(ax, dndx("x"))
        finish(ax)
    h = [plt.Rectangle((0, 0), 1, 1, fc=cmap(0.45), ec=NAVY, lw=1.0), plt.Line2D([], [], color=INK, lw=1.2),
         plt.Line2D([], [], color=GRAY, lw=1.2, ls=(0, (4, 2))), plt.Line2D([], [], color=HH170_C, lw=1.3)]
    labs = [TT_L, r"$f_{t\bar{t}}$", r"$t\bar{t}_R$", HH_L]                 # the caption gives F_R and "projection"
    if x["h1"] is not None:
        h.append(plt.Line2D([], [], color=HH125_C, lw=1.3)); labs.append(H1_L)
    fig.axes[1].legend(h, labs, loc="upper right", ncol=1, handlelength=1.4, handletextpad=0.4,
                       labelspacing=0.2, borderaxespad=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return int(ok_tt.sum()), W


def ratio_s_over_b(edges, h_s, h_b, n_s, n_b, min_count=25):
    """S/tt̄ of unit-area histograms, merging sparse bins (both >= min_count effective events)."""
    oe, orr = [edges[0]], []
    a_s = a_b = 0.0
    for i in range(len(h_s)):
        a_s += h_s[i]; a_b += h_b[i]
        last = i == len(h_s) - 1
        enough = a_s * n_s >= min_count and a_b * n_b >= min_count
        if enough or last:
            if last and not enough and orr:
                oe[-1] = edges[i + 1]
            else:
                oe.append(edges[i + 1]); orr.append(a_s / a_b if a_b > 0 else np.nan)
            a_s = a_b = 0.0
    return np.array(oe), np.array(orr)


def npad_upsilon(ref, vals, room=1.0, ends=(0.001, 0.995)):
    """υ exactly as ../paper/compare_y.py defines it for Fig. 5: y' = y + y_min with y_min = room x |lowest y of
    the reference samples|, rounded up to two figures; s' = -ln y'; υ = (s'_1 - s')/(s'_1 - s'_0) clipped to [0, 1],
    with s'_0, s'_1 the `ends` quantiles of s' pooled over the reference samples (Fig. 5: tt̄ and hh(125)).
    Events below -y_min (none in the reference samples) get υ = 0, the least tt̄-like."""
    lowest = min(float(np.min(v)) for v in ref.values())
    y_min = 0.0
    if lowest < 0:
        raw = room * -lowest
        tenth = 10.0 ** (np.floor(np.log10(raw)) - 1)
        y_min = float(np.ceil(raw / tenth) * tenth)
    pooled = np.concatenate([-np.log(v + y_min) for v in ref.values()])
    s0, s1 = np.quantile(pooled[np.isfinite(pooled)], ends)
    out = {}
    for k, v in vals.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            u = (s1 + np.log(v + y_min)) / (s1 - s0)
        out[k] = np.clip(np.nan_to_num(u, nan=0.0), 0, 1)
    return out, y_min


def transform(vals, y0_factor=2.5, edge_q=0.005):
    """y -> t as in paper_style_plots.transform (y0 from the most negative y; s' edges at the 0.5% / 99.5%
    quantiles over the unweighted samples)."""
    y0 = y0_factor * min(0.0, min(float(v.min()) for v in vals.values()))
    sp = {k: -np.log(v - y0) for k, v in vals.items()}
    allsp = np.concatenate(list(sp.values()))
    s0, s1 = np.quantile(allsp, edge_q), np.quantile(allsp, 1 - edge_q)
    return {k: np.clip((s1 - v) / (s1 - s0), 0, 1) for k, v in sp.items()}, y0


def fig3_plot(path, fr, panels, w_r, n_eff_r, extra):
    """y and υ panels side by side: tt̄ filled with the white-to-navy gradient in υ, signals as lines; the legend
    inside the y panel (no ratio: too busy)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("tt", ["#ffffff", NAVY])
    P = PANEL_DATA                                             # the panel side of Fig. 5
    T, GX, B, RT, TOP = 0.75, 0.12, 0.62, 0.15, 0.12           # inches: left title, gap, bottom, right, top
    W = T + P + GX + T + P + RT
    H = B + P + TOP
    fig = plt.figure(figsize=(W, H))
    for col, (v, edges, sym, xlab, shade_bins) in enumerate(panels):
        ax = place(fig, T + col * (P + GX + T), B, P, P, W, H)
        clip = lambda u: np.clip(u, edges[0], edges[-1] - 1e-9 * (edges[-1] - edges[0]))
        wd = np.diff(edges)
        h = {"tt": np.histogram(clip(v["tt"]), edges)[0] / len(v["tt"]),
             "ttR": np.histogram(clip(v["tt"]), edges, weights=w_r)[0] / w_r.sum(),
             "hh": np.histogram(clip(v["hh"]), edges)[0] / len(v["hh"])}
        if extra:
            h["h1"] = np.histogram(clip(v["h1"]), edges)[0] / len(v["h1"])
        colors = cmap(0.08 + 0.92 * shade_bins)
        for i in range(len(wd)):
            ax.fill_between([edges[i], edges[i + 1]], 0, h["tt"][i] / wd[i], color=colors[i], lw=0)
        ax.stairs(h["tt"] / wd, edges, color=NAVY, lw=1.0)
        ax.stairs(h["ttR"] / wd, edges, color=GRAY, lw=1.2, ls=(0, (4, 2)))
        ax.stairs(h["hh"] / wd, edges, color=HH170_C, lw=1.3)
        if extra:
            ax.stairs(h["h1"] / wd, edges, color=HH125_C, lw=1.3)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, 1.08 * max((hh / wd).max() for hh in h.values()))
        ax.set_xlabel(xlab)
        upright_ylabel(ax, dndx(sym))
        finish(ax)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=cmap(0.45), ec=NAVY, lw=1.0),
          plt.Line2D([], [], color=GRAY, lw=1.2, ls=(0, (4, 2))), plt.Line2D([], [], color=HH170_C, lw=1.3)]
    labs = [TT_L, r"$t\bar{t}_R$", HH_L]                       # F_R is given in the caption
    if extra:
        hs.append(plt.Line2D([], [], color=HH125_C, lw=1.3)); labs.append(H1_L)
    fig.axes[0].legend(hs, labs, loc="upper right", bbox_to_anchor=(1.03, 1.0), handlelength=1.3,
                       handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)   # nudged right, off the peak
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return W


def fig3(D, fr, path, level="reco", extra=False):
    from plotting import osde
    from wtb_reweight import weights
    sc = np.load(os.path.join(repo_paths.RESULTS, "osde_h170", f"scores_{level}.npz"))
    y_tt, y_hh = sc["train"], sc["hh"]           # tt̄: fit sample, leave-one-out; HH(170)
    # alignment check: the stored LOO scores are those of the reco-selected fit-sample events, in order
    f = osde.load_density(os.path.join(repo_paths.RESULTS, "osde_h170", f"f_ttbar_{level}.npz"))
    from plotting.angles import ANGLE_NAMES
    if level == "reco":
        ok, x_tt = D["reco_ok"], {a: D[f"tt_r_{a}"] for a in ANGLE_NAMES}
    else:                                        # parton_trig: parton angles of the events passing the lepton trigger
        ok = D["tt_p_trig"] & np.all([np.isfinite(D[f"tt_p_{a}"]) for a in ANGLE_NAMES], 0)
        x_tt = {a: D[f"tt_p_{a}"][ok] for a in ANGLE_NAMES}
    y_re = f(x_tt)
    assert len(y_re) == len(y_tt), (len(y_re), len(y_tt))
    corr = np.corrcoef(y_re, y_tt)[0, 1]
    w_r = weights(D["tt_p_cos_theta_prime"][ok], D["tt_p_cos_theta_prime_tilde"][ok], fr)
    n_eff_r = w_r.sum() ** 2 / (w_r ** 2).sum()
    # optional hh(125) (osde/: the same tt̄-only fit and identical tt̄ scores as osde_h170/)
    y_x = np.load(os.path.join(repo_paths.RESULTS, "osde", f"scores_{level}.npz"))["hh"] if extra else None
    # υ as in Fig. 5 (compare_y.py), with its parameters from tt̄ and hh(125), so υ is the same number in both
    y_ref = y_x if extra else np.load(os.path.join(repo_paths.RESULTS, "osde", f"scores_{level}.npz"))["hh"]
    t, y0 = npad_upsilon({"tt": y_tt, "x": y_ref}, {"tt": y_tt, "hh": y_hh, "x": y_ref})
    expo = int(np.floor(np.log10(np.percentile(y_tt, 99))))
    scale = 10.0 ** expo
    ylo = np.percentile(np.r_[y_tt, y_hh], 0.05)
    yhi = np.percentile(np.r_[y_tt, y_hh], 99.7)
    y_edges = np.linspace(ylo - 0.05 * (yhi - ylo), yhi, 41) / scale
    t_edges = np.linspace(0, 1, 34)
    # the gradient is linear in υ, the same mapping in both panels (υ is monotonic in y)
    order = np.argsort(y_tt)
    ups_of_y = lambda yv: np.interp(yv * scale, y_tt[order], t["tt"][order])
    lo, hi = np.percentile(t["tt"], [5, 60])
    shade = lambda u: np.clip((u - lo) / (hi - lo), 0, 1)
    yc, tc = 0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (t_edges[1:] + t_edges[:-1])
    panels = [({"tt": y_tt / scale, "hh": y_hh / scale, **({"h1": y_x / scale} if extra else {})}, y_edges, "y",
               rf"$y\ \ [10^{{{expo}}}]$", shade(ups_of_y(yc))),
              ({"tt": t["tt"], "hh": t["hh"], **({"h1": t["x"]} if extra else {})}, t_edges, r"\upsilon",
               r"$\upsilon$", shade(tc))]
    W = fig3_plot(path, fr, panels, w_r, n_eff_r, extra)
    return corr, n_eff_r, len(y_tt), len(y_hh), W


def upsilon(vals, weights=None, shift=True, edge_q=0.005):
    """υ = (s1 - s') / (s1 - s0), s' = -ln(y - y0): 0 = unlikely to be tt̄, as in Fig. 3 (y0 = 2.5 x the most negative
    y when shift, for the OSDE density that can go negative; none for a positive density)."""
    y0 = 2.5 * min(0.0, min(float(v.min()) for v in vals.values())) if shift else 0.0
    sp = {k: -np.log(v - y0) for k, v in vals.items()}
    allsp = np.concatenate(list(sp.values()))
    s0, s1 = np.quantile(allsp, edge_q), np.quantile(allsp, 1 - edge_q)
    return {k: np.clip((s1 - v) / (s1 - s0), 0, 1) for k, v in sp.items()}


def fig5(D, fr, path, corner=0.20):
    """Reco level: the two tt̄ scores, angular (υ_Ω from f_tt̄) and resonance masses (υ_m from the Cauchy factors),
    for tt̄, tt̄R and HH(170); new physics sits in the corner where both are small."""
    from scipy.ndimage import gaussian_filter
    from plotting import osde
    from plotting.angles import ANGLE_NAMES
    from wtb_reweight import weights
    f = osde.load_density(os.path.join(repo_paths.RESULTS, "osde_h170", "f_ttbar_reco.npz"))
    sc = np.load(os.path.join(repo_paths.RESULTS, "osde_h170", "scores_reco.npz"))
    yo_tt = sc["train"]                                             # leave-one-out, aligned (checked in fig3)
    yo_hh = f({a: D[f"hh_r_{a}"] for a in ANGLE_NAMES})
    hh_check = float(np.max(np.abs(yo_hh - sc["hh"]))) if len(sc["hh"]) == len(yo_hh) else np.nan
    ym = {"tt": np.ones(len(yo_tt)), "hh": np.ones(len(yo_hh))}
    for m in MASSES:
        x = D[f"tt_r_{m}"]
        M = np.median(x); q1, q3 = np.percentile(x, [25, 75]); W = 0.5 * (q3 - q1)
        for k in ym:
            ym[k] = ym[k] * W / np.pi / ((D[f"{k}_r_{m}"] - M) ** 2 + W ** 2)
    # each score as its tt̄ quantile: the fraction of tt̄ that is less tt̄-like, so tt̄ is uniform and (0,0) is the
    # corner where an event is in the tt̄ tail of both scores
    def q(ref, x):
        return np.searchsorted(np.sort(ref), x, side="right") / len(ref)
    qo = {"tt": q(yo_tt, yo_tt), "hh": q(yo_tt, yo_hh)}
    qm = {"tt": q(ym["tt"], ym["tt"]), "hh": q(ym["tt"], ym["hh"])}
    ok = D["reco_ok"]
    w_r = weights(D["tt_p_cos_theta_prime"][ok], D["tt_p_cos_theta_prime_tilde"][ok], fr)

    edges = np.linspace(0, 1, 11)
    H_tt = np.histogram2d(qo["tt"], qm["tt"], [edges, edges])[0] / len(yo_tt)
    sig = {"ttR": (np.histogram2d(qo["tt"], qm["tt"], [edges, edges], weights=w_r)[0] / w_r.sum(),
                   rf"$t\bar t_R$ ($F_R={fr:g}$)"),
           "hh": (np.histogram2d(qo["hh"], qm["hh"], [edges, edges])[0] / len(yo_hh), r"$HH$ (170)")}
    H, lab = sig["hh"]
    frac = {"tt": np.mean((qo["tt"] < corner) & (qm["tt"] < corner)),
            "ttR": np.sum(w_r * ((qo["tt"] < corner) & (qm["tt"] < corner))) / w_r.sum(),
            "hh": np.mean((qo["hh"] < corner) & (qm["hh"] < corner))}
    R = np.where(H_tt > 0, H / np.maximum(H_tt, 1e-12), np.nan)
    W5 = fig5_plot(path, edges, R, corner, frac["hh"] / frac["tt"])
    uo, um = qo, qm
    return frac, hh_check, float(np.corrcoef(uo["tt"], um["tt"])[0, 1]), W5


def fig5_plot(path, edges, R, corner, enrich):
    """One square panel: HH density / tt̄ density in the plane of the two tt̄ quantiles, colour bar on top."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("npad", [NAVY, "#FFFFFF", AMBER])
    norm = LogNorm(vmin=1 / 3, vmax=3)
    P, T, B, CB, TOP = 2.2, 0.78, 0.62, 0.14, 0.95             # inches: panel, left title, bottom, colour bar, above it
    W = T + P + 0.22
    H = B + P + 0.22 + CB + TOP
    fig = plt.figure(figsize=(W, H))
    ax = place(fig, T, B, P, P, W, H)
    im = ax.pcolormesh(edges, edges, R.T, cmap=cmap, norm=norm, rasterized=True)
    ax.plot([0, corner, corner], [corner, corner, 0], color=INK, lw=1.2, ls=(0, (4, 2)))
    ax.text(0.5 * corner, 0.5 * corner, rf"$\times{enrich:.1f}$", ha="center", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$q_\Omega\ \ (\mathrm{angles})$")
    ax.set_ylabel(r"$q_m$", rotation=0, ha="right", va="center", labelpad=6)
    finish(ax, no_values_y=False)
    cax = place(fig, T, B + P + 0.22, P, CB, W, H)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[1 / 3, 1 / 2, 1, 2, 3])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.set_xticklabels([r"$\frac{1}{3}$", r"$\frac{1}{2}$", "$1$", "$2$", "$3$"])
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.6)
    cb.ax.set_title(r"$HH\ /\ t\bar{t}\ \mathrm{density}$", pad=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return W



def printed_pt(width_in, include_in):
    """Text size in print when a figure width_in wide is set include_in wide."""
    return FONT * include_in / width_in


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fr", type=float, default=0.3, help="right-handed W helicity fraction of tt̄R")
    args = p.parse_args()
    repo_paths.results_dir("paper_figs")                       # creates results/paper_figs/
    D = load_inputs(os.path.join(OUT, "cache.npz"))
    col = 0.95 * 3.375                                         # \includegraphics[width=0.95\linewidth]
    scale5 = col / DATA_WIDTH_IN                                # Fig. 5's print scale: match its panels and text
    incl = lambda w: f"width={w * scale5 / 3.375:.3f}\\linewidth (panels {PANEL_DATA * scale5:.2f} in, text {FONT * scale5:.1f} pt)"
    n2, W2 = fig2(D, args.fr, os.path.join(OUT, "fig2_osde_projections_partontrig"), "parton_trig")
    print(f"fig2 parton+trig: {n2} tt̄ events; {W2:.2f} in wide -> include at {incl(W2)}")
    c2, ne2, nt2, nh2, W3 = fig3(D, args.fr, os.path.join(OUT, "fig3_discriminator_partontrig"), "parton_trig",
                                 extra=True)
    print(f"fig3 parton+trig: {nt2} tt̄ (LOO) / tt̄R n_eff {ne2:.0f} / {nh2} HH(170); corr(f, LOO) = {c2:.4f}; "
          f"{W3:.2f} in wide -> include at {incl(W3)}")
    frac, hh_check, rho, W5 = fig5(D, args.fr, os.path.join(OUT, "fig5_corner_reco"))
    scale = col / REF_WIDTH_IN                                 # the print scale of Fig. 4
    print(f"fig5: corner fractions {frac}; HH score check {hh_check:.1e}; {W5:.2f} in wide -> include at "
          f"width={W5 * scale / 3.375:.2f}\\linewidth for the Fig. 4 scale (text {FONT * scale:.1f} pt)")
    print(f"Fig. 4 itself: text {FONT * scale:.1f} pt")

if __name__ == "__main__":
    main()
