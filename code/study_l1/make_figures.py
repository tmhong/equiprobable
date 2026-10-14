#!/usr/bin/env python3
"""
Figures of the L1 feasibility study, drawn from study_l1/results.json (run emulate.py first):

  fig_precision.png   ΔAUC (hh vs tt̄) and per-event score fidelity vs datapath width W and ROM depth K
  fig_resources.png   resources of four designs by block, as % of a VP1802, against the 5% budget
  fig_latency.png     pipeline schedule (critical path + concurrent blocks) against the GEP budget

Sized for ~10-inch width with >= 14 pt text.  Colors: the dataviz reference palette (validated order).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "legend.frameon": False, "axes.titleweight": "semibold", "axes.titlelocation": "left",
})


def rows_by_name(R):
    return {r["name"]: r for r in R["rows"]}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(HERE, name + ".png"))


def errline(ax, x, y, e, color, marker, label, dx=0.0):
    x = np.asarray(x, float) + dx
    ax.errorbar(x, y, yerr=e, color=color, marker=marker, ms=8, mfc=color, mec=SURF, mew=2, capsize=0,
                elinewidth=1.5, lw=2, label=label, zorder=3)


def _logfmt(ax, ticks):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _legend(ax, items, loc):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=c, marker=m, ms=8, mec=SURF, mew=2, lw=2) for c, m, _ in items]
    ax.legend(h, [lab for *_, lab in items], loc=loc)


def fig_precision(R):
    rb = rows_by_name(R)
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 9), gridspec_kw={"hspace": 0.45, "wspace": 0.32})
    Ws = [10, 12, 14, 16, 18, 20, 24]
    Ks = [32, 64, 256, 1024, 4096]
    Kcs = [64, 256, 1024, 4096]
    FULL, MASS, CHI = (SLOT[0], "o", "full score (angles + BW)"), (SLOT[1], "s", "masses only (BW)"), \
        (SLOT[2], "^", "χ² ROMs (full score)")
    # (a) ΔAUC vs W
    a = ax[0, 0]
    rw = [rb[f"N=6 W={W} K=1024"] for W in Ws]
    for key, (col, mk, lab), dx in (("full", FULL, -0.2), ("mass", MASS, 0.2)):
        d = np.array([r[f"dauc_{key}"] for r in rw])
        errline(a, Ws, 100 * d[:, 0], 100 * d[:, 1], col, mk, lab, dx)
    a.axhline(0, color=INK2, lw=1)
    a.axvspan(15.4, 16.6, color=GRID, alpha=0.6, lw=0, zorder=0)
    a.set_xticks(Ws)
    a.set_xlabel("datapath width W (bits)")
    a.set_ylabel("ΔAUC  (×10⁻²)")
    a.set_title("(a) AUC vs bit width")
    _legend(a, [FULL, MASS], "lower right")
    # (b) ΔAUC vs ROM depth
    b = ax[0, 1]
    rk = [rb[f"N=6 W=18 K={K} (score ROMs)"] for K in Ks]
    rkc = [rb[f"N=6 W=18 Kχ={K} (χ² ROMs)"] for K in Kcs]
    lx = np.log2(Ks)
    for key, (col, mk, lab), dx in (("full", FULL, -0.12), ("mass", MASS, 0.0)):
        d = np.array([r[f"dauc_{key}"] for r in rk])
        errline(b, lx, 100 * d[:, 0], 100 * d[:, 1], col, mk, lab, dx)
    d = np.array([r["dauc_full"] for r in rkc])
    errline(b, np.log2(Kcs), 100 * d[:, 0], 100 * d[:, 1], CHI[0], CHI[1], CHI[2], 0.12)
    b.axhline(0, color=INK2, lw=1)
    b.axvspan(np.log2(256) - 0.35, np.log2(256) + 0.35, color=GRID, alpha=0.6, lw=0, zorder=0)
    b.set_xticks(np.log2(Ks))
    b.set_xticklabels([str(k) for k in Ks])
    b.set_xlabel("entries per 1-D ROM")
    b.set_ylabel("ΔAUC  (×10⁻²)")
    b.set_title("(b) AUC vs ROM depth")
    _legend(b, [(FULL[0], FULL[1], "score ROMs, full score"), (MASS[0], MASS[1], "score ROMs, masses only"), CHI],
            "lower right")
    ylo = min(a.get_ylim()[0], b.get_ylim()[0])
    yhi = max(a.get_ylim()[1], b.get_ylim()[1])
    a.set_ylim(ylo, yhi)
    b.set_ylim(ylo, yhi)
    # (c) per-event fidelity vs W
    c = ax[1, 0]
    for key, (col, mk, lab) in (("full", FULL), ("mass", MASS)):
        y = [r["fidelity"][f"score_rms_same_assignment_{key}"] for r in rw]
        c.plot(Ws, y, color=col, marker=mk, ms=8, mec=SURF, mew=2, zorder=3 if key == "full" else 4,
               mfc=col if key == "mass" else "none")
    c.set_yscale("log")
    _logfmt(c, [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    c.set_xticks(Ws)
    c.set_xlabel("datapath width W (bits)")
    c.set_ylabel("score RMS vs float")
    c.set_title("(c) per-event fidelity vs W")
    _legend(c, [(FULL[0], FULL[1], "full score"), (MASS[0], MASS[1], "masses only")], "upper right")
    # (d) per-event fidelity vs ROM depth
    dd = ax[1, 1]
    for key, (col, mk, lab) in (("full", FULL), ("mass", MASS)):
        y = [r["fidelity"][f"score_rms_same_assignment_{key}"] for r in rk]
        dd.plot(lx, y, color=col, marker=mk, ms=8, mec=SURF, mew=2, zorder=3 if key == "full" else 4,
                mfc=col if key == "mass" else "none")
    dd.set_yscale("log")
    _logfmt(dd, [0.02, 0.05, 0.1, 0.2, 0.5, 1])
    dd.set_xticks(np.log2(Ks))
    dd.set_xticklabels([str(k) for k in Ks])
    dd.set_xlabel("entries per 1-D ROM (score)")
    dd.set_ylabel("score RMS vs float")
    dd.set_title("(d) per-event fidelity vs K")
    _legend(dd, [(FULL[0], FULL[1], "full score"), (MASS[0], MASS[1], "masses only")], "upper right")
    same = [100 * np.mean(list(r["fidelity"]["same_assignment"].values())) for r in rw]
    fig.text(0.0, 0.02,
             "N = 6 leading jets, no b-tag, nominal L1 inputs (pT 0.5 GeV, η 0.025, φ 2π/256, MET 1 GeV); "
             "(a) K = Kχ = 1024, (b) W = 18.\n"
             "(a, b) ΔAUC(hh vs tt̄) relative to the float L1 chain (AUC 0.583 full, 0.582 masses only; offline "
             "reference 0.602 / 0.600);\nerror bars: paired bootstrap ±1σ. Shaded: recommended W = 16, K = 256.\n"
             "(c, d) RMS of the event score vs the same-input float chain, events with the same χ² pick "
             "(full and masses-only coincide).\nSame χ² pick vs W = 10, 12, 14, 16, 18, 24: "
             + ", ".join(f"{v:.0f}%" for v in same[:5]) + f", {same[-1]:.0f}%.", fontsize=13, color=INK2,
             va="top")
    save(fig, "fig_precision")


GROUPS = [("inputs + pairwise m²", ("input decode", "pairwise", "ν-jet")),
          ("ν p_z solver", ("ν p_z",)),
          ("χ² terms + hypotheses", ("χ² terms", "hypotheses")),
          ("masses + BW ROMs", ("masses",)),
          ("angle unit", ("angles",)),
          ("density + −ln + decision", ("density", "-ln"))]


def group_of(block):
    for i, (_, keys) in enumerate(GROUPS):
        if any(k in block for k in keys):
            return i
    raise KeyError(block)


def pick(R, variant, arch, N=6, W=16, K=256, lanes=4):
    for e in R["resources"]:
        if (e["N"], e["W"], e["K"], e["variant"], e["arch"]) == (N, W, K, variant, arch) and \
                (arch == "parallel" or e["lanes"] == lanes):
            return e
    raise KeyError((variant, arch))


def fig_resources(R):
    dev = R["meta"]["devices"]
    designs = [("masses only, parallel", pick(R, "masses", "parallel")),
               ("full score, parallel", pick(R, "full", "parallel")),
               ("masses only, time-mux", pick(R, "masses", "tmux")),
               ("full score, time-mux", pick(R, "full", "tmux"))]
    res = [("LUT", "LUTs"), ("FF", "flip-flops"), ("DSP", "DSP58 engines"), ("BRAM", "block RAM (36 kb)")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6), gridspec_kw={"hspace": 0.5, "wspace": 0.12}, sharey=True)
    for k, (rk, title) in enumerate(res):
        a = ax.flat[k]
        for j, (lab, e) in enumerate(designs):
            vals = np.zeros(len(GROUPS))
            for b in e["blocks"]:
                vals[group_of(b["block"])] += b[rk]
            pct = 100 * vals / dev["VP1802"][rk]
            left = 0.0
            for g in range(len(GROUPS)):
                if pct[g] <= 0:
                    continue
                a.barh(j, pct[g], left=left, height=0.62, color=SLOT[g], edgecolor=SURF, linewidth=1.5,
                       label=GROUPS[g][0] if (k == 0 and j == 1) or (k == 0 and j == 0 and g == 3) else None)
                left += pct[g]
            a.text(left + 0.08, j, f"{left:.2f}%", va="center", ha="left", fontsize=13, color=INK,
                   bbox={"facecolor": SURF, "edgecolor": "none", "pad": 1}, zorder=5)
        a.axvline(5.0, color=INK, lw=1.5)
        a.text(5.0, -0.75, "5% of VP1802 ", ha="right", va="center", fontsize=13, color=INK)
        vc = 5.0 * dev["VC1902"][rk] / dev["VP1802"][rk]
        a.axvline(vc, color=MUTED, lw=1.5)
        a.text(vc, 3.75, " 5% of VC1902", ha="left", va="center", fontsize=13, color=INK2)
        a.set_xlim(0, 5.6)
        a.set_ylim(3.6 + 0.4, -1.1)
        a.set_yticks(range(len(designs)))
        a.set_yticklabels([d[0] for d in designs])
        a.set_xlabel("% of VP1802")
        a.set_title(title, loc="left")
        a.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SLOT[g]) for g in range(len(GROUPS))]
    fig.legend(handles, [g[0] for g in GROUPS], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07),
               fontsize=14)
    fig.text(0.01, -0.12, "Analytic estimate (no synthesis): N = 6 jets, W = 16, K = Kχ = 256; time-mux = loops "
             "serialized inside the 48-BC TMUX period (4 hypothesis lanes).\nLUT/FF include ×1.3 for control and "
             "balancing. VP1802: 3.36 M LUT, 6.72 M FF, 14,352 DSP, 4,941 BRAM36; VC1902: 0.90 M, 1.80 M, 1,968, 967.",
             fontsize=13, color=INK2)
    save(fig, "fig_resources")


def fig_latency(R):
    clk = R["meta"]["clock_mhz"]
    designs = [("masses only, parallel", pick(R, "masses", "parallel")),
               ("full score, parallel", pick(R, "full", "parallel")),
               ("masses only, time-mux", pick(R, "masses", "tmux")),
               ("full score, time-mux", pick(R, "full", "tmux"))]
    fig, a = plt.subplots(figsize=(11, 6.4))
    ns = 1e3 / clk
    for j, (lab, e) in enumerate(designs):
        lanes_end = []                                   # greedy packing of the concurrent blocks
        for s in sorted(e["schedule"], key=lambda x: (x["lane"], x["start"])):
            g = group_of(s["block"])
            if s["lane"] == 0:
                y, h = j - 0.2, 0.32
            else:
                k = next((i for i, t in enumerate(lanes_end) if t <= s["start"]), len(lanes_end))
                if k == len(lanes_end):
                    lanes_end.append(0)
                lanes_end[k] = s["end"]
                y, h = j + 0.08 + 0.16 * k, 0.12
            a.barh(y, (s["end"] - s["start"]) * ns, left=s["start"] * ns, height=h, color=SLOT[g],
                   edgecolor=SURF, linewidth=1.5)
        tot = e["latency_clk"] * ns
        a.text(tot + 15, j - 0.2, f"{tot:.0f} ns ({e['latency_clk']} clk)", va="center", fontsize=13, color=INK,
               bbox={"facecolor": SURF, "edgecolor": "none", "pad": 1})
    budget = R["meta"]["tmux_bc"] * 25.0
    a.axvline(budget, color=INK, lw=1.5)
    a.text(budget - 10, -0.78, "GEP per-event budget\n= TMUX period 1.2 µs", ha="right", va="center", fontsize=13)
    a.set_xlim(0, 1450)
    a.set_ylim(len(designs) - 0.35, -1.1)
    a.set_yticks(range(len(designs)))
    a.set_yticklabels([d[0] for d in designs])
    a.set_xlabel(f"time after the L1 objects arrive (ns at {clk:.0f} MHz)")
    a.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SLOT[g]) for g in range(len(GROUPS))]
    fig.legend(handles, [g[0] for g in GROUPS], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               fontsize=14)
    fig.text(0.0, -0.13, "Thick bars: critical path; thin bars: blocks running concurrently. The whole ATLAS L0 "
             "budget of 10 µs is off scale (8–23× the totals).\nAnalytic pipeline depths (DSP58 3 clk, dot-product "
             "cascades, recip/rsqrt/sqrt units 12–18 clk, CORDIC W+1 clk); N = 6, W = 16, K = 256.\n"
             "At 360 MHz all four designs scale by 2/3 (full score, time-mux: 808 ns).", fontsize=13, color=INK2,
             va="top")
    save(fig, "fig_latency")


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    fig_precision(R)
    fig_resources(R)
    fig_latency(R)


if __name__ == "__main__":
    main()
