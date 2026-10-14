#!/usr/bin/env python3
"""
hh-specific plots.  In h -> W W* one W is off-shell (m_h < 2 m_W), so the hadronically
decaying "W" is either the on-shell W or the W*.  At parton level the on-shell one is the
heavier: hadronic W on-shell  <=>  m(q q') > m(l nu).  Events are aligned between levels,
so the same parton-level classification is applied to the gen and reco events.
With --mh > 2 m_W (e.g. the m_h = 170 GeV sample) both W are on shell: the split is then
heavier vs lighter W (labels only; the classification is the same).  --mh also places the
parton m_bb marker (default 125 GeV).

Output: <out>/w_had_mass_split.png (+ .json with the fractions); --out defaults to results/plots_hh

Example:
  python code/hh_plots.py --sample samples/hh_bblvqq/run_seed2001 --out results/plots_hh
  python code/hh_plots.py --sample samples/hh_bblvqq/run_m170_seed2170 --mh 170 --out results/plots_hh_h170
"""
import argparse
import json
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plotting import style  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_plots  # noqa: E402
import repo_paths  # noqa: E402
from plotting.physics import M_W  # noqa: E402

LEVELS = {"parton": ("#1f4e9c", ":"), "gen": ("#d1495b", "--"), "reco": ("#2a9d8f", "-")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--config", default=os.path.join(HERE, "plot_config.py"))
    p.add_argument("--out", default=os.path.join(repo_paths.RESULTS, "plots_hh"))
    p.add_argument("--mh", type=float, default=125.0,
                   help="Higgs mass of the sample in GeV (default 125): position of the parton m_bb marker; for "
                        "m_h > 2 m_W both W are on shell and the split is labelled heavier / lighter W")
    args = p.parse_args()

    cfg = runpy.run_path(args.config)
    opts = {"had_fermion": "down", "jet_assignment": "truth", **cfg.get("ANGLES", {})}
    ev, _ = make_plots.load_sample("hh", args.sample, list(LEVELS), None, opts)
    P = ev["parton"]
    m_qq, m_lv = np.asarray(P.w_had.mass, float), np.asarray(P.w_lep.mass, float)
    onshell = m_qq > m_lv                      # hadronic boson is the on-shell W
    both_onshell = args.mh > 2 * M_W           # h -> WW with both W on shell: "on-shell" = the heavier W

    bins = np.linspace(0, 200, 51)
    style.use()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.8), sharey=False)
    res = {"frac_had_onshell": float(onshell.mean())}
    if both_onshell:
        res["note"] = (f"m_h = {args.mh:g} GeV > 2 m_W: both W on shell; 'onshell' = hadronic W is the heavier W, "
                       "'offshell' = it is the lighter W")
    titles = ((r"hadronic $W$ = heavier $W$", r"hadronic $W$ = lighter $W$") if both_onshell else
              (r"hadronic $W$ on-shell", r"hadronic $W^*$ off-shell"))
    for ax, case, title in ((axs[0], onshell, titles[0]),
                            (axs[1], ~onshell, titles[1])):
        for lvl, (col, ls) in LEVELS.items():
            E = ev[lvl]
            sel = case if lvl == "parton" else case & np.asarray(E.reco_ok)
            m = np.asarray(E.w_had.mass, float)[sel]
            m = m[np.isfinite(m)]
            med = np.median(m) if len(m) else np.nan
            ax.hist(np.clip(m, bins[0], bins[-1] - 1e-6), bins, histtype="step", density=True, lw=1.6,
                    color=col, ls=ls, label=f"{lvl} ({len(m)}, median {med:.0f} GeV)")
            res[f"{'onshell' if case is onshell else 'offshell'}_{lvl}_median"] = float(med)
        ax.set_title(f"{title}: {100 * case.mean():.0f}% of events", fontsize=15)
        ax.set_xlabel(r"$m_{W,\mathrm{had}}$ [GeV]")
        ax.set_ylabel("normalized")
        ax.legend(fontsize=13, frameon=False)
        ax.set_xlim(bins[0], bins[-1])
    fig.suptitle((rf"$hh\to b\bar b\,WW$ ($m_h={args.mh:g}$ GeV)" if both_onshell else r"$hh\to b\bar b\,WW^*$")
                 + r": hadronic $W$ mass by case (gen/reco: $\chi^2$-selected jets)")
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "w_had_mass_split.png")
    fig.savefig(path, dpi=120)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    overview(ev, args.out, args.mh)


def overview(ev, out, mh=125.0):
    """One readable 1x3 figure: m_bb, leading-lepton pT, number of b-jets (parton/gen/reco)."""
    import awkward as ak
    style.use()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    # m_bb: parton is a delta function at m_h -> scale the axis to gen/reco and annotate
    bins = np.linspace(0, 300, 61)
    ymax = 0
    for lvl, (col, ls) in LEVELS.items():
        E = ev[lvl]
        if lvl == "parton":
            m = np.asarray((E.b + E.bbar).mass, float)
        else:
            two = ak.to_numpy(ak.num(E.bjets) >= 2)
            bj = E.bjets[two]
            m = np.asarray((bj[:, 0] + bj[:, 1]).mass, float)
        h, _ = np.histogram(np.clip(m, 0, 300 - 1e-6), bins, density=True)
        if lvl != "parton":
            ymax = max(ymax, h.max())
        axs[0].stairs(h, bins, color=col, ls=ls, lw=1.6, label=f"{lvl} (median {np.median(m):.0f} GeV)")
    axs[0].set_ylim(0, 1.35 * ymax)
    axs[0].annotate(r"parton: $\delta$ at $m_h$" + "\n(off scale)", xy=(mh, 1.3 * ymax), xytext=(mh + 60, 0.75 * ymax),
                    fontsize=13, color=LEVELS["parton"][0], arrowprops=dict(arrowstyle="->", color=LEVELS["parton"][0]))
    axs[0].set_xlabel(r"$m_{b\bar b}$ [GeV]  (gen/reco: two leading b-jets)")
    # leading lepton pT
    bins = np.linspace(0, 250, 51)
    for lvl, (col, ls) in LEVELS.items():
        pt = np.asarray(ak.to_numpy(ak.fill_none(ev[lvl].lep.pt, np.nan)), float)
        pt = pt[np.isfinite(pt)]
        axs[1].hist(np.clip(pt, 0, 250 - 1e-6), bins, histtype="step", density=True, color=col, ls=ls, lw=1.6,
                    label=f"{lvl} (mean {pt.mean():.0f} GeV)")
    axs[1].set_xlabel(r"leading lepton $p_T$ [GeV]")
    # number of b-jets
    bins = np.arange(-0.5, 5.5, 1)
    for lvl in ("gen", "reco"):
        col, ls = LEVELS[lvl]
        axs[2].hist(np.clip(ev[lvl].nbjets, 0, 4), bins, histtype="step", density=True, color=col, ls=ls, lw=1.6,
                    label=lvl)
    axs[2].set_xlabel("number of b-jets")
    for ax in axs:
        ax.set_ylabel("normalized")
        ax.legend(fontsize=14, frameon=False)
    decay = (rf"WW\to b\bar b\,\ell\nu\,q\bar q'$ ($m_h={mh:g}$ GeV)" if mh > 2 * M_W
             else r"WW^*\to b\bar b\,\ell\nu\,q\bar q'$")
    fig.suptitle(r"$hh\to b\bar b\," + decay + " at parton, gen and reco level", fontsize=17)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hh_overview.png"), dpi=120)


if __name__ == "__main__":
    main()
