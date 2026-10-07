#!/usr/bin/env python3
"""
Leptonic vs hadronic top: are (θ, φ, θ', φ') and their tilde versions distributed the same?

In the SM (V-A, massless fermions) the down-type quark in W -> q q' plays the role of the
charged lepton in W -> l nu, so at parton level cos θ' and cos θ~' (and the other pairs)
must agree.  At gen/reco the two sides are measured differently (lepton cuts + neutrino
reconstruction vs jet cuts, response and jet assignment), so they need not.

Compared: parton (all), gen and reco (events with a reconstructed tt̄), and reco restricted to
events whose jets are all correctly assigned (perfect flavour tagging + correct assignment).

Output: plots/lep_vs_had.png and plots/lep_vs_had.json (KS distance and p-value per pair)
"""
import argparse
import json
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_plots  # noqa: E402
from plotting import summary  # noqa: E402

PAIRS = [("cos_theta", "cos_theta_tilde", r"$\cos\theta$ vs $\cos\tilde\theta$", (-1, 1)),
         ("phi", "phi_tilde", r"$\phi$ vs $\tilde\phi$", (-np.pi, np.pi)),
         ("cos_theta_prime", "cos_theta_prime_tilde", r"$\cos\theta'$ vs $\cos\tilde\theta'$", (-1, 1)),
         ("phi_prime", "phi_prime_tilde", r"$\phi'$ vs $\tilde\phi'$", (-np.pi, np.pi))]


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov distance and asymptotic p-value."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a) - np.searchsorted(b, x, side="right") / len(b))))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)), 0, 1))
    return d, p


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--config", default=os.path.join(HERE, "plot_config.py"))
    p.add_argument("--out", default=os.path.join(HERE, "plots", "lep_vs_had"))
    args = p.parse_args()
    cfg = runpy.run_path(args.config)
    opts = {"had_fermion": "down", "jet_assignment": "truth", **cfg.get("ANGLES", {})}
    ev, _ = make_plots.load_sample("s", args.sample, ["parton", "gen", "reco"], None, opts)
    P, G, R = ev["parton"], ev["gen"], ev["reco"]
    cases = [("parton", P, np.ones(len(P), bool)), ("gen", G, np.asarray(G.reco_ok)),
             ("reco", R, np.asarray(R.reco_ok)), ("reco, correct jets", R, summary.correct_assignment(R, P))]
    fig, axs = plt.subplots(len(cases), len(PAIRS), figsize=(17, 13))
    res = {}
    for i, (lab, E, m) in enumerate(cases):
        res[lab] = {"n": int(m.sum())}
        for j, (a, b, title, rng) in enumerate(PAIRS):
            va, vb = np.asarray(E[a], float)[m], np.asarray(E[b], float)[m]
            ok = np.isfinite(va) & np.isfinite(vb)
            va, vb = va[ok], vb[ok]
            d, pv = ks(va, vb)
            res[lab][f"{a}|{b}"] = {"ks": d, "p": pv}
            ax = axs[i, j]
            edges = np.linspace(*rng, 26)
            ax.hist(va, edges, histtype="step", density=True, lw=1.6, color="#1f4e9c", label="leptonic top")
            ax.hist(vb, edges, histtype="step", density=True, lw=1.6, color="#d1495b", ls="--",
                    label="hadronic top (down-type $q$)")
            ax.set_title(f"{lab}: KS {d:.3f}, p = {pv:.2g}", fontsize=9)
            ax.set_xlim(*rng)
            ax.set_ylim(bottom=0)
            if i == len(cases) - 1:
                ax.set_xlabel(title)
            if j == 0:
                ax.set_ylabel(f"{lab}\n({m.sum()} events)", fontsize=9)
    axs[0, 0].legend(fontsize=8, frameon=False, loc="lower center")
    fig.suptitle(r"Leptonic vs hadronic top decay angles (hadronic side: down-type quark, truth-tagged)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=110)
    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    for lab, r in res.items():
        print(f"{lab:20s} n={r['n']:6d}  " + "  ".join(f"{k.split('|')[0]:16s} KS {v['ks']:.3f} p {v['p']:.2g}"
                                                     for k, v in r.items() if k != "n"))


if __name__ == "__main__":
    main()
