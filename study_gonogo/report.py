#!/usr/bin/env python3
"""
Tables (results.md) and the enrichment figure (fig_enrichment.png/.pdf) from study_gonogo/results.json.

Run: .venv/bin/python study_gonogo/report.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
SIG = [("etat", "η_t(343)"), ("stop", "stop (300, 120)"), ("H350", "H(350) → tt̄"), ("nospin", "tt̄, no spin corr."),
       ("tt_indep", "tt̄ independent (null)")]
METH = [("osde_cv", "f_tt̄, CV-truncated (2.5σ)"), ("osde_all", "f_tt̄, all terms"), ("flow", "flow (same angles)"),
        ("bw", "BW masses"), ("mtt_dens", "m_tt̄ density"), ("mtt_window", "best m_tt̄ window"),
        ("osde_cv+bw", "f_tt̄ + BW"), ("bdt_angles", "BDT angles (supervised)"), ("bdt_all", "BDT all (supervised)")]
LEVELS = [("parton_trig", "parton + trigger"), ("gen", "generator, χ²"), ("reco", "reconstructed, χ²"),
          ("matched", "reconstructed, truth-matched")]


def fmt(s, m, wp):
    r = s["scores"][m]
    return f"{r[f'R@{wp}']:.2f} ± {r[f'R@{wp}_err']:.2f}"


def main():
    L = []
    L.append("# Go/no-go results\n")
    L.append("Rule: `CRITERIA.md`. R = ε_S/ε_tt̄ at ε_tt̄ = 1% and 10% (bootstrap errors). "
             "tt̄ scores are leave-one-out (OSDE) or out-of-fold (flow) on the fit sample.\n")
    dec = R.get("decision", {})
    if dec:
        L.append("## Decision\n")
        for lvl, d in dec.items():
            L.append(f"- **{lvl}**: {d['verdict']}" + (f" (passes: {d['passes']})" if d["passes"] else ""))
        L.append("")
    for lvl, lname in LEVELS:
        if lvl not in R["levels"]:
            continue
        lev = R["levels"][lvl]
        L.append(f"## {lname}\n")
        L.append(f"tt̄ fit events: {lev['n_tt']}; CV-truncated terms: {lev['n_terms_cv']}; "
                 f"tt̄ with y ≤ 0: {100 * lev['frac_y_le_0_tt']['osde_cv']:.1f}% (CV), {100 * lev['frac_y_le_0_tt']['osde_all']:.1f}% (all)\n")
        sigs = [(k, n) for k, n in SIG if k in lev]
        L.append("| score | " + " | ".join(f"{n} R@1% | R@10% | AUC" for _, n in sigs) + " |")
        L.append("|---|" + "---|---|---|" * len(sigs))
        for m, mname in METH:
            row = [mname]
            for k, _ in sigs:
                s = lev[k]
                a = s["scores"][m]["auc"]
                row += [fmt(s, m, "0.01"), fmt(s, m, "0.1"), "" if a is None else f"{a:.3f}"]
            L.append("| " + " | ".join(row) + " |")
        L.append("\n" + "events: " + ", ".join(f"{n} {lev[k]['n']} (selection {100 * lev[k]['eff_sel']:.1f}%)" for k, n in sigs) + "\n")
        rules = [(k, n) for k, n in sigs if "rule" in lev[k]]
        if rules:
            L.append("Rule checks for f_tt̄, CV-truncated (2.5σ): (a) R ≥ 2 and > 3σ above 1; (b) beats BW, m_tt̄ density and the best "
                     "window by > 2σ; (c) not 2σ below the flow.\n")
            L.append("| signal | wp | R | (a) | (b) vs BW | (b) vs m_tt̄ dens. | (b) vs window | (c) vs flow | pass |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for k, n in rules:
                for wp, v in lev[k]["rule"].items():
                    b = lambda x: f"{v[x]['diff']:+.2f} ± {v[x]['sigma']:.2f} {'✓' if v[x]['pass'] else '✗'}"  # noqa: E731
                    L.append(f"| {n} | {float(wp):.0%} | {v['R']:.2f} ± {v['R_err']:.2f} | {'✓' if v['a_enrichment'] else '✗'} | "
                             f"{b('b_vs_bw')} | {b('b_vs_mtt_dens')} | {b('b_vs_mtt_window')} | {b('c_vs_flow')} | "
                             f"{'**PASS**' if v['pass'] else 'fail'} |")
            L.append("")
        ys = [(k, n) for k, n in sigs if "yields_140fb" in lev[k]]
        if ys and lvl in ("reco", "matched"):
            L.append("Yields at 140 fb⁻¹ (σ_tt̄ = 832 pb, BR = 0.288; η_t 8.8 pb, stop 10.57 pb):\n")
            L.append("| signal | cut | S | B | S/B | Z (stat) | Z (5% syst) | Z (10% syst) |")
            L.append("|---|---|---|---|---|---|---|---|")
            for k, n in ys:
                y = lev[k]["yields_140fb"]
                L.append(f"| {n} | none | {y['N_S']:.0f} | {y['N_B']:.0f} | {y['S/B_inclusive']:.4f} | | | |")
                for m in ("osde_cv", "flow", "bw", "mtt_window", "osde_cv+bw", "bdt_all"):
                    for wp in ("0.01", "0.1"):
                        c = y[f"{m}@{wp}"]
                        L.append(f"| {n} | {m} @ {float(wp):.0%} | {c['S']:.0f} | {c['B']:.0f} | {c['S/B']:.3f} | {c['Z_stat']:.1f} | "
                                 f"{c['Z_5pct']:.2f} | {c['Z_10pct']:.2f} |")
            L.append("")
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(L) + "\n")

    # figure: R at 1% (filled) and 10% (open) per score, signals in columns, reco and matched in rows
    lv = [(l, n) for l, n in LEVELS if l in ("reco", "matched") and l in R["levels"]]
    sg = [(k, n) for k, n in SIG if k != "tt_indep" and all(k in R["levels"][l] for l, _ in lv)]
    if not lv or not sg:
        return
    fig, axs = plt.subplots(len(lv), len(sg), figsize=(4.2 * len(sg), 3.6 * len(lv)), squeeze=False, sharey=True)
    cols = {"osde_cv": "#1f3b73", "osde_all": "#5b7bb5", "flow": "#7a7a7a", "bw": "#c28b2c", "mtt_dens": "#e3c27a",
            "mtt_window": "#b23a48", "osde_cv+bw": "#2a7f62", "bdt_angles": "#999999", "bdt_all": "#cccccc"}
    x = np.arange(len(METH))
    for i, (l, ln) in enumerate(lv):
        for j, (k, kn) in enumerate(sg):
            ax = axs[i, j]
            s = R["levels"][l][k]["scores"]
            for xi, (m, _) in zip(x, METH):
                for wp, mk, fill in (("0.01", "o", True), ("0.1", "s", False)):
                    r, e = s[m][f"R@{wp}"], s[m][f"R@{wp}_err"]
                    ax.errorbar(xi + (-0.15 if fill else 0.15), max(r, 0.03), yerr=e, fmt=mk, color=cols[m], ms=6,
                                mfc=cols[m] if fill else "white", lw=1.2, capsize=0)
            ax.axhline(1, color="0.5", lw=0.8)
            ax.axhline(2, color="0.5", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_ylim(0.03, 30)
            ax.set_xticks(x)
            ax.set_xticklabels([n for _, n in METH], rotation=60, ha="right", fontsize=8)
            if i == 0:
                ax.set_title(kn, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{ln}\n" + r"$\epsilon_S/\epsilon_{t\bar t}$", fontsize=10)
    fig.text(0.01, 0.005, "filled: at 1% of tt̄; open: at 10% of tt̄; grey line R = 1, dotted R = 2", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_enrichment.{ext}"), dpi=130)


if __name__ == "__main__":
    main()
