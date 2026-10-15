#!/usr/bin/env python3
"""
Pileup validation: branch sizes and analysis-level numbers for Delphes samples.

  .venv/bin/python code/analysis/pileup/validate_pileup.py nopu=samples/ttbar_bblvqq/run_seed12345:1000 \
      pu0=samples/ttbar_bblvqq/test_pu0 pu50=samples/ttbar_bblvqq/test_pu50 pu200=samples/ttbar_bblvqq/test_pu200

label=path[:max_events] (a relative path is taken from the repository top).  Uses the analysis code unchanged
(make_plots.load_sample, lib.summary.correct_assignment).  Prints a table and writes results/analysis/pileup/validation.json.
"""
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # code/
import repo_paths                                   # noqa: E402
from plotting import make_plots                     # noqa: E402
from lib import io, summary                    # noqa: E402

ANGLES = {"had_fermion": "down", "jet_assignment": "truth"}


def branch_sizes(root):
    """{top-level branch: (compressed MB, uncompressed MB) per 1000 events} (whole file)."""
    t = uproot.open(root)["Delphes"]
    out = {}
    for top in t.keys(recursive=False):
        b = t[top]
        comp = b.compressed_bytes + sum(s.compressed_bytes for s in b.itervalues(recursive=True))
        unc = b.uncompressed_bytes + sum(s.uncompressed_bytes for s in b.itervalues(recursive=True))
        out[top] = (comp / 1e6 * 1000 / t.num_entries, unc / 1e6 * 1000 / t.num_entries)
    return out, t.num_entries


def check_branches(root):
    """Every branch the analysis reads (lib/io.py DELPHES_BRANCHES) must exist."""
    keys = set(uproot.open(root)["Delphes"].keys())
    return [f"{c}.{f}" for c, fs in io.DELPHES_BRANCHES.items() if c != "Photon"
            for f in fs if f"{c}/{c}.{f}" not in keys]


def mean_err(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def frac_err(m):
    m = np.asarray(m, bool)
    p = m.mean()
    return p, np.sqrt(p * (1 - p) / len(m))


def analyse(label, path, nmax):
    root = os.path.join(path, "delphes_events.root")
    res = {"path": path, "file_MB": os.path.getsize(root) / 1e6}
    sizes, ntot = branch_sizes(root)
    res["entries_in_file"] = ntot
    res["kB_per_event"] = res["file_MB"] * 1e3 / ntot
    res["branch_MB_per_1000ev"] = {k: round(v[0], 3) for k, v in sizes.items()}
    res["missing_branches"] = check_branches(root)

    ev, xsec = make_plots.load_sample(label, path, ["parton", "gen", "reco"], nmax, ANGLES)
    P, G, R = ev["parton"], ev["gen"], ev["reco"]
    assert len(P) == len(G) == len(R), "parton / Delphes event counts differ"
    res["n_events"] = len(R)
    res["xsec_pb"] = xsec
    res["reco_njets"] = mean_err(R.njets)
    res["reco_nbjets"] = mean_err(R.nbjets)
    res["reco_nleptons_eq1"] = frac_err(R.nleptons == 1)
    res["reco_presel"] = frac_err(R.presel)
    res["reco_ok"] = frac_err(R.reco_ok)
    nu_pt = ak.to_numpy(ak.fill_none(P.nu.pt, np.nan)).astype(float)
    dmet = R.met - nu_pt
    q16, q50, q84 = np.percentile(dmet[np.isfinite(dmet)], [16, 50, 84])
    res["met_minus_nupt"] = {"mean": float(np.nanmean(dmet)), "std": float(np.nanstd(dmet)),
                             "median": float(q50), "hw68": float((q84 - q16) / 2)}
    gdmet = G.met - nu_pt
    res["gen_met_minus_nupt_hw68"] = float(summary.central(gdmet)[1])
    ok = np.asarray(R.reco_ok)
    correct = np.asarray(summary.correct_assignment(R, P))
    res["correct_of_reco_ok"] = frac_err(correct[ok])
    res["correct_of_all"] = frac_err(correct)
    res["mW_had_median_reco_ok"] = float(np.nanmedian(np.asarray(R.w_had.mass)[ok]))
    res["mt_had_median_reco_ok"] = float(np.nanmedian(np.asarray(R.t_had.mass)[ok]))
    # particle level must not see pileup
    res["gen_njets"] = mean_err(G.njets)
    res["gen_nbjets"] = mean_err(G.nbjets)
    res["gen_reco_ok"] = frac_err(G.reco_ok)
    t = uproot.open(root)["Delphes"]
    n = len(R)
    res["_genjet_count"] = ak.to_numpy(ak.num(t["GenJet/GenJet.PT"].array(entry_stop=n))).tolist()
    res["_genjet_pt_sum"] = ak.to_numpy(ak.sum(t["GenJet/GenJet.PT"].array(entry_stop=n), axis=1)).tolist()
    res["raw_genjet_mult"] = float(np.mean(res["_genjet_count"]))
    if "Rho" in t.keys(recursive=False):
        rho = t["Rho/Rho.Rho"].array(entry_stop=n)
        res["rho_central_mean_GeV"] = float(ak.mean(rho[:, 1]))   # 2nd GridRange: |eta| < 2.5
    return res


def main():
    results = {}
    for arg in sys.argv[1:]:
        label, spec = arg.split("=", 1)
        path, _, nmax = spec.partition(":")
        results[label] = analyse(label, os.path.join(repo_paths.REPO, path) if not os.path.isabs(path) else path,
                                 int(nmax) if nmax else None)

    labels = list(results)
    rows = [("events", lambda r: f"{r['n_events']}"),
            ("file kB/event", lambda r: f"{r['kB_per_event']:.1f}"),
            ("missing branches", lambda r: ",".join(r["missing_branches"]) or "none"),
            ("<rho> |eta|<2.5 [GeV]", lambda r: f"{r.get('rho_central_mean_GeV', float('nan')):.1f}"),
            ("reco <njets>", lambda r: "%.3f ± %.3f" % r["reco_njets"]),
            ("reco <nbjets>", lambda r: "%.3f ± %.3f" % r["reco_nbjets"]),
            ("reco 1 lepton", lambda r: "%.3f ± %.3f" % r["reco_nleptons_eq1"]),
            ("reco presel", lambda r: "%.3f ± %.3f" % r["reco_presel"]),
            ("reco_ok", lambda r: "%.3f ± %.3f" % r["reco_ok"]),
            ("MET-pT(nu) std", lambda r: f"{r['met_minus_nupt']['std']:.1f}"),
            ("MET-pT(nu) hw68", lambda r: f"{r['met_minus_nupt']['hw68']:.1f}"),
            ("MET-pT(nu) median", lambda r: f"{r['met_minus_nupt']['median']:.1f}"),
            ("chi2 correct / reco_ok", lambda r: "%.3f ± %.3f" % r["correct_of_reco_ok"]),
            ("chi2 correct / all", lambda r: "%.3f ± %.3f" % r["correct_of_all"]),
            ("m_W had median", lambda r: f"{r['mW_had_median_reco_ok']:.1f}"),
            ("m_t had median", lambda r: f"{r['mt_had_median_reco_ok']:.1f}"),
            ("gen <njets>", lambda r: "%.3f ± %.3f" % r["gen_njets"]),
            ("gen <nbjets>", lambda r: "%.3f ± %.3f" % r["gen_nbjets"]),
            ("gen raw GenJet mult", lambda r: f"{r['raw_genjet_mult']:.3f}"),
            ("gen MET-pT(nu) hw68", lambda r: f"{r['gen_met_minus_nupt_hw68']:.1f}")]
    print("\n" + " | ".join(["quantity"] + labels))
    for name, f in rows:
        print(" | ".join([name] + [f(results[l]) for l in labels]))
    print("\nbranch sizes, compressed MB per 1000 events:")
    allb = sorted({b for r in results.values() for b in r["branch_MB_per_1000ev"]},
                  key=lambda b: -max(r["branch_MB_per_1000ev"].get(b, 0) for r in results.values()))
    for b in allb:
        print(" | ".join([b] + [f"{results[l]['branch_MB_per_1000ev'].get(b, float('nan')):.2f}" for l in labels]))
    # identical hard events (same seed): GenJets must agree event by event
    for a in labels:
        for b in labels:
            if a < b and results[a]["_genjet_count"][:5] == results[b]["_genjet_count"][:5]:
                same = (results[a]["_genjet_count"] == results[b]["_genjet_count"] and
                        np.allclose(results[a]["_genjet_pt_sum"], results[b]["_genjet_pt_sum"]))
                print(f"GenJets {a} vs {b}: event-by-event identical = {same}")
    for r in results.values():
        r.pop("_genjet_count"), r.pop("_genjet_pt_sum")
    out = os.path.join(repo_paths.results_dir("pileup"), "validation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
