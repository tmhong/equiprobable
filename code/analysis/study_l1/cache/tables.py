"""Scratch: print markdown tables from results.json for design.md."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))   # code/
import repo_paths  # noqa: E402
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo_paths.RESULTS, "study_l1", "results.json")))
rows = {r["name"]: r for r in R["rows"]}
def f4(v): return "–" if v is None else f"{v:.3f}"
def d(r, k):
    v = r.get(f"dauc_{k}")
    return "–" if not v else f"{v[0]:+.4f} ± {v[1]:.4f}"
def sa(r):
    s = r.get("fidelity", {}).get("same_assignment") or r.get("same_assignment_as_ref") or {}
    return "–" if not s else f"{100*s.get('tt_test', float('nan')):.1f} / {100*s.get('hh', float('nan')):.1f}"
print("## ladder / inputs / recommended / l1set / stress")
print("| configuration | AUC full | AUC masses | AUC angles | AUC χ²min | ΔAUC full | ΔAUC masses | same pick tt̄/hh (%) |")
print("|---|---|---|---|---|---|---|---|")
for r in R["rows"]:
    if r["group"] in ("ladder", "inputs", "recommended", "l1set", "stress"):
        print(f"| {r['name']} | {f4(r['auc_full'])} | {f4(r['auc_mass'])} | {f4(r['auc_angles'])} | {f4(r['auc_chi2min'])} | {d(r,'full')} | {d(r,'mass')} | {sa(r)} |")
print("\n## W / K / Kchi scans")
print("| configuration | AUC full | AUC masses | ΔAUC full | ΔAUC masses | same pick tt̄/hh (%) | score RMS (same pick) full / masses | rank corr. masses | saturations |")
print("|---|---|---|---|---|---|---|---|---|")
for r in R["rows"]:
    if r["group"] in ("W", "K", "Kchi"):
        fd = r["fidelity"]
        print(f"| {r['name']} | {f4(r['auc_full'])} | {f4(r['auc_mass'])} | {d(r,'full')} | {d(r,'mass')} | {sa(r)} | {fd['score_rms_same_assignment_full']:.3f} / {fd['score_rms_same_assignment_mass']:.3f} | {fd['rank_corr_mass']:.4f} | {r.get('saturations', {})} |")
print("\n## working points and tails")
for name in ("offline reference (N=6, b-tag, truth d-type)", "float L1 N=6 (+ massless objects)", "recommended: full score", "recommended: masses only (no angle unit)", "N=6 W=16 K=256 full, L1 event set", "N=6 W=16 K=256 masses only, L1 event set"):
    r = rows[name]
    for k in ("full", "mass"):
        e = r.get(f"eff_tt_at_hh_{k}")
        if e: print(f"{name} [{k}] eff_tt at eff_hh 0.9/0.5/0.1: {e['0.9']:.3f} / {e['0.5']:.3f} / {e['0.1']:.3f}")
        t = r.get(f"tail_{k}")
        if t: print("    tail:", "; ".join(f"acc_tt {a}: eff_hh {v['eff_hh']:.3f} (float {v['eff_hh_ref']:.3f}), same hh {100*v['overlap_hh']:.0f}%, same tt {100*v['overlap_tt']:.0f}%" for a, v in t.items()))
    if "dauc_full_minus_mass" in r: print("    full - masses:", r["dauc_full_minus_mass"])
print("\n## calibration constants (recommended)")
print(rows["recommended: full score"]["cal"], rows["float L1 N=6 (+ massless objects)"]["cal"])
print("\n## resources")
def res(e):
    t = e["total"]; f = e["frac"]
    return (f"LUT {t['LUT']:.0f} | FF {t['FF']:.0f} | DSP {t['DSP']:.0f} | BRAM {t['BRAM']:.1f} | "
            f"VP1802 {100*f['VP1802']['LUT']:.2f}/{100*f['VP1802']['FF']:.2f}/{100*f['VP1802']['DSP']:.2f}/{100*f['VP1802']['BRAM']:.2f} | "
            f"VC1902 {100*f['VC1902']['LUT']:.1f}/{100*f['VC1902']['FF']:.1f}/{100*f['VC1902']['DSP']:.1f}/{100*f['VC1902']['BRAM']:.1f} | "
            f"lat {e['latency_clk']} clk {e['latency_ns']:.0f} ns | II {e['ii_clk']}")
for e in R["resources"]:
    if (e["W"], e["K"]) in ((16, 256),) or e["arch"] == "tmux":
        print(f"N={e['N']} {e['variant']:6s} {e['arch']:8s} lanes={e['lanes']} W={e['W']} K={e['K']}: {res(e)}")
print("\n## W/K dependence (N=6 full parallel)")
for e in R["resources"]:
    if e["N"] == 6 and e["arch"] == "parallel":
        print(f"{e['variant']:6s} W={e['W']} K={e['K']}: {res(e)}")
print("\n## blocks (recommended)")
for v, e in R["recommended_resources"].items():
    print(v)
    for b in e["blocks"]:
        print(f"| {b['block']} | {b['LUT']:.0f} | {b['FF']:.0f} | {b['DSP']:.0f} | {b['BRAM']:.1f} | {b['lat']} |")
    print("schedule:", [(s['block'][:20], s['start'], s['end'], s['lane']) for s in e.get("schedule", [])])
