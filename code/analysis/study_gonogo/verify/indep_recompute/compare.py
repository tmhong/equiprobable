import json, os, sys
H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, "..", "..", "..", ".."))  # code/
sys.path.insert(0, ROOT)
import repo_paths
me = json.load(open(os.path.join(repo_paths.RESULTS, "study_gonogo/verify/indep_recompute/recompute.json")))
ref = json.load(open(os.path.join(repo_paths.RESULTS, "study_gonogo/results.json")))
worst = []
for lv in ["reco", "matched", "parton_trig", "gen"]:
    a, b = me[lv], ref["levels"][lv]
    print(f"== {lv}: n_tt {a['n_tt']} vs {b['n_tt']}, n_fit {a['n_fit']}, terms_cv {a['n_terms_cv']} vs {b['n_terms_cv']}, "
          f"frac_y<=0 cv {a['frac_y_le_0_tt']['osde_cv']:.4f} vs {b['frac_y_le_0_tt']['osde_cv']:.4f} all {a['frac_y_le_0_tt']['osde_all']:.4f} vs {b['frac_y_le_0_tt']['osde_all']:.4f}")
    for k in ["etat", "stop", "H350", "nospin", "tt_indep"]:
        x, y = a[k], b[k]
        print(f"  {k}: n {x['n']} vs {y['n']}, eff_sel {x['eff_sel']:.5f} vs {y['eff_sel']:.5f}")
        for m in ["osde_cv", "osde_all", "bw", "mtt_dens"]:
            row = []
            for w in ["0.01", "0.1"]:
                r1, r2, e = x["scores"][m][f"R@{w}"], y["scores"][m][f"R@{w}"], y["scores"][m][f"R@{w}_err"]
                pull = (r1 - r2) / e if e > 0 else 0.0
                worst.append((abs(pull), lv, k, m, w, r1, r2, e))
                row.append(f"R@{w} {r1:.3f} vs {r2:.3f}±{e:.3f} ({pull:+.2f}σ)")
            da = x["scores"][m]["auc"] - y["scores"][m]["auc"]
            row.append(f"AUC {x['scores'][m]['auc']:.4f} vs {y['scores'][m]['auc']:.4f} (Δ{da:+.1e})")
            print(f"    {m:9s} " + "  ".join(row))
        if "mtt_window" in x:
            row = []
            for w in ["0.01", "0.1"]:
                r2, e = y["scores"]["mtt_window"][f"R@{w}"], y["scores"]["mtt_window"][f"R@{w}_err"]
                W = x["mtt_window"]
                row.append(f"R@{w}: rep {r2:.2f}±{e:.2f} | evenodd {W['evenodd@'+w]:.2f} random-half {W['halfsig_random@'+w]:.2f} "
                           f"fullCV {W['fullcv@'+w]:.2f} full-signal {W['full@'+w]:.2f} [{W['full_window@'+w][0]:.0f},{W['full_window@'+w][1]:.0f}]")
                worst.append((abs(W['evenodd@'+w] - r2) / e if e > 0 else 0, lv, k, "mtt_window(evenodd)", w, W['evenodd@'+w], r2, e))
            print("    window   " + "\n             ".join(row))
        if "rule" in x:
            for w in ["0.01", "0.1"]:
                c, d = x["rule"][w], y["rule"][w]
                print(f"    rule@{w}: R {c['R']:.3f}±{c['R_err']:.3f} (rep {d['R']:.3f}±{d['R_err']:.3f}) a={c['a']} (rep {d['a_enrichment']}); "
                      f"b_bw {c['b_bw']['diff']:+.2f}±{c['b_bw']['sigma']:.2f} {c['b_bw']['pass']} (rep {d['b_vs_bw']['diff']:+.2f}±{d['b_vs_bw']['sigma']:.2f} {d['b_vs_bw']['pass']}); "
                      f"b_dens {c['b_mtt_dens']['diff']:+.2f}±{c['b_mtt_dens']['sigma']:.2f} {c['b_mtt_dens']['pass']} (rep {d['b_vs_mtt_dens']['diff']:+.2f}±{d['b_vs_mtt_dens']['sigma']:.2f} {d['b_vs_mtt_dens']['pass']}); "
                      f"b_win(evenodd) {c['b_win_evenodd']['diff']:+.2f}±{c['b_win_evenodd']['sigma']:.2f} {c['b_win_evenodd']['pass']} (rep {d['b_vs_mtt_window']['diff']:+.2f}±{d['b_vs_mtt_window']['sigma']:.2f} {d['b_vs_mtt_window']['pass']}); "
                      f"b_win(full) {c['b_win_full']['diff']:+.2f}±{c['b_win_full']['sigma']:.2f} {c['b_win_full']['pass']}; "
                      f"c_flow(rep) {d['c_vs_flow']['diff']:+.2f}±{d['c_vs_flow']['sigma']:.2f} {d['c_vs_flow']['pass']}; "
                      f"PASS mine {c['pass_with_evenodd_window']}/{c['pass_with_full_window']} rep {d['pass']}")
worst.sort(reverse=True)
print("\nLargest pulls (mine - reported)/reported_err:")
for w in worst[:12]:
    print("  %.2f  %s %s %s @%s  %.3f vs %.3f ± %.3f" % w)
