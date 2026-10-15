#!/usr/bin/env python3
"""
Paired version of stop_met_test.py: saves per-event scores (nominal vs oracle E_T^miss, same events) and computes
the paired-bootstrap uncertainty of R(nominal) - R(oracle) for f_tt (CV) at 1% and 10%, gen and reco.
Run: cd ttbar_bblvqq && OMP_NUM_THREADS=4 .venv/bin/python code/analysis/study_gonogo/verify/physics_skeptic/stop_met_paired.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stop_met_test as T  # noqa: E402

from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
import repo_paths  # noqa: E402


def main():
    path = os.path.join(repo_paths.SAMPLES, "stop_bblvqq/run_seed4002")
    parton, cpx, cpy = T.lhe_info(os.path.join(path, "parton_events.lhe.gz"))
    root = os.path.join(path, "delphes_events.root")
    out, save = {}, {}
    rng = np.random.default_rng(11)
    for level in ("gen", "reco"):
        f_all = osde.load_density(repo_paths.results(T.F_FILES[level]))
        f_cv = f_all.truncated(f_all.significance() > 2.5)
        z = np.load(repo_paths.results("study_gonogo/cache/tt_fit.npz"))
        s_tt = -f_cv.loo({a: z[f"{level}/{a}"] for a in ANGLE_NAMES})
        S = {}
        for oracle in (False, True):
            X, ok, ptchi = T.build(root, parton, cpx, cpy, level, oracle)
            S[oracle] = -f_cv(X)
        save[f"{level}/tt"] = s_tt
        save[f"{level}/nominal"], save[f"{level}/oracle"], save[f"{level}/ptchi"] = S[False], S[True], ptchi[ok]
        res = {}
        for wp in (0.01, 0.10):
            d = []
            for b in range(301):
                ib = np.arange(len(s_tt)) if b == 0 else rng.integers(0, len(s_tt), len(s_tt))
                is_ = np.arange(len(S[False])) if b == 0 else rng.integers(0, len(S[False]), len(S[False]))
                d.append([T.R(S[False][is_], s_tt[ib], wp), T.R(S[True][is_], s_tt[ib], wp)])
            d = np.array(d)
            res[f"R@{wp:g}"] = {"nominal": float(d[0, 0]), "oracle": float(d[0, 1]),
                                "diff": float(d[0, 0] - d[0, 1]), "diff_sigma_paired": float(np.std(d[1:, 0] - d[1:, 1]))}
        # Spearman-like check: correlation of the per-event score shift with pT(chi chi)
        shift = S[False] - S[True]
        res["corr(score shift, pT chi chi)"] = float(np.corrcoef(shift, ptchi[ok])[0, 1])
        out[level] = res
        print(level, json.dumps(res), flush=True)
    np.savez_compressed(os.path.join(T.OUT, "stop_met_scores.npz"), **save)
    json.dump(out, open(os.path.join(T.OUT, "stop_met_paired.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
