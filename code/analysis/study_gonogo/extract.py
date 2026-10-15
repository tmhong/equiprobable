#!/usr/bin/env python3
"""
Per-sample arrays for the go/no-go analysis, cached in results/analysis/study_gonogo/cache/<key>.npz.

For each level in (parton_trig, gen, reco) and the events that osde_fit.py uses (finite angles; gen/reco: a
reconstructed tt̄ system, i.e. reco_ok of the chi2 assignment): the ten angles, m_W,had, m_t,lep, m_t,had and m_tt̄,
plus n_all (generated events).  The ΔR-matched level comes from study_truthmatch/build_matched.py.

For the tt̄ fit sample the angles are checked against osde_fit.py's cache, so that leave-one-out scores are exact.

Run: OMP_NUM_THREADS=3 .venv/bin/python code/analysis/study_gonogo/extract.py KEY=PATH[,PATH...] ...   (PATH relative to the repo top)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))          # code/
sys.path.insert(0, CODE)
from plotting import make_plots  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
import repo_paths  # noqa: E402

CACHE = repo_paths.results_dir("study_gonogo", "cache", create=False)
OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
LEVELS = ("parton_trig", "gen", "reco")


def one(path):
    ev, _ = make_plots.load_sample(os.path.basename(os.path.normpath(path)), path, list(LEVELS), None, OPTS)
    out = {}
    for lvl in LEVELS:
        e = ev[lvl]
        ang = {a: np.asarray(e[a], float) for a in ANGLE_NAMES}
        ok = np.all([np.isfinite(v) for v in ang.values()], axis=0)
        if lvl not in ("parton", "parton_trig"):
            ok &= np.asarray(e.reco_ok)
        m = {"m_w_had": e.w_had.mass, "m_t_lep": e.t_lep.mass, "m_t_had": e.t_had.mass,
             "mtt": (e.t_lep + e.t_had).mass}
        for a, v in ang.items():
            out[f"{lvl}/{a}"] = v[ok]
        for k, v in m.items():
            out[f"{lvl}/{k}"] = np.asarray(v, float)[ok]
        out[f"{lvl}/n_all"] = np.array(len(e))
    cache = os.path.join(path, "angles_down_truth.npz")
    if os.path.exists(cache):                       # the OSDE fit sample: same events, same order
        z = np.load(cache)
        for lvl in LEVELS:
            if f"{lvl}/Y" in z:
                same = all(len(z[f"{lvl}/{a}"]) == len(out[f"{lvl}/{a}"]) and
                           np.allclose(z[f"{lvl}/{a}"], out[f"{lvl}/{a}"]) for a in ANGLE_NAMES)
                print(f"  {lvl}: {len(out[f'{lvl}/Y'])} events, identical to osde_fit cache: {same}")
                out[f"{lvl}/same_as_fit_cache"] = np.array(same)
    return out


def main():
    os.makedirs(CACHE, exist_ok=True)
    for arg in sys.argv[1:]:
        key, paths = arg.split("=", 1)
        parts = [one(os.path.join(repo_paths.REPO, p)) for p in paths.split(",")]
        out = {}
        for k in parts[0]:
            if k.endswith("/n_all"):
                out[k] = np.array(sum(int(p[k]) for p in parts))
            elif k.endswith("same_as_fit_cache"):
                if len(parts) == 1:                    # only meaningful for a single sample (the OSDE fit sample)
                    out[k] = parts[0][k]
            else:
                out[k] = np.concatenate([p[k] for p in parts])
        np.savez_compressed(os.path.join(CACHE, f"{key}.npz"), **out)
        print(key, {lvl: (int(len(out[f'{lvl}/Y'])), int(out[f'{lvl}/n_all'])) for lvl in LEVELS}, flush=True)


if __name__ == "__main__":
    main()
