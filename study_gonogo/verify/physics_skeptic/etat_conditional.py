#!/usr/bin/env python3
"""
Is the f_tt (CV) enrichment for eta_t a spin effect or a proxy for its threshold kinematics?
R = eps_S/eps_tt at eps_tt = 10% computed (a) inclusively and (b) conditionally in bins of m_tt (the tt threshold is
taken from the tt fit sample's LOO scores in the same bin), then averaged over the signal's m_tt distribution.
Also the same for a supervised BDT on angles + m_tt versus m_tt alone (does adding angles help at fixed m_tt?).
Cache only.  Run: cd ttbar_bblvqq && OMP_NUM_THREADS=4 .venv/bin/python study_gonogo/verify/physics_skeptic/etat_conditional.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402

F = {"parton_trig": "osde/f_ttbar_parton_trig.npz", "gen": "osde/f_ttbar_gen.npz", "reco": "osde/f_ttbar_reco.npz"}


def cond_R(s, m_s, b, m_b, edges, wp):
    """Signal-weighted average of the per-bin enrichment, with the tt threshold set per m_tt bin."""
    num, tot = 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        ms, mb = (m_s >= lo) & (m_s < hi), (m_b >= lo) & (m_b < hi)
        if ms.sum() < 50 or mb.sum() < 200:
            continue
        thr = np.quantile(b[mb], 1 - wp)
        num += (s[ms] > thr).sum()
        tot += ms.sum()
    return num / tot / wp, tot


def main():
    out = {}
    rng = np.random.default_rng(3)
    for level in ("parton_trig", "gen", "reco"):
        f_all = osde.load_density(os.path.join(ROOT, F[level]))
        f_cv = f_all.truncated(f_all.significance() > 2.5)
        z = np.load(os.path.join(ROOT, "study_gonogo/cache/tt_fit.npz"))
        b = -f_cv.loo({a: z[f"{level}/{a}"] for a in ANGLE_NAMES})
        m_b = z[f"{level}/mtt"]
        edges = np.quantile(m_b, np.linspace(0, 1, 11))          # tt m_tt deciles
        edges[0], edges[-1] = 0, 1e9
        out[level] = {}
        for key in ("etat", "H350", "stop", "nospin", "tt_indep"):
            zs = np.load(os.path.join(ROOT, f"study_gonogo/cache/{key}.npz"))
            s = -f_cv({a: zs[f"{level}/{a}"] for a in ANGLE_NAMES})
            m_s = zs[f"{level}/mtt"]
            r = {}
            for wp in (0.01, 0.10):
                inc = float(np.mean(s > np.quantile(b, 1 - wp)) / wp)
                c, n = cond_R(s, m_s, b, m_b, edges, wp)
                reps = []
                for _ in range(100):
                    i, j = rng.integers(0, len(s), len(s)), rng.integers(0, len(b), len(b))
                    reps.append(cond_R(s[i], m_s[i], b[j], m_b[j], edges, wp)[0])
                r[f"wp{wp:g}"] = {"inclusive": round(inc, 3), "conditional_on_mtt_decile": round(float(c), 3),
                                  "cond_err": round(float(np.std(reps)), 3), "n_used": int(n)}
            out[level][key] = r
            print(level, key, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(HERE, "etat_conditional.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
