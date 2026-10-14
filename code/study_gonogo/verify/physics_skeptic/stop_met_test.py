#!/usr/bin/env python3
"""
Physics-skeptic check (no existing file is modified): is the stop's reco-level angular "anomaly" caused by the
neutralinos' contribution to E_T^miss corrupting the W-constrained neutrino, rather than by spin structure?

For the stop sample (and, as a control, the tt_indep sample) the gen and reco events are rebuilt exactly as in
make_plots.load_sample, then rebuilt a second time with an oracle E_T^miss from which the true neutralino pT
(LHE, pid 1000022) has been subtracted.  The angles are recomputed and scored with the same f_tt (CV-truncated) as
analyze.py; R = eps_S/eps_tt at eps_tt = 1% and 10% with the tt fit sample's leave-one-out scores as reference.
Also: supervised BDT (angles only), both versions; and R binned in the neutralino pT fraction of E_T^miss.

Run: cd ttbar_bblvqq && OMP_NUM_THREADS=4 .venv/bin/python study_gonogo/verify/physics_skeptic/stop_met_test.py
"""
import json
import os
import sys

import awkward as ak
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from plotting import angles, io, osde, physics  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

CHUNK = 10000
F_FILES = {"gen": "osde/f_ttbar_gen.npz", "reco": "osde/f_ttbar_reco.npz"}


def lhe_info(lhe):
    parts, _ = io.read_lhe(lhe, None)
    parton = physics.build_parton(parts, np.ones(len(parts)))
    inv = (parts.status == 1) & (abs(parts.pid) == 1000022)
    cpx = ak.to_numpy(ak.sum(parts.px[inv], axis=1))
    cpy = ak.to_numpy(ak.sum(parts.py[inv], axis=1))
    return parton, cpx, cpy


def build(root, parton, cpx, cpy, level, oracle):
    n_tot = io.delphes_entries(root)
    chunks = []
    for lo in range(0, n_tot, CHUNK):
        d = io.read_delphes(root, entry_start=lo, max_events=min(lo + CHUNK, n_tot))
        ev = physics.build_gen(d) if level == "gen" else physics.build_reco(d)
        if oracle:
            sl = slice(lo, lo + len(ev.met))
            mpx = ev.met * np.cos(ev.met_phi) - cpx[sl]
            mpy = ev.met * np.sin(ev.met_phi) - cpy[sl]
            ev.met, ev.met_phi = np.hypot(mpx, mpy), np.arctan2(mpy, mpx)
            physics._reconstruct_ttbar(ev)
        chunks.append(ev)
        del d
    ev = physics.Event.concat(chunks)
    angles.assign_had_quarks(ev, parton, "truth")
    angles.add_angles(ev, "down")
    X = {a: np.asarray(ev[a], float) for a in ANGLE_NAMES}
    ok = np.all([np.isfinite(v) for v in X.values()], axis=0) & np.asarray(ev.reco_ok)
    return {a: v[ok] for a, v in X.items()}, ok, np.hypot(cpx, cpy)


def R(s, b, wp):
    return float(np.mean(s > np.quantile(b, 1 - wp)) / wp)


def boot_R(s, b, wp, nb=200, seed=1):
    rng = np.random.default_rng(seed)
    v = [R(s[rng.integers(0, len(s), len(s))], b[rng.integers(0, len(b), len(b))], wp) for _ in range(nb)]
    return float(np.std(v))


def bdt(F_tt, F_s, seed=0):
    rng = np.random.default_rng(seed)
    ft, fs = rng.permutation(len(F_tt)) % 2, rng.permutation(len(F_s)) % 2
    o_tt, o_s = np.zeros(len(F_tt)), np.zeros(len(F_s))
    for f in (0, 1):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
        clf.fit(np.vstack([F_tt[ft != f], F_s[fs != f]]), np.r_[np.zeros((ft != f).sum()), np.ones((fs != f).sum())])
        o_tt[ft == f] = clf.predict_proba(F_tt[ft == f])[:, 1]
        o_s[fs == f] = clf.predict_proba(F_s[fs == f])[:, 1]
    return o_tt, o_s


def main():
    out = {}
    samples = {"stop": os.path.join(ROOT, "samples/stop_bblvqq/run_seed4002")}
    for level in ("gen", "reco"):
        f_all = osde.load_density(os.path.join(ROOT, F_FILES[level]))
        f_cv = f_all.truncated(f_all.significance() > 2.5)
        z = np.load(os.path.join(ROOT, "study_gonogo/cache/tt_fit.npz"))
        Xtt = {a: z[f"{level}/{a}"] for a in ANGLE_NAMES}
        s_tt = -f_cv.loo(Xtt)
        A_tt = np.column_stack([Xtt[a] for a in ANGLE_NAMES])
        zc = np.load(os.path.join(ROOT, "study_gonogo/cache/stop.npz"))
        for key, path in samples.items():
            parton, cpx, cpy = lhe_info(os.path.join(path, "parton_events.lhe.gz"))
            root = os.path.join(path, "delphes_events.root")
            res = {}
            for oracle in (False, True):
                X, ok, ptchi = build(root, parton, cpx, cpy, level, oracle)
                if not oracle:   # must reproduce the cache exactly
                    same = len(X["Y"]) == len(zc[f"{level}/Y"]) and all(np.allclose(X[a], zc[f"{level}/{a}"]) for a in ANGLE_NAMES)
                    res["reproduces_cache"] = bool(same)
                s = -f_cv(X)
                A_s = np.column_stack([X[a] for a in ANGLE_NAMES])
                b_tt, b_s = bdt(A_tt, A_s)
                tag = "oracle_met" if oracle else "nominal"
                r = {"n": int(len(s)), "eff_sel": float(ok.mean())}
                for wp in (0.01, 0.10):
                    r[f"osde_cv R@{wp:g}"] = [R(s, s_tt, wp), boot_R(s, s_tt, wp)]
                    r[f"bdt_angles R@{wp:g}"] = [R(b_s, b_tt, wp), boot_R(b_s, b_tt, wp)]
                # R of osde_cv vs the neutralino pT (true), nominal only
                if not oracle:
                    pt = ptchi[ok]
                    edges = np.quantile(pt, [0, 0.25, 0.5, 0.75, 1.0])
                    r["osde_cv R@0.01 by pT(chi chi) quartile"] = []
                    for i in range(4):
                        m = (pt >= edges[i]) & (pt <= edges[i + 1])
                        r["osde_cv R@0.01 by pT(chi chi) quartile"].append(
                            [float(edges[i]), float(edges[i + 1]), R(s[m], s_tt, 0.01), boot_R(s[m], s_tt, 0.01)])
                res[tag] = r
                print(level, key, tag, json.dumps(r), flush=True)
            out[f"{level}/{key}"] = res
    json.dump(out, open(os.path.join(HERE, "stop_met_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
