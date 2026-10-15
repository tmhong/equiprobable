#!/usr/bin/env python3
"""
Reco level with perfect jet assignment: Delphes jets matched to the LHE partons by Delta R.

For every event of a sample the four LHE quarks (b_had, b_lep, q_down, q_up; for non-tt̄ samples the b pairing is
the one closest to m_t, as in physics.build_parton) are matched one-to-one to the selected reco jets (pT > 25 GeV,
|eta| < 2.5), greedily, smallest Delta R first, Delta R < DR_MAX.  An event is kept when it passes the usual
preselection (1 lepton, >= 4 jets, >= 1 b-tag) and all four quarks are matched.  The jets then take the roles of
their partons (no chi2); the neutrino is still reconstructed from E_T^miss with the m_W constraint, keeping the root
whose m(l nu b_lep) is closer to m_t.  The ten angles are computed with plotting/angles.add_angles.

Writes
  <sample>/angles_down_dRmatch.npz     reco/<angle> for the matched events, the osde_fit.py cache format
                                       (fit with --config code/study_truthmatch/config_dRmatch.py --levels reco)
  results/study_truthmatch/data_<name>.npz   event indices, masses, the chi2-assignment angles of the same events,
                                       parton angles (for the tt̄_R weights) and the matching statistics

Run: OMP_NUM_THREADS=3 .venv/bin/python code/study_truthmatch/build_matched.py [--max-events N] [--dr 0.4] SAMPLE ...
(SAMPLE relative to the repo top, e.g. samples/ttbar_bblvqq/run_seed1001)
"""
import argparse
import json
import os
import sys

import awkward as ak
import numpy as np
import vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import make_plots  # noqa: E402
from plotting import angles, physics  # noqa: E402
import repo_paths  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}   # the chi2 comparison uses the paper's labelling
PARTS = ("b_had", "b_lep", "q_down", "q_up")
TAG = "down_dRmatch"
OUT = repo_paths.results_dir("study_truthmatch")


def arr4(v):
    """(n, 4) px, py, pz, E from a vector / awkward record (NaN where missing)."""
    return np.stack([np.asarray(ak.to_numpy(ak.fill_none(getattr(v, a), np.nan)), float)
                     for a in ("px", "py", "pz", "E")], axis=1)


def v4(a):
    return vector.zip({"px": a[:, 0], "py": a[:, 1], "pz": a[:, 2], "E": a[:, 3]})


def mass(a):
    return np.sqrt(np.clip(a[:, 3] ** 2 - a[:, 0] ** 2 - a[:, 1] ** 2 - a[:, 2] ** 2, 0, None))


def match(reco, parton, dr_max):
    """Greedy one-to-one matching of the four partons to the selected reco jets.  Returns jet index (n, 4), -1 = none."""
    n = len(reco)
    nj = int(ak.max(ak.num(reco.jets))) if n else 0
    jets = ak.pad_none(reco.jets, nj, clip=True)
    jeta = ak.to_numpy(ak.fill_none(jets.eta, np.nan)).astype(float)
    jphi = ak.to_numpy(ak.fill_none(jets.phi, np.nan)).astype(float)
    D = np.full((n, 4, nj), 99.0)
    for i, k in enumerate(PARTS):
        p = getattr(parton, k)
        peta = np.asarray(ak.to_numpy(ak.fill_none(p.eta, np.nan)), float)
        pphi = np.asarray(ak.to_numpy(ak.fill_none(p.phi, np.nan)), float)
        dphi = np.angle(np.exp(1j * (jphi - pphi[:, None])))
        d = np.hypot(jeta - peta[:, None], dphi)
        D[:, i, :] = np.where(np.isfinite(d), d, 99.0)
    idx = np.full((n, 4), -1)
    rows = np.arange(n)
    for _ in range(4):
        flat = D.reshape(n, -1).argmin(1)
        i, j = flat // nj, flat % nj
        good = D.reshape(n, -1)[rows, flat] < dr_max
        r = rows[good]
        idx[r, i[good]] = j[good]
        D[r, i[good], :] = 99.0
        D[r, :, j[good]] = 99.0
    return idx, jets


def build(path, dr_max, max_events):
    name = os.path.basename(os.path.normpath(path))
    ev, _ = make_plots.load_sample(name, path, ["parton", "reco"], max_events, OPTS)
    parton, reco = ev["parton"], ev["reco"]
    n = len(reco)
    assert len(parton) == n, (len(parton), n)

    idx, jets = match(reco, parton, dr_max)
    all4 = (idx >= 0).all(1)
    J = np.stack([np.asarray(ak.to_numpy(ak.fill_none(getattr(jets, a), np.nan)), float)
                  for a in ("px", "py", "pz", "E")], axis=-1)            # (n, nj, 4)
    rows = np.arange(n)
    M = {k: np.where(all4[:, None], J[rows, np.clip(idx[:, i], 0, None)], np.nan) for i, k in enumerate(PARTS)}

    # neutrino: E_T^miss + m_W constraint, root with m(l nu b_lep) closest to m_t
    lep = arr4(reco.lep)
    mpx, mpy = reco.met * np.cos(reco.met_phi), reco.met * np.sin(reco.met_phi)
    pzs = physics.neutrino_pz(lep[:, 0], lep[:, 1], lep[:, 2], lep[:, 3], mpx, mpy)       # (2, n)
    nus = [np.stack([mpx, mpy, pz, np.sqrt(mpx ** 2 + mpy ** 2 + pz ** 2)], 1) for pz in pzs]
    dm = [np.abs(mass(lep + nu + M["b_lep"]) - physics.M_TOP) for nu in nus]
    nu = np.where((dm[0] <= dm[1])[:, None], nus[0], nus[1])

    w_lep, w_had = lep + nu, M["q_down"] + M["q_up"]
    t_lep, t_had = w_lep + M["b_lep"], w_had + M["b_had"]
    m = physics.Event(weight=np.ones(n))
    m.lep, m.nu = v4(lep), v4(nu)
    m.q_down, m.q_up, m.b_lep, m.b_had = (v4(M[k]) for k in ("q_down", "q_up", "b_lep", "b_had"))
    m.w_lep, m.w_had, m.t_lep, m.t_had = v4(w_lep), v4(w_had), v4(t_lep), v4(t_had)
    angles.add_angles(m, OPTS["had_fermion"])
    ang_m = {a: np.asarray(m[a], float) for a in angles.ANGLE_NAMES}
    presel = np.asarray(reco.presel, bool)
    ok = presel & all4 & np.all([np.isfinite(v) for v in ang_m.values()], axis=0)

    # the chi2 assignment of the same events, for a like-for-like comparison
    ang_c = {a: np.asarray(reco[a], float) for a in angles.ANGLE_NAMES}
    chi_ok = np.asarray(reco.reco_ok, bool) & np.all([np.isfinite(v) for v in ang_c.values()], axis=0)
    C = {"b_had": arr4(reco.b_had), "b_lep": arr4(reco.b_lep), "q_down": arr4(reco.q_down), "q_up": arr4(reco.q_up)}
    same = lambda a, b: np.all(np.abs(a - b) < 1e-3, axis=1)
    chi_right = same(C["b_had"], M["b_had"]) & same(C["b_lep"], M["b_lep"]) & \
        same(C["q_down"], M["q_down"]) & same(C["q_up"], M["q_up"])
    chi_jets = (same(C["b_had"], M["b_had"]) | same(C["b_had"], M["b_lep"])) & \
        (same(C["q_down"], M["q_down"]) | same(C["q_down"], M["q_up"]))
    ang_p = {a: np.asarray(parton[a], float) for a in angles.ANGLE_NAMES}

    # osde_fit.py cache (reco level only), next to the sample
    flat = {f"reco/{a}": v[ok] for a, v in ang_m.items()}
    flat["reco/n_all"] = np.array(n)
    if max_events is None:
        np.savez(os.path.join(path, f"angles_{TAG}.npz"), **flat)

    stats = {"sample": path, "dr_max": dr_max, "n_all": int(n), "n_presel": int(presel.sum()),
             "n_all4_matched": int(all4.sum()), "n_matched_presel": int(ok.sum()),
             "n_chi2_reco_ok": int(chi_ok.sum()), "n_both": int((ok & chi_ok).sum()),
             "chi2_all_roles_right_of_matched": float(chi_right[ok & chi_ok].mean()) if (ok & chi_ok).any() else None,
             "chi2_same_jets_of_matched": float((chi_right | chi_jets)[ok & chi_ok].mean()) if (ok & chi_ok).any() else None,
             "per_parton_match_frac": {k: float((idx[:, i] >= 0).mean()) for i, k in enumerate(PARTS)}}
    out = {"ok": ok, "chi_ok": chi_ok, "chi_right": chi_right,
           **{f"m_{a}": v for a, v in ang_m.items()}, **{f"c_{a}": v for a, v in ang_c.items()},
           **{f"p_{a}": v for a, v in ang_p.items()},
           "m_m_w_had": mass(w_had), "m_m_t_lep": mass(t_lep), "m_m_t_had": mass(t_had), "m_mtt": mass(t_lep + t_had),
           "c_mtt": mass(arr4(reco.t_lep) + arr4(reco.t_had)), "n_generated": np.array(n),
           "c_m_w_had": mass(arr4(reco.w_had)), "c_m_t_lep": mass(arr4(reco.t_lep)), "c_m_t_had": mass(arr4(reco.t_had)),
           "parton_lep_trigger": np.asarray(parton.lep_trigger, bool)}
    suffix = "" if max_events is None else f"_test{max_events}"
    np.savez_compressed(os.path.join(OUT, f"data_{name}{suffix}.npz"), **out)
    json.dump(stats, open(os.path.join(OUT, f"stats_{name}{suffix}.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples", nargs="+")
    p.add_argument("--dr", type=float, default=0.4)
    p.add_argument("--max-events", type=int, default=None, help="test run (does not write the angle cache)")
    a = p.parse_args()
    for s in a.samples:
        build(os.path.join(repo_paths.REPO, s) if not os.path.isabs(s) else s, a.dr, a.max_events)


if __name__ == "__main__":
    main()
