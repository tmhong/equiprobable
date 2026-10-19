#!/usr/bin/env python3
"""
The candidate list of a reconstructed event: the "encoder" of the study_vae idea.

The paper keeps one assignment per event, the lowest chi2 over (jet permutation x neutrino solution).  Here every
selected event keeps its K lowest-chi2 candidates, each with its chi2 and its ten angles, so that a score can average
over the assignment ambiguity instead of committing to one candidate.  The chi2, the permutations (six leading jets,
min(n_b, 2) b-tagged jets in the two b slots) and the two W-mass neutrino solutions are those of
lib/physics._reconstruct_ttbar; the down-type W_had jet of each candidate is chosen by truth matching to the
parton-level quarks, as in the paper (jet_assignment "truth").

Writes results/analysis/study_vae/cand_<key>.npz: chi2 (n, K) (inf = no candidate), the ten angles <name> (n, K) and
reco_ok (n,) (the paper's selected events: preselection and a finite best chi2).

Run: OMP_NUM_THREADS=6 .venv/bin/python code/analysis/study_vae/candidates.py KEY=PATH[,PATH...] ... [--k 8]
"""
import argparse
import os
import sys
from itertools import permutations

import awkward as ak
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CODE)
import repo_paths  # noqa: E402
from lib import physics  # noqa: E402
from lib.angles import ANGLE_NAMES, as_array, decay_angles, production_angles  # noqa: E402
from plotting import make_plots  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}


def eta_phi(p):
    """Pseudorapidity and azimuth of (n, 4) four-vectors."""
    pt = np.hypot(p[:, 0], p[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.arcsinh(p[:, 2] / pt), np.arctan2(p[:, 1], p[:, 0])


def delta_r(a, b):
    ea, pa = eta_phi(a)
    eb, pb = eta_phi(b)
    dphi = (pa - pb + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(ea - eb, dphi)


def candidates(ev, parton, k):
    """chi2 (n, k) and the ten angles {name: (n, k)} of the k lowest-chi2 candidates of every event."""
    c = physics.CUTS
    nmax = c["reco_max_jets"]
    nev = len(ev.weight)
    jets = ak.pad_none(ev.jets[:, :nmax], nmax, clip=True)

    def dense(x, fill=np.nan):
        return ak.to_numpy(ak.fill_none(x, fill)).astype(float)

    jpx, jpy, jpz, je = (dense(getattr(jets, a)) for a in ("px", "py", "pz", "E"))
    jb = dense(jets.btag, 0).astype(bool)
    lep = as_array(ev.lep)                                            # (n, 4)
    lpx, lpy, lpz, le = lep.T
    mpx, mpy = ev.met * np.cos(ev.met_phi), ev.met * np.sin(ev.met_phi)
    nupz = physics.neutrino_pz(lpx, lpy, lpz, le, mpx, mpy)          # (2, n)
    nue = np.sqrt(mpx**2 + mpy**2 + nupz**2)
    wl = np.stack([lpx + mpx + 0 * nupz, lpy + mpy + 0 * nupz, lpz + nupz, le + nue])   # (4, 2, n)

    def mass(px, py, pz, e):
        return np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0, None))

    perms = np.array([p for p in permutations(range(nmax), 4) if p[2] < p[3]])
    bh, bl, q1, q2 = perms.T
    need = np.minimum(ev.nbjets, 2)
    nperm = len(perms)
    top_idx = np.zeros((nev, k), dtype=int)
    top_chi2 = np.full((nev, k), np.inf)
    for lo in range(0, nev, 20000):                                   # as in physics._reconstruct_ttbar
        s = slice(lo, lo + 20000)
        J = np.stack([jpx[s], jpy[s], jpz[s], je[s]])
        W = J[:, :, q1] + J[:, :, q2]
        TH = W + J[:, :, bh]
        chi2_had = ((mass(*W) - physics.M_W) / c["sigma_w"])**2 + ((mass(*TH) - physics.M_TOP) / c["sigma_top"])**2
        chi2 = np.concatenate([
            chi2_had + ((mass(*(wl[:, isol, s, None] + J[:, :, bl])) - physics.M_TOP) / c["sigma_top"])**2
            for isol in (0, 1)], axis=1)                                # (n, 2 nperm)
        ntag_slots = np.tile(jb[s][:, bh].astype(int) + jb[s][:, bl].astype(int), 2)
        chi2 = np.where((ntag_slots == need[s, None]) & np.isfinite(chi2), chi2, np.inf)
        order = np.argsort(chi2, axis=1, kind="stable")[:, :k]
        top_idx[s] = order
        top_chi2[s] = np.take_along_axis(chi2, order, axis=1)

    rows = np.arange(nev)
    J = np.stack([jpx, jpy, jpz, je])                                 # (4, n, nmax)
    q_down, q_up = as_array(parton.q_down), as_array(parton.q_up)
    out = {a: np.full((nev, k), np.nan) for a in ANGLE_NAMES}
    for j in range(k):
        isol, iperm = top_idx[:, j] // nperm, top_idx[:, j] % nperm
        sel = perms[iperm]

        def jet(slot):
            return J[:, rows, sel[:, slot]].T                          # (n, 4)

        w_lep = wl[:, isol, rows].T
        t_lep = w_lep + jet(1)
        qa, qb = jet(2), jet(3)
        w_had = qa + qb
        t_had = w_had + jet(0)
        swap = delta_r(qa, q_down) + delta_r(qb, q_up) < delta_r(qa, q_up) + delta_r(qb, q_down)
        f_had = np.where(swap[:, None], qa, qb)                       # the down-type jet, by truth matching
        Y, cT = production_angles(t_lep, t_had)
        ct, ph, ctp, php = decay_angles(t_lep, w_lep, lep)
        ctt, pht, cttp, phtp = decay_angles(t_had, w_had, f_had)
        vals = dict(zip(ANGLE_NAMES, (Y, cT, ct, ph, ctp, php, ctt, pht, cttp, phtp)))   # ANGLE_NAMES order
        ok = np.isfinite(top_chi2[:, j])
        for a in ANGLE_NAMES:
            out[a][:, j] = np.where(ok, vals[a], np.nan)
    return top_chi2, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples", nargs="+", help="KEY=PATH[,PATH...] (paths relative to the repository top)")
    p.add_argument("--k", type=int, default=8, help="candidates kept per event")
    a = p.parse_args()
    out_dir = repo_paths.results_dir("study_vae")
    for arg in a.samples:
        key, paths = arg.split("=", 1)
        parts = []
        for path in paths.split(","):
            ev_all, _ = make_plots.load_sample(key, os.path.join(repo_paths.REPO, path), ["parton", "reco"], None, OPTS)
            ev, parton = ev_all["reco"], ev_all["parton"]
            chi2, ang = candidates(ev, parton, a.k)
            best_ok = np.asarray(ev.reco_ok, bool)
            # consistency: the first candidate is the paper's chi2 assignment
            same = np.allclose(chi2[best_ok, 0], np.asarray(ev.reco_chi2)[best_ok], rtol=1e-9, atol=1e-9)
            for nm in ANGLE_NAMES:                                     # the paper's reco angles of the best candidate
                same &= np.allclose(ang[nm][best_ok, 0], np.asarray(ev[nm], float)[best_ok], equal_nan=True, atol=1e-7)
            print(f"[{key}] {path}: {len(best_ok)} events, {best_ok.sum()} selected; best candidate = paper's chi2 "
                  f"assignment and angles: {same}", flush=True)
            parts.append(dict(chi2=chi2, reco_ok=best_ok, **ang))
        merged = {kk: np.concatenate([q[kk] for q in parts]) for kk in parts[0]}
        np.savez_compressed(os.path.join(out_dir, f"cand_{key}.npz"), **merged)
        print(f"[{key}] wrote cand_{key}.npz: {len(merged['reco_ok'])} events", flush=True)


if __name__ == "__main__":
    main()
