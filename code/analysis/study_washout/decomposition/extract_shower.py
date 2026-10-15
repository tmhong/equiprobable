"""
Pythia-record truth for the decomposition: for each role (b_lep, b_had, q_down, q_up, lep, nu)
  h_<role>  the decay product as first produced by the (last-copy) t / W / h, i.e. AFTER the
            ISR recoil of the hard system but before its own FSR  ("post-ISR parton")
  l_<role>  its last copy before hadronization (after FSR)            ("pre-hadronization")
Roles are assigned by PID from the LHE event (same pairing as the parton level).
usage (from the repository top; SAMPLE_DIR and OUT relative to the current directory):
  python code/study_washout/decomposition/extract_shower.py tt samples/ttbar_bblvqq/run_seed1001 100000 \
      results/study_washout/decomposition/cache_tt_shower.npz
(cache_hh_shower.npz goes to results/study_washout<OUT_SUFFIX>/decomposition, next to cache_hh.npz.)
"""
import sys, os, time
import numpy as np, awkward as ak, uproot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # code/
from plotting import io, physics

label, path, nmax, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
t0 = time.time()
parts, _ = io.read_lhe(os.path.join(path, "parton_events.lhe.gz"), nmax)
P = physics.build_parton(parts, np.ones(len(parts)))
ROLES = ["b_lep", "b_had", "q_down", "q_up", "lep", "nu"]
role_pid = {r: ak.to_numpy(ak.fill_none(P[r].pid, 0)).astype(int) for r in ROLES}
n = len(role_pid["lep"])
print(label, "LHE", n, f"{time.time()-t0:.0f}s", flush=True)

tree = uproot.open(os.path.join(path, "delphes_events.root"))["Delphes"]
br = ["Particle.PID", "Particle.Status", "Particle.M1", "Particle.D1", "Particle.D2",
      "Particle.Px", "Particle.Py", "Particle.Pz", "Particle.E"]
res = {f"{k}_{r}": np.full((n, 4), np.nan) for k in ("h", "l") for r in ROLES}
CH = 5000
for lo in range(0, n, CH):
    hi = min(n, lo + CH)
    a = tree.arrays(br, entry_start=lo, entry_stop=hi)
    counts = ak.to_numpy(ak.num(a["Particle.PID"]))
    off = np.concatenate([[0], np.cumsum(counts)[:-1]])
    evi = np.repeat(np.arange(hi - lo), counts)
    offs = off[evi]
    pid = ak.to_numpy(ak.flatten(a["Particle.PID"])).astype(int)
    st = ak.to_numpy(ak.flatten(a["Particle.Status"]))
    m1 = ak.to_numpy(ak.flatten(a["Particle.M1"]))
    d1 = ak.to_numpy(ak.flatten(a["Particle.D1"]))
    d2 = ak.to_numpy(ak.flatten(a["Particle.D2"]))
    p4 = np.stack([ak.to_numpy(ak.flatten(a[f"Particle.{c}"])) for c in ("Px", "Py", "Pz", "E")], 1)
    N = len(pid)
    mom = np.where(m1 >= 0, offs + m1, -1)
    mpid = np.where(mom >= 0, np.abs(pid[np.clip(mom, 0, N - 1)]), 0)
    first = (mom >= 0) & np.isin(mpid, (6, 24, 25)) & (np.abs(pid) != mpid) & \
            np.isin(np.abs(pid), (1, 2, 3, 4, 5, 11, 12, 13, 14)) & np.isin(st, (1, 23))
    # last copy: follow daughters with the same PID
    cur = np.where(first)[0]
    last = cur.copy()
    for _ in range(60):
        D1 = np.where(d1[last] >= 0, offs[last] + d1[last], -1)
        D2 = np.where(d2[last] >= 0, offs[last] + d2[last], -1)
        nxt = np.full(len(last), -1)
        for cand in (D1, D2, np.where(D1 >= 0, D1 + 1, -1)):
            okc = (cand >= 0) & (cand < N) & (nxt < 0)
            cc = np.clip(cand, 0, N - 1)
            okc &= (pid[cc] == pid[last]) & (evi[cc] == evi[last])
            nxt = np.where(okc, cc, nxt)
        if not (nxt >= 0).any():
            break
        last = np.where(nxt >= 0, nxt, last)
    fe, fpid = evi[cur], pid[cur]
    for r in ROLES:
        want = role_pid[r][lo:hi]
        sel = fpid == want[fe]
        # first occurrence per event
        e_sel, idx_sel, last_sel = fe[sel], cur[sel], last[sel]
        _, u = np.unique(e_sel, return_index=True)
        res[f"h_{r}"][lo + e_sel[u]] = p4[idx_sel[u]]
        res[f"l_{r}"][lo + e_sel[u]] = p4[last_sel[u]]
    print(label, hi, f"{time.time()-t0:.0f}s", flush=True)
found = np.all([np.isfinite(res[f"h_{r}"][:, 0]) for r in ROLES], 0)
print("all roles found:", found.mean())
np.savez_compressed(out, found=found, **res)
