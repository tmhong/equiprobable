"""Extract per-event arrays (parton, truth-matched gen, chi2 gen) for the washout mechanism study.
usage: python code/analysis/study_washout/mechanism/extract.py NAME PATH MAXEV   -> results/analysis/study_washout/mechanism/data_NAME.npz
       OUT_SUFFIX=_h170 python code/analysis/study_washout/mechanism/extract.py hh PATH MAXEV   -> results/analysis/study_washout_h170/mechanism/data_hh.npz
PATH: the sample directory, relative to the repository top (e.g. samples/hh_bblvqq/run_seed2001)"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, awkward as ak
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))   # code/
import repo_paths
PROJ = repo_paths.REPO
SUFFIX = os.environ.get("OUT_SUFFIX", "")      # non-empty: write into the sibling results/analysis/study_washout<SUFFIX>/mechanism
OUT = repo_paths.results_dir("study_washout" + SUFFIX, "mechanism")
from plotting import make_plots
from lib import angles as A, physics as P

name, path, maxev = sys.argv[1], sys.argv[2], int(sys.argv[3])
ev, _ = make_plots.load_sample(name, os.path.join(PROJ, path), ["parton", "gen"], maxev,
                               {"had_fermion": "down", "jet_assignment": "truth"})
pa, ge = ev["parton"], ev["gen"]
n = len(ge)
import uproot
def truth_partons(root, nev):
    """Pythia-record partons after ISR recoil: b, bbar (status 23 from t/h), W_had quarks (status 23
    from W/h), prompt e/mu and nu (status 1 with W/h ancestor). (n,4) arrays px,py,pz,E."""
    t = uproot.open(root)["Delphes"]
    res = {k: [] for k in ("b", "bbar", "qd", "qu", "lep", "nu")}
    for lo in range(0, nev, 5000):
        a = t.arrays(["Particle.PID", "Particle.Status", "Particle.M1", "Particle.Px", "Particle.Py", "Particle.Pz", "Particle.E"],
                     entry_start=lo, entry_stop=min(lo + 5000, nev))
        part = ak.zip({"pid": a["Particle.PID"], "m1": a["Particle.M1"]})
        st = a["Particle.Status"]; pid = a["Particle.PID"]; apid = abs(pid)
        mpid = abs(pid[ak.where(a["Particle.M1"] >= 0, a["Particle.M1"], 0)])
        anc = abs(P._ancestor(part)[0])
        v = ak.zip({"px": a["Particle.Px"], "py": a["Particle.Py"], "pz": a["Particle.Pz"], "E": a["Particle.E"]})
        def one(mask, sortby=None):
            x = v[mask]
            if sortby is not None:
                x = x[ak.argsort(-np.hypot(x.px, x.py), axis=1)]
            x = ak.firsts(x)
            return np.stack([ak.to_numpy(ak.fill_none(x[c], np.nan)).astype(float) for c in ("px", "py", "pz", "E")], -1)
        heavy = (mpid == 6) | (mpid == 25)
        res["b"].append(one((pid == 5) & (st == 23) & heavy))
        res["bbar"].append(one((pid == -5) & (st == 23) & heavy))
        wq = (st == 23) & (apid <= 4) & ((mpid == 24) | (mpid == 25))
        res["qd"].append(one(wq & (apid % 2 == 1)))
        res["qu"].append(one(wq & (apid % 2 == 0)))
        bos = (anc == 24) | (anc == 25)
        res["lep"].append(one((st == 1) & ((apid == 11) | (apid == 13)) & bos, sortby=True))
        res["nu"].append(one((st == 1) & ((apid == 12) | (apid == 14)) & bos, sortby=True))
    return {k: np.concatenate(v) for k, v in res.items()}
TR = truth_partons(os.path.join(PROJ, path, "delphes_events.root"), n)

out = {}
arr = A.as_array

def m4(a):
    return np.sqrt(np.clip(a[:, 3]**2 - (a[:, :3]**2).sum(1), 0, None))

def angles10(tl, wl, lep, th, wh, fd):
    Y, cT = A.production_angles(tl, th)
    c, p, cp, pp = A.decay_angles(tl, wl, lep)
    ct, pt_, cpt, ppt = A.decay_angles(th, wh, fd)
    return dict(zip(A.ANGLE_NAMES, [Y, cT, c, p, cp, pp, ct, pt_, cpt, ppt]))

# ---------------- parton
P4 = {k: arr(pa[k]) for k in ["t_lep", "t_had", "w_lep", "w_had", "b_lep", "b_had", "lep", "nu", "q_down", "q_up", "b", "bbar"]}
for k in ["t_lep", "t_had", "w_lep", "w_had"]:
    out[f"p_m_{k}"] = m4(P4[k])
out["p_m_bb"] = m4(P4["b"] + P4["bbar"])
out["p_m_lvqq"] = m4(P4["w_lep"] + P4["w_had"])
out["p_m_tt"] = m4(P4["t_lep"] + P4["t_had"])
for a in A.ANGLE_NAMES:
    out[f"p_{a}"] = np.asarray(pa[a], float)

# ---------------- gen: chi2 result
out["g_reco_ok"] = np.asarray(ge.reco_ok)
out["g_presel"] = np.asarray(ge.presel)
out["g_njets"] = ge.njets; out["g_nbjets"] = ge.nbjets; out["g_nlep"] = ge.nleptons
for a in A.ANGLE_NAMES:
    out[f"c_{a}"] = np.asarray(ge[a], float)
G = {k: arr(ge[k]) for k in ["t_lep", "t_had", "w_lep", "w_had", "b_lep", "b_had", "q_down", "q_up", "nu"]}
for k in ["t_lep", "t_had", "w_lep", "w_had"]:
    out[f"c_m_{k}"] = m4(G[k])
out["c_chi2"] = np.asarray(ge.reco_chi2)
lep = arr(ge.lep)
out["c_nu_pz"] = G["nu"][:, 2]
out["p_nu_pz"] = P4["nu"][:, 2]

# ---------------- gen jets (all), truth matching
NJ = 10
jets = ak.pad_none(ge.jets, NJ, clip=True)
J = np.stack([ak.to_numpy(ak.fill_none(getattr(jets, c), np.nan)).astype(float) for c in ("px", "py", "pz", "E")], -1)  # (n,NJ,4)
jb = ak.to_numpy(ak.fill_none(jets.btag, False)).astype(bool)
jpt = np.hypot(J[..., 0], J[..., 1]); jeta = np.arcsinh(J[..., 2] / jpt); jphi = np.arctan2(J[..., 1], J[..., 0])

def eta_phi(a):
    pt = np.hypot(a[:, 0], a[:, 1]); return np.arcsinh(a[:, 2] / pt), np.arctan2(a[:, 1], a[:, 0])

def dr_to_jets(a):
    e, p = eta_phi(a)
    dphi = np.angle(np.exp(1j * (jphi - p[:, None])))
    d = np.hypot(jeta - e[:, None], dphi)
    return np.where(np.isfinite(d), d, 99.)

parts = ["b_had", "b_lep", "q_down", "q_up"]
blep_is_b = ak.to_numpy(ak.fill_none(pa.b_lep.pid, 5)) == 5        # LHE pairing (tt: true; hh: closest to m_t)
TRP = {"b_lep": np.where(blep_is_b[:, None], TR["b"], TR["bbar"]), "b_had": np.where(blep_is_b[:, None], TR["bbar"], TR["b"]),
       "q_down": TR["qd"], "q_up": TR["qu"], "lep": TR["lep"], "nu": TR["nu"]}
# (a') Pythia-truth partons after ISR recoil, no acceptance
wl_a = TRP["lep"] + TRP["nu"]; wh_a = TRP["q_down"] + TRP["q_up"]
tl_a = wl_a + TRP["b_lep"]; th_a = wh_a + TRP["b_had"]
for k, v in angles10(tl_a, wl_a, TRP["lep"], th_a, wh_a, TRP["q_down"]).items():
    out[f"a_{k}"] = v
out["a_m_t_lep"], out["a_m_t_had"], out["a_m_w_lep"], out["a_m_w_had"] = m4(tl_a), m4(th_a), m4(wl_a), m4(wh_a)
out["a_ok"] = np.all(np.isfinite(np.concatenate([tl_a, th_a], 1)), 1)
out["a_sys_pt"] = np.hypot((tl_a + th_a)[:, 0], (tl_a + th_a)[:, 1])
out["p_sys_pt"] = np.hypot((P4["t_lep"] + P4["t_had"])[:, 0], (P4["t_lep"] + P4["t_had"])[:, 1])
for k in ("b_lep", "b_had", "q_down", "q_up"):
    out[f"a_pt_{k}"] = np.hypot(TRP[k][:, 0], TRP[k][:, 1]); out[f"a_eta_{k}"] = eta_phi(TRP[k])[0] if False else np.arcsinh(TRP[k][:, 2] / np.hypot(TRP[k][:, 0], TRP[k][:, 1]))
D = np.stack([dr_to_jets(TRP[k]) for k in parts], 1)   # (n,4,NJ)
idx = np.full((n, 4), -1); used = np.zeros((n, NJ), bool)
# greedy global matching: smallest dR first
Dw = D.copy()
for _ in range(4):
    flat = Dw.reshape(n, -1).argmin(1)
    i, j = flat // NJ, flat % NJ
    dmin = Dw.reshape(n, -1)[np.arange(n), flat]
    good = dmin < 0.4
    rows = np.arange(n)[good]
    idx[rows, i[good]] = j[good]
    Dw[rows, i[good], :] = 99.; Dw[rows, :, j[good]] = 99.
    Dw[np.arange(n)[~good]] = 99.
matched = (idx >= 0).all(1) & np.isfinite(lep[:, 0])
out["t_matched"] = matched
rows = np.arange(n)
TJ = {k: np.where(matched[:, None], J[rows, np.clip(idx[:, i], 0, None)], np.nan) for i, k in enumerate(parts)}
out["t_idx"] = idx
# which chi2 jets equal the matched jets
def same(a, b):
    return np.all(np.abs(a - b) < 1e-3, axis=1)
chi_q = [arr(ge.q1), arr(ge.q2)]
out["t_chi_bhad_ok"] = same(G["b_had"], TJ["b_had"]); out["t_chi_blep_ok"] = same(G["b_lep"], TJ["b_lep"])
out["t_chi_bswap"] = same(G["b_had"], TJ["b_lep"]) & same(G["b_lep"], TJ["b_had"])
out["t_chi_qq_ok"] = (same(chi_q[0], TJ["q_down"]) | same(chi_q[0], TJ["q_up"])) & \
                     (same(chi_q[1], TJ["q_down"]) | same(chi_q[1], TJ["q_up"]))
out["t_in6"] = (idx < 6).all(1) & matched
out["t_bt"] = jb[rows, np.clip(idx[:, 0], 0, None)] & jb[rows, np.clip(idx[:, 1], 0, None)]

# (b) truth-matched jets + true nu
nu_true = TRP["nu"]
wl_b = lep + nu_true
wh_b = TJ["q_down"] + TJ["q_up"]
tl_b = wl_b + TJ["b_lep"]; th_b = wh_b + TJ["b_had"]
for k, v in angles10(tl_b, wl_b, lep, th_b, wh_b, TJ["q_down"]).items():
    out[f"t_{k}"] = v
out["t_m_t_lep"], out["t_m_t_had"], out["t_m_w_lep"], out["t_m_w_had"] = m4(tl_b), m4(th_b), m4(wl_b), m4(wh_b)

# (b2) truth-matched jets + MET nu with m_W constraint (solution closest to m_t for lv b)
met, mphi = ge.met, ge.met_phi
mpx, mpy = met * np.cos(mphi), met * np.sin(mphi)
pzs = P.neutrino_pz(lep[:, 0], lep[:, 1], lep[:, 2], lep[:, 3], mpx, mpy)
best = None
cands = []
for s in (0, 1):
    nu = np.stack([mpx, mpy, pzs[s], np.sqrt(mpx**2 + mpy**2 + pzs[s]**2)], -1)
    cands.append(nu)
mt = [np.abs(m4(lep + nu + TJ["b_lep"]) - 172.5) for nu in cands]
nu_m = np.where((mt[0] <= mt[1])[:, None], cands[0], cands[1])
wl_m = lep + nu_m; tl_m = wl_m + TJ["b_lep"]
for k, v in angles10(tl_m, wl_m, lep, th_b, wh_b, TJ["q_down"]).items():
    out[f"n_{k}"] = v
out["n_m_t_lep"], out["n_m_w_lep"] = m4(tl_m), m4(wl_m)
out["n_nu_pz"] = nu_m[:, 2]
# (b3) true neutrino but MET pT replaced (pT from GenMET, pz true) -- isolates the m_W pz constraint
# -- skipped; instead (b4): truth jets, chi2 nu choice irrelevant. keep (b),(b2),(c)

# ---------------- hypothesis-independent gen observables
bj = np.where(jb[..., None], J, np.nan)
order_b = np.argsort(~jb, axis=1, kind="stable")       # b-tagged first, pt order preserved
Jb = np.take_along_axis(J, order_b[..., None], 1)
Jl = np.take_along_axis(J, np.argsort(jb, axis=1, kind="stable")[..., None], 1)
nb = ge.nbjets; nj = ge.njets
b1, b2 = Jb[:, 0], Jb[:, 1]
l1, l2 = Jl[:, 0], Jl[:, 1]
out["h_m_bb"] = np.where(nb >= 2, m4(b1 + b2), np.nan)
def dR(a, b):
    ea, pa_ = eta_phi(a); eb, pb = eta_phi(b)
    return np.hypot(ea - eb, np.angle(np.exp(1j * (pa_ - pb))))
out["h_dR_bb"] = np.where(nb >= 2, dR(b1, b2), np.nan)
out["h_pt_bb"] = np.where(nb >= 2, np.hypot((b1 + b2)[:, 0], (b1 + b2)[:, 1]), np.nan)
okl = (nj - nb) >= 2
out["h_m_jj"] = np.where(okl, m4(l1 + l2), np.nan)            # two leading non-b jets
out["h_m_ljj"] = np.where(okl, m4(lep + l1 + l2), np.nan)     # visible h->WW* mass
out["h_dR_l_jj"] = np.where(okl, dR(lep, l1 + l2), np.nan)
vis = lep + l1 + l2
ptv = np.hypot(vis[:, 0], vis[:, 1]); mv = m4(vis)
etv = np.sqrt(ptv**2 + mv**2)
out["h_mT_WW"] = np.where(okl, np.sqrt(np.clip((etv + met)**2 - (vis[:, 0] + mpx)**2 - (vis[:, 1] + mpy)**2, 0, None)), np.nan)
out["h_mtw"] = ge.mtw
out["h_met"] = met
out["h_lep_pt"] = np.hypot(lep[:, 0], lep[:, 1])
out["h_HT"] = np.nansum(np.where(np.isfinite(jpt), jpt, 0), 1)
out["h_njets"] = nj; out["h_nbjets"] = nb
out["h_dR_lb_min"] = np.where(nb >= 1, np.fmin(dR(lep, b1), np.where(nb >= 2, dR(lep, b2), 99)), np.nan)
if OUT:
    os.makedirs(OUT, exist_ok=True)
np.savez_compressed(os.path.join(OUT, f"data_{name}.npz"), **{k: v for k, v in out.items() if v is not None})
print("saved", name, n, "matched", matched.mean(), "reco_ok", out["g_reco_ok"].mean())
