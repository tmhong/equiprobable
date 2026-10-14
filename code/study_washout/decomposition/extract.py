"""Extract aligned parton + gen arrays needed for the parton->gen decomposition into an npz cache."""
import sys, os, time
import numpy as np, awkward as ak
sys.path.insert(0, '/Users/TMHONG/Claude-code/ttbar_bblvqq')
import make_plots
from plotting.angles import as_array, ANGLE_NAMES

label, path, nmax, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
t0 = time.time()
ev, xsec = make_plots.load_sample(label, path, ['parton', 'gen'], nmax,
                                  {"had_fermion": "down", "jet_assignment": "truth"})
P, G = ev['parton'], ev['gen']
print(label, len(P), len(G), f"{time.time()-t0:.0f}s")
d = {}
for k in ["b_lep", "b_had", "q_down", "q_up", "lep", "nu"]:
    d[f"p_{k}"] = as_array(P[k])
for a in ANGLE_NAMES:
    d[f"p_ang_{a}"] = np.asarray(P[a], float)
    d[f"g_ang_{a}"] = np.asarray(G[a], float)
for k in ["b_lep", "b_had", "q_down", "q_up", "lep", "nu"]:
    d[f"g_{k}"] = as_array(G[k])
d["g_reco_ok"] = np.asarray(G.reco_ok, bool)
d["g_met"] = np.asarray(ak.fill_none(G.met, np.nan), float)
d["g_met_phi"] = np.asarray(ak.fill_none(G.met_phi, np.nan), float)
d["g_nleptons"] = np.asarray(G.nleptons)
NJ = 20
jets = ak.pad_none(G.jets, NJ, clip=True)
for c in ["px", "py", "pz", "E", "pt", "eta", "phi"]:
    d[f"j_{c}"] = ak.to_numpy(ak.fill_none(getattr(jets, c), np.nan)).astype(float)
d["j_btag"] = ak.to_numpy(ak.fill_none(jets.btag, False)).astype(bool)
d["njets"] = ak.to_numpy(ak.num(G.jets))
np.savez_compressed(out, **d)
print("saved", out, f"{time.time()-t0:.0f}s")
