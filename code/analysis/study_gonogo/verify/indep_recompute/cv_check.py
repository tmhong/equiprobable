"""Is |c|/sigma > 2.5 the 5-fold LSCV choice for the reco (chi2) and the matched (dR) tt fits?"""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from lib import osde
import repo_paths
ANG = ["Y", "cos_Theta", "cos_theta", "phi", "cos_theta_prime", "phi_prime",
       "cos_theta_tilde", "phi_tilde", "cos_theta_prime_tilde", "phi_prime_tilde"]
T = [0, 0.5, 1.0, np.sqrt(2), 2, 2.5, 3, 4, 5, 7, 10, 15, 20]
z = np.load(os.path.join(repo_paths.RESULTS, "study_gonogo/cache/tt_fit.npz"))
Xr = {a: z["reco/" + a] for a in ANG}
m = np.load(os.path.join(repo_paths.RESULTS, "study_truthmatch/data_run_seed1001.npz"))
ok = m["ok"].astype(bool)
Xm = {a: m["m_" + a][ok] for a in ANG}
for name, X, ff in (("reco", Xr, "osde/f_ttbar_reco.npz"), ("matched", Xm, "study_truthmatch/osde_dRmatch/f_ttbar_reco.npz")):
    f = osde.load_density(os.path.join(repo_paths.RESULTS, ff))
    s1, s2, cnt = f.fold_sums(X, 5)
    cv, err, kept = osde.cv_threshold_scan(s1, s2, cnt, T)
    i = int(np.argmin(cv))
    print(name, "t_cv =", T[i], " LSCV(t)-min in units of its SE:",
          " ".join(f"{t:.2g}:{(c - cv[i]) / max(e, 1e-30):+.2f}" for t, c, e in zip(T, cv, err)))
