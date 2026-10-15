#!/usr/bin/env python3
"""Is the leave-one-out tt reference calibrated?  Split the reco tt fit sample in halves A/B, fit OSDE on A,
truncate at |c|/sigma > T using A's errors, then R = eps_B / eps_A(LOO) at fixed eps (should be 1)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # code/
sys.path.insert(0, ROOT)
from analysis.study_gonogo import analyze as A
from lib import osde
import repo_paths
f0 = osde.load_density(repo_paths.results(A.F_FILES["reco"]))
tt = A.load("tt_fit", "reco")
X = tt["X"]; n = len(X["Y"])
WPS = (0.005, 0.01, 0.02, 0.05, 0.1)
res = {T: [] for T in (0, 2.5)}
for split in range(6):
    rng = np.random.default_rng(100 + split)
    perm = rng.permutation(n); ia, ib = perm[: n // 2], perm[n // 2:]
    XA = {k: v[ia] for k, v in X.items()}; XB = {k: v[ib] for k, v in X.items()}
    f = osde.TTbarDensity(int(f0.n_legendre), int(f0.n_hermite), float(f0.y_scale), int(f0.j1max), int(f0.j2max)).fit(XA)
    for T in res:
        g = f if T == 0 else f.truncated(f.significance() > T)
        sa, sb = -g.loo(XA), -g(XB)
        res[T].append([np.mean(sb > np.quantile(sa, 1 - w)) / w for w in WPS])
for T, v in res.items():
    v = np.array(v)
    print(f"T={T}: R(held-out tt vs LOO tt) at {WPS}: mean " + " ".join(f"{m:.3f}" for m in v.mean(0)) +
          "  sd over splits " + " ".join(f"{s:.3f}" for s in v.std(0)) + f"  (stat per split ~ {np.sqrt(1/(0.01*n/2)):.2f}@1%)")
