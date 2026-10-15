"""Sensitivity of R(f_tt) at reco/matched to the truncation threshold, and a 2-fold check in which the coefficient
fit AND the 2.5-sigma mask are chosen on one half of tt and tt is scored on the other half (no selection leakage)."""
import os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from recompute import get, basis, f_eval, self_term, own_fit, R_at, SIG, FFILE, RES  # own helpers only
from lib import osde
import repo_paths  # noqa: E402  (code/ is on sys.path via recompute)
for level in ("reco", "matched"):
    f = osde.load_density(repo_paths.results(FFILE[level]))
    tt = get("tt_fit", level)
    n = len(tt["mtt"])
    ft = basis(f, tt["X"])
    sig = np.abs(f.c) / np.where(f.c_err > 0, f.c_err, np.inf)
    D = {k: get(k, level) for k in ("etat", "stop", "H350", "tt_indep")}
    F = {k: basis(f, d["X"]) for k, d in D.items()}
    print(f"== {level}")
    for t in (2.0, 2.5, 3.0, 4.0, 5.0):
        m = sig > t
        c = np.where(m, f.c, 0)
        stt = -(n * f_eval(c, ft) - self_term(m, ft)) / (n - 1)
        print(f"  t={t}: terms {m.sum():4d}  " + "  ".join(
            f"{k} {R_at(-f_eval(c, F[k]), stt, .01):.2f}/{R_at(-f_eval(c, F[k]), stt, .1):.2f}" for k in D))
    # 2-fold, mask chosen on the training half
    rng = np.random.default_rng(3)
    res = {k: [] for k in D}
    for rep in range(5):
        perm = rng.permutation(n)
        halves = (perm[: n // 2], perm[n // 2:])
        for a, b in ((0, 1), (1, 0)):
            ia, ib = halves[a], halves[b]
            fa = tuple(x[ia] for x in ft)
            fb = tuple(x[ib] for x in ft)
            c, e = own_fit(fa)
            m = np.abs(c) / np.where(e > 0, e, np.inf) > 2.5
            c = np.where(m, c, 0)
            sb = -f_eval(c, fb)
            for k in D:
                ss = -f_eval(c, F[k])
                res[k].append((R_at(ss, sb, .01), R_at(ss, sb, .1)))
    print("  2-fold honest (half-size fit, own 2.5-sigma mask), mean +- spread over 10 splits: " + "  ".join(
        f"{k} {np.mean([r[0] for r in v]):.2f}+-{np.std([r[0] for r in v]):.2f}/{np.mean([r[1] for r in v]):.2f}+-{np.std([r[1] for r in v]):.2f}"
        for k, v in res.items()))
