"""Quick look: is the reco-level stop f_tt tail tied to the leptonic-top mass (neutrino reconstruction)?
R of f_tt (CV, LOO for tt) using the global tt cut, within bins of |m_t_lep - median_tt| / halfIQR_tt."""
import os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from recompute import get, basis, f_eval, self_term, FFILE, RES
from lib import osde
import repo_paths  # noqa: E402  (code/ is on sys.path via recompute)
for level in ("gen", "reco", "matched"):
    f = osde.load_density(repo_paths.results(FFILE[level]))
    tt = get("tt_fit", level); n = len(tt["mtt"])
    sig = np.abs(f.c) / np.where(f.c_err > 0, f.c_err, np.inf); m = sig > 2.5; c = np.where(m, f.c, 0)
    ft = basis(f, tt["X"])
    stt = -(n * f_eval(c, ft) - self_term(m, ft)) / (n - 1)
    out = []
    for key in ("stop", "etat"):
        d = get(key, level)
        ss = -f_eval(c, basis(f, d["X"]))
        for wp in (0.01, 0.1):
            t = np.quantile(stt, 1 - wp)
            for mk in ("m_t_lep", "m_t_had"):
                M = np.median(tt["m"][mk]); q1, q3 = np.quantile(tt["m"][mk], [.25, .75]); W = (q3 - q1) / 2
                ut, us = np.abs(tt["m"][mk] - M) / W, np.abs(d["m"][mk] - M) / W
                row = []
                for lo, hi in ((0, 1), (1, 3), (3, 1e9)):
                    bt, bs = (ut >= lo) & (ut < hi), (us >= lo) & (us < hi)
                    et, es = np.mean(stt[bt] > t), np.mean(ss[bs] > t)
                    row.append(f"[{lo},{hi if hi < 1e8 else 'inf'}) fS={bs.mean():.2f} fB={bt.mean():.2f} R={es / et:.2f}")
                out.append(f"  {key} wp {wp:g} {mk}: " + "; ".join(row))
    print(f"== {level}"); print("\n".join(out))
