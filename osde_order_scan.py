#!/usr/bin/env python3
"""
How many terms does each angle need?  One-dimensional order scan per angle and level.

For an orthonormal 1-D basis φ_k, adding term k changes the integrated squared error by
σ_k^2 - c_k^2; with |ĉ_k|^2 - σ_k^2 an unbiased estimate of c_k^2, the cross-validation-like
criterion  J(K) = Σ_{k<=K} (2 σ_k^2 - |ĉ_k|^2)  is minimized at the best order K.
(Fourier order m contributes its cos and sin terms together.)

Mapping to the 10-D basis of plotting/osde.py:
  cos θ (Legendre degree)  and φ (harmonic order)   -> j1max
  cos θ' (Legendre degree)                          -> j2max
  φ' (harmonic order)                               -> min(j1max, j2max)
  cos Θ (Legendre degree K)                         -> n_legendre = K + 1
  Y (Hermite order K, scale s)                      -> n_hermite = K + 1
(likewise for the hadronic-side angles)

Output: osde/order_scan.json and a printed table.
"""
import argparse
import json
import os
import sys
from math import factorial, pi, sqrt

import numpy as np
from numpy.polynomial import hermite, legendre

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KIND = {"cos_theta": "leg", "phi": "four", "cos_theta_prime": "leg", "phi_prime": "four",
        "cos_theta_tilde": "leg", "phi_tilde": "four", "cos_theta_prime_tilde": "leg",
        "phi_prime_tilde": "four", "cos_Theta": "leg", "Y": "herm"}
KMAX = {"leg": 24, "four": 16, "herm": 24}


def terms(v, kind, k, scale):
    """Orthonormal basis values of order k (list of arrays: 1 for Legendre/Hermite, 2 for Fourier)."""
    if kind == "leg":
        return [sqrt((2 * k + 1) / 2) * legendre.Legendre.basis(k)(v)]
    if kind == "four":
        if k == 0:
            return [np.full_like(v, 1 / sqrt(2 * pi))]
        return [np.cos(k * v) / sqrt(pi), np.sin(k * v) / sqrt(pi)]
    t = v / scale
    return [hermite.hermval(t, [0] * k + [1]) * np.exp(-t * t / 2) / sqrt(2.0 ** k * factorial(k) * sqrt(pi) * scale)]


def best_order(v, kind, scale=1.0):
    n = len(v)
    J, cum = [], 0.0
    for k in range(KMAX[kind] + 1):
        for b in terms(v, kind, k, scale):
            c, var = b.mean(), b.var() / n
            cum += 2 * var - c * c
        J.append(cum)
    return int(np.argmin(J)), J


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--tag", default="down_truth")
    p.add_argument("--levels", default="parton,gen,reco")
    p.add_argument("--out", default=os.path.join(HERE, "osde", "order_scan.json"))
    args = p.parse_args()
    z = np.load(os.path.join(args.train, f"angles_{args.tag}.npz"))
    out = {}
    print(f"{'angle':24s} " + "  ".join(f"{l:>8s}" for l in args.levels.split(",")) + "   (best 1-D order)")
    for a, kind in KIND.items():
        row = []
        for lvl in args.levels.split(","):
            v = z[f"{lvl}/{a}"]
            # Hermite scale: std(Y) of the fit sample (the paper uses 1; std(Y) needs far fewer terms
            # when the measured Y distribution is narrower, as at gen/reco)
            scale = float(np.std(v)) if kind == "herm" else 1.0
            k, _ = best_order(v, kind, scale)
            out.setdefault(lvl, {})[a] = k
            if kind == "herm":
                out[lvl]["_y_scale"] = round(scale, 3)
            row.append(k)
        print(f"{a:24s} " + "  ".join(f"{k:8d}" for k in row))
    for lvl, o in out.items():
        j1 = max(o["cos_theta"], o["phi"], o["cos_theta_tilde"], o["phi_tilde"])
        j2 = max(o["cos_theta_prime"], o["cos_theta_prime_tilde"])
        jphi = max(o["phi_prime"], o["phi_prime_tilde"])
        o["_required"] = {"j1max": j1, "j2max": j2, "min_j1_j2_for_phi_prime": jphi,
                          "n_legendre": o["cos_Theta"] + 1, "n_hermite": o["Y"] + 1, "y_scale": o["_y_scale"]}
        print(f"{lvl}: needs j1max >= {j1}, j2max >= {j2}, min(j1max,j2max) >= {jphi}, "
              f"n_legendre >= {o['cos_Theta'] + 1}, n_hermite >= {o['Y'] + 1} (Hermite scale {o['_y_scale']})")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
