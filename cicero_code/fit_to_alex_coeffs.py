#!/usr/bin/env python3
"""
apply_coeffs.py
===============

Evaluate an ALREADY-FITTED OSDE coefficient tensor on a different sample.

Intended use: coefficients fitted on TRUTH-level angles, evaluated on
RECONSTRUCTED angles, to see how much of the truth-level discrimination
survives reconstruction.

Because the coefficients contain no information from the samples being
evaluated, there is no train/test split and no overfitting concern -- 100%
of both input samples is used. That is a real statistical gain over the
self-fitted version.

Basis conventions are imported from recon_osde.py so they cannot drift.
m_parameters, nN and the M_function argument order below were verified
identical between recon_osde.py and truth_osde.py.

Run:
    python3 apply_coeffs.py \
        --coeffs  osde_saved/coeffs_truth.npy \
        --top_file rootFiles/ttbar_angles_ttbar.root \
        --hh_file  rootFiles/hh_angles_ttbar.root

Optional:
    --topk 50          keep only the 50 largest-|c| coefficients
    --coeffs_hh FILE   also supply an HH-fitted tensor -> log-likelihood ratio
"""

import argparse
import os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the basis functions from the existing script rather than copying
# them, so the implementations are guaranteed identical. recon_osde.py guards
# its main() with __name__, so importing it runs nothing.
try:
    from recon_osde import M_function, hermite_func, legendre_func
except ImportError as exc:
    raise SystemExit(
        f"Could not import basis functions from recon_osde.py: {exc}\n"
        "Run this script from the directory containing recon_osde.py."
    )

# --- basis definition. MUST match the script that produced the coefficients. --
M_PARAMETERS = [(0,0,0,0), (1,0,0,0), (1,0,1,0), (1,0,-1,0),
                (0,1,0,0), (1,1,0,0), (1,1,1,0), (1,1,-1,0),
                (1,1,0,1), (1,1,0,-1), (1,1,1,1), (1,1,-1,-1),
                (1,1,1,-1), (1,1,-1,1), (0,2,0,0), (1,2,0,0),
                (1,2,1,0), (1,2,-1,0), (1,2,0,1), (1,2,0,-1),
                (1,2,1,1), (1,2,1,-1), (1,2,-1,1), (1,2,-1,-1)]
NN = 5

BRANCHES = ["t_theta", "t_phi", "t_thetaprime", "t_phistar",
            "t~_theta", "t~_phi", "t~_thetastar", "t~_phistar",
            "rapidity", "cosprodangle"]


def load_sample(path, tag):
    """Read the ten angle branches and convert the four cos(theta) to angles."""
    with uproot.open(path) as f:
        tree = f["ttbar"]
        b = tree.arrays(BRANCHES, library="np")
    for name in ["t_theta", "t_thetaprime", "t~_theta", "t~_thetastar"]:
        b[name] = np.arccos(np.clip(b[name], -1.0, 1.0))
    n = len(b["t_theta"])
    print(f"  {tag:<8}: {n} events from {os.path.basename(path)}")
    return b, n


def eval_g(coeffs, b, batch=2000):
    """Contract the coefficient tensor against the basis evaluated on sample b.

    g(Theta) = sum_ijkl c_ijkl phi_i(Omega_t) psi_j(Omega_tbar) H_k(Y) P_l(cosTheta)

    Batched over events: the intermediate of the full contraction would be
    24*24*5*N complex128, which is several GB for N ~ 80k.
    """
    n = len(b["t_theta"])
    out = np.zeros(n, dtype=np.float64)

    for s in range(0, n, batch):
        e = min(s + batch, n)
        sl = slice(s, e)

        top = np.array([M_function(b["t_phi"][sl], b["t_theta"][sl],
                                   b["t_phistar"][sl], b["t_thetaprime"][sl], *p)
                        for p in M_PARAMETERS])                     # (24, nb)
        bar = np.array([M_function(b["t~_phi"][sl], b["t~_theta"][sl],
                                   b["t~_phistar"][sl], b["t~_thetastar"][sl], *p)
                        for p in M_PARAMETERS])                     # (24, nb)
        her = np.array([hermite_func(b["rapidity"][sl], k)
                        for k in range(NN)], dtype=np.complex128)   # (5, nb)
        leg = np.array([legendre_func(b["cosprodangle"][sl], k)
                        for k in range(NN)], dtype=np.complex128)   # (5, nb)

        # NOTE: no np.conj here. The fit conjugated the basis when forming the
        # coefficients (c = <conj(phi psi H P)>); the reconstruction of the
        # density uses the un-conjugated basis.
        out[sl] = np.real(np.einsum("ijkl,ix,jx,kx,lx->x",
                                    coeffs, top, bar, her, leg, optimize=True))
        if (s // batch) % 10 == 0:
            print(f"    ... {e}/{n}", end="\r", flush=True)
    print(f"    ... {n}/{n} done   ")
    return out


def total_variation(x, y, bins):
    """0.5 * sum |p_x - p_y|: the best separation any single cut can achieve."""
    hx, _ = np.histogram(x, bins=bins)
    hy, _ = np.histogram(y, bins=bins)
    px = hx / max(hx.sum(), 1)
    py = hy / max(hy.sum(), 1)
    return 0.5 * np.abs(px - py).sum()


def roc(sig, bkg, n_pts=400):
    """ROC treating LARGE score as signal-like (ttbar). Returns eps_sig, eps_bkg, auc."""
    lo = min(sig.min(), bkg.min())
    hi = max(sig.max(), bkg.max())
    cuts = np.linspace(lo, hi, n_pts)
    es = np.array([(sig >= c).mean() for c in cuts])
    eb = np.array([(bkg >= c).mean() for c in cuts])
    order = np.argsort(eb)
    auc = np.trapezoid(es[order], eb[order])
    return es, eb, abs(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", required=True, help="fitted coefficient .npy")
    ap.add_argument("--coeffs_hh", default=None,
                    help="optional HH-fitted .npy -> log-likelihood ratio")
    ap.add_argument("--top_file", default="rootFiles/ttbar_angles_ttbar.root")
    ap.add_argument("--hh_file",  default="rootFiles/hh_angles_ttbar.root")
    ap.add_argument("--topk", type=int, default=0,
                    help="keep only the K largest-|c| coefficients (0 = keep all)")
    ap.add_argument("--plots_dir", default="plots/alex_coeff")
    ap.add_argument("--save_dir",  default="osde_alex_coeff")
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    coeffs = np.load(args.coeffs)
    expect = (len(M_PARAMETERS), len(M_PARAMETERS), NN, NN)
    print(f"\ncoefficients: {args.coeffs}  shape {coeffs.shape}")
    if coeffs.shape != expect:
        raise SystemExit(
            f"\nSHAPE MISMATCH: got {coeffs.shape}, expected {expect}.\n"
            f"  {expect} is the 4-block layout (top, topbar, rapidity, cosprod).\n"
            f"  A 3-block tensor would be {expect[:3]} and needs the einsum\n"
            f"  'ijk,ix,jx,kx->x' instead. Check which script made this file."
        )
    nz = np.count_nonzero(coeffs)
    print(f"  nonzero: {nz} / {coeffs.size} ({100*nz/coeffs.size:.1f}%)")

    if args.topk > 0:
        flat = np.abs(coeffs).ravel()
        thr = np.sort(flat)[::-1][min(args.topk, flat.size) - 1]
        keep = np.abs(coeffs) >= thr
        coeffs = np.where(keep, coeffs, 0)
        print(f"  truncated to top {args.topk}: {np.count_nonzero(coeffs)} kept")

    print("\nreading samples (full, no train/test split needed):")
    t_b, n_t = load_sample(args.top_file, "ttbar")
    h_b, n_h = load_sample(args.hh_file,  "HH")

    print("\nevaluating g on ttbar:")
    g_t = eval_g(coeffs, t_b)
    print("evaluating g on HH:")
    g_h = eval_g(coeffs, h_b)

    score_t, score_h, label = g_t, g_h, r"$g(\mathbf{\Theta})$"
    suffix = "g"

    if args.coeffs_hh:
        c_hh = np.load(args.coeffs_hh)
        if c_hh.shape != expect:
            raise SystemExit(f"--coeffs_hh shape {c_hh.shape} != {expect}")
        print("\nevaluating g_HH on ttbar:")
        gh_t = eval_g(c_hh, t_b)
        print("evaluating g_HH on HH:")
        gh_h = eval_g(c_hh, h_b)
        # A density estimate can go slightly negative where statistics are thin,
        # so take |.| and floor the denominator before dividing.
        neg = (np.sum(g_t < 0) + np.sum(g_h < 0) +
               np.sum(gh_t < 0) + np.sum(gh_h < 0))
        tot = 2 * (n_t + n_h)
        print(f"\n  negative density values: {neg}/{tot} ({100*neg/tot:.2f}%)")
        eps = 1e-14
        score_t = np.log((np.abs(g_t)  + eps) / (np.abs(gh_t) + eps))
        score_h = np.log((np.abs(g_h)  + eps) / (np.abs(gh_h) + eps))
        label = r"$\log\,[\,g_{t\bar t}/g_{hh}\,]$"
        suffix = "llr"

    # drop any non-finite values rather than letting them poison the binning
    mt = np.isfinite(score_t); mh = np.isfinite(score_h)
    if (~mt).any() or (~mh).any():
        print(f"  dropping non-finite: ttbar {np.sum(~mt)}, HH {np.sum(~mh)}")
    score_t, score_h = score_t[mt], score_h[mh]

    allv = np.concatenate([score_t, score_h])
    bins = np.linspace(np.percentile(allv, 0.1), np.percentile(allv, 99.9), 101)

    tv = total_variation(score_t, score_h, bins)
    es, eb, auc = roc(score_t, score_h)

    print("\n================= RESULT =================")
    print(f"  ttbar mean  : {score_t.mean():.6g}   std {score_t.std():.6g}")
    print(f"  HH    mean  : {score_h.mean():.6g}   std {score_h.std():.6g}")
    print(f"  TV distance : {tv:.4f}   (noise floor ~0.02 at these statistics)")
    print(f"  ROC AUC     : {auc:.4f}   (0.5 = no separation)")
    print("==========================================\n")

    # --- g distributions ---
    nt, _ = np.histogram(score_t, bins=bins, density=True)
    nh, _ = np.histogram(score_h, bins=bins, density=True)
    plt.figure(figsize=(7, 5))
    plt.stairs(nt, bins, label=r"$t\bar t$", linewidth=1.5)
    plt.stairs(nh, bins, label="HH", linewidth=1.5)
    plt.xlabel(label)
    plt.ylabel("density")
    plt.title(f"Truth-fitted coefficients applied to reco angles\nTV = {tv:.3f}, AUC = {auc:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.plots_dir}/applied_{suffix}_dist.png", dpi=140)
    plt.close()

    # --- ROC ---
    plt.figure(figsize=(6, 6))
    plt.plot(eb, es, label=f"OSDE (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="no separation")
    plt.xlabel(r"$\epsilon_{hh}$")
    plt.ylabel(r"$\epsilon_{t\bar t}$")
    plt.title("ROC: truth-fitted coefficients on reco angles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.plots_dir}/applied_{suffix}_roc.png", dpi=140)
    plt.close()

    np.savez(f"{args.save_dir}/applied_{suffix}.npz",
             score_tt=score_t, score_hh=score_h,
             eps_tt=es, eps_hh=eb, auc=auc, tv=tv)
    print(f"wrote {args.plots_dir}/ and {args.save_dir}/applied_{suffix}.npz\n")


if __name__ == "__main__":
    main()
