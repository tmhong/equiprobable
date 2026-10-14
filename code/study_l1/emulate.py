#!/usr/bin/env python3
"""
Bit-true fixed-point emulation of an L1 (ATLAS Phase-II L0 Global Trigger) implementation of the
"angles + Breit-Wigner" tt̄-density score, run on the reco-level samples.

    s = -ln( f_angles(Ω) - y0 ) + Σ_k ln[(m_k - M_k)^2 + W_k^2]        (high s = not tt̄)

f_angles: the paper-basis OSDE density results/osde/f_ttbar_reco.npz truncated to |c|/σ > 10 (31 terms);
masses: m_W,had, m_t,lep, m_t,had of the χ² tt̄ reconstruction.  See results/study_l1/design.md for the design.

Chain (every operator emulated with its rounding; numpy float64 carries exact fixed-point values,
which is bit-exact for W <= 24 because products of two <= 26-bit operands are exact in float64):

  L1 objects (pT, η, φ codes; MET Ex, Ey)
    -> decode LUTs (cos, sin φ; cosh, sinh, sech, tanh η) and 4 multipliers per object
    -> pairwise invariants m²_ij = 2 p_i·p_j (massless objects)  [shared by all hypotheses]
    -> ν p_z from the m_W constraint (reciprocal, sqrt units)
    -> χ² terms from LUTs indexed by m² (sqrt-free), m²_jjb and m²_lνb as sums of pairwise m²
    -> all C(N,2)(N-2)(N-3) assignments x 2 ν solutions, argmin
    -> masses (3 sqrt units) -> Breit-Wigner surprisal LUTs            (masses-only score ends here)
    -> ten angles without explicit boosts (aberration, invariant ratios, CORDIC atan2)
    -> 25 one-dimensional basis-function LUTs -> 27 real product terms -> f -> -ln LUT -> score

Inputs: results/study_l1/cache/*.npz from extract.py.  Output: results/study_l1/results.json and a printed table.
Deterministic (no random numbers except the bootstrap, which uses a fixed seed).

    python code/study_l1/emulate.py              # everything (~2 min)
    python code/study_l1/emulate.py --selftest   # only the consistency checks
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations, permutations
from math import comb, log2, pi, sqrt

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # code/
sys.path.insert(0, ROOT)
import repo_paths  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES, decay_angles, production_angles  # noqa: E402

M_W, M_TOP, SIG_W, SIG_T = 80.4, 172.5, 10.0, 15.0     # χ² of plotting/physics.py
DENSITY = os.path.join(repo_paths.RESULTS, "osde", "f_ttbar_reco.npz")
SIG_CUT = 10.0                                         # keep |ĉ|/σ > 10  (31 terms at reco)
MASSES = ("m_w_had", "m_t_lep", "m_t_had")

# Fixed-point formats: integer bits I (incl. sign for signed) of each signal class; F = W - I.
FMT = {
    "unit": 2,     # dimensionless in [-2, 2): cos, sin, sech, tanh, unit vectors, β, cos θ
    "hyp": 4,      # cosh η, sinh η (|η| < 2.5 -> < 6.2)
    "ratio": 5,    # γ, βγ (tops: γ < 16), boost-frame components
    "mom": 13,     # GeV: object momenta/energies (±4096)
    "momT": 14,    # GeV: composite momenta (tops, tt̄; ±8192)
    "m2": 22,      # GeV² unsigned: pairwise/composite invariant masses squared (m < 2048 GeV)
    "m2s": 23,     # GeV² signed (ν solver)
    "mass": 11,    # GeV unsigned: masses after sqrt
    "chi": 9,      # χ² term (unsigned, saturates at 512)
    "chis": 11,    # sum of three χ² terms
    "fval": 2,     # basis-function values (|value| < 2)
    "f": 2,        # f · 2^F_SHIFT (f < 5.5e-5 -> < 0.9)
    "s": 6,        # surprisal terms (±32)
    "angle": 1,    # azimuth in units of π, (-1, 1]
    "Y": 4,        # rapidity
}
F_SHIFT = 14
CHI_WINDOW = {"W": 2.0 ** 16, "T": 2.0 ** 18}         # χ² LUT index ranges in m² (m < 256 / 512 GeV)
BW_WINDOW = {"m_w_had": 512.0, "m_t_lep": 1024.0, "m_t_had": 1024.0}   # BW LUT ranges in m (GeV)
Y_WINDOW = 4.0

INPUT_SCHEMES = {
    # LSBs of the L1 objects (jets and lepton: pT, η, φ = 2π/2^bits; MET: Ex, Ey)
    "fine": {"pt": 0.25, "eta": 0.0125, "phi_bits": 9, "met": 0.25},
    "nominal": {"pt": 0.5, "eta": 0.025, "phi_bits": 8, "met": 1.0},
    "coarse": {"pt": 1.0, "eta": 0.1, "phi_bits": 6, "met": 2.0},
}


# =============================================================================================
# 1. Fixed-point arithmetic model and bit-true operator units
# =============================================================================================
class Arith:
    """Rounding model of the datapath.  W = None: IEEE float64 (the float reference)."""

    def __init__(self, W=None):
        self.W = W
        self.sat = {}

    def fx(self, x, I, signed=True, tag=None):
        """ap_fixed<W,I> (ap_ufixed if unsigned): round to nearest, saturate."""
        if self.W is None:
            return x
        F = self.W - I
        s = np.floor(np.ldexp(x, F) + 0.5)
        lo, hi = (-2.0 ** (self.W - 1), 2.0 ** (self.W - 1) - 1) if signed else (0.0, 2.0 ** self.W - 1)
        if tag is not None:
            with np.errstate(invalid="ignore"):
                nsat = int(np.count_nonzero((s < lo) | (s > hi)))
            self.sat[tag] = self.sat.get(tag, 0) + nsat
        return np.ldexp(np.clip(s, lo, hi), -F)

    def fl(self, x):
        """W-bit mantissa with free exponent: an operand after leading-zero count and shift."""
        if self.W is None:
            return x
        m, e = np.frexp(x)
        return np.ldexp(np.floor(np.ldexp(m, self.W - 1) + 0.5), e - (self.W - 1))

    def _kaddr(self):
        return min(10, self.W - 2)

    def recip(self, x):
        """1/x, x > 0: normalize; 2^k-entry seed ROM; one Newton-Raphson step; W-bit mantissa out."""
        if self.W is None:
            return 1.0 / x
        m, e = np.frexp(self.fl(x))                     # x = m 2^e, m in [0.5, 1)
        k = self._kaddr()
        a = np.floor((2 * m - 1) * 2 ** k)
        y = self.fx(1.0 / (0.5 + (a + 0.5) / 2 ** (k + 1)), 2, False)
        y = self.fx(y * self.fx(2 - self.fx(m * y, 2, False), 2, False), 2, False)
        return self.fl(np.ldexp(y, -e))

    def rsqrt(self, x):
        """1/sqrt(x), x > 0: exponent parity + k-1 mantissa bits address the seed ROM; one NR step."""
        if self.W is None:
            return 1.0 / np.sqrt(x)
        m, e = np.frexp(self.fl(x))
        odd = (e % 2) != 0
        k = self._kaddr() - 1
        a = np.floor((2 * m - 1) * 2 ** k)
        mc = 0.5 + (a + 0.5) / 2 ** (k + 1)
        mm, mmc = np.where(odd, m / 2, m), np.where(odd, mc / 2, mc)
        y = self.fx(1.0 / np.sqrt(mmc), 2, False)
        t = self.fx(mm * self.fx(y * y, 3, False), 2, False)
        y = self.fx(y * self.fx((3 - t) / 2, 2, False), 2, False)
        return self.fl(np.ldexp(y, -np.where(odd, (e + 1) // 2, e // 2)))

    def sqrt(self, x):
        """sqrt(x) = x · rsqrt(x) (0 for x <= 0)."""
        pos = x > 0
        r = self.rsqrt(np.where(pos, x, 1.0))
        return np.where(pos, self.fl(self.fl(x) * r), 0.0)

    def ln(self, x, K=None):
        """ln x, x > 0: (e-1)·ln2 + ROM[log2(K) mantissa bits] (ln of the bin centre); K=None exact."""
        m, e = np.frexp(self.fl(x))
        if K is None:
            lnm = np.log(2 * m)
        else:
            k = int(round(log2(K)))
            lnm = np.log(1 + (np.floor((2 * m - 1) * 2 ** k) + 0.5) / 2 ** k)
        lnm = self.fx(lnm, 1, False)
        ln2 = self.fx(np.log(2.0), 1, False)
        return (e - 1) * ln2 + lnm

    def atan2(self, y, x):
        """CORDIC vectoring (W-2 iterations, inputs normalized to W bits): angle in units of π."""
        if self.W is None:
            return np.arctan2(y, x) / pi
        W = self.W
        _, e = np.frexp(np.maximum(np.abs(x), np.abs(y)))
        X = np.floor(np.ldexp(x, W - 2 - e) + 0.5)
        Y = np.floor(np.ldexp(y, W - 2 - e) + 0.5)
        neg = X < 0
        Z = np.where(neg, np.where(Y >= 0, 1.0, -1.0) * 2.0 ** (W - 1), 0.0)   # pre-rotation by ±π
        X, Y = np.where(neg, -X, X), np.where(neg, -Y, Y)
        for i in range(W - 2):
            d = np.where(Y >= 0, 1.0, -1.0)
            X, Y = X + d * np.floor(np.ldexp(Y, -i)), Y - d * np.floor(np.ldexp(X, -i))
            Z = Z + d * np.floor(np.arctan(2.0 ** -i) / pi * 2.0 ** (W - 1) + 0.5)
        Z = np.where(Z <= -2.0 ** (W - 1), Z + 2.0 ** W, Z)
        return np.ldexp(Z, -(W - 1))


def lut(x, fn, lo, hi, K, periodic=False):
    """1-D function ROM with K entries over [lo, hi) (value at the bin centre); K = None: exact."""
    if K is None:
        return fn(x)
    i = np.floor((x - lo) * (K / (hi - lo)))
    i = np.mod(i, K) if periodic else np.clip(i, 0, K - 1)
    return fn(lo + (i + 0.5) * (hi - lo) / K)


# =============================================================================================
# 2. Data
# =============================================================================================
def load(name):
    z = np.load(os.path.join(repo_paths.RESULTS, "study_l1", "cache", f"{name}.npz"))
    return {k: z[k] for k in z.files}


def event_set(d, kind):
    """'offline': the reference events (offline preselection with ≥1 b-tag, χ² ok);
    'l1': exactly one lepton and ≥ 4 jets (no b-tag requirement: none at L1)."""
    if kind == "offline":
        return np.flatnonzero(d["reco_ok"])
    return np.flatnonzero((d["nleptons"] == 1) & (d["njets"] >= 4))


SMEAR = {"jet": 0.10, "lep": 0.05, "met": 15.0}   # resolution stress test: extra σ(pT)/pT, σ(Ex,y) GeV


def add_noise(D, seed=2024):
    """Fixed standard-normal numbers per event for the resolution stress test (deterministic)."""
    for i, (name, d) in enumerate(sorted(D.items())):
        rng = np.random.default_rng(seed + i)
        n = len(d["met"])
        d["z_jet"], d["z_lep"], d["z_met"] = rng.standard_normal((n, 8)), rng.standard_normal(n), rng.standard_normal((n, 2))


def l1_inputs(d, idx, N, scheme, smear=False):
    """L1 objects: N leading jets (pT, η, φ; NaN-padded), lepton, MET (Ex, Ey); quantized if scheme.
    smear: extra Gaussian resolution (SMEAR) before quantization, as a stress test; jets keep the
    offline pT order."""
    pt, eta, phi = (d[f"jet_{k}"][idx, :N].copy() for k in ("pt", "eta", "phi"))
    lpt, leta, lphi = (d[f"lep_{k}"][idx].copy() for k in ("pt", "eta", "phi"))
    ex, ey = d["met"][idx] * np.cos(d["met_phi"][idx]), d["met"][idx] * np.sin(d["met_phi"][idx])
    if smear:
        pt = pt * np.clip(1 + SMEAR["jet"] * d["z_jet"][idx, :N], 0.2, None)
        lpt = lpt * np.clip(1 + SMEAR["lep"] * d["z_lep"][idx], 0.2, None)
        ex, ey = ex + SMEAR["met"] * d["z_met"][idx, 0], ey + SMEAR["met"] * d["z_met"][idx, 1]
    if scheme is not None:
        q = INPUT_SCHEMES[scheme]
        dphi = 2 * pi / 2 ** q["phi_bits"]

        def qpt(x):
            return np.round(x / q["pt"]) * q["pt"]

        def qeta(x):
            return np.round(x / q["eta"]) * q["eta"]

        def qphi(x):
            c = np.round(x / dphi)
            return (np.mod(c + 2 ** (q["phi_bits"] - 1), 2 ** q["phi_bits"]) - 2 ** (q["phi_bits"] - 1)) * dphi

        pt, eta, phi, lpt, leta, lphi = qpt(pt), qeta(eta), qphi(phi), qpt(lpt), qeta(leta), qphi(lphi)
        ex, ey = np.round(ex / q["met"]) * q["met"], np.round(ey / q["met"]) * q["met"]
    return {"pt": pt, "eta": eta, "phi": phi, "lpt": lpt, "leta": leta, "lphi": lphi, "ex": ex, "ey": ey}


# =============================================================================================
# 3. Float reference chains
# =============================================================================================
def _p4(pt, eta, phi, m):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.stack([px, py, pz, np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)], -1)


def _mass(p):
    return np.sqrt(np.clip(p[..., 3] ** 2 - p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2, 0, None))


def _nu_pz(L, mex, mey, mw=M_W):
    lpx, lpy, lpz, le = L.T
    lpt2 = lpx ** 2 + lpy ** 2
    mu = 0.5 * mw ** 2 + lpx * mex + lpy * mey
    a = mu * lpz / lpt2
    disc = a ** 2 - (le ** 2 * (mex ** 2 + mey ** 2) - mu ** 2) / lpt2
    root = np.sqrt(np.clip(disc, 0, None))
    return np.stack([a - root, a + root])


def assignments(N):
    """(bh, bl, q1, q2) with q1 < q2 (the order of plotting/physics.py)."""
    return np.array([p for p in permutations(range(N), 4) if p[2] < p[3]])


def float_chain(d, idx, N=6, use_btag=True, massless=False, dtype="truth", scheme=None, smear=False):
    """Float χ² reconstruction (same algorithm as plotting/physics.py, generalized) + the ten angles
    from plotting/angles.py.  With N=6, use_btag, massive objects and truth down-type assignment
    it reproduces the offline reference exactly."""
    inp = l1_inputs(d, idx, N, scheme, smear)
    n = len(idx)
    jm = np.zeros((n, N)) if massless else d["jet_mass"][idx, :N]
    J = _p4(inp["pt"], inp["eta"], inp["phi"], jm)
    L = _p4(inp["lpt"], inp["leta"], inp["lphi"], np.zeros(n) if massless else d["lep_mass"][idx])
    mex, mey = inp["ex"], inp["ey"]
    pz = _nu_pz(L, mex, mey)
    NU = np.stack([np.broadcast_to(mex, pz.shape), np.broadcast_to(mey, pz.shape), pz,
                   np.sqrt(mex ** 2 + mey ** 2 + pz ** 2)], -1)
    perms = assignments(N)
    bh, bl, q1, q2 = perms.T
    Wh = J[:, q1] + J[:, q2]
    c_had = ((_mass(Wh) - M_W) / SIG_W) ** 2 + ((_mass(Wh + J[:, bh]) - M_TOP) / SIG_T) ** 2
    chi2 = np.concatenate([c_had + ((_mass(L[:, None] + NU[s][:, None] + J[:, bl]) - M_TOP) / SIG_T) ** 2
                           for s in (0, 1)], axis=1)
    if use_btag:
        jb = d["jet_btag"][idx, :N]
        need = np.minimum(d["nbjets"][idx], 2)
        nt = np.tile(jb[:, bh].astype(int) + jb[:, bl].astype(int), 2)
        chi2 = np.where((nt == need[:, None]) & np.isfinite(chi2), chi2, np.inf)
    else:
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
    best = np.argmin(chi2, axis=1)
    rows = np.arange(n)
    isol, ip = best // len(perms), best % len(perms)
    sel = perms[ip]
    o = {"ok": np.isfinite(chi2[rows, best]), "chi2": chi2[rows, best], "isol": isol, "sel": sel}
    b_had, b_lep, j1, j2 = (J[rows, sel[:, k]] for k in range(4))
    nu = NU[isol, rows]
    w_had, w_lep = j1 + j2, L + nu
    t_had, t_lep = w_had + b_had, w_lep + b_lep
    if dtype == "truth":
        swap = _dr(j1, d["parton_q_down"][idx]) + _dr(j2, d["parton_q_up"][idx]) < \
            _dr(j1, d["parton_q_up"][idx]) + _dr(j2, d["parton_q_down"][idx])
    else:                                   # 'softer': the lower-pT W_had jet is the down-type one
        swap = np.zeros(n, bool)
    down = np.where(swap[:, None], j1, j2)
    X = {}
    X["Y"], X["cos_Theta"] = production_angles(t_lep, t_had)
    X["cos_theta"], X["phi"], X["cos_theta_prime"], X["phi_prime"] = decay_angles(t_lep, w_lep, L)
    X["cos_theta_tilde"], X["phi_tilde"], X["cos_theta_prime_tilde"], X["phi_prime_tilde"] = \
        decay_angles(t_had, w_had, down)
    o["angles"] = X
    o["masses"] = {"m_w_had": _mass(w_had), "m_t_lep": _mass(t_lep), "m_t_had": _mass(t_had)}
    return o


def _dr(a, b):
    def ep(p):
        return np.arcsinh(p[:, 2] / np.hypot(p[:, 0], p[:, 1])), np.arctan2(p[:, 1], p[:, 0])
    ea, pa = ep(a)
    eb, pb = ep(b)
    return np.sqrt((ea - eb) ** 2 + ((pa - pb + pi) % (2 * pi) - pi) ** 2)


# =============================================================================================
# 4. The angular density as LUT products
# =============================================================================================
class DensityLUT:
    """f(Ω) = Σ_k w_k Re(c_k M_lep(a_k) M_had(b_k)) p_l(cosΘ) h_n(Y), conjugate pairs merged (w=2).
    Per-top factor M = R1(cosθ) R2(cosθ') exp(-i(m'φ + mφ')), R1 = sqrt(2j1+1)/sqrt(4π) d^j1_m'm,
    R2 = sqrt(2j2+1)/sqrt(4π) d^j2_m0: every factor is a 1-D function (one ROM each)."""

    def __init__(self, path=DENSITY, sig=SIG_CUT):
        full = osde.load_density(path)
        self.f = full.truncated(full.significance() > sig)
        f = self.f
        kept = np.flatnonzero(f.c)
        partner = f.conjugate_partner()
        self.terms = []
        for i in kept:
            p = partner[i]
            if p < i and p in kept:
                continue
            a, b, l, n = np.unravel_index(i, f.shape)
            self.terms.append({"flat": int(i), "lep": tuple(f.midx[a]), "had": tuple(f.midx[b]), "l": int(l),
                               "n": int(n), "c": complex(f.c.flat[i]), "w": 1.0 if p == i else 2.0,
                               "label": osde.describe(f, i)[0]})
        self.n_kept = len(kept)
        side = {"lep": set(), "had": set()}
        for t in self.terms:
            side["lep"].add(t["lep"])
            side["had"].add(t["had"])
        self.funcs = {s: sorted(v) for s, v in side.items()}
        self.r1 = {s: sorted({(j1, mp, m) for j1, j2, mp, m in v if j1 > 0}) for s, v in side.items()}
        self.r2 = {s: sorted({(j2, m) for j1, j2, mp, m in v if j2 > 0}) for s, v in side.items()}
        self.ls = sorted({t["l"] for t in self.terms if t["l"] > 0})
        self.ns = sorted({t["n"] for t in self.terms})
        self.prod_pairs = sorted({(t["l"], t["n"]) for t in self.terms})

    def n_onedim(self):
        """Number of distinct non-constant 1-D functions (ROMs): cos/sin of an azimuth count as 2."""
        n = len(self.ls) + len(self.ns)
        for s in ("lep", "had"):
            n += len(self.r1[s]) + len(self.r2[s])
            n += 2 * any(mp != 0 for _, _, mp, _ in self.funcs[s]) + 2 * any(m != 0 for *_, m in self.funcs[s])
        return n

    @staticmethod
    def _r1(j1, mp, m):
        return lambda c: sqrt(2 * j1 + 1) / sqrt(4 * pi) * osde.wigner_small_d(j1, mp, m, np.arccos(np.clip(c, -1, 1)))

    @staticmethod
    def _r2(j2, m):
        return lambda c: sqrt(2 * j2 + 1) / sqrt(4 * pi) * osde.wigner_small_d(j2, m, 0, np.arccos(np.clip(c, -1, 1)))

    def evaluate(self, X, A, K):
        """f · 2^F_SHIFT from the angles X (cos values; azimuths in units of π), datapath A, ROM size K."""
        fv, I = FMT["fval"], FMT["unit"]
        out = {}
        for s, (ph, ct, php, ctp) in {"lep": ("phi", "cos_theta", "phi_prime", "cos_theta_prime"),
                                      "had": ("phi_tilde", "cos_theta_tilde", "phi_prime_tilde",
                                              "cos_theta_prime_tilde")}.items():
            R1 = {k: A.fx(lut(X[ct], self._r1(*k), -1, 1, K), fv) for k in self.r1[s]}
            R2 = {k: A.fx(lut(X[ctp], self._r2(*k), -1, 1, K), fv) for k in self.r2[s]}
            c1, s1 = (A.fx(lut(X[ph], fn, -1, 1, K, True), I) for fn in (lambda a: np.cos(pi * a), lambda a: np.sin(pi * a)))
            c2, s2 = (A.fx(lut(X[php], fn, -1, 1, K, True), I) for fn in (lambda a: np.cos(pi * a), lambda a: np.sin(pi * a)))
            c0 = A.fx(np.full_like(X[ct], 1 / sqrt(4 * pi)), fv)
            for key in self.funcs[s]:
                j1, j2, mp, m = key
                r = A.fx((R1[(j1, mp, m)] if j1 else c0) * (R2[(j2, m)] if j2 else c0), fv)
                # exp(-i m' φ) exp(-i m φ'), |m'|, |m| <= 1
                er, ei = np.ones_like(r), np.zeros_like(r)
                if mp:
                    er, ei = c1, -mp * s1
                if m:
                    er, ei = (A.fx(er * c2 + ei * m * s2, I), A.fx(ei * c2 - er * m * s2, I)) if mp else (c2, -m * s2)
                out[(s, key)] = (A.fx(r * er, fv), A.fx(r * ei, fv))
        P = {l: A.fx(lut(X["cos_Theta"], lambda u, l=l: sqrt((2 * l + 1) / 2) * osde.legendre.legval(u, [0] * l + [1]),
                         -1, 1, K), fv) for l in self.ls}
        P[0] = A.fx(np.full_like(X["Y"], sqrt(0.5)), fv)
        sc = self.f.y_scale
        H = {n: A.fx(lut(X["Y"], lambda y, n=n: osde.hermite_functions(y, n + 1, sc)[:, n], -Y_WINDOW, Y_WINDOW, K),
                     fv) for n in self.ns}
        PH = {(l, n): A.fx(P[l] * H[n], fv) for l, n in self.prod_pairs}
        f = 0.0
        for t in self.terms:
            lr, li = out[("lep", t["lep"])]
            hr, hi = out[("had", t["had"])]
            zr, zi = A.fx(lr * hr - li * hi, fv), A.fx(lr * hi + li * hr, fv)
            cr, ci = (A.fl(np.array(t["w"] * x * 2.0 ** F_SHIFT)) for x in (t["c"].real, t["c"].imag))
            re = A.fx(cr * zr - ci * zi, FMT["f"])
            f = A.fx(f + A.fx(re * PH[(t["l"], t["n"])], FMT["f"]), FMT["f"], tag="f_acc")
        return f


# =============================================================================================
# 5. The bit-true L1 chain
# =============================================================================================
def l1_chain(d, idx, cfg, A, dens):
    """Bit-true chain from the L1 objects to masses, ten angles and f (before the calibrated LUTs).
    cfg: N (jets), inputs (scheme or None), K (score ROM size or None), Kchi (χ² ROM size or None),
    angles (bool)."""
    N, K, Kc = cfg["N"], cfg["K"], cfg["Kchi"]
    inp = l1_inputs(d, idx, N, cfg["inputs"], cfg.get("smear", False))
    n = len(idx)
    U, H, R, MOM, MOMT, M2 = FMT["unit"], FMT["hyp"], FMT["ratio"], FMT["mom"], FMT["momT"], FMT["m2"]
    valid = np.isfinite(inp["pt"])                                      # (n, N)
    pt = np.where(valid, inp["pt"], 0.0)
    eta = np.where(valid, inp["eta"], 0.0)
    phi = np.where(valid, inp["phi"], 0.0)
    # ---- stage 0: decode the codes (ROMs indexed by the η, φ codes) -> four-vectors
    cph, sph = A.fx(np.cos(phi), U), A.fx(np.sin(phi), U)
    ch, sh = A.fx(np.cosh(eta), H), A.fx(np.sinh(eta), H)
    J = np.stack([A.fx(pt * cph, MOM, tag="mom"), A.fx(pt * sph, MOM, tag="mom"),
                  A.fx(pt * sh, MOM, tag="mom"), A.fx(pt * ch, MOM, tag="mom")], -1)       # (n, N, 4)
    lcph, lsph = A.fx(np.cos(inp["lphi"]), U), A.fx(np.sin(inp["lphi"]), U)
    lch, lsh = A.fx(np.cosh(inp["leta"]), H), A.fx(np.sinh(inp["leta"]), H)
    lpt = inp["lpt"]
    L = np.stack([A.fx(lpt * lcph, MOM), A.fx(lpt * lsph, MOM), A.fx(lpt * lsh, MOM), A.fx(lpt * lch, MOM)], -1)
    ex, ey = A.fx(inp["ex"], MOM, tag="mom"), A.fx(inp["ey"], MOM, tag="mom")

    def m2(a, b):              # 2 a·b of massless objects: 4-DSP cascade, one rounding
        return A.fx(2 * (a[..., 3] * b[..., 3] - a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
                         - a[..., 2] * b[..., 2]), M2, signed=False, tag="m2")

    # ---- stage 1: pairwise invariants of the jets and lepton-jet
    m2jj = np.zeros((n, N, N))
    for i, j in combinations(range(N), 2):
        m2jj[:, i, j] = m2jj[:, j, i] = m2(J[:, i], J[:, j])
    m2lj = m2(L[:, None], J)                                                  # (n, N)
    # ---- stage 2: neutrino p_z from the m_W constraint
    mu = A.fx(0.5 * M_W ** 2 + L[:, 0] * ex + L[:, 1] * ey, FMT["m2s"], tag="m2s")
    kappa = A.fx(mu * A.fl(1.0 / lpt), MOM, tag="mom")          # 1/pT_l: ROM indexed by the lepton pT code
    D2 = kappa * kappa - ex * ex - ey * ey
    D = A.fx(A.sqrt(np.where(D2 > 0, A.fl(D2), 0.0)), MOM, tag="mom")
    NU = []
    for sgn in (-1.0, 1.0):
        pz = A.fx(kappa * lsh + sgn * D * lch, MOM, tag="mom")
        e = A.fx(A.sqrt(A.fl(ex * ex + ey * ey + pz * pz)), MOM, tag="mom")
        NU.append(np.stack([ex, ey, pz, e], -1))
    NU = np.stack(NU)                                                         # (2, n, 4)
    m2ln = m2(L[None], NU)                                                    # (2, n)
    m2nj = m2(NU[:, :, None], J[None])                                        # (2, n, N)
    # ---- stage 3: χ² terms (ROMs indexed by m²), hypotheses, argmin
    CH = FMT["chi"]

    def chi_lut(x, M, S, win):
        return A.fx(lut(x, lambda v: ((np.sqrt(np.clip(v, 0, None)) - M) / S) ** 2, 0.0, win, Kc), CH, False, tag="chi")

    chiW = chi_lut(m2jj, M_W, SIG_W, CHI_WINDOW["W"])                       # (n, N, N)
    m2t = np.zeros((n, N, N, N))
    for i, j, k in combinations(range(N), 3):
        v = A.fx(m2jj[:, i, j] + m2jj[:, i, k] + m2jj[:, j, k], M2, False, tag="m2")
        for a, b, c in permutations((i, j, k)):
            m2t[:, a, b, c] = v
    m2tl = A.fx(m2ln[:, :, None] + m2lj[None] + m2nj, M2, False, tag="m2")    # (2, n, N)
    chiTl = chi_lut(m2tl, M_TOP, SIG_T, CHI_WINDOW["T"])
    s_best = (chiTl[1] < chiTl[0]).astype(int)                               # ν solution per b_lep (tie: 0)
    chiTl_min = np.minimum(chiTl[0], chiTl[1])                               # (n, N)
    perms = assignments(N)
    bh, bl, q1, q2 = perms.T
    m2th = m2t[:, q1, q2, bh]                                                # (n, H)
    chiTh = chi_lut(m2th, M_TOP, SIG_T, CHI_WINDOW["T"])
    chi = A.fx(chiW[:, q1, q2] + chiTh + chiTl_min[:, bl], FMT["chis"], False, tag="chi")
    ok_h = valid[:, bh] & valid[:, bl] & valid[:, q1] & valid[:, q2]
    chi = np.where(ok_h, chi, np.inf)
    best = np.argmin(chi, axis=1)
    rows = np.arange(n)
    sel = perms[best]
    isol = s_best[rows, sel[:, 1]]
    out = {"sel": sel, "isol": isol, "chi2": chi[rows, best]}
    # ---- stage 4: masses of the chosen hypothesis (3 sqrt units)
    m2_sel = {"m_w_had": m2jj[rows, sel[:, 2], sel[:, 3]], "m_t_had": m2th[rows, best],
              "m_t_lep": m2tl[isol, rows, sel[:, 1]]}
    out["masses"] = {k: A.fx(A.sqrt(A.fl(v)), FMT["mass"], False, tag="mass") for k, v in m2_sel.items()}
    if not cfg.get("angles", True):
        return out
    # ---- stage 5: the ten angles of the chosen hypothesis, without explicit boosts
    jb, jl, j1, j2 = (sel[:, k] for k in range(4))
    nu = NU[isol, rows]
    obj = {"lep": L, "nu": nu, "bl": J[rows, jl], "q1": J[rows, j1], "q2": J[rows, j2], "bh": J[rows, jb]}

    def unitdir(cp, sp, e_):          # unit direction of a massless object from its (η, φ) code ROMs
        se, th = A.fx(1 / np.cosh(e_), U), A.fx(np.tanh(e_), U)
        return np.stack([A.fx(cp * se, U), A.fx(sp * se, U), th], -1)

    udir = {"lep": unitdir(lcph, lsph, inp["leta"])}
    for k, jj in (("bl", jl), ("q1", j1), ("q2", j2), ("bh", jb)):
        udir[k] = unitdir(cph[rows, jj], sph[rows, jj], eta[rows, jj])
    pm2 = {("lep", "nu"): m2ln[isol, rows], ("lep", "bl"): m2lj[rows, jl], ("nu", "bl"): m2nj[isol, rows, jl]}
    for a, ja in (("q1", j1), ("q2", j2), ("bh", jb), ("bl", jl)):
        pm2[("lep", a)] = m2lj[rows, ja]
        pm2[("nu", a)] = m2nj[isol, rows, ja]
    for (a, ja), (b, jb_) in combinations((("bl", jl), ("q1", j1), ("q2", j2), ("bh", jb)), 2):
        pm2[(a, b)] = m2jj[rows, ja, jb_]
    X = {}
    # production: Y from two logarithms; cos Θ from a dimensionless boost of the leptonic-top velocity
    tl = A.fx(obj["lep"] + obj["nu"] + obj["bl"], MOMT, tag="momT")
    th = A.fx(obj["q1"] + obj["q2"] + obj["bh"], MOMT, tag="momT")
    T = A.fx(tl + th, MOMT, tag="momT")
    tiny = 2.0 ** -8
    X["Y"] = A.fx(0.5 * (A.ln(np.maximum(T[:, 3] + T[:, 2], tiny), K) - A.ln(np.maximum(T[:, 3] - T[:, 2], tiny), K)),
                  FMT["Y"])
    MT2 = sum(pm2[k] for k in pm2)             # Σ of the 15 pairwise m² of the six objects
    MT = A.sqrt(A.fl(MT2))
    gam = A.fx(T[:, 3] * A.recip(MT), R, tag="ratio")
    beta = A.fx(T[:, :3] * A.recip(T[:, 3])[:, None], U)
    v = A.fx(tl[:, :3] * A.recip(tl[:, 3])[:, None], U)
    bv = A.fx(np.sum(beta * v, -1), U)
    g = A.fx(A.fl(T[:, 3] * T[:, 3]) * A.recip(A.fl(MT * (T[:, 3] + MT))), R, tag="ratio")   # γ²/(γ+1)
    kk = A.fx(gam - g * bv, R, tag="ratio")
    u = A.fx(v - kk[:, None] * beta, R, tag="ratio")
    X["cos_Theta"] = np.clip(A.fx(u[:, 2] * A.rsqrt(A.fl(np.sum(u * u, -1))), U), -1, 1)
    # decay angles of each top
    for side, (f_, o_, b_, sfx) in {"lep": ("lep", "nu", "bl", ""), "had": ("q2", "q1", "bh", "_tilde")}.items():
        t = tl if side == "lep" else th
        rP = A.rsqrt(A.fl(np.sum(t[:, :3] ** 2, -1)))
        rpT = A.rsqrt(A.fl(t[:, 0] ** 2 + t[:, 1] ** 2))
        z = A.fx(t[:, :3] * rP[:, None], U)
        y = np.stack([A.fx(-t[:, 1] * rpT, U), A.fx(t[:, 0] * rpT, U), np.zeros(n)], -1)
        x = np.stack([A.fx(y[:, 1] * z[:, 2], U), A.fx(-y[:, 0] * z[:, 2], U), A.fx(y[:, 0] * z[:, 1] - y[:, 1] * z[:, 0], U)], -1)
        mt2 = pm2[(f_, o_) if (f_, o_) in pm2 else (o_, f_)] + pm2[_key(pm2, f_, b_)] + pm2[_key(pm2, o_, b_)]
        rmt = A.rsqrt(A.fl(mt2))
        P = A.fl(np.sum(t[:, :3] ** 2, -1) * rP)
        ga = A.fx(t[:, 3] * rmt, R, tag="ratio")
        bga = A.fx(P * rmt, R, tag="ratio")
        bh_, fh_ = udir[b_], udir[f_]
        cb, bx, by = (A.fx(np.sum(bh_ * ax, -1), U) for ax in (z, x, y))
        cf, fxx, fyy = (A.fx(np.sum(fh_ * ax, -1), U) for ax in (z, x, y))
        nb = A.fx(bga - ga * cb, R, tag="ratio")
        db = A.fx(ga - bga * cb, R, tag="ratio")
        X["cos_theta" + sfx] = np.clip(A.fx(nb * A.recip(db), U), -1, 1)
        X["phi" + sfx] = A.atan2(-by, -bx)
        mfb, mob = pm2[_key(pm2, f_, b_)], pm2[_key(pm2, o_, b_)]
        X["cos_theta_prime" + sfx] = np.clip(A.fx((mfb - mob) * A.recip(np.maximum(mfb + mob, 1.0)), U), -1, 1)
        gz = A.fx(ga * cf - bga, R, tag="ratio")
        Yp = A.fx(db * A.fx(by * fxx - bx * fyy, U), R, tag="ratio")
        Xp = A.fx(-nb * A.fx(bx * fxx + by * fyy, U) - A.fx(bx * bx + by * by, U) * gz, R, tag="ratio")
        X["phi_prime" + sfx] = A.atan2(Yp, Xp)
    out["angles"] = X
    out["f"] = dens.evaluate(X, A, K)
    return out


def _key(pm2, a, b):
    return (a, b) if (a, b) in pm2 else (b, a)


# =============================================================================================
# 6. Calibration and score
# =============================================================================================
def calibrate(masses, f=None):
    """M, W (median, half-IQR) of each tt̄ mass; y0 = 2.5·min f and the clip floor (as study_salvage)."""
    cal = {k: (float(np.median(masses[k])), float(0.5 * np.subtract(*np.percentile(masses[k], [75, 25]))))
           for k in MASSES}
    if f is not None:
        fmin = float(np.min(f))
        y0 = 2.5 * fmin if fmin < 0 else 0.0
        cal["y0"], cal["zmin"] = y0, (fmin - y0) * 1e-3
    return cal


def score_float(masses, f, cal):
    sm = sum(np.log((masses[k] - cal[k][0]) ** 2 + cal[k][1] ** 2) for k in MASSES)
    if f is None:
        return {"mass": sm}
    sa = -np.log(np.clip(f - cal["y0"], 1e-300 + cal["zmin"], None))
    return {"mass": sm, "angles": sa, "full": sa + sm}


def score_fixed(masses, f, cal, A, K):
    """BW ROMs indexed by m (window per mass, K entries), -ln(f - y0) by the LN unit (f in 2^F_SHIFT units)."""
    S = FMT["s"]
    sm = 0.0
    for k in MASSES:
        M, Wd = cal[k]
        sm = A.fx(sm + A.fx(lut(masses[k], lambda m, M=M, Wd=Wd: np.log((m - M) ** 2 + Wd ** 2), 0.0,
                                BW_WINDOW[k], K), S, tag="s"), S + 1, tag="s")
    if f is None:
        return {"mass": sm}
    y0 = A.fx(cal["y0"], FMT["f"])
    zmin = max(cal["zmin"], 2.0 ** -(A.W - FMT["f"])) if A.W else cal["zmin"]
    z = np.maximum(A.fx(f - y0, FMT["f"] + 1), zmin)
    sa = A.fx(-A.ln(z, K) + F_SHIFT * np.log(2.0), S, tag="s")
    return {"mass": sm, "angles": sa, "full": A.fx(sa + sm, S + 1, tag="s")}


# =============================================================================================
# 7. Metrics
# =============================================================================================
def auc(s_tt, s_hh):
    return float(roc_auc_score(np.r_[np.zeros(len(s_tt)), np.ones(len(s_hh))], np.r_[s_tt, s_hh]))


def eff_tt_at_hh(s_tt, s_hh, eff_hh):
    """tt̄ fraction kept when the cut keeps a fraction eff_hh of hh (keep s > cut)."""
    cut = np.quantile(s_hh, 1 - eff_hh)
    return float(np.mean(s_tt > cut))


def tail(s_tt, s_hh, ref_tt, ref_hh, acc_tt):
    """At the cut keeping a fraction acc_tt of tt̄: hh efficiency, and the fraction of the events the
    reference (float) selects at its own cut that this score also selects (hh, tt̄)."""
    c, cr = np.quantile(s_tt, 1 - acc_tt), np.quantile(ref_tt, 1 - acc_tt)
    sel_h, ref_sel_h = s_hh > c, ref_hh > cr
    sel_t, ref_sel_t = s_tt > c, ref_tt > cr
    return {"eff_hh": float(sel_h.mean()), "eff_hh_ref": float(ref_sel_h.mean()),
            "overlap_hh": float((sel_h & ref_sel_h).sum() / max(ref_sel_h.sum(), 1)),
            "overlap_tt": float((sel_t & ref_sel_t).sum() / max(ref_sel_t.sum(), 1))}


def paired_bootstrap(s_tt, s_hh, r_tt, r_hh, n=200, seed=11):
    """Mean and std of AUC(s) - AUC(r) over bootstrap replicas of the same events."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        it, ih = rng.integers(0, len(s_tt), len(s_tt)), rng.integers(0, len(s_hh), len(s_hh))
        d.append(auc(s_tt[it], s_hh[ih]) - auc(r_tt[it], r_hh[ih]))
    return float(np.mean(d)), float(np.std(d))


# =============================================================================================
# 8. Resource and latency model (analytic; to be validated with Vitis HLS / hls4ml + Vivado)
# =============================================================================================
DEVICES = {   # AMD Versal; LUT, FF, DSP58, BRAM36, URAM288
    "VP1802": {"LUT": 3360896, "FF": 6721792, "DSP": 14352, "BRAM": 4941, "URAM": 2549},
    "VC1902": {"LUT": 899840, "FF": 1799680, "DSP": 1968, "BRAM": 967, "URAM": 463},
}
CLOCK_MHZ = 240.0
TMUX_BC = 48                    # GEP time-multiplexing period: a new event every 48 BC = 1.2 µs
OVERHEAD = 1.3                  # control, pipeline balancing and routing on LUT/FF (assumption)


def _ops(W, K, Kc):
    """Cost of one operator instance (LUT, FF, DSP58, BRAM36) and its latency in clock cycles, at
    datapath width W (all W <= 24 fit one 27x24 DSP58 multiplier)."""
    lw = int(np.ceil(log2(W)))

    def rom(k, w=W):                       # LUTRAM up to 256 entries, block RAM above (dual-port 1Kx18 = 0.5)
        if k <= 256:
            return {"LUT": 1.25 * w * int(np.ceil(k / 64)), "FF": w, "DSP": 0, "BRAM": 0, "lat": 1}
        return {"LUT": 0, "FF": w, "DSP": 0, "BRAM": max(0.5, np.ceil(k * w / 18432) / 2), "lat": 2}

    norm = 2 * W * lw                      # leading-zero count + barrel shifters (in and out)
    return {
        "mul": {"LUT": 0, "FF": W, "DSP": 1, "BRAM": 0, "lat": 3},
        "dot2": {"LUT": 0, "FF": W, "DSP": 2, "BRAM": 0, "lat": 4},
        "dot3": {"LUT": 0, "FF": W, "DSP": 3, "BRAM": 0, "lat": 5},
        "dot4": {"LUT": 0, "FF": W, "DSP": 4, "BRAM": 0, "lat": 6},
        "add": {"LUT": W, "FF": W, "DSP": 0, "BRAM": 0, "lat": 1},
        "cmpsel": {"LUT": 2 * W + 8, "FF": W + 8, "DSP": 0, "BRAM": 0, "lat": 1},
        "rom_K": rom(K), "rom_Kc": rom(Kc), "rom_in": rom(256), "rom_4k": rom(4096),
        "recip": {"LUT": norm + W, "FF": 5 * W, "DSP": 2, "BRAM": 0.5, "lat": 12},
        "rsqrt": {"LUT": norm + W, "FF": 6 * W, "DSP": 3, "BRAM": 0.5, "lat": 15},
        "sqrt": {"LUT": norm + W, "FF": 7 * W, "DSP": 4, "BRAM": 0.5, "lat": 18},
        "ln": {"LUT": norm + W, "FF": 3 * W, "DSP": 1, **{k: v for k, v in rom(K).items() if k == "BRAM"}, "lat": 8},
        "cordic": {"LUT": 3 * (W + 2) * (W - 2), "FF": 3 * (W + 2) * (W - 2), "DSP": 0, "BRAM": 0, "lat": W + 1},
    }


def resource_model(N, W, K, Kc, variant="full", arch="parallel", lanes=4, n_onedim=25, n_terms=27):
    """Analytic operator inventory per block -> LUT/FF/DSP/BRAM and pipeline latency (clock cycles).

    arch 'parallel': every operator instantiated; one event in flight (II = latency, which is far below the
        48-BC TMUX period, so no inter-event pipelining or delay lines are needed).
    arch 'tmux': loops serialized inside the TMUX period: one object decoder, one 4-DSP pairwise unit,
        3 χ² ROMs, `lanes` hypothesis units with running minima, one shared sqrt for the masses, one top
        unit used for both tops, one complex MAC for the density terms.
    LUT and FF include a factor OVERHEAD for control, multiplexers and pipeline balancing."""
    op = _ops(W, K, Kc)
    njj, nlj, nnj = comb(N, 2), N, 2 * N + 2
    ntrip, H = comb(N, 3), comb(N, 2) * (N - 2) * (N - 3)
    nobj = N + 1
    par = arch == "parallel"
    blocks = []

    def block(name, items, lat, ii=1, path=True):
        r = {"LUT": 0.0, "FF": 0.0, "DSP": 0.0, "BRAM": 0.0}
        for o, cnt in items:
            for k in r:
                r[k] += op[o][k] * cnt
        r["LUT"] *= OVERHEAD
        r["FF"] *= OVERHEAD
        blocks.append({"block": name, **{k: round(v, 1) for k, v in r.items()}, "lat": int(lat), "ii": int(ii),
                       "critical": path})

    if par:
        block(f"input decode ({nobj} objects: η/φ ROMs, 4 mul each)", [("rom_in", 4 * nobj), ("mul", 4 * nobj)], 4)
        block(f"pairwise m² jets+lepton ({njj + nlj} 4-DSP dots)", [("dot4", njj + nlj)], 6, path=False)
    else:
        block("input decode (1 object/clock)", [("rom_in", 4), ("mul", 4)], 4 + nobj, nobj)
        block(f"pairwise m² jets+lepton (1 dot unit, {njj + nlj} pairs)", [("dot4", 1)], 6 + njj + nlj, njj + nlj,
              path=False)
    block("ν p_z solver (1/pT ROM, 3 sqrt)", [("rom_4k", 1), ("dot2", 3), ("mul", 1), ("dot3", 3), ("sqrt", 3),
                                              ("add", 4)], 4 + 3 + 5 + 18 + 4 + 5 + 18)
    if par:
        block(f"ν-jet and ℓν m² ({nnj} dots)", [("dot4", nnj)], 6)
        block(f"χ² terms ({njj + ntrip + 2 * N} ROM lookups, ν choice)",
              [("rom_Kc", njj + ntrip + 2 * N), ("add", 2 * ntrip + 4 * N), ("cmpsel", N)],
              2 + op["rom_Kc"]["lat"] + 1)
        block(f"{H} hypotheses + argmin tree", [("add", 2 * H), ("cmpsel", H - 1)], 2 + int(np.ceil(log2(H))))
        block("masses: mux, 3 sqrt, 3 BW ROMs, sum", [("sqrt", 3), ("rom_K", 3), ("add", 2), ("cmpsel", 4)],
              1 + 18 + op["rom_K"]["lat"] + 2 + 1)
    else:
        block(f"ν-jet and ℓν m² (1 dot unit, {nnj} pairs)", [("dot4", 1)], 6 + nnj, nnj)
        nser = max(njj, ntrip, 2 * N)
        block(f"χ² terms (3 ROMs, {nser} clocks)", [("rom_Kc", 3), ("add", 4), ("cmpsel", 1)],
              2 + op["rom_Kc"]["lat"] + 1 + nser, nser)
        nh = int(np.ceil(H / lanes))
        block(f"{H} hypotheses ({lanes} lanes, running min)", [("add", 2 * lanes), ("cmpsel", 2 * lanes)],
              2 + nh + int(np.ceil(log2(max(lanes, 2)))), nh)
        block("masses: mux, 1 shared sqrt, BW ROMs, sum", [("sqrt", 1), ("rom_K", 3), ("add", 2), ("cmpsel", 4)],
              1 + 18 + 2 + op["rom_K"]["lat"] + 2 + 1, 3)
    if variant == "full":
        # production: 15 mul + 2 dot3, 2 ln, 1 sqrt, 4 recip, 1 rsqrt, ~35 adders (see l1_chain)
        block("angles: production (Y, cos Θ)", [("mul", 15), ("dot3", 2), ("ln", 2), ("sqrt", 1), ("recip", 4),
                                                ("rsqrt", 1), ("add", 35)],
              2 + 4 + 18 + 1 + 3 + 12 + 3 + 3 + 1 + 3 + 1 + 5 + 15 + 3)
        ntop = 2 if par else 1
        block(f"angles: top-decay unit x{ntop} (cos θ, φ, cos θ', φ')",
              [("mul", 20 * ntop), ("dot2", 8 * ntop), ("dot3", 5 * ntop), ("rsqrt", 3 * ntop), ("recip", 2 * ntop),
               ("cordic", 2 * ntop), ("rom_in", 4 * ntop), ("add", 12 * ntop)],
              2 + 5 + 15 + 3 + 4 + 5 + 4 + 4 + (W + 1) + (0 if par else 2), 1 if par else 2)
        nmul = 21 + 6 + 3 * n_terms
        if par:
            block(f"density: {n_onedim} 1-D ROMs, {n_terms} product terms, adder tree",
                  [("rom_K", n_onedim), ("mul", nmul), ("add", n_terms)],
                  op["rom_K"]["lat"] + 3 + 3 + 3 + 4 + 3 + int(np.ceil(log2(n_terms))))
        else:
            block(f"density: {n_onedim} 1-D ROMs, complex MAC over {n_terms} terms",
                  [("rom_K", n_onedim), ("mul", 8), ("dot2", 2), ("add", 2)],
                  op["rom_K"]["lat"] + 3 + 3 + 3 + 4 + 3 + n_terms, n_terms)
        block("-ln(f - y0) unit, final sum, threshold", [("ln", 1), ("add", 3), ("cmpsel", 1)], 1 + 8 + 1 + 1)
    tot = {k: round(sum(b[k] for b in blocks), 1) for k in ("LUT", "FF", "DSP", "BRAM")}
    # critical path: decode -> ν -> ν-jet m² -> χ² terms -> hypotheses -> masses; the full score continues
    # after the hypothesis mux (parallel to the mass sqrt) through max(production, top) -> density -> -ln.
    L = [b["lat"] for b in blocks]
    to_sel = L[0] + L[2] + L[3] + L[4] + L[5]
    crit = to_sel + L[6]
    if variant == "full":
        crit = max(crit, to_sel + 1 + max(L[7], L[8]) + L[9] + L[10])
    # schedule (start, end) per block; lane 0 = critical path, lane 1 = concurrent
    sched, t = [], 0
    for i, lane in ((0, 0), (2, 0), (3, 0), (4, 0), (5, 0)):
        sched.append({"block": blocks[i]["block"], "start": t, "end": t + L[i], "lane": lane})
        t += L[i]
    sched.insert(1, {"block": blocks[1]["block"], "start": L[0], "end": L[0] + L[1], "lane": 1})
    if variant == "full":
        sched.append({"block": blocks[6]["block"], "start": t, "end": t + L[6], "lane": 1})
        first, second = (7, 8) if L[7] >= L[8] else (8, 7)
        sched.append({"block": blocks[first]["block"], "start": t + 1, "end": t + 1 + L[first], "lane": 0})
        sched.append({"block": blocks[second]["block"], "start": t + 1, "end": t + 1 + L[second], "lane": 1})
        t += 1 + L[first]
        for i in (9, 10):
            sched.append({"block": blocks[i]["block"], "start": t, "end": t + L[i], "lane": 0})
            t += L[i]
    else:
        sched.append({"block": blocks[6]["block"], "start": t, "end": t + L[6], "lane": 0})
    ii = max(max(b["ii"] for b in blocks), 1)
    if par:
        ii = crit                      # one event in flight
    return {"N": N, "W": W, "K": K, "Kchi": Kc, "variant": variant, "arch": arch, "lanes": lanes if not par else None,
            "blocks": blocks, "schedule": sched, "total": tot, "latency_clk": int(crit), "ii_clk": int(ii),
            "latency_ns": round(crit / CLOCK_MHZ * 1e3, 1), "ii_ns": round(ii / CLOCK_MHZ * 1e3, 1),
            "tmux_period_clk": int(TMUX_BC * 25e-3 * CLOCK_MHZ),
            "frac": {dev: {k: round(tot[k] / DEVICES[dev][k], 5) for k in ("LUT", "FF", "DSP", "BRAM")}
                     for dev in DEVICES}}


# =============================================================================================
# 9. Driver
# =============================================================================================
class Study:
    def __init__(self):
        self.D = {k: load(k) for k in ("tt_test", "tt_calib", "hh")}
        add_noise(self.D)
        self.dens = DensityLUT()
        self.cache = {}

    def run_float(self, name, N=6, use_btag=True, massless=False, dtype="truth", scheme=None, kind="offline",
                  smear=False):
        out = {}
        for k, d in self.D.items():
            idx = event_set(d, kind)
            o = float_chain(d, idx, N, use_btag, massless, dtype, scheme, smear)
            if not o["ok"].all():
                raise RuntimeError(f"{name}: {(~o['ok']).sum()} events without a valid hypothesis")
            o["f"] = self.dens.f(o["angles"])
            out[k] = o
        cal = calibrate(out["tt_calib"]["masses"], out["tt_calib"]["f"])
        sc = {k: score_float(out[k]["masses"], out[k]["f"], cal) for k in ("tt_test", "hh")}
        sc["chi2"] = {"tt_test": out["tt_test"]["chi2"], "hh": out["hh"]["chi2"]}
        res = {"name": name, "cfg": {"N": N, "btag": use_btag, "massless": massless, "dtype": dtype,
                                     "inputs": scheme, "W": None, "K": None, "Kchi": None, "events": kind,
                                     "smear": smear, "angles": True},
               "cal": cal, "scores": sc, "reco": out}
        self.cache[name] = res
        return res

    def run_fixed(self, name, N, inputs="nominal", W=18, K=1024, Kchi=1024, kind="offline", angles=True,
                  smear=False):
        A = Arith(W)
        cfg = {"N": N, "inputs": inputs, "K": K, "Kchi": Kchi, "angles": angles, "smear": smear}
        out = {k: l1_chain(d, event_set(d, kind), cfg, A, self.dens) for k, d in self.D.items()}
        cal = calibrate(out["tt_calib"]["masses"], out["tt_calib"].get("f"))
        sc = {k: score_fixed(out[k]["masses"], out[k].get("f"), cal, A, K) for k in ("tt_test", "hh")}
        sc["chi2"] = {"tt_test": out["tt_test"]["chi2"], "hh": out["hh"]["chi2"]}
        res = {"name": name, "cfg": {"N": N, "btag": False, "massless": True, "dtype": "softer", "inputs": inputs,
                                     "W": W, "K": K, "Kchi": Kchi, "events": kind, "smear": smear, "angles": angles},
               "cal": cal, "scores": sc, "reco": out, "sat": dict(A.sat)}
        self.cache[name] = res
        return res


def _same_assignment(res, ref):
    same = {}
    for k in ("tt_test", "hh"):
        a, b = res["reco"][k], ref["reco"][k]
        if len(a["sel"]) == len(b["sel"]):
            same[k] = float(np.mean(np.all(a["sel"] == b["sel"], 1) & (a["isol"] == b["isol"])))
    return same


def summarize(res, ref=None, ref2=None, boot=True):
    """AUCs, working points, tail agreement and assignment agreement of one configuration.
    ref: the physics reference (float chain without input quantization);
    ref2: the same inputs in float arithmetic with exact functions (arithmetic and ROM fidelity)."""
    from scipy.stats import spearmanr
    sc = res["scores"]
    tt, hh = sc["tt_test"], sc["hh"]
    row = {"name": res["name"], "cfg": res["cfg"], "n_tt": int(len(tt["mass"])), "n_hh": int(len(hh["mass"])),
           "auc_full": auc(tt["full"], hh["full"]) if "full" in tt else None,
           "auc_mass": auc(tt["mass"], hh["mass"]),
           "auc_angles": auc(tt["angles"], hh["angles"]) if "angles" in tt else None,
           "auc_chi2min": auc(sc["chi2"]["tt_test"], sc["chi2"]["hh"]),
           "cal": {k: (list(v) if isinstance(v, tuple) else v) for k, v in res["cal"].items()}}
    for key in ("full", "mass"):
        if key in tt:
            row[f"eff_tt_at_hh_{key}"] = {str(e): eff_tt_at_hh(tt[key], hh[key], e) for e in (0.9, 0.5, 0.1)}
    if "full" in tt and boot:
        row["dauc_full_minus_mass"] = paired_bootstrap(tt["full"], hh["full"], tt["mass"], hh["mass"])
    if "sat" in res:
        row["saturations"] = {k: v for k, v in res["sat"].items() if v}
    if ref is not None:
        rt, rh = ref["scores"]["tt_test"], ref["scores"]["hh"]
        for key in ("full", "mass"):
            if key not in tt or key not in rt:
                continue
            row[f"tail_{key}"] = {str(a): tail(tt[key], hh[key], rt[key], rh[key], a) for a in (0.10, 0.05, 0.01)}
            if boot:
                row[f"dauc_{key}"] = paired_bootstrap(tt[key], hh[key], rt[key], rh[key])
        row["same_assignment_as_ref"] = _same_assignment(res, ref)
        row["ref"] = ref["name"]
    if ref2 is not None:
        fid = {"ref": ref2["name"], "same_assignment": _same_assignment(res, ref2)}
        for key in ("full", "mass"):
            if key not in tt or key not in ref2["scores"]["tt_test"]:
                continue
            a = np.r_[tt[key], hh[key]]
            b = np.r_[ref2["scores"]["tt_test"][key], ref2["scores"]["hh"][key]]
            dd = a - b - np.median(a - b)
            fid[f"score_rms_{key}"] = float(np.sqrt(np.mean(dd ** 2)))
            fid[f"score_q99_{key}"] = float(np.quantile(np.abs(dd), 0.99))
            fid[f"rank_corr_{key}"] = float(spearmanr(a, b).statistic)
            same = np.concatenate([np.all(res["reco"][k]["sel"] == ref2["reco"][k]["sel"], 1) &
                                   (res["reco"][k]["isol"] == ref2["reco"][k]["isol"]) for k in ("tt_test", "hh")])
            ds = a[same] - b[same]
            fid[f"score_rms_same_assignment_{key}"] = float(np.sqrt(np.mean((ds - np.median(ds)) ** 2)))
        row["fidelity"] = fid
    return row


def selftest(st):
    """The fixed-point code path with W = None, exact functions and unquantized inputs must reproduce
    the float chain (massless, no b-tag, softer down-type) event by event."""
    print("selftest: fixed-point code path in float mode vs float chain (plotting/angles.py)")
    ok = True
    for N in (6, 5, 4):
        fl = st.run_float(f"_float{N}", N, False, True, "softer")
        fx = st.run_fixed(f"_fx{N}", N, None, None, None, None)
        for k in ("tt_test", "hh"):
            a, b = fx["reco"][k], fl["reco"][k]
            same = np.mean(np.all(a["sel"] == b["sel"], 1) & (a["isol"] == b["isol"]))
            dm = max(np.max(np.abs(a["masses"][m] - b["masses"][m])) for m in MASSES)
            da = 0.0
            for nm in ANGLE_NAMES:
                x, y = a["angles"][nm], b["angles"][nm]
                if nm.startswith("phi"):
                    x = x * pi
                    dd = np.abs((x - y + pi) % (2 * pi) - pi)
                else:
                    dd = np.abs(x - y)
                da = max(da, float(np.max(dd)))
            dfv = float(np.max(np.abs(a["f"] / 2 ** F_SHIFT - b["f"])) / np.max(np.abs(b["f"])))
            print(f"  N={N} {k:8s} same assignment {same:.4f}  max|Δm| {dm:.2e} GeV  max|Δangle| {da:.2e}  "
                  f"max|Δf|/max f {dfv:.2e}")
            ok &= same > 0.999 and dm < 1e-6 and da < 1e-6 and dfv < 1e-9
        ra, rb = summarize(fx, boot=False), summarize(fl, boot=False)
        print(f"  N={N} AUC full {ra['auc_full']:.5f} vs {rb['auc_full']:.5f}; mass {ra['auc_mass']:.5f} vs {rb['auc_mass']:.5f}")
    # bit-true units against their float functions
    A = Arith(18)
    x = np.exp(np.linspace(-6, 12, 5001))
    for nm, fn, ref in (("recip", A.recip, 1 / x), ("rsqrt", A.rsqrt, 1 / np.sqrt(x)), ("sqrt", A.sqrt, np.sqrt(x))):
        print(f"  W=18 {nm:6s} max rel. error {np.max(np.abs(fn(x) / ref - 1)):.2e}")
    ang = np.linspace(-pi + 1e-3, pi, 3001)
    err = np.abs(A.atan2(3.7 * np.sin(ang), 3.7 * np.cos(ang)) * pi - ang)
    print(f"  W=18 CORDIC atan2 max error {np.max(np.minimum(err, 2 * pi - err)):.2e} rad")
    print(f"  W=18 ln max error {np.max(np.abs(A.ln(x, 1024) - np.log(x))):.2e}")
    return ok


RECOMMENDED = {"N": 6, "inputs": "nominal", "W": 16, "K": 256, "Kchi": 256}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="only run the consistency checks")
    ap.add_argument("--out", default=os.path.join(repo_paths.results_dir("study_l1"), "results.json"))
    args = ap.parse_args()
    t0 = time.time()
    st = Study()
    if args.selftest:
        sys.exit(0 if selftest(st) else 1)
    passed = selftest(st)
    rows = []

    def add(res, ref=None, ref2=None, group=""):
        r = summarize(res, ref, ref2)
        r["group"] = group
        rows.append(r)
        f = lambda v: "  -   " if v is None else f"{v:.4f}"  # noqa: E731
        d = r.get("dauc_full") or r.get("dauc_mass")
        sa = r.get("fidelity", {}).get("same_assignment", r.get("same_assignment_as_ref", {}))
        print(f"{r['name']:50s} {f(r['auc_full'])} {f(r['auc_mass'])} {f(r['auc_angles'])} {r['auc_chi2min']:.4f}"
              + (f"  {d[0]:+.4f}±{d[1]:.4f}" if d else " " * 17)
              + (f"  {sa.get('tt_test', float('nan')):.3f}" if sa else "")
              + (f"  rms {r['fidelity'].get('score_rms_full', r['fidelity'].get('score_rms_mass')):.4f}"
                 if "fidelity" in r else ""), flush=True)
        return r

    print(f"\n{'configuration':50s} {'full':6s} {'mass':6s} {'angles':6s} {'χ²min':6s}  ΔAUC vs float ref   same-assign  score rms")
    print("=== float ladder: offline reference -> L1 restrictions (offline event set) ===")
    ref0 = st.run_float("offline reference (N=6, b-tag, truth d-type)")
    add(ref0, group="ladder")
    add(st.run_float("+ softer jet = down-type", 6, True, False, "softer"), ref0, group="ladder")
    add(st.run_float("+ no b-tag in χ²", 6, False, False, "softer"), ref0, group="ladder")
    fl = {}
    for N in (6, 5, 4):
        fl[N] = st.run_float(f"float L1 N={N} (+ massless objects)", N, False, True, "softer")
        add(fl[N], ref0, group="ladder")
    print("=== quantized L1 inputs, float arithmetic, exact functions ===")
    q = {}
    for N in (6, 5, 4):
        for sch in ("fine", "nominal", "coarse"):
            q[(N, sch)] = st.run_fixed(f"N={N} inputs {sch}, float arithmetic", N, sch, None, None, None)
            add(q[(N, sch)], fl[N], group="inputs")
    print("=== datapath width W (nominal inputs, K = Kχ = 1024) ===")
    for N in (6, 5):
        for W in (10, 12, 14, 16, 18, 20, 24):
            add(st.run_fixed(f"N={N} W={W} K=1024", N, "nominal", W, 1024, 1024), fl[N], q[(N, "nominal")], group="W")
    print("=== ROM depth (nominal inputs, W = 18) ===")
    for N in (6, 5):
        for K in (32, 64, 256, 1024, 4096):
            add(st.run_fixed(f"N={N} W=18 K={K} (score ROMs)", N, "nominal", 18, K, 1024), fl[N], q[(N, "nominal")],
                group="K")
        for Kc in (64, 256, 1024, 4096):
            add(st.run_fixed(f"N={N} W=18 Kχ={Kc} (χ² ROMs)", N, "nominal", 18, 1024, Kc), fl[N], q[(N, "nominal")],
                group="Kchi")
        add(st.run_fixed(f"N={N} W=float K=Kχ=1024 (ROMs only)", N, "nominal", None, 1024, 1024), fl[N],
            q[(N, "nominal")], group="K")
    print("=== recommended configurations (N=6, nominal inputs, W=16, K=Kχ=256) ===")
    R = RECOMMENDED
    rec_full = add(st.run_fixed("recommended: full score", R["N"], R["inputs"], R["W"], R["K"], R["Kchi"]), fl[6],
                   q[(6, "nominal")], group="recommended")
    rec_mass = add(st.run_fixed("recommended: masses only (no angle unit)", R["N"], R["inputs"], R["W"], R["K"],
                                R["Kchi"], angles=False), fl[6], q[(6, "nominal")], group="recommended")
    add(st.run_fixed("N=5, W=16, K=256, masses only", 5, "nominal", 16, 256, 256, angles=False), fl[5],
        q[(5, "nominal")], group="recommended")
    print("=== L1 event set (1 lepton, >= 4 jets, no b-tag requirement) ===")
    for N in (6, 5):
        f1 = st.run_float(f"float L1 N={N}, L1 event set", N, False, True, "softer", kind="l1")
        add(f1, group="l1set")
        add(st.run_fixed(f"N={N} W=16 K=256 full, L1 event set", N, "nominal", 16, 256, 256, kind="l1"), f1,
            group="l1set")
        add(st.run_fixed(f"N={N} W=16 K=256 masses only, L1 event set", N, "nominal", 16, 256, 256, kind="l1",
                         angles=False), f1, group="l1set")
    print(f"=== resolution stress test (extra σ: jets {SMEAR['jet']:.0%}, lepton {SMEAR['lep']:.0%}, "
          f"MET {SMEAR['met']:.0f} GeV per component) ===")
    fs = st.run_float("float L1 N=6, smeared inputs", 6, False, True, "softer", smear=True)
    add(fs, fl[6], group="stress")
    add(st.run_fixed("recommended full, smeared inputs", 6, "nominal", 16, 256, 256, smear=True), fs, group="stress")
    add(st.run_fixed("recommended masses only, smeared inputs", 6, "nominal", 16, 256, 256, smear=True,
                     angles=False), fs, group="stress")

    # resources and latency
    dens = st.dens
    kw = {"n_onedim": dens.n_onedim(), "n_terms": len(dens.terms)}
    res_tab = []
    for N in (6, 5, 4):
        for variant in ("full", "masses"):
            for W in (12, 14, 16, 18, 24):
                for K in (64, 256, 1024):
                    res_tab.append(resource_model(N, W, K, K, variant, "parallel", **kw))
            for lanes in (1, 4, 12):
                res_tab.append(resource_model(N, 16, 256, 256, variant, "tmux", lanes, **kw))
    rec_res = {v: resource_model(R["N"], R["W"], R["K"], R["Kchi"], v, "parallel", **kw) for v in ("masses", "full")}
    print("\n=== resources (analytic) of the recommended configuration, parallel, "
          f"{CLOCK_MHZ:.0f} MHz ===")
    for v, r in rec_res.items():
        t = r["total"]
        print(f"  {v:7s} LUT {t['LUT']:7.0f} FF {t['FF']:7.0f} DSP {t['DSP']:4.0f} BRAM36 {t['BRAM']:5.1f} | "
              + " | ".join(f"{dev}: " + " ".join(f"{k} {100 * x:.2f}%" for k, x in r["frac"][dev].items())
                           for dev in DEVICES)
              + f" | latency {r['latency_clk']} clk = {r['latency_ns']:.0f} ns")
    out = {"meta": {"date": time.strftime("%Y-%m-%d"), "density": os.path.relpath(DENSITY, repo_paths.REPO),
                    "kept_terms": dens.n_kept, "real_terms": len(dens.terms), "onedim_functions": dens.n_onedim(),
                    "terms": [{k: (v if k != "c" else [v.real, v.imag]) for k, v in t.items()} for t in dens.terms],
                    "samples": {"tt_test": "samples/ttbar_bblvqq/run_seed12345 (10k)",
                                "tt_calib": "samples/ttbar_bblvqq/run_seed1001 (first 30k)",
                                "hh": "samples/hh_bblvqq/run_seed2001 (20k)"},
                    "formats_integer_bits": FMT, "f_shift": F_SHIFT, "input_schemes": INPUT_SCHEMES,
                    "chi_windows_m2": CHI_WINDOW, "bw_windows_m": BW_WINDOW, "y_window": Y_WINDOW,
                    "smear": SMEAR, "devices": DEVICES, "clock_mhz": CLOCK_MHZ, "tmux_bc": TMUX_BC,
                    "lut_ff_overhead": OVERHEAD, "selftest_passed": bool(passed), "recommended": RECOMMENDED},
           "reference_check": {"expected (study_salvage/reco_extension.json)":
                               {"angles+BW": 0.6019, "masses": 0.6003, "angles": 0.5182},
                               "obtained": {"angles+BW": rows[0]["auc_full"], "masses": rows[0]["auc_mass"],
                                            "angles": rows[0]["auc_angles"]}},
           "rows": rows, "resources": res_tab, "recommended_resources": rec_res}
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
