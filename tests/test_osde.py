"""
Checks of the basis functions and OSDE (run: .venv/bin/python -m pytest -q tests).
"""
import os
import sys
from itertools import product
from math import pi, sqrt

import numpy as np
import pytest
from numpy.polynomial import legendre

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plotting import angles, osde  # noqa: E402

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- Wigner d ----
def test_small_d_closed_forms():
    b = np.linspace(0, pi, 7)
    c, s = np.cos(b), np.sin(b)
    known = {(1, 1, 1): (1 + c) / 2, (1, 1, 0): -s / sqrt(2), (1, 1, -1): (1 - c) / 2,
             (1, 0, 0): c, (1, 0, 1): s / sqrt(2),
             (2, 0, 0): (3 * c**2 - 1) / 2, (2, 1, 0): -sqrt(3 / 2) * s * c,
             (2, 2, 0): sqrt(3 / 8) * s**2, (2, 1, 1): (1 + c) * (2 * c - 1) / 2}
    for (j, mp, m), ref in known.items():
        assert np.allclose(osde.wigner_small_d(j, mp, m, b), ref, atol=1e-12), (j, mp, m)


def test_small_d_vs_sympy():
    sympy = pytest.importorskip("sympy")
    from sympy.physics.quantum.spin import Rotation
    beta = sympy.Symbol("b")
    for j in range(3):
        for mp in range(-j, j + 1):
            for m in range(-j, j + 1):
                expr = sympy.lambdify(beta, Rotation.d(j, mp, m, beta).doit(), "numpy")
                for b in (0.3, 1.1, 2.5):
                    assert abs(complex(expr(b)) - osde.wigner_small_d(j, mp, m, b)) < 1e-12, (j, mp, m, b)


# -------------------------------------------------------- orthonormality ----
def m_gram(midx, n=12):
    u, wu = legendre.leggauss(n)
    ph = -pi + 2 * pi * (np.arange(n) + 0.5) / n
    wp = np.full(n, 2 * pi / n)
    g = np.meshgrid(ph, u, ph, u, indexing="ij")
    w = np.einsum("a,b,c,d->abcd", wp, wu, wp, wu).ravel()
    M = osde.m_functions(*(x.ravel() for x in g), midx)
    return np.einsum("k,ki,kj->ij", w, M.conj(), M)


def test_m_count_and_orthonormality():
    midx = osde.m_indices()
    assert len(midx) == 24
    assert np.allclose(m_gram(midx), np.eye(24), atol=1e-12)


def test_m_conjugation_closure():
    # conj(D^j_{m'm}) = (-1)^{m'-m} D^j_{-m',-m}, and M = D^{j1}_{m'm} D^{j2}_{m0}, so
    # conj(M_{m'm}) = (-1)^{m'} M_{-m',-m}: the set is closed, so f = sum c B is real
    midx = osde.m_indices()
    x = [RNG.uniform(-pi, pi, 50), RNG.uniform(-1, 1, 50), RNG.uniform(-pi, pi, 50), RNG.uniform(-1, 1, 50)]
    M = osde.m_functions(*x, midx)
    pos = {k: i for i, k in enumerate(midx)}
    for i, (j1, j2, mp, m) in enumerate(midx):
        k = pos[(j1, j2, -mp, -m)]
        assert np.allclose(M[:, i].conj(), (-1) ** mp * M[:, k], atol=1e-12)


def test_legendre_hermite_orthonormal():
    u, w = legendre.leggauss(20)
    P = osde.legendre_functions(u, 5)
    assert np.allclose(P.T @ (w[:, None] * P), np.eye(5), atol=1e-12)
    y = np.linspace(-25, 25, 200001)
    for scale in (1.0, 0.7):
        H = osde.hermite_functions(y, 5, scale)
        assert np.allclose(np.trapezoid(H[:, :, None] * H[:, None, :], y, axis=0), np.eye(5), atol=1e-9)


def test_total_count():
    f = osde.TTbarDensity()
    assert np.prod(f.shape) == 14400


# ------------------------------------------------------------- toy OSDE ----
def toy_events(n, F0=0.7, FL=0.3):
    """Independent angles from densities inside the span (coefficients known analytically).
    cos θ', cos θ~' ~ W helicity distribution; cos Θ ~ (1 + P2)/2; Y ~ N(0, 1); rest uniform."""
    FR = 1 - F0 - FL

    def helicity(n):
        out = np.empty(0)
        while len(out) < n:
            c = RNG.uniform(-1, 1, 3 * n)
            pdf = 3 / 8 * (1 - c)**2 * FL + 3 / 4 * (1 - c**2) * F0 + 3 / 8 * (1 + c)**2 * FR
            out = np.concatenate([out, c[RNG.uniform(0, 0.75, 3 * n) < pdf]])
        return out[:n]

    def costheta_big(n):
        out = np.empty(0)
        while len(out) < n:
            c = RNG.uniform(-1, 1, 3 * n)
            out = np.concatenate([out, c[RNG.uniform(0, 1, 3 * n) < (1 + (3 * c**2 - 1) / 2) / 2]])
        return out[:n]

    X = {k: RNG.uniform(-pi, pi, n) for k in angles.ANGLE_NAMES if k.startswith("phi")}
    X.update(cos_theta=RNG.uniform(-1, 1, n), cos_theta_tilde=RNG.uniform(-1, 1, n),
             cos_theta_prime=helicity(n), cos_theta_prime_tilde=helicity(n),
             cos_Theta=costheta_big(n), Y=RNG.normal(0, 1, n))
    return X


def toy_truth(f, F0=0.7, FL=0.3):
    """
    Exact coefficients of the toy density.  For the angular part
    f = 1/(4π) [θ, φ] x 1/(2π) [φ'] x g(cos θ'), so c = ∫ M* f = sqrt(2 j2 + 1)/(4π) <P_j2>_g
    for j1 = 0 (m' = m = 0) and 0 otherwise.
    """
    FR = 1 - F0 - FL
    mom = [1.0, (FR - FL) / 2, 0.1 - 0.3 * F0]                    # <P_0>, <P_1>, <P_2> of g
    cm = np.array([sqrt(2 * j2 + 1) / (4 * pi) * mom[j2] if j1 == 0 else 0.0
                   for j1, j2, mp, m in f.midx], dtype=complex)
    cp = np.array([sqrt(1 / 2), 0, sqrt(5 / 2) / 5, 0, 0])         # <p_l> for (1 + P_2)/2
    cy = np.zeros(f.n_hermite)
    cy[0] = 1 / (sqrt(2) * pi ** 0.25)                              # N(0,1) = h_0 / (sqrt2 π^1/4)
    return np.einsum("j,i,l,n->jiln", cm, cm, cp, cy)


def test_toy_recovery():
    f = osde.TTbarDensity()
    X = toy_events(200000)
    f.fit(X)
    truth = toy_truth(f)
    pull = (f.c.real - truth.real) / f.c_err
    # for real basis functions (m' = m = 0 for both tops) the error estimate is exact
    real_b = np.array([mp == 0 and m == 0 for _, _, mp, m in f.midx])
    sel = real_b[:, None, None, None] & real_b[None, :, None, None] & np.ones(f.shape, bool)
    assert abs(np.mean(pull[sel])) < 0.15
    assert 0.9 < np.std(pull[sel]) < 1.1
    assert np.std(pull) < 1.1
    # the large, known coefficients are recovered precisely
    big = abs(truth) > 1e-3
    assert np.all(abs(f.c[big] - truth[big]) < 5 * f.c_err[big])
    # f is real and the projections reproduce the generating densities
    re, im = f({k: v[:2000] for k, v in X.items()}, imag=True)
    assert np.max(abs(im)) < 1e-10 * np.max(abs(re))
    u = np.linspace(-0.95, 0.95, 9)
    g = f.projection("cos_theta_prime", u)
    ref = 3 / 8 * (1 - u)**2 * 0.3 + 3 / 4 * (1 - u**2) * 0.7
    assert np.allclose(g, ref, atol=0.02)
    assert abs(f.integral() - 1) < 0.01


# ------------------------------------------------- extended (block) basis ----
def test_high_j_small_d_stable():
    # orthonormality in j of d^j_{m'm} up to j = 24 (exact Gauss-Legendre)
    u, w = legendre.leggauss(80)
    th = np.arccos(u)
    for mp, m in ((0, 0), (3, -2), (5, 5)):
        js = range(max(abs(mp), abs(m)), 25)
        G = np.array([[np.sum(w * osde.wigner_small_d(j1, mp, m, th) * osde.wigner_small_d(j2, mp, m, th))
                       * sqrt((2 * j1 + 1) * (2 * j2 + 1)) / 2 for j2 in js] for j1 in js])
        assert np.allclose(G, np.eye(len(G)), atol=1e-12)
    # agrees with the explicit Wigner sum at low j
    b = np.linspace(0.05, 3.0, 9)
    for j in range(5):
        for mp in range(-j, j + 1):
            for m in range(-j, j + 1):
                assert np.allclose(osde.wigner_small_d(j, mp, m, b), osde.wigner_small_d_explicit(j, mp, m, b),
                                   atol=1e-13)


def test_extended_blocks_orthonormal_and_disjoint():
    g = osde.ExtendedDensity(J1=3, J2=3, n_legendre=3, n_hermite=3, n_legendre_hi=6, n_hermite_hi=6)
    # factor set orthonormal (quadrature exact for these degrees)
    G = m_gram(g.m_all, n=10)
    assert np.allclose(G, np.eye(len(g.m_all)), atol=1e-11)
    # blocks are disjoint subsets of the product basis -> the union is orthonormal
    seen = set()
    for _, a, b, l, n in g.blocks:
        for t in product(a, b, l, n):
            assert t not in seen
            seen.add(t)
    assert len(seen) == g.n_terms
    # closed under conjugation
    part = g.conjugate_partner()
    assert np.array_equal(np.sort(part), np.arange(g.n_terms))


def test_extended_reduces_to_paper_basis():
    X = toy_events(20000)
    f0 = osde.TTbarDensity().fit(X)
    g = osde.ExtendedDensity().fit(X)          # J1=1, J2=2, 5 Legendre, 5 Hermite: block A only
    assert g.n_terms == 14400
    sub = {k: v[:500] for k, v in X.items()}
    assert np.allclose(f0(sub), g(sub), rtol=1e-10, atol=1e-14)
    # integrals agree to the precision of the (trapezoid) factor integrals of TTbarDensity
    assert abs(f0.integral() - g.integral()) < 1e-6
    u = np.linspace(-0.9, 0.9, 7)
    assert np.allclose(f0.projection("cos_theta_prime", u), g.projection("cos_theta_prime", u), atol=1e-10)
    assert np.allclose(f0.projection("phi_prime", u), g.projection("phi_prime", u), atol=1e-10)


def test_extended_captures_sharp_marginal():
    # cos θ' with a sharp acceptance-like edge: flat on [-0.6, 1]; needs high Legendre orders
    n = 200000
    X = toy_events(n)
    X["cos_theta_prime"] = RNG.uniform(-0.6, 1, n)
    lo = osde.ExtendedDensity().fit(X)
    hi = osde.ExtendedDensity(J1=1, J2=12).fit(X)
    u = np.linspace(-0.95, 0.95, 39)
    truth = np.where(u > -0.6, 1 / 1.6, 0)
    away = np.abs(u + 0.6) > 0.15                  # away from the discontinuity (Gibbs)
    err_lo = np.max(np.abs(lo.projection("cos_theta_prime", u) - truth)[away])
    err_hi = np.max(np.abs(hi.projection("cos_theta_prime", u) - truth)[away])
    assert err_hi < 0.5 * err_lo and err_hi < 0.12


def test_leave_one_out_exact():
    # loo() must equal refitting without event k and evaluating at x_k (brute force)
    X = toy_events(400)
    for dens in (osde.TTbarDensity(n_legendre=3, n_hermite=3),
                 osde.ExtendedDensity(J1=2, J2=3, n_legendre=3, n_hermite=3, n_legendre_hi=5, n_hermite_hi=5)):
        full = dens.fit(X)
        loo = full.loo(X)
        for k in (0, 17, 399):
            keep = np.arange(400) != k
            ref = type(dens)(**({"n_legendre": 3, "n_hermite": 3} if isinstance(dens, osde.TTbarDensity) else
                                {"J1": 2, "J2": 3, "n_legendre": 3, "n_hermite": 3, "n_legendre_hi": 5,
                                 "n_hermite_hi": 5}))
            ref.fit({key: v[keep] for key, v in X.items()})
            assert np.isclose(loo[k], ref({key: v[k:k + 1] for key, v in X.items()})[0], rtol=1e-9, atol=1e-15)
