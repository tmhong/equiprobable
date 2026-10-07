"""
Checks of the basis functions and OSDE (run: .venv/bin/python -m pytest -q tests).
"""
import os
import sys
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
