"""
Orthonormal basis of the paper and orthogonal series density estimation (OSDE).

  f_tt̄(x) = sum_{j, j~, l, n} c_{j j~ l n} M_j(θ) M_j~(θ~) p_l(cos Θ) h_n(Y)

  M^{j1 j2}_{m'm}(φ,θ,φ',θ') = (1/4π) sqrt(2j1+1) sqrt(2j2+1)
                               D^{j1}_{m'm}(φ,θ,0) D^{j2}_{m0}(φ',θ',0),   j1 <= 1, j2 <= 2
                               -> 24 orthonormal functions per top decay (in dΩ dΩ')
  p_l(u) = sqrt((2l+1)/2) P_l(u)                  normalized Legendre, l < n_legendre
  h_n(Y) = (2^n n! sqrt(π) s)^(-1/2) H_n(Y/s) exp(-(Y/s)^2/2)
                                                  normalized Hermite functions, n < n_hermite
                                                  (s = y_scale; the paper's choice is s = 1)

  D^j_{m'm}(α,β,γ) = exp(-i m' α) d^j_{m'm}(β) exp(-i m γ)   (Wigner D, z-y-z, passive)

Coefficients: c = ∫ B* f dx, estimated by the sample mean ĉ = (1/N) Σ_k B*(x_k), with
dx = dΩ dΩ' dΩ~ dΩ~' dcosΘ dY.  The basis factorizes, so fitting and evaluating use
einsum contractions and never build an (N x 14400) array.

Angles are passed as a dict of arrays keyed by lib.angles.ANGLE_NAMES.
"""
from itertools import product
from math import factorial, pi, sqrt

import numpy as np
from numpy.polynomial import hermite, legendre

LEP = ("phi", "cos_theta", "phi_prime", "cos_theta_prime")
HAD = ("phi_tilde", "cos_theta_tilde", "phi_prime_tilde", "cos_theta_prime_tilde")


# ----------------------------------------------------------------------------
# Wigner functions and the M basis
# ----------------------------------------------------------------------------
def wigner_small_d(j, mp, m, beta):
    """Wigner small-d d^j_{m'm}(beta) (integer j), via Jacobi polynomials (numerically stable
    to high j; agrees with the explicit Wigner sum to 1e-14)."""
    from scipy.special import eval_jacobi, gammaln
    beta = np.asarray(beta, dtype=float)
    k = min(j + m, j - m, j + mp, j - mp)
    if k == j + m:
        a, lam = mp - m, mp - m
    elif k == j - m:
        a, lam = m - mp, 0
    elif k == j + mp:
        a, lam = m - mp, 0
    else:
        a, lam = mp - m, mp - m
    b = 2 * j - 2 * k - a
    pref = (-1) ** lam * np.exp(0.5 * (gammaln(2 * j - k + 1) + gammaln(k + 1)
                                       - gammaln(k + a + 1) - gammaln(k + b + 1)))
    return pref * np.sin(beta / 2) ** a * np.cos(beta / 2) ** b * eval_jacobi(k, a, b, np.cos(beta))


def wigner_small_d_explicit(j, mp, m, beta):
    """Explicit Wigner sum (reference implementation; loses precision at large j)."""
    beta = np.asarray(beta, dtype=float)
    c, s = np.cos(beta / 2), np.sin(beta / 2)
    pref = sqrt(factorial(j + mp) * factorial(j - mp) * factorial(j + m) * factorial(j - m))
    out = np.zeros_like(beta)
    for k in range(max(0, m - mp), min(j + m, j - mp) + 1):
        den = factorial(j + m - k) * factorial(k) * factorial(mp - m + k) * factorial(j - mp - k)
        out = out + (-1) ** (mp - m + k) * pref / den * c ** (2 * j + m - mp - 2 * k) * s ** (mp - m + 2 * k)
    return out


def wigner_D(j, mp, m, alpha, beta):
    """D^j_{m'm}(alpha, beta, 0)."""
    return np.exp(-1j * mp * np.asarray(alpha)) * wigner_small_d(j, mp, m, beta)


def m_indices(j1max=1, j2max=2):
    """[(j1, j2, m', m)] of the M functions: |m'| <= j1, |m| <= min(j1, j2)."""
    out = []
    for j1, j2 in product(range(j1max + 1), range(j2max + 1)):
        for mp in range(-j1, j1 + 1):
            for m in range(-min(j1, j2), min(j1, j2) + 1):
                out.append((j1, j2, mp, m))
    return out


def m_functions(phi, cos_theta, phi_p, cos_theta_p, idx):
    """(N, len(idx)) complex values of the M functions."""
    th, thp = np.arccos(np.clip(cos_theta, -1, 1)), np.arccos(np.clip(cos_theta_p, -1, 1))
    cols = [(sqrt(2 * j1 + 1) * sqrt(2 * j2 + 1) / (4 * pi))
            * wigner_D(j1, mp, m, phi, th) * wigner_D(j2, m, 0, phi_p, thp)
            for j1, j2, mp, m in idx]
    return np.stack(cols, axis=-1)


def legendre_functions(u, n):
    """(N, n) normalized Legendre polynomials sqrt((2l+1)/2) P_l(u)."""
    u = np.asarray(u, float)
    return np.stack([sqrt((2 * l + 1) / 2) * legendre.legval(u, [0] * l + [1]) for l in range(n)], axis=-1)


def hermite_functions(y, n, scale=1.0):
    """(N, n) normalized Hermite functions in y/scale (orthonormal in dy)."""
    t = np.asarray(y, float) / scale
    return np.stack([hermite.hermval(t, [0] * k + [1]) * np.exp(-t * t / 2)
                     / sqrt(2.0 ** k * factorial(k) * sqrt(pi) * scale) for k in range(n)], axis=-1)


# ----------------------------------------------------------------------------
# The density
# ----------------------------------------------------------------------------
class TTbarDensity:
    """f_tt̄(x) on the ten angles, with coefficients from OSDE."""

    def __init__(self, n_legendre=5, n_hermite=5, y_scale=1.0, j1max=1, j2max=2):
        self.n_legendre, self.n_hermite, self.y_scale = n_legendre, n_hermite, y_scale
        self.j1max, self.j2max = j1max, j2max
        self.midx = m_indices(j1max, j2max)
        self.c = self.c_err = None
        self.n_fit = 0

    @property
    def shape(self):
        return (len(self.midx), len(self.midx), self.n_legendre, self.n_hermite)

    def top_functions(self, phi, cos_theta, phi_p, cos_theta_p):
        """The per-top angular basis (here the paper's M functions); subclasses override this."""
        return m_functions(phi, cos_theta, phi_p, cos_theta_p, self.midx)

    def factors(self, X):
        """Basis factors (M_lep, M_had, p_l, h_n) evaluated on the events in X."""
        return (self.top_functions(*(X[k] for k in LEP)),
                self.top_functions(*(X[k] for k in HAD)),
                legendre_functions(X["cos_Theta"], self.n_legendre),
                hermite_functions(X["Y"], self.n_hermite, self.y_scale))

    def fit(self, X, chunk=20000):
        """OSDE: ĉ = mean(B*), with the standard error of each (complex) coefficient."""
        n = len(X["Y"])
        s1 = np.zeros(self.shape, complex)
        s2 = np.zeros(self.shape)
        for lo in range(0, n, chunk):
            ml, mh, p, h = self.factors({k: v[lo:lo + chunk] for k, v in X.items()})
            s1 += np.einsum("kj,ki,kl,kn->jiln", ml.conj(), mh.conj(), p, h, optimize=True)
            s2 += np.einsum("kj,ki,kl,kn->jiln", abs(ml) ** 2, abs(mh) ** 2, p ** 2, h ** 2, optimize=True)
        self.c = s1 / n
        self.c_err = np.sqrt(np.clip(s2 / n - abs(self.c) ** 2, 0, None) / n)
        self.n_fit = n
        return self

    def fold_sums(self, X, k=5, chunk=20000):
        """Per-fold sums of B* and |B|^2 (event i goes to fold i % k), for cross-validation."""
        n = len(X["Y"])
        s1 = np.zeros((k,) + self.shape, complex)
        s2 = np.zeros((k,) + self.shape)
        counts = np.zeros(k)
        for lo in range(0, n, chunk):
            ml, mh, p, h = self.factors({key: v[lo:lo + chunk] for key, v in X.items()})
            fold = (np.arange(lo, lo + len(p)) % k)
            for i in range(k):
                m = fold == i
                if m.any():
                    s1[i] += np.einsum("kj,ki,kl,kn->jiln", ml[m].conj(), mh[m].conj(), p[m], h[m], optimize=True)
                    s2[i] += np.einsum("kj,ki,kl,kn->jiln", abs(ml[m]) ** 2, abs(mh[m]) ** 2, p[m] ** 2,
                                       h[m] ** 2, optimize=True)
                    counts[i] += m.sum()
        return s1, s2, counts

    def loo(self, X):
        """Leave-one-out f on the *training* events: removes each event's own contribution
        (1/N) Σ_j |B_j(x_k)|^2 to the coefficients, i.e. the self-inclusion (overfitting) bias.
        Assumes X is the sample the coefficients were fit on."""
        ml, mh, p, h = self.factors(X)
        kept = np.abs(self.c) > 0
        self_term = np.einsum("kj,ki,kl,kn,jiln->k", abs(ml) ** 2, abs(mh) ** 2, p ** 2, h ** 2,
                              kept.astype(float), optimize=True)
        n = self.n_fit
        return (n * self(X) - self_term) / (n - 1)

    def significance(self):
        """|ĉ| / σ(ĉ) for every coefficient."""
        return np.abs(self.c) / np.where(self.c_err > 0, self.c_err, np.inf)

    def truncated(self, keep):
        """Copy of this density keeping only coefficients where `keep` is True."""
        g = TTbarDensity(self.n_legendre, self.n_hermite, self.y_scale, self.j1max, self.j2max)
        g.c, g.c_err, g.n_fit = np.where(keep, self.c, 0), self.c_err, self.n_fit
        return g

    def conjugate_partner(self):
        """Flat index of the conjugate partner of each coefficient: conj(B_k) = ± B_partner(k)."""
        pos = {key: i for i, key in enumerate(self.midx)}
        pm = np.array([pos[(j1, j2, -mp, -m)] for j1, j2, mp, m in self.midx])
        idx = np.arange(int(np.prod(self.shape))).reshape(self.shape)
        return idx[pm][:, pm].ravel()

    def __call__(self, X, imag=False, chunk=20000):
        """f(x_k) for each event (real part; imag=True also returns the imaginary part)."""
        out = []
        for lo in range(0, len(X["Y"]), chunk):
            ml, mh, p, h = self.factors({k: v[lo:lo + chunk] for k, v in X.items()})
            out.append(np.einsum("jiln,kj,ki,kl,kn->k", self.c, ml, mh, p, h, optimize=True))
        f = np.concatenate(out) if out else np.zeros(0, complex)
        return (f.real, f.imag) if imag else f.real

    # ---- integrals and one-dimensional projections ------------------------
    def _m_integrals(self, n=12):
        """Quadrature (exact for this basis) over (φ, cosθ, φ', cosθ').
        Returns the grid nodes and the M values times quadrature weights."""
        u, wu = legendre.leggauss(n)
        ph = -pi + 2 * pi * (np.arange(n) + 0.5) / n
        wp = np.full(n, 2 * pi / n)
        g = np.meshgrid(ph, u, ph, u, indexing="ij")
        w = np.einsum("a,b,c,d->abcd", wp, wu, wp, wu).ravel()
        vals = self.top_functions(*(x.ravel() for x in g))
        return [x.ravel() for x in g], w, vals

    def _y_integral(self):
        y = np.linspace(-20, 20, 40001) * self.y_scale
        return np.trapezoid(hermite_functions(y, self.n_hermite, self.y_scale), y, axis=0)

    def integral(self):
        """∫ f dx (1 if the density lies in the span; the Y expansion makes it approximate)."""
        _, w, vals = self._m_integrals()
        im = w @ vals
        ip = np.trapezoid(legendre_functions(np.linspace(-1, 1, 2001), self.n_legendre),
                          np.linspace(-1, 1, 2001), axis=0)
        return np.einsum("jiln,j,i,l,n->", self.c, im, im, ip, self._y_integral()).real

    def projection(self, name, grid):
        """One-dimensional marginal density of f in angle `name` on `grid`."""
        _, w, vals = self._m_integrals()
        im = w @ vals
        u = np.linspace(-1, 1, 2001)
        ip = np.trapezoid(legendre_functions(u, self.n_legendre), u, axis=0)
        iy = self._y_integral()
        grid = np.asarray(grid, float)
        fl, fh, fp, fy = im, im, ip, iy
        if name == "Y":
            fy = hermite_functions(grid, self.n_hermite, self.y_scale)
            return np.einsum("jiln,j,i,l,kn->k", self.c, fl, fh, fp, fy).real
        if name == "cos_Theta":
            fp = legendre_functions(grid, self.n_legendre)
            return np.einsum("jiln,j,i,kl,n->k", self.c, fl, fh, fp, fy).real
        group = LEP if name in LEP else HAD
        fm = self._m_partial(name, grid)
        if group is LEP:
            return np.einsum("jiln,kj,i,l,n->k", self.c, fm, fh, fp, fy).real
        return np.einsum("jiln,j,ki,l,n->k", self.c, fl, fm, fp, fy).real

    def _m_partial(self, name, grid, n=12):
        """∫ M over the three M variables other than `name`, with `name` set to grid values."""
        group = LEP if name in LEP else HAD
        pos = group.index(name)
        u, wu = legendre.leggauss(n)
        ph = -pi + 2 * pi * (np.arange(n) + 0.5) / n
        wp = np.full(n, 2 * pi / n)
        axes = [(ph, wp), (u, wu), (ph, wp), (u, wu)]
        out = []
        for x in grid:
            pts = [np.array([x]) if i == pos else axes[i][0] for i in range(4)]
            wts = [np.array([1.0]) if i == pos else axes[i][1] for i in range(4)]
            g = np.meshgrid(*pts, indexing="ij")
            w = np.einsum("a,b,c,d->abcd", *wts).ravel()
            out.append(w @ self.top_functions(*(v.ravel() for v in g)))
        return np.array(out)

    # ---- persistence -------------------------------------------------------
    def save(self, path, **meta):
        np.savez(path, c=self.c, c_err=self.c_err, n_fit=self.n_fit, midx=np.array(self.midx),
                 n_legendre=self.n_legendre, n_hermite=self.n_hermite, y_scale=self.y_scale,
                 j1max=self.j1max, j2max=self.j2max, meta=repr(meta))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        f = cls(int(z["n_legendre"]), int(z["n_hermite"]), float(z["y_scale"]), int(z["j1max"]), int(z["j2max"]))
        f.c, f.c_err, f.n_fit = z["c"], z["c_err"], int(z["n_fit"])
        return f


KT_THRESHOLD = np.sqrt(2.0)   # Kronmal-Tarter: keep a term iff |ĉ|^2 > 2 σ^2 (minimizes MISE)


def cv_threshold_scan(s1, s2, counts, thresholds):
    """
    k-fold least-squares cross-validation of the significance threshold t (keep |ĉ|/σ > t,
    always keeping the constant term).  For an orthonormal basis
        ISE(f̂) - ∫f^2 = ∫f̂^2 - 2 E[f̂] = Σ_kept |ĉ|^2 - 2 Re Σ_kept ĉ conj(ĉ_heldout),
    with ĉ (and σ) from the other folds and ĉ_heldout from the held-out fold.
    Returns (mean LSCV per threshold, its standard error over folds, mean number kept).
    """
    k = len(counts)
    tot1, tot2, ntot = s1.sum(axis=0), s2.sum(axis=0), counts.sum()
    scores = np.zeros((k, len(thresholds)))
    kept = np.zeros((k, len(thresholds)))
    for i in range(k):
        n = ntot - counts[i]
        c = (tot1 - s1[i]) / n
        err = np.sqrt(np.clip((tot2 - s2[i]) / n - abs(c) ** 2, 0, None) / n)
        sig = np.abs(c) / np.where(err > 0, err, np.inf)
        sig.flat[0] = np.inf
        c_held = s1[i] / counts[i]
        cross = (c * c_held.conj()).real
        for j, t in enumerate(thresholds):
            keep = sig > t
            scores[i, j] = np.sum(abs(c[keep]) ** 2) - 2 * np.sum(cross[keep])
            kept[i, j] = keep.sum()
    return scores.mean(axis=0), scores.std(axis=0) / np.sqrt(k), kept.mean(axis=0)


def describe(f, flat_index):
    """Physics label of a basis term: the non-constant factors and what they probe."""
    if getattr(f, "sh", False):
        a, b, l, n = np.unravel_index(flat_index, f.shape)
        parts = [f"{side}[Y{l1}^{m1}(θ,φ) Y{l2}^{m2}(θ',φ')]" for side, (l1, l2, m1, m2)
                 in (("lep", f.midx[a]), ("had", f.midx[b])) if (l1, l2) != (0, 0)]
        if l:
            parts.append(f"P{l}(cosΘ)")
        if n:
            parts.append(f"H{n}(Y)")
        return (" x ".join(parts) or "constant"), "spherical-harmonic term"
    if hasattr(f, "term"):
        _, ma, mb, l, n = f.term(flat_index)
    else:
        a, b, l, n = np.unravel_index(flat_index, f.shape)
        ma, mb = f.midx[a], f.midx[b]
    parts, tags = [], []
    for side, (j1, j2, mp, m) in (("lep", ma), ("had", mb)):
        if (j1, j2) == (0, 0):
            continue
        parts.append(f"{side}[j1={j1},j2={j2},m'={mp},m={m}]")
        if j1 == 0:
            tags.append(f"W helicity ({side})" if j2 <= 2 else f"high-order W decay ({side})")
        elif j2 == 0:
            tags.append(f"top polarization ({side}, {'longitudinal' if mp == 0 else 'transverse'})")
        else:
            tags.append(f"top pol. x W helicity interference ({side})")
    if ma[0] >= 1 and mb[0] >= 1:
        tags = ["tt spin correlation"] + tags
    if l:
        parts.append(f"P{l}(cosΘ)")
        tags.append("production angle")
    if n:
        parts.append(f"H{n}(Y)")
        tags.append("rapidity")
    return (" x ".join(parts) or "constant"), "; ".join(dict.fromkeys(tags)) or "normalization"


# ----------------------------------------------------------------------------
# Extended (block) basis for gen/reco: the paper's product basis plus high-order blocks
# ----------------------------------------------------------------------------
def single_top_indices(j1max, j2max):
    """M functions that shape one top's angular distribution at high order: all (θ, φ) terms
    (j2 = 0, any m') and all terms with m' = 0 (θ, θ', φ' structure).  Closed under conjugation."""
    out = []
    for j1 in range(j1max + 1):
        for j2 in range(j2max + 1):
            for mp in range(-j1, j1 + 1):
                for m in range(-min(j1, j2), min(j1, j2) + 1):
                    if j2 == 0 or mp == 0:
                        out.append((j1, j2, mp, m))
    return out


def _m_factor_integrals(idx, names, grid=None, which=None, n=96):
    """Integrals of M functions over (φ, cosθ, φ', cosθ'), optionally leaving `which`
    (one of 'phi', 'cos_theta', 'phi_prime', 'cos_theta_prime') free on `grid`.
    Separable: φ integrals are 2π δ_{m'0} and 2π δ_{m0}; θ integrals by Gauss-Legendre."""
    u, w = legendre.leggauss(n)
    th = np.arccos(u)
    rows = []
    g = None if grid is None else np.asarray(grid, float)
    for j1, j2, mp, m in idx:
        pref = sqrt(2 * j1 + 1) * sqrt(2 * j2 + 1) / (4 * pi)
        fphi = (2 * pi * (mp == 0)) if which != "phi" else np.exp(-1j * mp * g)
        fth = (w @ wigner_small_d(j1, mp, m, th)) if which != "cos_theta" else \
            wigner_small_d(j1, mp, m, np.arccos(np.clip(g, -1, 1)))
        fphp = (2 * pi * (m == 0)) if which != "phi_prime" else np.exp(-1j * m * g)
        fthp = (w @ wigner_small_d(j2, m, 0, th)) if which != "cos_theta_prime" else \
            wigner_small_d(j2, m, 0, np.arccos(np.clip(g, -1, 1)))
        rows.append(pref * fphi * fth * fphp * fthp * (np.ones_like(g) if g is not None else 1.0))
    out = np.array(rows, dtype=complex)
    return out.T if g is not None else out


class ExtendedDensity:
    """
    f(x) = Σ_blocks Σ c B over a union of disjoint subsets of the full orthonormal product basis
    (so the union is orthonormal and OSDE applies unchanged):
      A  paper block: M_lep(j1<=1, j2<=2) x M_had(same) x p_l (l < n_legendre) x h_n (n < n_hermite)
      L  leptonic single-top block: single_top_indices(J1, J2) minus the paper set, x M_had,0 x p_0 x h_n
      H  hadronic single-top block (same, with the tops exchanged)
      P  high Legendre: M_0 x M_0 x p_l (n_legendre <= l < n_legendre_hi) x h_n (n < n_hermite)
      Y  high Hermite:  M_0 x M_0 x p_0 x h_n (n_hermite <= n < n_hermite_hi)
    Coefficients are stored flat (block A first, so index 0 is the constant term).
    """

    def __init__(self, J1=1, J2=2, n_legendre=5, n_hermite=5, n_legendre_hi=None, n_hermite_hi=None,
                 y_scale=1.0):
        self.J1, self.J2, self.y_scale = J1, J2, y_scale
        self.nl, self.nh = n_legendre, n_hermite
        self.nl_hi = max(n_legendre_hi or n_legendre, n_legendre)
        self.nh_hi = max(n_hermite_hi or n_hermite, n_hermite)
        self.m_lo = m_indices(1, 2)
        lo = set(self.m_lo)
        self.m_hi = [k for k in single_top_indices(J1, J2) if k not in lo]
        self.m_all = self.m_lo + self.m_hi
        z0 = [0]                                   # index of the constant M (0,0,0,0) in m_all
        nlo, nall = len(self.m_lo), len(self.m_all)
        lo_i, hi_i = list(range(nlo)), list(range(nlo, nall))
        r = lambda a, b: list(range(a, b))        # noqa: E731
        self.blocks = [("A", lo_i, lo_i, r(0, self.nl), r(0, self.nh))]
        if hi_i:
            self.blocks += [("L", hi_i, z0, [0], r(0, self.nh)), ("H", z0, hi_i, [0], r(0, self.nh))]
        if self.nl_hi > self.nl:
            self.blocks.append(("P", z0, z0, r(self.nl, self.nl_hi), r(0, self.nh)))
        if self.nh_hi > self.nh:
            self.blocks.append(("Y", z0, z0, [0], r(self.nh, self.nh_hi)))
        self.sizes = [len(a) * len(b) * len(l) * len(n) for _, a, b, l, n in self.blocks]
        self.c = self.c_err = None
        self.n_fit = 0

    @property
    def n_terms(self):
        return int(sum(self.sizes))

    shape = property(lambda self: (self.n_terms,))

    def factors(self, X):
        return (m_functions(*(X[k] for k in LEP), self.m_all),
                m_functions(*(X[k] for k in HAD), self.m_all),
                legendre_functions(X["cos_Theta"], self.nl_hi),
                hermite_functions(X["Y"], self.nh_hi, self.y_scale))

    def _blockwise(self, fac, conj):
        ml, mh, p, h = fac
        if conj:
            ml, mh = ml.conj(), mh.conj()
        out = []
        for _, a, b, l, n in self.blocks:
            out.append(np.einsum("kj,ki,kl,kn->jiln", ml[:, a], mh[:, b], p[:, l], h[:, n],
                                 optimize=True).ravel())
        return np.concatenate(out)

    def _sums(self, X, chunk=5000, k=1):
        nobs = len(X["Y"])
        s1 = np.zeros((k, self.n_terms), complex)
        s2 = np.zeros((k, self.n_terms))
        counts = np.zeros(k)
        for lo in range(0, nobs, chunk):
            fac = self.factors({key: v[lo:lo + chunk] for key, v in X.items()})
            fold = np.arange(lo, lo + len(fac[2])) % k
            for i in range(k):
                msk = fold == i
                if not msk.any():
                    continue
                sub = tuple(x[msk] for x in fac)
                s1[i] += self._blockwise(sub, conj=True)
                s2[i] += self._blockwise(tuple(abs(x) ** 2 for x in sub), conj=False).real
                counts[i] += msk.sum()
        return s1, s2, counts

    def fit(self, X):
        s1, s2, counts = self._sums(X)
        n = counts.sum()
        self.c = s1[0] / n
        self.c_err = np.sqrt(np.clip(s2[0] / n - abs(self.c) ** 2, 0, None) / n)
        self.n_fit = int(n)
        return self

    def fold_sums(self, X, k=5):
        return self._sums(X, k=k)

    def _evaluate(self, fac, coef):
        ml, mh, p, h = fac
        out, pos = 0, 0
        for (_, a, b, l, n), size in zip(self.blocks, self.sizes):
            cb = coef[pos:pos + size].reshape(len(a), len(b), len(l), len(n))
            pos += size
            out = out + np.einsum("jiln,kj,ki,kl,kn->k", cb, ml[:, a], mh[:, b], p[:, l], h[:, n], optimize=True)
        return out

    def __call__(self, X, imag=False, chunk=5000):
        res = []
        for lo in range(0, len(X["Y"]), chunk):
            res.append(self._evaluate(self.factors({k: v[lo:lo + chunk] for k, v in X.items()}), self.c))
        f = np.concatenate(res)
        return (f.real, f.imag) if imag else f.real

    def loo(self, X, chunk=5000):
        kept = (np.abs(self.c) > 0).astype(float)
        out = []
        for lo in range(0, len(X["Y"]), chunk):
            fac = self.factors({k: v[lo:lo + chunk] for k, v in X.items()})
            f = self._evaluate(fac, self.c).real
            self_term = self._evaluate(tuple(abs(x) ** 2 for x in fac), kept).real
            out.append((self.n_fit * f - self_term) / (self.n_fit - 1))
        return np.concatenate(out)

    def significance(self):
        return np.abs(self.c) / np.where(self.c_err > 0, self.c_err, np.inf)

    def truncated(self, keep):
        g = ExtendedDensity(self.J1, self.J2, self.nl, self.nh, self.nl_hi, self.nh_hi, self.y_scale)
        g.c, g.c_err, g.n_fit = np.where(keep, self.c, 0), self.c_err, self.n_fit
        return g

    def conjugate_partner(self):
        pos = {key: i for i, key in enumerate(self.m_all)}
        pm = np.array([pos[(j1, j2, -mp, -m)] for j1, j2, mp, m in self.m_all])
        out, off = [], 0
        for (_, a, b, l, n), size in zip(self.blocks, self.sizes):
            ia = {v: i for i, v in enumerate(a)}
            ib = {v: i for i, v in enumerate(b)}
            idx = np.arange(size).reshape(len(a), len(b), len(l), len(n))
            pa = [ia[pm[v]] for v in a]
            pb = [ib[pm[v]] for v in b]
            out.append(off + idx[pa][:, pb].ravel())
            off += size
        return np.concatenate(out)

    def term(self, flat):
        """(block, lep M index, had M index, l, n) of a flat coefficient index."""
        off = 0
        for (name, a, b, l, n), size in zip(self.blocks, self.sizes):
            if flat < off + size:
                ia, ib, il, i_n = np.unravel_index(flat - off, (len(a), len(b), len(l), len(n)))
                return name, self.m_all[a[ia]], self.m_all[b[ib]], l[il], n[i_n]
            off += size
        raise IndexError(flat)

    # ---- integrals and projections ------------------------------------------
    def _factor_vectors(self, name=None, grid=None):
        """Per-factor integral vectors (or values on grid for the free variable)."""
        im = _m_factor_integrals(self.m_all, LEP)
        u = np.linspace(-1, 1, 4001)
        ip = np.trapezoid(legendre_functions(u, self.nl_hi), u, axis=0)
        y = np.linspace(-25, 25, 50001) * self.y_scale
        iy = np.trapezoid(hermite_functions(y, self.nh_hi, self.y_scale), y, axis=0)
        fl, fh, fp, fy = im, im, ip, iy
        if name in LEP:
            fl = _m_factor_integrals(self.m_all, LEP, grid, ("phi", "cos_theta", "phi_prime", "cos_theta_prime")[LEP.index(name)])
        elif name in HAD:
            fh = _m_factor_integrals(self.m_all, HAD, grid, ("phi", "cos_theta", "phi_prime", "cos_theta_prime")[HAD.index(name)])
        elif name == "cos_Theta":
            fp = legendre_functions(grid, self.nl_hi)
        elif name == "Y":
            fy = hermite_functions(grid, self.nh_hi, self.y_scale)
        return fl, fh, fp, fy

    def _contract(self, fl, fh, fp, fy):
        """Σ_blocks c · (factor vectors); a factor given on a grid (2-D) becomes the output axis."""
        out, pos = 0, 0
        for (_, a, b, l, n), size in zip(self.blocks, self.sizes):
            cb = self.c[pos:pos + size].reshape(len(a), len(b), len(l), len(n))
            pos += size
            sub = [vec[..., ix] for vec, ix in ((fl, a), (fh, b), (fp, l), (fy, n))]
            ops = ",".join(("k" + c) if v.ndim == 2 else c for v, c in zip(sub, "jiln"))
            free = "k" if any(v.ndim == 2 for v in sub) else ""
            out = out + np.einsum(f"jiln,{ops}->{free}", cb, *sub, optimize=True)
        return np.real(out)

    def integral(self):
        return float(self._contract(*self._factor_vectors()))

    def projection(self, name, grid):
        return self._contract(*self._factor_vectors(name, np.asarray(grid, float)))

    # ---- persistence -----------------------------------------------------------
    def save(self, path, **meta):
        np.savez(path, kind="extended", c=self.c, c_err=self.c_err, n_fit=self.n_fit,
                 params=np.array([self.J1, self.J2, self.nl, self.nh, self.nl_hi, self.nh_hi]),
                 y_scale=self.y_scale, meta=repr(meta))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        J1, J2, nl, nh, nl_hi, nh_hi = (int(v) for v in z["params"])
        f = cls(J1, J2, nl, nh, nl_hi, nh_hi, float(z["y_scale"]))
        f.c, f.c_err, f.n_fit = z["c"], z["c_err"], int(z["n_fit"])
        return f


# ----------------------------------------------------------------------------
# Spherical-harmonic product basis (for comparison with the paper's M functions)
# ----------------------------------------------------------------------------
def sh_indices(l1max=1, l2max=2):
    """[(l1, l2, m1, m2)] of Y_{l1}^{m1}(θ,φ) Y_{l2}^{m2}(θ',φ'), stored like (j1, j2, m', m) so the
    conjugate partner is (l1, l2, -m1, -m2) as for the M functions."""
    return [(l1, l2, m1, m2) for l1 in range(l1max + 1) for l2 in range(l2max + 1)
            for m1 in range(-l1, l1 + 1) for m2 in range(-l2, l2 + 1)]


def sh_functions(phi, cos_theta, phi_p, cos_theta_p, idx):
    """(N, len(idx)) values of Y_{l1}^{m1}(θ, φ) Y_{l2}^{m2}(θ', φ') (orthonormal in dΩ dΩ')."""
    from scipy.special import sph_harm_y
    th, thp = np.arccos(np.clip(cos_theta, -1, 1)), np.arccos(np.clip(cos_theta_p, -1, 1))
    cache = {}

    def Y(l, m, t, p, key):
        if (key, l, m) not in cache:
            cache[(key, l, m)] = sph_harm_y(l, m, t, p)
        return cache[(key, l, m)]
    return np.stack([Y(l1, m1, th, phi, 0) * Y(l2, m2, thp, phi_p, 1) for l1, l2, m1, m2 in idx], axis=-1)


class SHDensity(TTbarDensity):
    """Same construction as TTbarDensity, but each top's angular factor is the product of two
    ordinary spherical harmonics, Y_{l1}^{m1}(θ,φ) Y_{l2}^{m2}(θ',φ'), l1 <= l1max, l2 <= l2max
    ((l1max+1)^2 (l2max+1)^2 functions per top), instead of the spin-weighted M functions."""

    sh = True

    def __init__(self, n_legendre=5, n_hermite=5, y_scale=1.0, l1max=1, l2max=2):
        super().__init__(n_legendre, n_hermite, y_scale)
        self.l1max, self.l2max = l1max, l2max
        self.midx = sh_indices(l1max, l2max)

    def top_functions(self, phi, cos_theta, phi_p, cos_theta_p):
        return sh_functions(phi, cos_theta, phi_p, cos_theta_p, self.midx)

    def truncated(self, keep):
        g = SHDensity(self.n_legendre, self.n_hermite, self.y_scale, self.l1max, self.l2max)
        g.c, g.c_err, g.n_fit = np.where(keep, self.c, 0), self.c_err, self.n_fit
        return g

    def save(self, path, **meta):
        np.savez(path, kind="sh", c=self.c, c_err=self.c_err, n_fit=self.n_fit,
                 params=np.array([self.n_legendre, self.n_hermite, self.l1max, self.l2max]),
                 y_scale=self.y_scale, meta=repr(meta))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        nl, nh, l1, l2 = (int(v) for v in z["params"])
        f = cls(nl, nh, float(z["y_scale"]), l1, l2)
        f.c, f.c_err, f.n_fit = z["c"], z["c_err"], int(z["n_fit"])
        return f


def load_density(path):
    """Load a TTbarDensity (paper basis), ExtendedDensity or SHDensity file."""
    z = np.load(path, allow_pickle=False)
    kind = str(z["kind"]) if "kind" in z else "paper"
    return {"extended": ExtendedDensity, "sh": SHDensity}.get(kind, TTbarDensity).load(path)
