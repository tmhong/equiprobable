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

Angles are passed as a dict of arrays keyed by plotting.angles.ANGLE_NAMES.
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
    """Wigner small-d d^j_{m'm}(beta) (integer j), explicit Wigner formula."""
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

    def factors(self, X):
        """Basis factors (M_lep, M_had, p_l, h_n) evaluated on the events in X."""
        return (m_functions(*(X[k] for k in LEP), self.midx),
                m_functions(*(X[k] for k in HAD), self.midx),
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
        vals = m_functions(*(x.ravel() for x in g), self.midx)
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
            out.append(w @ m_functions(*(v.ravel() for v in g), self.midx))
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
    a, b, l, n = np.unravel_index(flat_index, f.shape)
    parts, tags = [], []
    for side, (j1, j2, mp, m) in (("lep", f.midx[a]), ("had", f.midx[b])):
        if (j1, j2) == (0, 0):
            continue
        parts.append(f"{side}[j1={j1},j2={j2},m'={mp},m={m}]")
        if j1 == 0:
            tags.append(f"W helicity ({side})")
        elif j2 == 0:
            tags.append(f"top polarization ({side}, {'longitudinal' if mp == 0 else 'transverse'})")
        else:
            tags.append(f"top pol. x W helicity interference ({side})")
    if f.midx[a][0] == 1 and f.midx[b][0] == 1:
        tags = ["tt spin correlation"] + tags
    if l:
        parts.append(f"P{l}(cosΘ)")
        tags.append("production angle")
    if n:
        parts.append(f"H{n}(Y)")
        tags.append("rapidity")
    return (" x ".join(parts) or "constant"), "; ".join(dict.fromkeys(tags)) or "normalization"
