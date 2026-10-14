"""
KL divergence carried by a score, D_KL(r || 1) = ∫_0^1 r ln r dq, where q is the tt̄ quantile of the score (tt̄ uniform
in q) and r the density of the signal in q.  Since tt̄ is uniform, D_KL(r || 1) = -H(q_signal), the negative
differential entropy of the signal's q values.

Estimator: Vasicek m-spacing entropy with Ebrahimi's boundary correction (m = round(sqrt(n)/2)), with the bias of the
estimator for a uniform sample of the same size subtracted (it is not zero for finite n), and bootstrap errors that
resample both the tt̄ reference and the signal.
"""
import numpy as np


def quantile(ref, x):
    """Fraction of the reference with a lower score than x (ties count half): the tt̄ quantile q."""
    r = np.sort(ref)
    lo = np.searchsorted(r, x, side="left")
    hi = np.searchsorted(r, x, side="right")
    return (lo + hi) / (2.0 * len(r))


def entropy_mspacing(x, m=None):
    """Vasicek m-spacing estimate of the differential entropy, with Ebrahimi's correction at the ends."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    m = m or max(1, int(round(np.sqrt(n) / 2)))
    i = np.arange(n)
    up = x[np.minimum(i + m, n - 1)]
    dn = x[np.maximum(i - m, 0)]
    c = np.where(i < m, 1 + i / m, np.where(i >= n - m, 1 + (n - 1 - i) / m, 2.0))
    d = np.maximum(up - dn, 1e-12)
    return float(np.mean(np.log(n * d / (c * m))))


def null_bias(n, reps=200, seed=1):
    rng = np.random.default_rng(seed)
    return float(np.mean([entropy_mspacing(rng.random(n)) for _ in range(reps)]))


def kl_score(y_ref, y_sig, nboot=200, seed=2):
    """D_KL(r||1) with r the density of the signal in the tt̄ quantile of y (low y = unlike tt̄), and its error."""
    b = null_bias(len(y_sig))
    est = lambda ref, sig: -(entropy_mspacing(quantile(ref, sig)) - b)  # noqa: E731
    d0 = est(y_ref, y_sig)
    rng = np.random.default_rng(seed)
    reps = [est(y_ref[rng.integers(0, len(y_ref), len(y_ref))], y_sig[rng.integers(0, len(y_sig), len(y_sig))])
            for _ in range(nboot)]
    return d0, float(np.std(reps))
