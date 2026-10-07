import numpy as np
from sklearn.metrics import roc_auc_score

def auc1d(x_h, x_t, lo, hi, nb=25):
    """AUC of the binned likelihood ratio (best 1D discriminant), 2-fold to avoid bias."""
    x_h, x_t = x_h[np.isfinite(x_h)], x_t[np.isfinite(x_t)]
    e = np.linspace(lo, hi, nb + 1)
    sc, y = [], []
    for k in (0, 1):
        tr_h, te_h = x_h[k::2], x_h[1 - k::2]; tr_t, te_t = x_t[k::2], x_t[1 - k::2]
        hh_, _ = np.histogram(np.clip(tr_h, lo, hi - 1e-9), e); ht_, _ = np.histogram(np.clip(tr_t, lo, hi - 1e-9), e)
        r = (hh_ + 1) / hh_.sum() / ((ht_ + 1) / ht_.sum())
        f = lambda x: r[np.clip(np.digitize(np.clip(x, lo, hi - 1e-9), e) - 1, 0, nb - 1)]
        sc += [f(te_h), f(te_t)]; y += [np.ones(len(te_h)), np.zeros(len(te_t))]
    return roc_auc_score(np.concatenate(y), np.concatenate(sc))

