"""
Summary numbers for validation and the slides (make_slides.py).

All functions take Events from make_plots.load_sample(); comparisons to the parton
level rely on LHE and Delphes events being aligned (event i <-> event i).
"""
import numpy as np

from . import angles


def cutflow(ev):
    """[(label, n_pass)] for the default bblvqq preselection."""
    steps = [("All events", np.ones(len(ev), bool)),
             ("exactly 1 lepton", ev.nleptons == 1),
             (r"$\geq 4$ jets", ev.njets >= 4),
             (r"$\geq 1$ b-jet", ev.nbjets >= 1),
             (r"$\geq 2$ b-jets", ev.nbjets >= 2)]
    out, m = [], np.ones(len(ev), bool)
    for name, cut in steps:
        m = m & np.asarray(cut)
        out.append((name, int(m.sum())))
    out.append((r"$\chi^2$ reconstruction (from $\geq 1$ b)", int(np.sum(ev.reco_ok))))
    return out


def helicity_fractions(cos):
    """
    W helicity fractions from the helicity-angle moments of
    3/8 (1-c)^2 F_L + 3/4 (1-c^2) F_0 + 3/8 (1+c)^2 F_R:
      F_0 = 2 - 5<c^2>,  F_R - F_L = 2<c>.
    Returns dict with F0, FL, FR and their statistical errors.
    """
    c = np.asarray(cos, float)
    c = c[np.isfinite(c)]
    n = len(c)
    F0, d = 2 - 5 * np.mean(c**2), 2 * np.mean(c)
    FL = (1 - F0 - d) / 2
    eF0 = 5 * np.std(c**2) / np.sqrt(n)
    ed = 2 * np.std(c) / np.sqrt(n)
    eFL = 0.5 * np.hypot(eF0, ed)
    return dict(n=n, F0=F0, FL=FL, FR=1 - F0 - FL, eF0=eF0, eFL=eFL, eFR=eFL)


def _dr(a, b):
    return np.nan_to_num(np.asarray(a.deltaR(b), dtype=float), nan=9.0)


def correct_assignment(ev, parton, dr=0.4):
    """Events where the chi2 picked the right jets for b_had, b_lep, q_down, q_up."""
    return (np.asarray(ev.reco_ok)
            & (_dr(ev.b_had, parton.b_had) < dr) & (_dr(ev.b_lep, parton.b_lep) < dr)
            & (_dr(ev.q_down, parton.q_down) < dr) & (_dr(ev.q_up, parton.q_up) < dr))


def central(x):
    """(median, 68% half-width) of finite values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return q50, (q84 - q16) / 2


def angle_comparison(ev, parton, mask):
    """{angle: (correlation, median |delta|)} between ev and parton angles for mask."""
    out = {}
    for a in angles.ANGLE_NAMES:
        x, y = np.asarray(ev[a])[mask], np.asarray(parton[a])[mask]
        d = x - y
        if a.startswith("phi"):
            d = (d + np.pi) % (2 * np.pi) - np.pi
        out[a] = (np.corrcoef(x, y)[0, 1], np.median(np.abs(d)))
    return out


def summarize(events):
    """All numbers used on the slides, from {level: Event}."""
    P = events.get("parton")
    S = {"levels": list(events)}
    for lvl, ev in events.items():
        s = S[lvl] = {"n": len(ev)}
        s["hel_lep"] = helicity_fractions(ev.cos_theta_prime)
        s["hel_had"] = helicity_fractions(ev.cos_theta_prime_tilde)
        if lvl == "parton":
            continue
        s["cutflow"] = cutflow(ev)
        s["m_thad"], s["m_whad"] = central(ev.t_had.mass), central(ev.w_had.mass)
        if P is not None:
            ok = correct_assignment(ev, P)
            s["n_reco"], s["n_correct"] = int(np.sum(ev.reco_ok)), int(ok.sum())
            s["m_thad_correct"] = central(np.asarray(ev.t_had.mass)[ok])
            s["m_whad_correct"] = central(np.asarray(ev.w_had.mass)[ok])
            s["angles_all"] = angle_comparison(ev, P, np.asarray(ev.reco_ok))
            s["angles_correct"] = angle_comparison(ev, P, ok)
            s["nu_pz"] = central((np.asarray(ev.nu.pz) - np.asarray(P.nu.pz))[np.asarray(ev.reco_ok)])
    return S
