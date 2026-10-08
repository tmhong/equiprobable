#!/usr/bin/env python3
"""
Realistic quark tagging for the hadronic-side angles.

The paper assumes the down-type W_had daughter is known (perfect up/down tagging).  In reality the
only handle is c-tagging (W -> c s: the c-tagged jet is up-type, so the other is the down-type s)
plus a weak kinematic preference.  Here the down-type label is replaced by a probability:

  w = P(softer W jet is down-type | c-tags, kinematics)                       (Bayes)
  hypotheses: W -> c s or u d (equal), with c (or u) as the harder or the softer jet;
  prior P(softer is down) = p_s measured on tt̄ (30k events of the fit sample, truth labels);
  tag likelihood per jet: c-flavour -> eps_c, b -> eps_b, other -> eps_l  (jets are tagged
  at random with these efficiencies using the true flavour: Delphes jet flavour at reco,
  the quark id at parton level)

  y = w f(Ω[down = softer]) + (1 - w) f(Ω[down = harder])

i.e. the tt̄ density of the event as observed (unlabelled), with f fit on truth-labelled tt̄.
Compared: truth labels, softer jet always down, symmetrized (w = 1/2: no tagging at all),
c-tag at two illustrative ATLAS-like working points, and an ideal c-tagger.
Scores: angles only, and angles + the Breit-Wigner mass factors (the masses do not depend on
the labelling).  Output: study_salvage/tagging.json
"""
import json
import os
import sys

import awkward as ak
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import make_plots  # noqa: E402
from salvage import surprisal  # noqa: E402
from simplest import PDG, bw_surprisal, cauchy_surprisal  # noqa: E402
from plotting import angles as A  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402

OPTS = {"had_fermion": "down", "jet_assignment": "truth"}
# signal sample (SIGNAL_SAMPLE) and output suffix (OUT_SUFFIX, e.g. "_h170"); defaults = hh(125), tagging.json
HH_DEFAULT = "samples/hh_bblvqq/run_seed2001"
HH = os.environ.get("SIGNAL_SAMPLE", HH_DEFAULT)
SUFFIX = os.environ.get("OUT_SUFFIX", "")
if os.path.normpath(HH) != HH_DEFAULT and not SUFFIX:
    sys.exit("SIGNAL_SAMPLE is not hh(125): set OUT_SUFFIX (e.g. _h170) so the hh(125) outputs are not overwritten")
WPS = {"c-tag tight (eps_c 30%, b 11%, light 1.4%)": (0.30, 0.11, 0.014),
       "c-tag loose (eps_c 45%, b 30%, light 8%)": (0.45, 0.30, 0.08),
       "ideal c-tag (eps_c 100%, no mistag)": (1.0, 0.0, 0.0)}
HAD = ["cos_theta_tilde", "phi_tilde", "cos_theta_prime_tilde", "phi_prime_tilde"]


def arr(v):
    return np.stack([np.asarray(v.px, float), np.asarray(v.py, float), np.asarray(v.pz, float), np.asarray(v.E, float)], 1)


def extract(path, level, nmax=None):
    ev, _ = make_plots.load_sample("x", path, ["parton", level] if level != "parton" else ["parton"], nmax, OPTS)
    e = ev[level]
    if level == "parton":
        ok = np.ones(len(e), bool)
        harder, softer = A.as_array(e.quarks[:, 0]), A.as_array(e.quarks[:, 1])
        pid = np.abs(ak.to_numpy(e.quarks.pid[:, :2]))
        fl_h, fl_s = pid[:, 0], pid[:, 1]
        masses = {k: np.asarray(fn, float) for k, fn in (("m_w_lep", e.w_lep.mass), ("m_w_had", e.w_had.mass),
                                                          ("m_t_lep", e.t_lep.mass), ("m_t_had", e.t_had.mass))}
    else:
        ok = np.asarray(e.reco_ok)
        harder, softer = A.as_array(e.q1), A.as_array(e.q2)
        fl_h, fl_s = np.abs(np.asarray(e.q1_flavor)), np.abs(np.asarray(e.q2_flavor))
        masses = {k: np.asarray(fn, float) for k, fn in (("m_w_had", e.w_had.mass), ("m_t_lep", e.t_lep.mass),
                                                          ("m_t_had", e.t_had.mass))}
    t_had, w_had = A.as_array(e.t_had), A.as_array(e.w_had)
    base = {a: np.asarray(e[a], float) for a in ANGLE_NAMES}          # truth-labelled angles
    alt = {}
    for name, fj in (("softer", softer), ("harder", harder)):
        ct, ph, ctp, php = A.decay_angles(t_had, w_had, fj)
        alt[name] = {**base, "cos_theta_tilde": ct, "phi_tilde": ph, "cos_theta_prime_tilde": ctp,
                     "phi_prime_tilde": php}
    # which jet is truly the down-type one: compare the truth-labelled angle with the 'softer' version
    softer_is_down = np.isclose(alt["softer"]["cos_theta_prime_tilde"], base["cos_theta_prime_tilde"])
    ok &= np.all([np.isfinite(base[a]) for a in ANGLE_NAMES], 0)
    cut = lambda d: {a: v[ok] for a, v in d.items()}  # noqa: E731
    return {"truth": cut(base), "softer": cut(alt["softer"]), "harder": cut(alt["harder"]),
            "softer_is_down": softer_is_down[ok], "fl_h": fl_h[ok], "fl_s": fl_s[ok],
            "m": {k: v[ok] for k, v in masses.items()}}


def tag(fl, wp, rng):
    ec, eb, el = wp
    p = np.where(fl == 4, ec, np.where(fl == 5, eb, el))
    return rng.random(len(fl)) < p


def posterior_softer_down(th, ts, wp, p_s):
    """P(softer jet is down-type | tags on harder (th), softer (ts)) for W -> cs / ud, equal rates.
    The c-tag likelihood of a true c jet is eps_c; of a light (u, d, s) jet eps_l."""
    ec, _, el = (np.clip(x, 1e-4, 1 - 1e-4) for x in wp)   # avoid 0/0 for tag patterns impossible under the WP
    L = lambda t, e: np.where(t, e, 1 - e)  # noqa: E731
    # softer is down: up-type = harder.  cs: harder = c;  ud: both light
    a = p_s * 0.5 * (L(th, ec) * L(ts, el) + L(th, el) * L(ts, el))
    b = (1 - p_s) * 0.5 * (L(ts, ec) * L(th, el) + L(ts, el) * L(th, el))
    return a / (a + b)


def main():
    rng = np.random.default_rng(11)
    out = {}
    for level, fname in (("parton", "f_ttbar_parton.npz"), ("reco", "f_ttbar_reco.npz")):
        ref = extract(os.path.join(ROOT, "samples/ttbar_bblvqq/run_seed1001"), level, 30000)
        tt = extract(os.path.join(ROOT, "samples/ttbar_bblvqq/run_seed12345"), level)
        hh = extract(os.path.join(ROOT, HH), level)
        p_s = float(np.mean(ref["softer_is_down"]))
        lab = np.r_[np.zeros(len(tt["fl_h"])), np.ones(len(hh["fl_h"]))]
        # mass surprisal (labelling independent)
        sm = 0
        for k in tt["m"]:
            if level == "parton":
                sm = sm + np.r_[bw_surprisal(tt["m"][k], *PDG[k]), bw_surprisal(hh["m"][k], *PDG[k])]
            else:
                x = ref["m"][k]; x = x[np.isfinite(x)]
                M = np.median(x); q1, q3 = np.percentile(x, [25, 75]); W = 0.5 * (q3 - q1)
                sm = sm + np.r_[cauchy_surprisal(tt["m"][k], M, W), cauchy_surprisal(hh["m"][k], M, W)]
        full = osde.load_density(os.path.join(ROOT, "osde", fname))
        res = {"p_softer_is_down_tt": p_s, "n_tt": int((lab == 0).sum()), "n_hh": int(lab.sum()),
               "frac_cs_tt": float(np.mean((tt["fl_h"] == 4) | (tt["fl_s"] == 4)))}
        for fname_, f in (("all terms", full), ("23-ish terms (>10 sigma)", full.truncated(full.significance() > 10))):
            fr = f(ref["truth"])
            ev_f = {s: {k: f(d[s]) for k, d in (("tt", tt), ("hh", hh))} for s in ("truth", "softer", "harder")}
            strategies = {"truth (paper assumption)": None, "softer jet = down": "softer",
                          "symmetrized (no tagging)": 0.5}
            strategies.update({wp: wp for wp in WPS})
            for sname, how in strategies.items():
                ys, wcorrect = {}, None
                for k, d in (("tt", tt), ("hh", hh)):
                    if how is None:
                        ys[k] = ev_f["truth"][k]
                    elif how == "softer":
                        ys[k] = ev_f["softer"][k]
                    else:
                        if isinstance(how, float):
                            w = np.full(len(d["fl_h"]), how)
                        else:
                            wp = WPS[how]
                            w = posterior_softer_down(tag(d["fl_h"], wp, rng), tag(d["fl_s"], wp, rng), wp, p_s)
                        if k == "tt" and not isinstance(how, float):
                            wcorrect = float(np.mean((w > 0.5) == d["softer_is_down"]))
                        ys[k] = w * ev_f["softer"][k] + (1 - w) * ev_f["harder"][k]
                sa = np.r_[surprisal(ys["tt"], fr), surprisal(ys["hh"], fr)]
                res[f"{fname_} | {sname}"] = {"angles": roc_auc_score(lab, sa), "angles+BW": roc_auc_score(lab, sa + sm),
                                              "tt label correct (w>0.5)": wcorrect}
        res["masses only (BW)"] = roc_auc_score(lab, sm)
        out[level] = res
        print(level, json.dumps(res, indent=1), flush=True)
    json.dump(out, open(os.path.join(HERE, f"tagging{SUFFIX}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
