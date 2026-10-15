"""
Build analysis-level objects for the three levels of the bblvqq samples:

  build_parton(lhe_particles)  -> MG5 partons: tops, Ws, b's, lepton, neutrino, quarks
  build_gen(delphes)           -> particle level: GenJets, prompt leptons, GenMET
  build_reco(delphes)          -> Delphes reco: jets, b-jets, leptons, MET

Every builder returns an `Event` (attribute access to awkward arrays of
vector Momentum4D objects), so plot expressions look like
    ev.jets.pt, (ev.top_had).mass, ev.lep.eta, ev.nbjets ...

Object cuts live in CUTS below; edit them there (or in plot_config.py via
physics.CUTS.update(...)).
"""
from itertools import permutations

import awkward as ak
import numpy as np
import vector

vector.register_awkward()

M_W, M_TOP = 80.4, 172.5

CUTS = {
    "jet_pt": 25.0, "jet_eta": 2.5,
    "el_pt": 25.0, "el_eta": 2.47,
    "mu_pt": 25.0, "mu_eta": 2.5,
    # chi2 reconstruction: resolutions (GeV) and max number of jets considered
    "sigma_w": 10.0, "sigma_top": 15.0, "reco_max_jets": 6,
}


class Event(dict):
    """dict with attribute access: ev.jets == ev['jets']."""
    __getattr__ = dict.__getitem__

    def __setattr__(self, k, v):
        self[k] = v

    def __len__(self):
        return len(self["weight"])

    @staticmethod
    def concat(parts):
        """Concatenate Events built from consecutive chunks of the same sample."""
        out = Event()
        for k in parts[0]:
            vals = [p[k] for p in parts]
            if isinstance(vals[0], np.ndarray):
                out[k] = np.concatenate(vals)
            elif isinstance(vals[0], ak.Array):
                out[k] = ak.concatenate(vals)
            else:
                out[k] = vals[0]
        return out

    def mask(self, sel):
        """New Event keeping only events where sel is True (per-event arrays are filtered)."""
        sel = np.asarray(ak.to_numpy(ak.fill_none(ak.Array(sel), False)) if not isinstance(sel, np.ndarray) else sel,
                         dtype=bool)
        n = len(self)
        out = Event()
        for k, v in self.items():
            try:
                out[k] = v[sel] if len(v) == n else v
            except TypeError:          # scalars
                out[k] = v
        return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def p4_ptetaphim(pt, eta, phi, m, **extra):
    return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": m, **extra}, with_name="Momentum4D")


def p4_pxpypze(px, py, pz, e, **extra):
    return ak.zip({"px": px, "py": py, "pz": pz, "E": e, **extra}, with_name="Momentum4D")


def sort_pt(objs):
    return objs[ak.argsort(objs.pt, ascending=False)]


def first(objs, mask):
    """Single object per event: first element of objs[mask] (None if absent)."""
    return ak.firsts(objs[mask])


def neutrino_pz(lpx, lpy, lpz, le, met_px, met_py, mw=M_W):
    """
    Both W-mass constrained solutions for the neutrino pz, shape (2, n).
    If there is no real solution, both entries are the real part.
    """
    lpt2 = lpx**2 + lpy**2
    mu = 0.5 * mw**2 + lpx * met_px + lpy * met_py
    a = mu * lpz / lpt2
    disc = a**2 - (le**2 * (met_px**2 + met_py**2) - mu**2) / lpt2
    root = np.sqrt(np.clip(disc, 0, None))
    return np.stack([a - root, a + root])


# --------------------------------------------------------------------------
# parton level (LHE)
# --------------------------------------------------------------------------
def build_parton(particles, weight):
    """
    Parton-level objects from the LHE record.  Works for tt̄ and for hh (or any
    b b̄ l nu q q' final state): t_lep = W_lep + b_lep and t_had = W_had + b_had.
    tt̄: the b pairing is the true one (l+ comes from t -> b W+).
    otherwise: the pairing that best fits the tt̄ hypothesis (both m_Wb closest
    to m_t), so tt̄-hypothesis quantities such as the angles can be computed.
    """
    p = particles
    v = p4_pxpypze(p.px, p.py, p.pz, p.e, pid=p.pid)
    out = (p.status == 1)
    apid = abs(p.pid)

    ev = Event(weight=np.asarray(weight))
    ev.lep = first(v, out & ((apid == 11) | (apid == 13) | (apid == 15)))
    ev.nu = first(v, out & ((apid == 12) | (apid == 14) | (apid == 16)))
    ev.b = first(v, out & (p.pid == 5))
    ev.bbar = first(v, out & (p.pid == -5))
    ev.quarks = sort_pt(v[out & (apid <= 4)])
    ev.lep_charge = -np.sign(ak.to_numpy(ev.lep.pid))     # pid 11 = e-
    ev.lep_flavour = abs(ak.to_numpy(ev.lep.pid))
    ev.w_lep = ev.lep + ev.nu
    ev.w_had = ev.quarks[:, 0] + ev.quarks[:, 1]
    qid = abs(ev.quarks.pid)
    ev.q_down = first(ev.quarks, (qid == 1) | (qid == 3))   # d, s, d̄, s̄
    ev.q_up = first(ev.quarks, (qid == 2) | (qid == 4))     # u, c, ū, c̄

    def chi2(b_lep, b_had):
        return np.asarray(((ev.w_lep + b_lep).mass - M_TOP)**2 + ((ev.w_had + b_had).mass - M_TOP)**2)

    is_tt = ak.to_numpy(ak.any(abs(p.pid) == 6, axis=1))
    b_with_lep = np.where(is_tt, ev.lep_charge > 0, chi2(ev.b, ev.bbar) < chi2(ev.bbar, ev.b))
    ev.b_lep = ak.where(b_with_lep, ev.b, ev.bbar)
    ev.b_had = ak.where(b_with_lep, ev.bbar, ev.b)
    ev.t_lep = ev.w_lep + ev.b_lep
    ev.t_had = ev.w_had + ev.b_had
    ev.ttbar = ev.t_lep + ev.t_had
    # "parton + lepton trigger": the charged lepton passes the offline single-lepton cuts
    lpt = ak.to_numpy(ak.fill_none(ev.lep.pt, 0.0))
    leta = np.abs(ak.to_numpy(ak.fill_none(ev.lep.eta, 99.0)))
    is_e = ev.lep_flavour == 11
    ev.lep_trigger = np.where(is_e, (lpt > CUTS["el_pt"]) & (leta < CUTS["el_eta"]),
                              (lpt > CUTS["mu_pt"]) & (leta < CUTS["mu_eta"])) & (ev.lep_flavour != 15)
    return ev


# --------------------------------------------------------------------------
# particle level (Delphes GenJet / Particle / GenMissingET)
# --------------------------------------------------------------------------
def build_gen(d):
    c = CUTS
    part = d["Particle"]
    ev = Event(weight=_delphes_weight(d))

    # prompt leptons: status-1 e/mu whose ancestry (skipping self-copies) is a W, Z, h
    # or tau (in h -> W W* -> l nu q q the lepton can come straight from the h)
    anc, anc_idx = _ancestor(part)
    anc = abs(anc)
    # for leptons from a tau, require the tau itself to come from a boson (not a B/D decay)
    tau_anc = abs(_ancestor(part)[0])[ak.where(anc_idx >= 0, anc_idx, 0)]
    boson = (anc == 24) | (anc == 23) | (anc == 25)
    from_boson_tau = (anc == 15) & ((tau_anc == 24) | (tau_anc == 23) | (tau_anc == 25))
    apid = abs(part.pid)
    is_lep = (part.status == 1) & ((apid == 11) | (apid == 13)) & (boson | from_boson_tau)
    prompt = p4_ptetaphim(part.pt, part.eta, part.phi, part.mass, charge=part.charge, flavour=apid)[is_lep]
    is_el = prompt.flavour == 11
    ev.leptons = sort_pt(prompt[ak.where(is_el, (prompt.pt > c["el_pt"]) & (abs(prompt.eta) < c["el_eta"]),
                                         (prompt.pt > c["mu_pt"]) & (abs(prompt.eta) < c["mu_eta"]))])

    gj = d["GenJet"]
    jets = p4_ptetaphim(gj.pt, gj.eta, gj.phi, gj.mass)
    jets = jets[(jets.pt > c["jet_pt"]) & (abs(jets.eta) < c["jet_eta"])]
    # GenJets are clustered from all visible particles, so the prompt lepton also
    # appears as a jet: remove jets within dR < 0.4 of a prompt lepton (pT > 10).
    jets = jets[~_near(jets, prompt[prompt.pt > 10], 0.4)]

    # b-label: GenJet within dR<0.4 of a b quark with pT > 5 GeV
    bq = p4_ptetaphim(part.pt, part.eta, part.phi, part.mass)
    bq = bq[(abs(part.pid) == 5) & (part.pt > 5)]
    jets = ak.with_field(jets, _near(jets, bq, 0.4), "btag")
    ev.jets = sort_pt(jets)
    ev.bjets = ev.jets[ev.jets.btag]
    ev.ljets = ev.jets[~ev.jets.btag]

    met = d["GenMissingET"]
    ev.met = ak.firsts(met.met)
    ev.met_phi = ak.firsts(met.phi)
    _finish(ev)
    return ev


def _near(objs, others, dr):
    """Per object: is any of `others` (same event) within deltaR < dr?"""
    pairs = ak.cartesian([objs, others], nested=True)
    return ak.any(pairs["0"].deltaR(pairs["1"]) < dr, axis=-1)


def _ancestor(part, max_steps=40):
    """(pid, index in event) of the first ancestor (via M1) whose pid differs from the
    particle's own, i.e. skipping the particle's shower copies; (0, -1) if none."""
    counts = ak.to_numpy(ak.num(part.pid))
    offsets = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    pid = ak.to_numpy(ak.flatten(part.pid))
    m1 = ak.to_numpy(ak.flatten(part.m1))
    n = len(pid)
    cur = np.arange(n)
    anc = np.zeros(n, dtype=pid.dtype)
    anc_idx = np.full(n, -1)
    active = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        mother_local = m1[cur]
        has_mother = active & (mother_local >= 0)
        mother = np.clip(np.where(has_mother, offsets + mother_local, cur), 0, n - 1)
        differs = has_mother & (pid[mother] != pid)
        anc = np.where(differs, pid[mother], anc)
        anc_idx = np.where(differs, mother - offsets, anc_idx)
        active = has_mother & ~differs
        cur = np.where(active, mother, cur)
        if not active.any():
            break
    return ak.unflatten(anc, counts), ak.unflatten(anc_idx, counts)


def _ancestor_pid(part, max_steps=40):
    return _ancestor(part, max_steps)[0]


# --------------------------------------------------------------------------
# reco level (Delphes)
# --------------------------------------------------------------------------
def build_reco(d):
    c = CUTS
    ev = Event(weight=_delphes_weight(d))

    j = d["Jet"]
    jets = p4_ptetaphim(j.pt, j.eta, j.phi, j.mass, btag=(j.btag & 1) == 1, flavor=j.flavor)
    jets = jets[(jets.pt > c["jet_pt"]) & (abs(jets.eta) < c["jet_eta"])]
    ev.jets = sort_pt(jets)
    ev.bjets = ev.jets[ev.jets.btag]
    ev.ljets = ev.jets[~ev.jets.btag]

    e, m = d["Electron"], d["Muon"]
    els = p4_ptetaphim(e.pt, e.eta, e.phi, 0.000511 * ak.ones_like(e.pt), charge=e.charge, flavour=11 * ak.ones_like(e.charge))
    mus = p4_ptetaphim(m.pt, m.eta, m.phi, 0.10566 * ak.ones_like(m.pt), charge=m.charge, flavour=13 * ak.ones_like(m.charge))
    els = els[(els.pt > c["el_pt"]) & (abs(els.eta) < c["el_eta"])]
    mus = mus[(mus.pt > c["mu_pt"]) & (abs(mus.eta) < c["mu_eta"])]
    ev.electrons, ev.muons = els, mus
    ev.leptons = sort_pt(ak.concatenate([els, mus], axis=1))

    met = d["MissingET"]
    ev.met = ak.firsts(met.met)
    ev.met_phi = ak.firsts(met.phi)
    _finish(ev)
    return ev


def _delphes_weight(d):
    if "Event" in d and "weight" in d["Event"].fields:
        return ak.to_numpy(ak.fill_none(ak.firsts(d["Event"].weight), 1.0))
    return np.ones(len(next(iter(d.values()))))


# --------------------------------------------------------------------------
# common: counters, selection, chi2 ttbar reconstruction
# --------------------------------------------------------------------------
def _finish(ev):
    ev.njets = ak.to_numpy(ak.num(ev.jets))
    ev.nbjets = ak.to_numpy(ak.num(ev.bjets))
    ev.nleptons = ak.to_numpy(ak.num(ev.leptons))
    ev.met = ak.to_numpy(ak.fill_none(ev.met, 0.0))
    ev.met_phi = ak.to_numpy(ak.fill_none(ev.met_phi, 0.0))
    ev.lep = ak.firsts(ev.leptons)
    ev.mtw = np.sqrt(np.clip(2 * ak.to_numpy(ak.fill_none(ev.lep.pt, 0.0)) * ev.met *
                             (1 - np.cos(ak.to_numpy(ak.fill_none(ev.lep.phi, 0.0)) - ev.met_phi)), 0, None))
    # default preselection: exactly 1 lepton, >= 4 jets, >= 1 b-jet
    ev.presel = (ev.nleptons == 1) & (ev.njets >= 4) & (ev.nbjets >= 1)
    _reconstruct_ttbar(ev)


def _reconstruct_ttbar(ev):
    """
    chi2 = ((m_jj - mW)/sW)^2 + ((m_jjb - mt)/st)^2 + ((m_lvb - mt)/st)^2
    minimised over assignments (b_had, b_lep, q1, q2) of the leading jets,
    requiring min(nbjets, 2) b-tagged jets in the two b slots, and over the two
    W-mass-constrained neutrino pz solutions (so the top-mass term resolves the
    two-fold ambiguity).
    Results (NaN for events failing preselection): ev.reco_chi2, ev.w_had,
    ev.t_had, ev.b_had, ev.q1, ev.q2 (W_had jets, pT-ordered), ev.w_lep, ev.t_lep,
    ev.b_lep, ev.nu, ev.ttbar.
    """
    c = CUTS
    nmax = c["reco_max_jets"]
    nev = len(ev.weight)
    jets = ak.pad_none(ev.jets[:, :nmax], nmax, clip=True)

    def dense(x, fill=np.nan):
        return ak.to_numpy(ak.fill_none(x, fill)).astype(float)

    jpx, jpy, jpz, je = (dense(getattr(jets, a)) for a in ("px", "py", "pz", "E"))
    jb = dense(jets.btag, 0).astype(bool)

    lep = ev.lep
    lpx, lpy, lpz, le = (dense(getattr(lep, a)) for a in ("px", "py", "pz", "E"))
    mpx, mpy = ev.met * np.cos(ev.met_phi), ev.met * np.sin(ev.met_phi)
    nupz = neutrino_pz(lpx, lpy, lpz, le, mpx, mpy)                 # (2, nev)
    nue = np.sqrt(mpx**2 + mpy**2 + nupz**2)
    wl = np.stack([lpx + mpx + 0 * nupz, lpy + mpy + 0 * nupz, lpz + nupz, le + nue])  # (4, 2, nev)

    def mass(px, py, pz, e):
        return np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0, None))

    perms = np.array([p for p in permutations(range(nmax), 4) if p[2] < p[3]])
    bh, bl, q1, q2 = perms.T
    need = np.minimum(ev.nbjets, 2)
    nperm = len(perms)
    best = np.zeros(nev, dtype=int)                         # index into (nu solution, perm)
    best_chi2 = np.full(nev, np.inf)
    # process in chunks: the (event x permutation) arrays get large otherwise
    for lo in range(0, nev, 20000):
        s = slice(lo, lo + 20000)
        J = np.stack([jpx[s], jpy[s], jpz[s], je[s]])      # (4, n, nmax)
        W = J[:, :, q1] + J[:, :, q2]                      # (4, n, nperm)
        TH = W + J[:, :, bh]
        chi2_had = ((mass(*W) - M_W) / c["sigma_w"])**2 + ((mass(*TH) - M_TOP) / c["sigma_top"])**2
        chi2 = np.concatenate([
            chi2_had + ((mass(*(wl[:, isol, s, None] + J[:, :, bl])) - M_TOP) / c["sigma_top"])**2
            for isol in (0, 1)], axis=1)                   # (n, 2 * nperm)
        ntag_slots = np.tile(jb[s][:, bh].astype(int) + jb[s][:, bl].astype(int), 2)
        chi2 = np.where((ntag_slots == need[s, None]) & np.isfinite(chi2), chi2, np.inf)
        best[s] = np.argmin(chi2, axis=1)
        best_chi2[s] = chi2[np.arange(chi2.shape[0]), best[s]]
    ok = np.asarray(ev.presel) & np.isfinite(best_chi2)

    rows = np.arange(nev)
    isol, iperm = best // nperm, best % nperm
    sel = perms[iperm]                                      # (nev, 4) jet indices
    J = np.stack([jpx, jpy, jpz, je])

    def jet(slot):
        return J[:, rows, sel[:, slot]]

    wl = wl[:, isol, rows]                                  # chosen neutrino solution
    nu = np.stack([mpx, mpy, nupz[isol, rows], nue[isol, rows]])
    W = jet(2) + jet(3)
    TH = W + jet(0)
    TL = wl + jet(1)

    def pick(arr):
        return np.where(ok, arr, np.nan)

    def v4(a):
        return vector.zip({"px": a[0], "py": a[1], "pz": a[2], "E": a[3]})

    ev.reco_ok = ok
    ev.reco_chi2 = np.where(ok, best_chi2, np.nan)
    ev.nu = v4(pick(nu))
    ev.w_had, ev.t_had, ev.t_lep = v4(pick(W)), v4(pick(TH)), v4(pick(TL))
    ev.w_lep = v4(np.where(ok, wl, np.nan))
    ev.b_had, ev.b_lep = v4(pick(jet(0))), v4(pick(jet(1)))
    ev.q1, ev.q2 = v4(pick(jet(2))), v4(pick(jet(3)))      # q1 is the higher-pT jet
    if "flavor" in ak.fields(jets):                         # Delphes truth flavour of the W_had jets (reco only)
        fl = dense(jets.flavor, 0)
        ev.q1_flavor = np.where(ok, fl[rows, sel[:, 2]], 0).astype(int)
        ev.q2_flavor = np.where(ok, fl[rows, sel[:, 3]], 0).astype(int)
    ev.ttbar = ev.t_had + ev.t_lep
