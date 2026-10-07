"""
The ten tt̄ angles x = {Y, Θ, θ, φ, θ', φ', θ̃, φ̃, θ̃', φ̃'} as defined in the paper
(main.tex, "Angular Distributions"):

  Y        rapidity of the tt̄ system in the lab frame
  Θ        polar angle (w.r.t. the beam axis) of the leptonic top in the tt̄ rest frame
  θ, φ     polar/azimuthal angles of the W momentum in the top rest frame, with axes
             z = direction of the top momentum in the lab frame
             y = beam × z  (normal to the plane of the beam axis and the top momentum)
             x = y × z
  θ', φ'   spherical coordinates of the fermion in the W rest frame, reached by the
           standard boost from the top rest frame, with axes
             z' = direction of the W momentum (in the top rest frame)
             y' = z × z'  (lies in the xy plane)
             x' = y' × z'
           i.e. (x', y', z') = R(φ, θ, 0)(x, y, z), matching D(φ, θ, 0).

Unadorned names are for the leptonic top (fermion = charged lepton); "_tilde" names
are for the hadronic top (fermion = down-type quark by default, the analogue of the
charged lepton: W+ -> l+ nu  <->  W+ -> u d̄).

add_angles(ev) stores on the Event:
  Y, cos_Theta,
  cos_theta, phi, cos_theta_prime, phi_prime,
  cos_theta_tilde, phi_tilde, cos_theta_prime_tilde, phi_prime_tilde
(azimuths in (-π, π]; NaN where the event has no reconstructed tt̄ system).
"""
import awkward as ak
import numpy as np
import vector

ANGLE_NAMES = ["Y", "cos_Theta",
               "cos_theta", "phi", "cos_theta_prime", "phi_prime",
               "cos_theta_tilde", "phi_tilde", "cos_theta_prime_tilde", "phi_prime_tilde"]

BEAM = np.array([0.0, 0.0, 1.0])


def as_array(v):
    """Four-vector (vector/awkward Momentum4D, possibly with None) -> (n, 4) [px, py, pz, E]."""
    cols = []
    for c in ("px", "py", "pz", "E"):
        x = getattr(v, c)
        x = ak.to_numpy(ak.fill_none(x, np.nan)) if isinstance(x, ak.Array) else np.asarray(x)
        cols.append(x.astype(float))
    return np.stack(cols, axis=-1)


def unit(v):
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def dot(a, b):
    return np.sum(a * b, axis=-1)


def boost_to_rest(p, frame):
    """Boost four-vectors p (n, 4) into the rest frame of four-vectors frame (n, 4)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        b = frame[:, :3] / frame[:, 3:4]
        b2 = dot(b, b)
        gamma = 1.0 / np.sqrt(1.0 - b2)
        bp = dot(b, p[:, :3])
        coef = np.where(b2 > 0, (gamma - 1.0) / b2, 0.0)
        p3 = p[:, :3] + (coef * bp - gamma * p[:, 3])[:, None] * b
        e = gamma * (p[:, 3] - bp)
    return np.concatenate([p3, e[:, None]], axis=1)


def decay_angles(top, w, fermion):
    """
    (cos θ, φ, cos θ', φ') for top -> W (-> fermion ...) b.
    All inputs are lab-frame (n, 4) arrays.
    """
    # top rest frame axes
    z = unit(top[:, :3])
    y = unit(np.cross(BEAM, z))
    x = np.cross(y, z)
    w_t = boost_to_rest(w, top)
    f_t = boost_to_rest(fermion, top)
    wdir = unit(w_t[:, :3])
    cos_theta = dot(wdir, z)
    phi = np.arctan2(dot(wdir, y), dot(wdir, x))

    # W rest frame axes (standard boost from the top rest frame along the W direction)
    zp = wdir
    yp = unit(np.cross(z, zp))
    xp = np.cross(yp, zp)
    f_w = unit(boost_to_rest(f_t, w_t)[:, :3])
    cos_theta_p = dot(f_w, zp)
    phi_p = np.arctan2(dot(f_w, yp), dot(f_w, xp))
    return cos_theta, phi, cos_theta_p, phi_p


def production_angles(t_lep, t_had):
    """(Y, cos Θ): tt̄ rapidity in the lab and leptonic-top polar angle in the tt̄ frame."""
    tt = t_lep + t_had
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = 0.5 * np.log((tt[:, 3] + tt[:, 2]) / (tt[:, 3] - tt[:, 2]))
    cos_Theta = dot(unit(boost_to_rest(t_lep, tt)[:, :3]), BEAM)
    return Y, cos_Theta


def assign_had_quarks(ev, parton=None, mode="truth"):
    """
    Decide which of the two W_had jets (ev.q1 = harder, ev.q2 = softer) is the
    down-type quark at gen/reco level; sets ev.q_down and ev.q_up.
      mode "truth":  match to the parton-level quarks (needs the aligned LHE event;
                     the paper assumes up/down tagging).  Falls back to "softer".
      mode "softer": the lower-pT jet is the down-type one
      mode "harder": the higher-pT jet is the down-type one
    """
    q1, q2 = as_array(ev.q1), as_array(ev.q2)
    if mode == "truth" and parton is not None:
        def dr(a, b):
            va, vb = _to_vec(a), _to_vec(b)
            return np.asarray(va.deltaR(vb), dtype=float)
        d, u = as_array(parton.q_down), as_array(parton.q_up)
        swap = dr(q1, d) + dr(q2, u) < dr(q1, u) + dr(q2, d)      # q1 is the down-type
    elif mode in ("truth", "softer"):
        swap = np.zeros(len(q1), bool)
    elif mode == "harder":
        swap = np.ones(len(q1), bool)
    else:
        raise ValueError(f"unknown jet assignment mode {mode}")
    down = np.where(swap[:, None], q1, q2)
    up = np.where(swap[:, None], q2, q1)
    ev.q_down, ev.q_up = _to_vec(down), _to_vec(up)
    return ev


def _to_vec(a):
    return vector.zip({"px": a[:, 0], "py": a[:, 1], "pz": a[:, 2], "E": a[:, 3]})


def add_angles(ev, had_fermion="down"):
    """
    Compute the ten angles for an Event that has t_lep, w_lep, lep, t_had, w_had,
    q_down and q_up.  had_fermion = "down" (paper default) or "up" selects which
    W_had daughter defines the primed hadronic angles.
    """
    t_lep, w_lep, lep = as_array(ev.t_lep), as_array(ev.w_lep), as_array(ev.lep)
    t_had, w_had = as_array(ev.t_had), as_array(ev.w_had)
    f_had = as_array({"down": ev.q_down, "up": ev.q_up}[had_fermion])

    ev.Y, ev.cos_Theta = production_angles(t_lep, t_had)
    ev.cos_theta, ev.phi, ev.cos_theta_prime, ev.phi_prime = decay_angles(t_lep, w_lep, lep)
    (ev.cos_theta_tilde, ev.phi_tilde,
     ev.cos_theta_prime_tilde, ev.phi_prime_tilde) = decay_angles(t_had, w_had, f_had)
    return ev
