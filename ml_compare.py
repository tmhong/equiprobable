#!/usr/bin/env python3
"""
Is the OSDE density f_tt̄ helping?  tt̄ vs hh at parton level (or parton+trig).

 1. As an anomaly detector: f_tt̄ vs other UNSUPERVISED detectors trained only on tt̄ with the
    same angles (Gaussian-mixture density, autoencoder).
 2. Supervised upper bound: classifiers trained on tt̄ vs hh with (a) our ten angles and
    (b) our angles + the parton momenta.
 3. Does y = f_tt̄(x) help ML?  The same classifiers with y (and s' = -ln(y - y0)) added as
    inputs.  For tt̄ fit-sample events y is the leave-one-out value (no self-inclusion bias).

Inputs
  A0  the ten angles x = {Y, Θ, θ, φ, θ', φ', θ~, φ~, θ~', φ~'} exactly as used by f_tt̄
  A1  the same information with each azimuth encoded as (cos φ, sin φ) (periodic)
  B   lab-frame four-vectors of the six final-state partons (ℓ, ν, b, b̄, q1, q2): contains
      more information than the angles (e.g. invariant masses)
Models
  f_tt̄ (paper basis, all terms and CV-truncated): unsupervised, never sees hh
  BDT (gradient boosting), MLP, FT-Transformer (one token per feature) on A0/A1/B,
  particle transformer (one token per parton, pairwise ln m_ij and ΔR_ij attention bias) on B
Test set (identical for every method): the independent tt̄ sample + a held-out 40% of hh.
Checks: shuffled labels and tt̄-vs-tt̄ must give AUC ≈ 0.5; train vs test AUC (overtraining).

Output: ml/roc_<level>.png, ml/results_<level>.json
"""
import argparse
import json
import os
import runpy
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_plots  # noqa: E402
from plotting import osde, style  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402

torch.set_num_threads(8)
SEED = 20261007


# ----------------------------------------------------------------------------- inputs
def angle_features(X, periodic):
    cols = []
    for a in ANGLE_NAMES:
        v = np.asarray(X[a], float)
        if periodic and a.startswith("phi"):
            cols += [np.cos(v), np.sin(v)]
        else:
            cols.append(v)
    return np.stack(cols, axis=1)


PARTONS = ["lep", "nu", "b", "bbar", "q1", "q2"]


def parton_tokens(ev):
    """(N, 6, 4) [pT, eta, phi, ln E] and (N, 6, 4) [px, py, pz, E] of the six partons."""
    objs = {"lep": ev.lep, "nu": ev.nu, "b": ev.b, "bbar": ev.bbar, "q1": ev.quarks[:, 0], "q2": ev.quarks[:, 1]}
    feat, p4 = [], []
    for k in PARTONS:
        o = objs[k]
        feat.append(np.stack([np.log(np.asarray(o.pt, float)), np.asarray(o.eta, float), np.asarray(o.phi, float),
                              np.log(np.asarray(o.E, float))], axis=1))
        p4.append(np.stack([np.asarray(getattr(o, c), float) for c in ("px", "py", "pz", "E")], axis=1))
    return np.stack(feat, axis=1), np.stack(p4, axis=1)


def pair_features(p4, eta, phi):
    """(N, 6, 6, 2) pairwise ln m_ij^2 and ln ΔR_ij (ParT-style interaction features)."""
    s = p4[:, :, None, :] + p4[:, None, :, :]
    m2 = np.clip(s[..., 3] ** 2 - (s[..., :3] ** 2).sum(-1), 1e-3, None)
    dphi = (phi[:, :, None] - phi[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    dr = np.sqrt((eta[:, :, None] - eta[:, None, :]) ** 2 + dphi ** 2)
    return np.stack([np.log(m2), np.log(np.clip(dr, 1e-3, None))], axis=-1)


# ----------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, nin, width=128, depth=3, drop=0.1):
        super().__init__()
        layers, d = [], nin
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(drop)]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FTTransformer(nn.Module):
    """Feature tokenizer + transformer: each scalar feature becomes a token (value x w_f + b_f)."""
    def __init__(self, nfeat, d=48, heads=4, layers=3, drop=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(nfeat, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(nfeat, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        tok = x[:, :, None] * self.w + self.b
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1)
        return self.head(self.enc(tok)[:, 0]).squeeze(-1)


class ParticleTransformer(nn.Module):
    """One token per parton (kinematics + type embedding); attention biased by an MLP of the
    pairwise features (ln m_ij^2, ln ΔR_ij), as in the Particle Transformer."""
    def __init__(self, nfeat=4, npart=6, d=48, heads=4, layers=3, drop=0.1):
        super().__init__()
        self.heads = heads
        self.embed = nn.Sequential(nn.Linear(nfeat, d), nn.GELU(), nn.Linear(d, d))
        self.ptype = nn.Parameter(torch.randn(npart, d) * 0.1)
        self.pair = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, heads * layers))
        self.layers = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
                                     for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                  for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, pair):
        n, p, _ = x.shape
        h = self.embed(x) + self.ptype
        bias = self.pair(pair)                                    # (n, p, p, heads * layers)
        for i, (att, ffn, nrm) in enumerate(zip(self.layers, self.ffn, self.norm)):
            b = bias[..., i * self.heads:(i + 1) * self.heads].permute(0, 3, 1, 2).reshape(n * self.heads, p, p)
            a, _ = att(nrm(h), nrm(h), nrm(h), attn_mask=b)
            h = h + a
            h = h + ffn(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


def train_torch(model, tr, va, epochs=40, lr=2e-3, bs=512, patience=6):
    """Class-balanced BCE, AdamW, cosine schedule, early stopping on validation loss."""
    torch.manual_seed(SEED)
    *xtr, ytr = [torch.as_tensor(a, dtype=torch.float32) for a in tr]
    *xva, yva = [torch.as_tensor(a, dtype=torch.float32) for a in va]
    pos_w = (ytr == 0).sum() / (ytr == 1).sum()
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(*[x[idx] for x in xtr]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(predict_logits(model, xva), yva))
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict_logits(model, xs, bs=4096):
    model.eval()
    with torch.no_grad():
        n = len(xs[0])
        return torch.cat([model(*[torch.as_tensor(x[i:i + bs], dtype=torch.float32) for x in xs])
                          for i in range(0, n, bs)])


def predict(model, xs):
    return predict_logits(model, xs).numpy()


def standardize(train, *others):
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return [(a - mu) / sd for a in (train,) + others]


# ----------------------------------------------------------------------------- main
def bootstrap_auc(y, s, n=200):
    rng = np.random.default_rng(7)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    aucs = [roc_auc_score(np.r_[np.zeros(len(i0)), np.ones(len(i1))],
                          np.r_[s[rng.choice(i0, len(i0))], s[rng.choice(i1, len(i1))]]) for _ in range(n)]
    return float(np.std(aucs))


class AutoEncoder(nn.Module):
    def __init__(self, nin, hidden=(64, 32), latent=4):
        super().__init__()
        enc, d = [], nin
        for h in hidden:
            enc += [nn.Linear(d, h), nn.GELU()]
            d = h
        self.enc = nn.Sequential(*enc, nn.Linear(d, latent))
        dec, d = [], latent
        for h in hidden[::-1]:
            dec += [nn.Linear(d, h), nn.GELU()]
            d = h
        self.dec = nn.Sequential(*dec, nn.Linear(d, nin))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_autoencoder(x_tr, x_va, epochs=40):
    torch.manual_seed(SEED)
    model = AutoEncoder(x_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    xt, xv = torch.as_tensor(x_tr, dtype=torch.float32), torch.as_tensor(x_va, dtype=torch.float32)
    best, state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(perm), 512):
            b = xt[perm[i:i + 512]]
            opt.zero_grad()
            loss = ((model(b) - b) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(((model(xv) - xv) ** 2).mean())
        if vl < best - 1e-5:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(state)
    return model


class RealNVP(nn.Module):
    """Normalizing flow (affine coupling layers, alternating masks) -> exact log-density."""
    def __init__(self, dim, layers=8, hidden=128):
        super().__init__()
        self.masks = [torch.arange(dim) % 2 == (i % 2) for i in range(layers)]
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                                 nn.Linear(hidden, 2 * dim)) for _ in range(layers)])

    def log_prob(self, x):
        logdet = torch.zeros(len(x))
        for m, net in zip(self.masks, self.nets):
            m = m.float()
            st = net(x * m)
            s_, t_ = st.chunk(2, dim=1)
            s_ = torch.tanh(s_) * (1 - m)
            t_ = t_ * (1 - m)
            x = x * m + (1 - m) * (x * torch.exp(s_) + t_)
            logdet = logdet + s_.sum(1)
        return -0.5 * (x ** 2).sum(1) - 0.5 * x.shape[1] * np.log(2 * np.pi) + logdet


def train_flow(x_tr, x_va, epochs=40):
    torch.manual_seed(SEED)
    flow = RealNVP(x_tr.shape[1])
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    xt, xv = torch.as_tensor(x_tr, dtype=torch.float32), torch.as_tensor(x_va, dtype=torch.float32)
    best, state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        flow.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(perm), 512):
            opt.zero_grad()
            loss = -flow.log_prob(xt[perm[i:i + 512]]).mean()
            loss.backward()
            opt.step()
        flow.eval()
        with torch.no_grad():
            vl = float(-flow.log_prob(xv).mean())
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in flow.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    flow.load_state_dict(state)
    return flow


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="samples/ttbar_bblvqq/run_seed1001")
    p.add_argument("--test", default="samples/ttbar_bblvqq/run_seed12345")
    p.add_argument("--hh", default="samples/hh_bblvqq/run_seed2001")
    p.add_argument("--level", default="parton", choices=["parton", "parton_trig"])
    p.add_argument("--osde", default="osde", help="directory with f_ttbar_<level>.npz (paper basis)")
    p.add_argument("--out", default="ml")
    p.add_argument("--config", default=os.path.join(HERE, "plot_config.py"))
    p.add_argument("--quick", action="store_true", help="few epochs (testing only)")
    p.add_argument("--replot", action="store_true", help="only redraw the ROC figures from saved curves")
    args = p.parse_args()
    if args.replot:
        replot(args.level, args.out)
        return
    os.makedirs(args.out, exist_ok=True)
    cfg = runpy.run_path(args.config)
    opts = {"had_fermion": "down", "jet_assignment": "truth", **cfg.get("ANGLES", {})}
    epochs = 4 if args.quick else 40

    ev = {}
    for name, path in (("tt", args.train), ("tt_indep", args.test), ("hh", args.hh)):
        e, _ = make_plots.load_sample(name, path, [args.level], None, opts)
        ev[name] = e[args.level]
    rng = np.random.default_rng(SEED)
    n_hh, n_tt = len(ev["hh"]), len(ev["tt"])
    perm, tperm = rng.permutation(n_hh), rng.permutation(n_tt)
    hh_tr, hh_va, hh_te = perm[:int(0.5 * n_hh)], perm[int(0.5 * n_hh):int(0.6 * n_hh)], perm[int(0.6 * n_hh):]
    tt_tr, tt_va = tperm[:int(0.9 * n_tt)], tperm[int(0.9 * n_tt):]
    print(f"[ml] {args.level}: tt train {len(tt_tr)} val {len(tt_va)}, test {len(ev['tt_indep'])} (independent); "
          f"hh train {len(hh_tr)} val {len(hh_va)} test {len(hh_te)}")

    def angles_of(e):
        return {a: np.asarray(e[a], float) for a in ANGLE_NAMES}

    # ---- y = f_tt(x): leave-one-out on the tt fit sample, plain evaluation elsewhere
    f = osde.load_density(os.path.join(args.osde, f"f_ttbar_{args.level}.npz"))
    yv = {"tt": f.loo(angles_of(ev["tt"])), "tt_indep": f(angles_of(ev["tt_indep"])), "hh": f(angles_of(ev["hh"]))}
    y0 = 2.5 * min(0.0, min(float(v.min()) for v in yv.values()))
    yfeat = {k: np.stack([v / np.std(yv["tt"]), -np.log(v - y0)], axis=1) for k, v in yv.items()}

    def momenta(e):
        feat, _ = parton_tokens(e)
        return feat.reshape(len(feat), -1)

    feats = {"angles": lambda k: angle_features(angles_of(ev[k]), True),
             "angles+y": lambda k: np.hstack([angle_features(angles_of(ev[k]), True), yfeat[k]]),
             "angles+momenta": lambda k: np.hstack([angle_features(angles_of(ev[k]), True), momenta(ev[k])]),
             "angles+momenta+y": lambda k: np.hstack([angle_features(angles_of(ev[k]), True), momenta(ev[k]),
                                                      yfeat[k]])}

    def split(fn):
        ft, fi, fh = fn("tt"), fn("tt_indep"), fn("hh")
        cat = lambda a, b: np.concatenate([a, b])                     # noqa: E731
        lab = lambda a, b: np.concatenate([np.zeros(len(a)), np.ones(len(b))])  # noqa: E731
        return ((cat(ft[tt_tr], fh[hh_tr]), lab(tt_tr, hh_tr)), (cat(ft[tt_va], fh[hh_va]), lab(tt_va, hh_va)),
                (cat(fi, fh[hh_te]), lab(fi, hh_te)))

    res, rocs, test_scores = {}, {}, {}

    def record(group, name, s_test, y_test, s_train=None, y_train=None, t0=None):
        auc = roc_auc_score(y_test, s_test)
        if group != "control":
            test_scores[name] = np.asarray(s_test, float)
        fpr, tpr, _ = roc_curve(y_test, s_test)
        rocs[name] = (group, fpr, tpr, auc)
        r = {"group": group, "auc_test": float(auc), "auc_err": bootstrap_auc(y_test, s_test),
             "tt_rej_at_hh50": float(1 / max(np.interp(0.5, tpr, fpr), 1e-6))}
        if s_train is not None:
            r["auc_train"] = float(roc_auc_score(y_train, s_train))
        if t0:
            r["seconds"] = round(time.time() - t0, 1)
        res[name] = r
        print(f"[ml] {group:12s} {name:48s} AUC {auc:.4f} ± {r['auc_err']:.4f}"
              + (f"  (train {r['auc_train']:.4f})" if "auc_train" in r else ""), flush=True)

    y_te = np.concatenate([np.zeros(len(ev["tt_indep"])), np.ones(len(hh_te))])
    hh_te_angles = {k: v[hh_te] for k, v in angles_of(ev["hh"]).items()}

    # ---- 1. unsupervised (trained on tt only)
    for tag, fn in (("all terms", f"f_ttbar_{args.level}.npz"), ("CV-truncated", f"f_ttbar_{args.level}_simplified.npz")):
        path = os.path.join(args.osde, fn)
        if os.path.exists(path):
            g = osde.load_density(path)
            record("unsupervised", f"OSDE f_tt, paper basis ({tag})",
                   -np.concatenate([g(angles_of(ev["tt_indep"])), g(hh_te_angles)]), y_te)
    xa_tt = feats["angles"]("tt")
    xa_te = np.concatenate([feats["angles"]("tt_indep"), feats["angles"]("hh")[hh_te]])
    mu, sd = xa_tt[tt_tr].mean(0), xa_tt[tt_tr].std(0) + 1e-8
    t0 = time.time()
    gmm = GaussianMixture(n_components=32, covariance_type="full", random_state=SEED, max_iter=300)
    gmm.fit(((xa_tt[tt_tr] - mu) / sd)[:60000])
    record("unsupervised", "Gaussian mixture density (32 comp.)", -gmm.score_samples((xa_te - mu) / sd), y_te, t0=t0)
    t0 = time.time()
    ae = train_autoencoder((xa_tt[tt_tr] - mu) / sd, (xa_tt[tt_va] - mu) / sd, epochs=epochs)
    with torch.no_grad():
        xt = torch.as_tensor((xa_te - mu) / sd, dtype=torch.float32)
        err = ((ae(xt) - xt) ** 2).mean(1).numpy()
    record("unsupervised", "autoencoder (reconstruction error)", err, y_te, t0=t0)
    # normalizing flow on the raw ten angles (bounded, so no periodic embedding needed)
    xr_tt = angle_features(angles_of(ev["tt"]), False)
    xr_te = np.concatenate([angle_features(angles_of(ev["tt_indep"]), False),
                            angle_features(angles_of(ev["hh"]), False)[hh_te]])
    rmu, rsd = xr_tt[tt_tr].mean(0), xr_tt[tt_tr].std(0) + 1e-8
    t0 = time.time()
    flow = train_flow((xr_tt[tt_tr] - rmu) / rsd, (xr_tt[tt_va] - rmu) / rsd, epochs=epochs)
    with torch.no_grad():
        lp = flow.log_prob(torch.as_tensor((xr_te - rmu) / rsd, dtype=torch.float32)).numpy()
    record("unsupervised", "normalizing flow (RealNVP, 8 couplings)", -lp, y_te, t0=t0)

    # ---- 2./3. supervised, with and without y
    for setname, fn in feats.items():
        (xtr, ytr), (xva, yva), (xte, yte) = split(fn)
        t0 = time.time()
        bdt = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True,
                                             validation_fraction=0.1, random_state=SEED,
                                             class_weight="balanced").fit(xtr, ytr)
        record(setname, f"BDT [{setname}]", bdt.predict_proba(xte)[:, 1], yte, bdt.predict_proba(xtr)[:, 1], ytr, t0)
        xtr_s, xva_s, xte_s = standardize(xtr, xva, xte)
        t0 = time.time()
        mlp = train_torch(MLP(xtr.shape[1]), (xtr_s, ytr), (xva_s, yva), epochs=epochs)
        record(setname, f"MLP [{setname}]", predict(mlp, [xte_s]), yte, predict(mlp, [xtr_s]), ytr, t0)
        if "momenta" not in setname:      # with momenta the task is already saturated (AUC > 0.999); the
            t0 = time.time()               # particle transformer below is the attention model there
            ftt = train_torch(FTTransformer(xtr.shape[1]), (xtr_s, ytr), (xva_s, yva), epochs=epochs, lr=1e-3)
            record(setname, f"FT-Transformer [{setname}]", predict(ftt, [xte_s]), yte, predict(ftt, [xtr_s]), ytr, t0)

    # ---- particle transformer on the momenta (one token per parton, pairwise m, dR bias)
    def tok(k):
        feat, p4 = parton_tokens(ev[k])
        return feat, pair_features(p4, feat[..., 1], feat[..., 2])
    parts = {k: tok(k) for k in ("tt", "tt_indep", "hh")}

    def stack(sel_tt, key_tt, sel_hh):
        fe = np.concatenate([parts[key_tt][0][sel_tt], parts["hh"][0][sel_hh]])
        pr = np.concatenate([parts[key_tt][1][sel_tt], parts["hh"][1][sel_hh]])
        yy = np.concatenate([np.zeros(len(parts[key_tt][0][sel_tt])), np.ones(len(sel_hh))])
        return fe, pr, yy
    ftr, ptr, ytr = stack(tt_tr, "tt", hh_tr)
    fva, pva, yva = stack(tt_va, "tt", hh_va)
    fte, pte, yte = stack(slice(None), "tt_indep", hh_te)
    mu4, sd4 = ftr.reshape(-1, 4).mean(0), ftr.reshape(-1, 4).std(0) + 1e-8
    pmu, psd = ptr.reshape(-1, 2).mean(0), ptr.reshape(-1, 2).std(0) + 1e-8
    nf = lambda a: (a - mu4) / sd4       # noqa: E731
    npr = lambda a: (a - pmu) / psd      # noqa: E731
    t0 = time.time()
    part = train_torch(ParticleTransformer(), (nf(ftr), npr(ptr), ytr), (nf(fva), npr(pva), yva), epochs=epochs, lr=1e-3)
    sub = np.random.default_rng(3).choice(len(ytr), 30000, replace=False)     # both classes
    record("momenta", "Particle Transformer [momenta + pairwise m, dR]", predict(part, [nf(fte), npr(pte)]), yte,
           predict(part, [nf(ftr[sub]), npr(ptr[sub])]), ytr[sub], t0)

    # ---- controls
    (xtr, ytr), _, (xte, yte) = split(feats["angles"])
    chk = HistGradientBoostingClassifier(max_iter=200, random_state=SEED).fit(xtr, np.random.default_rng(1).permutation(ytr))
    record("control", "control: BDT, shuffled labels", chk.predict_proba(xte)[:, 1], yte)
    fa, fb = feats["angles"]("tt"), feats["angles"]("tt_indep")
    half = len(fb) // 2
    xa = np.concatenate([fa[tt_tr[:2 * half]], fb[:half]])
    ya = np.concatenate([np.zeros(2 * half), np.ones(half)])
    xb = np.concatenate([fa[tt_va[:len(fb) - half]], fb[half:]])
    yb = np.concatenate([np.zeros(len(fb) - half), np.ones(len(fb) - half)])
    chk2 = HistGradientBoostingClassifier(max_iter=200, random_state=SEED, class_weight="balanced").fit(xa, ya)
    record("control", "control: BDT, tt (fit) vs tt (independent)", chk2.predict_proba(xb)[:, 1], yb)

    # paired bootstrap of the key AUC differences (same test events -> correlated AUCs)
    paired = {}
    rngb = np.random.default_rng(11)
    i0, i1 = np.where(y_te == 0)[0], np.where(y_te == 1)[0]
    osde_cv = next((k for k in test_scores if k.startswith("OSDE") and "CV" in k), None)
    pairs = [(osde_cv, k) for k in test_scores if res[k]["group"] == "unsupervised" and not k.startswith("OSDE")]
    pairs += [(k.replace("[angles]", "[angles+y]"), k) for k in test_scores if k.endswith("[angles]")]
    pairs += [(k.replace("[angles+momenta]", "[angles+momenta+y]"), k) for k in test_scores if k.endswith("[angles+momenta]")]
    for a, b in pairs:
        if a not in test_scores or b not in test_scores:
            continue
        diffs = []
        for _ in range(200):
            idx = np.r_[rngb.choice(i0, len(i0)), rngb.choice(i1, len(i1))]
            yy = y_te[idx]
            diffs.append(roc_auc_score(yy, test_scores[a][idx]) - roc_auc_score(yy, test_scores[b][idx]))
        d = res[a]["auc_test"] - res[b]["auc_test"]
        paired[f"{a} - {b}"] = {"delta": float(d), "err": float(np.std(diffs))}
        print(f"[ml] paired: {a} - {b}: {d:+.4f} ± {np.std(diffs):.4f}", flush=True)
    with open(os.path.join(args.out, f"results_{args.level}.json"), "w") as fh:
        json.dump({"level": args.level, "n_test_tt": int(len(ev["tt_indep"])), "n_test_hh": int(len(hh_te)),
                   "results": res, "paired": paired}, fh, indent=1)

    np.savez(os.path.join(args.out, f"rocs_{args.level}.npz"),
             **{f"{i}|{g}|{n}|fpr": fpr for i, (n, (g, fpr, tpr, auc)) in enumerate(rocs.items())},
             **{f"{i}|{g}|{n}|tpr": tpr for i, (n, (g, fpr, tpr, auc)) in enumerate(rocs.items())})
    plot_rocs(rocs, args.level, args.out)


PANELS = [("a", "angles only: unsupervised ($t\\bar t$ only) vs supervised",
           lambda n, g: g in ("unsupervised", "control") or g == "angles"),
          ("b", "angles: does adding $y=f_{t\\bar t}$ help ML?", lambda n, g: g in ("angles", "angles+y")),
          ("c", "angles + momenta, with and without $y$",
           lambda n, g: g in ("angles+momenta", "angles+momenta+y", "momenta") or n.startswith("OSDE f_tt, paper basis (CV"))]
COLORS = {"OSDE": "k", "Gaussian": "#8ab17d", "autoencoder": "#6a4c93", "normalizing": "#c06c84", "BDT": "#e9a03b",
          "MLP": "#d1495b", "FT-Transformer": "#2a9d8f", "Particle": "#1f4e9c", "control": "0.6"}


def plot_rocs(rocs, level, out):
    """One legible figure per comparison: roc_<level>_{a,b,c}.png."""
    style.use()
    for key, title, sel in PANELS:
        fig, ax = plt.subplots(figsize=(10, 7.2))
        for name, (g, fpr, tpr, auc) in rocs.items():
            if not sel(name, g):
                continue
            col = next(c for k, c in COLORS.items() if name.startswith(k) or k in name)
            ls = "--" if ("+y" in g) else (":" if g == "control" else "-")
            lw = 3.0 if name.startswith("OSDE") else 2.0
            ax.plot(fpr, tpr, color=col, ls=ls, lw=lw, label=f"{name}: {auc:.3f}")
        ax.plot([0, 1], [0, 1], color="0.85", lw=1)
        ax.set_xlabel(r"$t\bar t$ efficiency (independent sample)")
        ax.set_ylabel(r"$hh$ efficiency")
        ax.set_title(f"{level.replace('_trig', '+trig')} level: {title}")
        ax.legend(fontsize=12, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"roc_{level}_{key}.png"), dpi=120)
        plt.close(fig)
    print(f"[ml] wrote {out}/roc_{level}_[abc].png")


def replot(level, out):
    z = np.load(os.path.join(out, f"rocs_{level}.npz"))
    res = json.load(open(os.path.join(out, f"results_{level}.json")))["results"]
    rocs = {}
    for k in sorted({k.rsplit("|", 1)[0] for k in z.files}, key=lambda k: int(k.split("|")[0])):
        _, g, n = k.split("|", 2)
        rocs[n] = (g, z[k + "|fpr"], z[k + "|tpr"], res[n]["auc_test"])
    plot_rocs(rocs, level, out)


if __name__ == "__main__":
    main()
