#!/usr/bin/env python3
"""
Rung 1 of study_vae: a plain VAE trained on the ten tt̄ angles, with the paper's OSDE density f as the yardstick.

A VAE is a density estimator: per event, the evidence lower bound (ELBO) bounds the log density it has learned,
    ln p_VAE(x) >= ELBO(x) = E_{z~q(z|x)}[ln p(x|z)] - D_KL(q(z|x) || p(z)),
so -ELBO(x) is the VAE's estimate of the surprisal, and the importance-weighted bound (IWAE, K samples) tightens it.
Our f is the true density of the same ten angles (parton level after the trigger, up to the OSDE truncation), in the
same measure dx = d cos theta dphi (x4) d cos Theta dY, so if the VAE has learned f then  -ELBO(x) ~ -ln f(x)  event by
event, with no free constant.  The decoders are chosen so that p(x|z) is a density in exactly that measure: a truncated
normal on [-1, 1] for the five cosines, a normal for Y and a von Mises distribution on the circle for the four azimuths.
The prior is the standard VAE prior, p(z) = N(0, 1) in a 10-dimensional latent space; nothing of f enters the training.

Trained on 80% of the tt̄ fit sample (run_seed1001, parton level after the trigger), early-stopped on the other 20%;
scored on the independent tt̄ sample (run_seed12345) and on the HH(170) and hh(125) samples.  Reports, on the independent
tt̄ sample, the event-by-event comparison of -ELBO and IWAE with s = -ln f (correlation, slope, mean values, residuals
binned in every angle) and the AUCs of every score against the two signals.

Writes results/analysis/study_vae/vae_angles.{json,md,png} and vae_angles_scores.npz (per-event scores).
Run: OMP_NUM_THREADS=6 .venv/bin/python code/analysis/study_vae/vae_angles.py [--latent 10] [--epochs 100] [--seed 1]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CODE)
import repo_paths  # noqa: E402
from lib import osde  # noqa: E402
from lib.angles import ANGLE_NAMES  # noqa: E402
from analysis.study_vae.score import auc, auc_err, shift_y0  # noqa: E402

T_CUT = 2.5
LEVEL = "parton_trig"
COS = [a for a in ANGLE_NAMES if a.startswith("cos")]            # 5 cosines, truncated normal on [-1, 1]
NC, NP = len(COS), 4
PHI = [a for a in ANGLE_NAMES if a.startswith("phi")]            # 4 azimuths, von Mises
OUT = repo_paths.results_dir("study_vae")
SAMPLES = {"tt_fit": "tt_fit", "tt_test": "tt_indep", "HH170": "HH170_all", "hh125": "hh125"}


def load(key):
    z = np.load(repo_paths.results("study_gonogo", "cache", f"{key}.npz"))
    X = {a: z[f"{LEVEL}/{a}"].astype(np.float64) for a in ANGLE_NAMES}
    ok = np.all([np.isfinite(v) for v in X.values()], axis=0)
    return {a: v[ok] for a, v in X.items()}


def features(X):
    """Encoder inputs: Y, the five cosines, and (cos, sin) of the four azimuths (14 features)."""
    cols = [X["Y"]] + [X[a] for a in COS] + [np.cos(X[a]) for a in PHI] + [np.sin(X[a]) for a in PHI]
    return np.stack(cols, axis=1).astype(np.float32)


def targets(X):
    """Decoder targets in the measure of f: Y, the five cosines, the four azimuths (10 columns)."""
    return np.stack([X["Y"]] + [X[a] for a in COS] + [X[a] for a in PHI], axis=1).astype(np.float32)


class VAE(nn.Module):
    def __init__(self, latent=10, hidden=128, n_feat=14):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * latent))
        # decoder outputs: Y (mu, logvar), 5 cosines (mu, logvar), 4 azimuths (2-vector for the mean angle, log kappa)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 + 2 * NC + 3 * NP))

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar.clamp(-10, 6)

    def log_px_z(self, t, z):
        """ln p(x|z) for targets t (n, 10) and latent z (..., n, latent): a density in dY dcos^5 dphi^4."""
        o = self.dec(z)
        mu_y, lv_y = o[..., 0], o[..., 1].clamp(-10, 6)
        mu_c, lv_c = o[..., 2:2 + NC], o[..., 2 + NC:2 + 2 * NC].clamp(-10, 6)
        v_phi = o[..., 2 + 2 * NC:2 + 2 * NC + 2 * NP].reshape(*o.shape[:-1], NP, 2)
        logk = o[..., 2 + 2 * NC + 2 * NP:].clamp(-6, 6)
        y, c, phi = t[..., 0], t[..., 1:1 + NC], t[..., 1 + NC:]
        ll = -0.5 * ((y - mu_y) ** 2 * torch.exp(-lv_y) + lv_y + math.log(2 * math.pi))
        sd = torch.exp(0.5 * lv_c)
        norm = torch.special.ndtr((1 - mu_c) / sd) - torch.special.ndtr((-1 - mu_c) / sd)     # mass inside [-1, 1]
        ll = ll + (-0.5 * ((c - mu_c) ** 2 * torch.exp(-lv_c) + lv_c + math.log(2 * math.pi))
                   - torch.log(norm.clamp_min(1e-12))).sum(-1)
        mu_phi = torch.atan2(v_phi[..., 1], v_phi[..., 0])
        kappa = torch.exp(logk)
        ll = ll + (kappa * torch.cos(phi - mu_phi) - math.log(2 * math.pi)
                   - (torch.log(torch.special.i0e(kappa)) + kappa)).sum(-1)       # ln I0(k) = ln i0e(k) + k
        return ll

    def elbo_terms(self, x, t, n_samples=1, free_bits=0.0):
        mu, logvar = self.encode(x)
        kl_dim = 0.5 * (torch.exp(logvar) + mu ** 2 - 1 - logvar)
        kl = (kl_dim.clamp_min(free_bits) if free_bits > 0 else kl_dim).sum(-1)
        eps = torch.randn(n_samples, *mu.shape, device=x.device)
        z = mu + torch.exp(0.5 * logvar) * eps
        rec = self.log_px_z(t.expand(n_samples, *t.shape), z).mean(0)
        return rec, kl                                            # ELBO = rec - kl

    @torch.no_grad()
    def iwae(self, x, t, k=64):
        """Importance-weighted estimate of ln p(x): log-mean-exp over k samples of ln p(x|z) + ln p(z) - ln q(z|x)."""
        mu, logvar = self.encode(x)
        eps = torch.randn(k, *mu.shape, device=x.device)
        z = mu + torch.exp(0.5 * logvar) * eps
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(-1)
        log_qz = -0.5 * (eps ** 2 + logvar + math.log(2 * math.pi)).sum(-1)
        lw = self.log_px_z(t.expand(k, *t.shape), z) + log_pz - log_qz
        return torch.logsumexp(lw, 0) - math.log(k)


def train(model, Xtr, Ttr, Xva, Tva, epochs, lr, bs, patience, seed, warmup=0, free_bits=0.0):
    """Maximizes the ELBO.  warmup > 0: the KL term is weighted by min(1, (epoch + 1) / warmup) during training (KL
    annealing); free_bits > 0: each latent dimension's KL counts only above free_bits nats (both against posterior
    collapse).  Validation always uses the plain ELBO."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, best_state, bad, hist = np.inf, None, 0, []
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        beta = min(1.0, (ep + 1) / warmup) if warmup > 0 else 1.0
        for lo in range(0, n, bs):
            idx = perm[lo:lo + bs]
            rec, kl = model.elbo_terms(Xtr[idx], Ttr[idx], free_bits=free_bits)
            loss = (beta * kl - rec).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            rec, kl = model.elbo_terms(Xva, Tva, n_samples=4)
            va = (kl - rec).mean().item()
        sched.step(va)
        hist.append((tot / n, va))
        if va < best - 1e-4:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if ep % 10 == 0 or bad == 0 and ep < 5:
            print(f"  epoch {ep:3d}  train -ELBO {tot / n:.4f}  val -ELBO {va:.4f}", flush=True)
    model.load_state_dict(best_state)
    return hist, best


def scores(model, X, n_samples=32, k=64):
    x, t = torch.tensor(features(X)), torch.tensor(targets(X))
    model.eval()
    out = {"neg_elbo": [], "iwae": [], "kl": [], "rec": []}
    with torch.no_grad():
        for lo in range(0, len(x), 4096):
            xb, tb = x[lo:lo + 4096], t[lo:lo + 4096]
            rec, kl = model.elbo_terms(xb, tb, n_samples=n_samples)
            out["neg_elbo"].append((kl - rec).numpy()); out["kl"].append(kl.numpy()); out["rec"].append((-rec).numpy())
            out["iwae"].append((-model.iwae(xb, tb, k=k)).numpy())
    return {k_: np.concatenate(v).astype(np.float64) for k_, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0, help="epochs of KL annealing (0: none)")
    p.add_argument("--free-bits", type=float, default=0.0, help="free bits per latent dimension (nats)")
    p.add_argument("--tag", default="", help="suffix for the output files")
    p.add_argument("--flow", action="store_true", help="also train the paper's RealNVP flow on the same split as a second yardstick")
    a = p.parse_args()
    tag = f"_{a.tag}" if a.tag else ""
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    t0 = time.time()
    data = {k: load(v) for k, v in SAMPLES.items()}

    # the yardstick: the paper's density, truncated at 2.5 sigma, and its shift y0 on the fit sample
    f_all = osde.load_density(repo_paths.results("osde", f"f_ttbar_{LEVEL}.npz"))
    f = f_all.truncated(f_all.significance() > T_CUT)
    y0 = shift_y0(f(data["tt_fit"]))
    fvals = {k: f(X) for k, X in data.items()}

    # the VAE: trained on 80% of the fit sample, early-stopped on 20%
    rng = np.random.default_rng(a.seed)
    n = len(data["tt_fit"]["Y"])
    perm = rng.permutation(n); ntr = int(0.8 * n)
    Xf, Tf = features(data["tt_fit"]), targets(data["tt_fit"])
    Xtr, Ttr = torch.tensor(Xf[perm[:ntr]]), torch.tensor(Tf[perm[:ntr]])
    Xva, Tva = torch.tensor(Xf[perm[ntr:]]), torch.tensor(Tf[perm[ntr:]])
    model = VAE(a.latent, a.hidden)
    hist, best_val = train(model, Xtr, Ttr, Xva, Tva, a.epochs, a.lr, 512, 10, a.seed, a.warmup, a.free_bits)
    print(f"trained {len(hist)} epochs, best val -ELBO {best_val:.4f}, {time.time() - t0:.0f} s", flush=True)

    S = {k: scores(model, X) for k, X in data.items()}
    for k in S:
        S[k]["s_f"] = -np.log(np.where(fvals[k] > 0, fvals[k], np.nan))
        S[k]["s_f_shift"] = -np.log(np.maximum(fvals[k] + y0, y0 / 100))
    vae_scores = ["neg_elbo", "iwae", "kl", "rec"]
    if a.flow:                                   # the paper's flow (ml_compare.RealNVP) on the same ten variables, same split
        from analysis import ml_compare
        mu_, sd_ = Tf[perm[:ntr]].mean(0), Tf[perm[:ntr]].std(0)
        flow = ml_compare.train_flow((Tf[perm[:ntr]] - mu_) / sd_, (Tf[perm[ntr:]] - mu_) / sd_, epochs=60)
        flow.eval()
        with torch.no_grad():
            for k, X in data.items():            # density in the measure of f: the Jacobian of the standardization
                xs = torch.as_tensor((targets(X) - mu_) / sd_, dtype=torch.float32)
                S[k]["s_flow"] = -np.concatenate([flow.log_prob(xs[lo:lo + 8192]).numpy() for lo in range(0, len(xs), 8192)]).astype(np.float64) \
                    + float(np.log(sd_).sum())
        vae_scores.append("s_flow")
        print(f"flow trained; mean -ln p_flow on tt_test {S['tt_test']['s_flow'].mean():.3f}", flush=True)
    np.savez_compressed(os.path.join(OUT, f"vae_angles{tag}_scores.npz"), **{f"{k}/{s}": v for k, d in S.items() for s, v in d.items()})

    # event-by-event comparison on the independent tt̄ sample
    T = S["tt_test"]; X = data["tt_test"]
    pos = np.isfinite(T["s_f"])
    cmp = {"n": int(len(pos)), "frac_f_positive": float(pos.mean()), "y0": y0,
           "mean": {k: float(np.mean(T[k][pos])) for k in ["s_f"] + vae_scores}}
    for k in [v for v in vae_scores if v in ("neg_elbo", "iwae", "s_flow")]:
        a_, b_ = np.polyfit(T["s_f"][pos], T[k][pos], 1)
        r = np.corrcoef(T["s_f"][pos], T[k][pos])[0, 1]
        from scipy.stats import spearmanr
        rho = spearmanr(T["s_f"][pos], T[k][pos]).correlation
        resid = T[k][pos] - T["s_f"][pos]
        cmp[k] = {"pearson": float(r), "spearman": float(rho), "slope": float(a_), "intercept": float(b_),
                  "resid_mean": float(resid.mean()), "resid_std": float(resid.std()),
                  "resid_by_angle": {}}
        for ang in ANGLE_NAMES:                                   # mean residual in 8 bins of each variable
            v = X[ang][pos]
            edges = np.quantile(v, np.linspace(0, 1, 9))
            b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 7)
            cmp[k]["resid_by_angle"][ang] = [float(resid[b == i].mean()) for i in range(8)]
    # AUCs against the two signals
    aucs = {}
    for sname in ["s_f_shift"] + vae_scores:
        for sig in ("HH170", "hh125"):
            b_, s_ = S["tt_test"][sname], S[sig][sname]
            aucs[f"{sname}/{sig}"] = [round(auc(b_, s_), 4), round(auc_err(b_, s_), 4)]
    res = {"config": vars(a), "n_events": {k: int(len(X["Y"])) for k, X in data.items()}, "epochs_run": len(hist),
           "best_val_neg_elbo": best_val, "history": hist, "comparison_tt_test": cmp, "auc": aucs,
           "kept_terms": int((f.c != 0).sum())}
    json.dump(res, open(os.path.join(OUT, f"vae_angles{tag}.json"), "w"), indent=1)

    # figure: -ELBO vs s on the independent tt̄ sample, and the residual in two angles
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].hexbin(T["s_f"][pos], T["neg_elbo"][pos], gridsize=60, bins="log", cmap="Blues")
    lo_, hi_ = np.percentile(T["s_f"][pos], [0.5, 99.5])
    ax[0].plot([lo_, hi_], [lo_, hi_], "r--", lw=1, label="−ELBO = −ln f")
    ax[0].set_xlabel("s = −ln f  (OSDE)"); ax[0].set_ylabel("−ELBO  (VAE)"); ax[0].legend(loc="upper left")
    ax[0].set_title(f"tt̄ test: r = {cmp['neg_elbo']['pearson']:.3f}, slope {cmp['neg_elbo']['slope']:.2f}")
    for i, ang in enumerate(("cos_theta_prime", "Y")):
        v = X[ang][pos]; resid = T["neg_elbo"][pos] - T["s_f"][pos]
        ax[1 + i].hexbin(v, resid, gridsize=60, bins="log", cmap="Blues")
        ax[1 + i].axhline(0, color="r", ls="--", lw=1)
        ax[1 + i].set_xlabel(ang); ax[1 + i].set_ylabel("−ELBO − s")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"vae_angles{tag}.png"), dpi=130)

    lines = [f"# study_vae rung 1: plain VAE on the ten angles vs the OSDE density (parton level after trigger)", "",
             f"VAE: latent {a.latent}, hidden {a.hidden}, warmup {a.warmup}, free bits {a.free_bits}, {len(hist)} epochs, "
             f"best validation −ELBO {best_val:.3f}; "
             f"f: {res['kept_terms']} terms, y0 = {y0:.2g}; events: {res['n_events']}", "",
             "## Event by event on the independent tt̄ sample (events with f > 0: "
             f"{100 * cmp['frac_f_positive']:.1f}%)", "",
             "| quantity | mean |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in cmp["mean"].items()] + [
             "", "| VAE score | Pearson r | Spearman | slope | intercept | residual mean ± std |", "|---|---|---|---|---|---|"] + [
             f"| {k} | {cmp[k]['pearson']:.3f} | {cmp[k]['spearman']:.3f} | {cmp[k]['slope']:.3f} | {cmp[k]['intercept']:.2f} | "
             f"{cmp[k]['resid_mean']:.3f} ± {cmp[k]['resid_std']:.3f} |" for k in cmp if isinstance(cmp[k], dict) and "pearson" in cmp[k]] + [
             "", "Mean residual (−ELBO − s) in 8 quantile bins of each variable:", ""] + [
             f"- {ang}: {[round(v, 2) for v in cmp['neg_elbo']['resid_by_angle'][ang]]}" for ang in ANGLE_NAMES] + [
             "", "## AUC against the independent tt̄ sample", "", "| score | HH170 | hh125 |", "|---|---|---|"] + [
             f"| {s} | {aucs[f'{s}/HH170'][0]:.3f} ± {aucs[f'{s}/HH170'][1]:.3f} | {aucs[f'{s}/hh125'][0]:.3f} ± {aucs[f'{s}/hh125'][1]:.3f} |"
             for s in ["s_f_shift"] + vae_scores]
    open(os.path.join(OUT, f"vae_angles{tag}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
