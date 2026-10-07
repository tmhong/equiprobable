# Running log: tt̄ / hh → bb̄ ℓν qq̄′ samples, angles and OSDE

Chronological record of requests, decisions and results in this working thread.
Newest entries are at the bottom. Numbers refer to the samples listed under "Samples".

## Samples (13 TeV, MG5_aMC 3.6.7 → Pythia8 → Delphes 3.5.1 ATLAS card, no pileup)
| sample | events | seed | use |
|---|---|---|---|
| `samples/ttbar_bblvqq/run_seed1001` | 100k | 1001 | tt̄ fit (training) sample for OSDE |
| `samples/ttbar_bblvqq/run_seed12345` | 10k | 12345 | tt̄ validation plots; independent tt̄ sample for f_tt̄ |
| `samples/hh_bblvqq/run_seed2001` | 20k | 2001 | hh (heft model) |

---

### 2026-10-07: setup and sample production
- **Request:** MG5 samples of tt̄ → bbWW → bbℓνqq at generator level, plus Delphes with the ATLAS card and zero pileup; Python scripts and easy-to-edit plotting; everything checked before hand-off.
- **Installed in `tools/` via `setup_tools.sh`:** MG5_aMC 3.6.7, Pythia8 and Delphes 3.5.1.
  - MG5's own Delphes build fails to link with ROOT 6.36, so Delphes is built with CMake instead.
  - MG5's version check crashes on exit, so runs are launched through `<proc>/bin/madevent`.
- **Process:** decay chains with spin correlations and no generator cuts. σ = 136.5 pb (LO).
- **Delphes output:** the Track/Tower/EFlow branches are dropped (−30% size). The remaining ~70 kB/event is mostly the full Particle record, which is kept because gen-level lepton finding needs it.
- **Plotting:** `make_plots.py` plus the editable `plot_config.py`, with parton/gen/reco overlays and a χ² tt̄ reconstruction.
- **Bugs found while validating and fixed:**
  - Prompt leptons were also being clustered as GenJets; they are now removed by overlap removal.
  - The Delphes slimming regex was wrong.
  - HepMC files were piling up in `work/`.
  - Plot names differing only in case collided on macOS.
- **Checks:**
  - LHE ↔ Delphes events are aligned 100%.
  - Truth-matched reco jets give m_t = 172 GeV and m_W = 82 GeV.
  - The broad χ² peaks come from wrong jet assignments (combinatorics).
  - Large lepton shifts between parton and gen level are real physics: Pythia's ISR recoil gives a median tt̄ p_T of 47 GeV.

### Angles (paper `main.tex`)
- **Implemented in `plotting/angles.py`:** Y, Θ, (θ,φ), (θ′,φ′) for the leptonic top, and the tilded angles for the hadronic top (with the down-type quark).
  - Top frame: z = top lab direction, y = beam × z, x = y × z.
  - W frame: z′ = Ŵ, y′ = z × z′, x′ = y′ × z′.
- **ν p_z:** both m_W solutions enter the χ², so the m_t term resolves the ambiguity, as in the paper.
- **Gen/reco down-type jet:** truth-matched to the LHE quarks, i.e. up/down tagging is assumed, as in the paper.
- **Checks:**
  - A synthetic round trip recovers the angles to 1e-14.
  - Parton-level W helicity fractions: F₀ = 0.715 ± 0.013 (lepton) and 0.705 ± 0.013 (down-type quark), vs SM 0.70.
- **ν p_z resolution (68% half-width):** 54 GeV (gen), 72 GeV (reco).

### Slides
- `make_slides.py` builds a Beamer deck (`slides/slides.pdf`) in which every number is recomputed from the samples.

### hh samples
- **Process:** `--process hh` with the heft model: `g g > h h, (h > b b~), (h > lep+ nul q q)` plus the c.c. (four-body h decay, since m_h < 2m_W).
- **Cross section:** σ = 0.11 fb. This is LO HEFT with MG defaults, **not** an SM prediction, so hh plots use unit area.
- **Fixes:**
  - Prompt leptons may now come straight from the h, which the old W/τ-only rule missed for half the hh events.
  - τ-descended leptons now require the τ to come from a boson; B/D → τ decays were giving fake extra leptons in 3% of events.
- **Parton-level hh:** the b pairing is the one closest to the tt̄ hypothesis.

### OSDE: f_tt̄ (`plotting/osde.py`, `osde_fit.py`, `tests/`)
- **Basis as in the paper:**
  - M^{j₁j₂}_{m′m} with j₁ ≤ 1, j₂ ≤ 2: 24 per top.
  - Normalized Legendre P_l(cos Θ), l < 5.
  - Normalized Hermite functions in Y (the weight factor e^{−Y²/2} is included; confirmed by the user).
  - Total 14,400 coefficients; ĉ = mean(B*) with standard errors.
- **Tests (7, pytest):**
  - Wigner d vs sympy.
  - Exact orthonormality.
  - Conjugation closure (f is real).
  - Toy with known coefficients: unit-width pulls.
- **Results** (fit on 100k tt̄; applied to the independent 10k tt̄ and 20k hh):
  - AUC(hh vs tt̄) = 0.65 (parton), 0.53 (gen), 0.51 (reco).
  - No single angle separates them at reco (AUC ≤ 0.53). An f_tt̄-only score can't flag hh where the tt̄ density is flat, and the tt̄-hypothesis χ² makes hh look tt̄-like.
- **Truncation of the expansions:** Y and cos Θ are imperfectly described with 5 terms. Hermite scale s = std(Y) fits Y 3–9× better than the paper's s = 1, which is kept as the default (`--y-scale`).

### Acceptance (`acceptance_plots.py`)
- Uses parton-level angle values before and after the gen/reco selections, so shape changes are acceptance only.
- **tt̄ effects:**
  - cos θ′ → −1 and φ′ ≈ 0: the lepton goes against the top's flight direction and fails p_T > 25 GeV.
  - |Y| > 2: η acceptance.
  - cos θ, cos θ̃ → +1: the b quark is soft.
- **hh:** the χ² turns its flat parton cos Θ into tt̄-like peaks. The reconstruction, more than the acceptance, makes hh look like tt̄.

### Train vs test differences in y = f(x) (user question)
- **Cause:** self-inclusion. Each fit event contributes (1/N) Σ_j |B_j(x_k)|² to its own f; that is 7%, 10% and 16% of f at parton, gen and reco.
- **Proof:** leave-one-out f on the fit sample matches the independent sample (KS 0.010–0.012 vs 0.04–0.10).
- `osde_fit.py` now uses leave-one-out for the fit sample.

### How many coefficients matter? (`osde_simplify.py`)
- **Criterion:** keep a term if |ĉ|/σ > t; the constant is always kept, and conjugate pairs are kept together (checked).
- **Choosing t:**
  - Kronmal–Tarter (t = √2) is MISE-optimal per term, but lets in ~16% of null terms.
  - 5-fold least-squares cross-validation picks t = 2.5: 326 (parton), 459 (gen), 342 (reco) terms. The curve is flat out to 5σ, which keeps 54, 153 and 127 terms.
- **Effect:** truncation lowers the f ≤ 0 fraction (parton 1.6% → 0.2%) and leaves the hh AUC unchanged.
- **Terms above 5σ:**
  - Parton: production, W helicity, and 22 tt̄ spin-correlation terms (~7σ); all imaginary parts are consistent with 0.
  - Reco: no spin correlations; acceptance-induced "polarization" terms instead.

### Questions answered
- **"test":** the coefficients come from the tt̄ fit sample, and f is applied to a separately generated tt̄ sample, which is "test" (now labelled "independent sample").
- **ROC:** every point is one y_cut, so the curve is a scan over cuts; the axis labels now say so.
- **GitLab:** fine. A local repo will be created; pushing needs the project URL and the user's auth.

### Δp_z (`nu_pz_plot.py`)
- **Resolution (68% half-width):** gen 54 GeV, reco 72 GeV; reco events with all jets correctly assigned: 49 GeV.
- **No real m_W solution:** 32% of reco events.
- **Choosing between two real solutions:** the top-mass χ² picks the one closer to the truth in 63% of events, vs 65% for the smaller-|p_z| rule. So the top-mass choice is not better here; the solution choice is coupled to the jet choice in the χ².

### Angle diagram (`figures/make_angle_diagram.py`)
- Panels (a) and (b) were dropped at the user's request.
- **Layout:** two panels, the top rest frame and the W rest frame (z to the right, x up, y out of the page), joined by a "boost along p̂_W" arrow.
- **Shading:** the x–z plane (the beam–top plane) and the xy-plane where φ is measured.
- **Right-angle markers** in both frames.

### Slides
- **Style:** University of Pittsburgh colors (Royal #003594, Gold #FFB81C) in a custom Beamer theme, with an outline and section slides.
- **Policy:** all results go in the slides (user request), with numbers recomputed by the scripts.
- **hh slide:** shows hh only (no tt̄) in a single 1×3 overview figure.
- **Hadronic W mass in hh, split by case:** the on-shell W is the hadronic one in 50% of events. When it's the W* (true median 32 GeV), the χ² still reconstructs it near m_W (median 95 / 89 GeV at gen / reco). This is the mechanism that makes hh look like tt̄.
- **OSDE per level, stated explicitly:** f^parton is fit on parton-level tt̄ and applied to parton-level samples; likewise gen→gen and reco→reco. The code always did this; the slides now say so.

### Git
- Local repository created (code, docs, diagram, slide PDF; no samples or tools). Pushing to gitlab.com needs the project URL and the user's authentication.
