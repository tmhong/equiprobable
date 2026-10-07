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

### Score plots, ROC, t (paper style)
- **Score plots:**
  - New linear-y panel that shows negative f.
  - The log panel has an extra left bin for y ≤ 0 (fractions per bin).
  - Surprisal is now computed for the shifted y′ = y + y₀, with y₀ = 1.05 × |min y| per level, so every event is included.
- **Paper's Fig. "tt_hh_dNdy_dNdt"** reproduced in the paper's style by `paper_style_plots.py`: blue-gradient tt̄ fill, orange hh, and a red hh/(hh + tt̄) ratio with an error band.
  - y′ = y − y₀, s′ = −ln y′, t = (s′₁ − s′)/(s′₁ − s′₀), with s′₀ and s′₁ the min and max over tt̄ and hh.
  - **The shape of t depends strongly on y₀:** the most negative event sets s′₁. The AUC does not depend on y₀.
  - **Open question to the user:** which y₀ does the paper use? The text says "the minimum", which makes s′ infinite for the most negative event. The text and the figure caption also differ in sign convention.

### Object definitions (slide 7)
- Four levels: parton (MG5 LHE, **no cuts**), parton + lepton trigger (offline lepton cuts only; 71.5% of events), particle level, reco.
- **E_T^miss:**
  - parton: p_T of the ν;
  - gen: GenMissingET, the vector sum of all neutrinos;
  - reco: MissingET, −Σ over calorimeter towers (Delphes ATLAS card).
- **Check:** muon and electron events show the same reco − gen MET along the lepton (−5.9 vs −5.2 GeV), so muons are not missed.

### Why f^gen and f^reco missed the data, and the extended basis (user: "specify how many and do it")
- **Cause:** the paper basis is at most linear in cos θ (j₁ ≤ 1), quadratic in cos θ′ (j₂ ≤ 2), and first-harmonic in φ′ (|m| ≤ 1). That is exactly right at parton level, but the acceptance edges need higher orders.
- **Code check:** f's projection equals a direct 1-D expansion at the same order (≤ 1.9% difference).
- **1-D cross-validated order scan** (`osde_order_scan.py`):
  - Parton level confirms j₁ ≤ 1, j₂ ≤ 2; cos Θ needs 11 Legendre terms and Y 11 Hermite terms.
  - Gen needs j₁ ≥ 13, j₂ ≥ 12; reco needs j₁ ≥ 18, j₂ ≥ 24 (capped at 16).
  - A full product basis at those orders would need more than 10⁶ terms per top pair.
- **Extended block basis** (`ExtendedDensity`):
  - Contents: the paper's full product, plus high-order single-top blocks (all (θ, φ) terms and all m′ = 0 terms), plus high-order cos Θ and Y blocks.
  - Still orthonormal, because the blocks are disjoint subsets of the full product basis.
  - Hermite scale = std(Y).
  - Wigner d switched to a stable Jacobi-polynomial form, accurate to j = 24.
  - Sizes: 14,436 (parton), 32,494 (gen), 49,867 (reco) terms.
- **Tests:** 12 pass. New: high-j Wigner d, block orthonormality and disjointness, exact reduction to the paper basis, and exact leave-one-out vs brute-force refits.
- **Results after CV truncation** (333 / 679 / 323 terms kept):
  - Projection χ²/ndf ≈ 0.3–1.6 for every angle at every level; gen Y is still 39.
  - Negative f in about 1.4% of events.
  - AUC unchanged: 0.666 / 0.535 / 0.512.
  - Conclusion: describing tt̄ better does not restore the hh separation.

### Angle diagram (further user feedback)
- φ made explicit: the W is projected onto the xy-plane (W⊥, solid gold arrow), with a dashed drop line parallel to z and the φ arc from x to W⊥ inside a gold quarter-disc for the xy-plane. Same for ℓ⊥ and φ′.
- No text inside or across the shaded areas; colour swatches and plain-language definitions are in the legend.

### Paper figure as reference (user re-sent `tt_hh_dNdy_dNdt.pdf`)
- **y₀ inferred from the figure:** its y axis starts at about −1.5×10⁻⁵ (≈ 2.5 × min y), and t is populated continuously from 0, which requires y₀ well below min y. So y₀ = 2.5 × min y is now the default (`--y0-factor`). This still needs confirming by the user.
- **Style matched:** no title over the figure (the level is printed inside the left panel), tick labels without trailing zeros, the legend in the upper right, and the left axis starting at y₀.

### Leptonic vs hadronic decay angles (user question)
- **Expectation:** in the SM the down-type quark plays the role of the charged lepton, so cos θ′ and cos θ̃′ (and the other pairs) should agree.
- **Parton level (100k):** they agree for every pair; KS p-values are 0.24, 0.54, 0.84 and 0.27.
- **Gen and reco:** they differ (cos θ′ KS distance 0.056 at gen, 0.080 at reco). They still differ with perfect flavour tagging and correct jets (0.097), because the two sides have different acceptance and resolution. See `lep_had_compare.py` and `plots/lep_vs_had.png`.

### t fills [0, 1] (user request)
- **Problem:** with s′₀ and s′₁ at the strict min and max, single extreme events stretch the range and leave empty bins (up to 18 of 50 for hh). No choice of y₀ fixes this.
- **Fix:** s′₀ and s′₁ are now the 0.5% and 99.5% quantiles of the combined sample, with events beyond them in the first or last bin. Result: no empty bins for tt̄ or hh, at any level, with either basis (`--edge-quantile`).

### Spherical-harmonic basis (user request)
- **Basis:** `SHDensity`, with Y_{l₁}^{m₁}(θ,φ)·Y_{l₂}^{m₂}(θ′,φ′), l₁ ≤ 1, l₂ ≤ 2, per top: 36 functions (vs 24 M), 32,400 coefficients. Tests: orthonormality, conjugation closure, toy recovery.
- **Comparison** (`basis_compare.py`): held-out ISE = Σ|ĉ|² − 2⟨f̂⟩ on the independent tt̄ sample, with paired differences vs M.
  - All terms: SH is worse at every level (10–15σ).
  - After CV truncation: SH ≈ M at gen and reco; M is better at parton level (+0.06 ± 0.02 ×10⁻⁶, about 3σ).
  - SH describes the trigger-sculpted φ′ better (χ²/ndf 7.4 vs 45.7), because l₂ = 2 allows a second φ′ harmonic.
  - The extended basis is best at gen and reco (> 8σ). The hh AUC is basis-independent.
- **Bug found and fixed:** `osde_simplify.py` saved the Kronmal–Tarter (√2σ) truncation as `_simplified`, not the CV-best one. The truncation was rerun for all bases and levels, and the document and slides were corrected.

### Parton + lepton trigger level (user request: "see it through")
- `parton_trig` = parton events whose lepton passes p_T > 25 GeV and |η| < 2.47 (e) / 2.5 (μ); 71.5% of tt̄.
- Carried through plots, angle caches, order scan, and OSDE in all three bases.
- Paper-basis AUC 0.669 (CV: 0.685). The trigger sculpts φ′ and cos θ′, so it needs j₁ ≥ 4 and j₂ ≥ 8.

### Is the OSDE approach helping? ML comparison (`ml_compare.py`, user request)
- **Test set:** the same for every method (independent tt̄ + 40% held-out hh).
- **Unsupervised** (trained on tt̄ only, same angles), parton level:
  - OSDE f_tt̄: 0.652 (all terms), 0.665 (CV);
  - normalizing flow (RealNVP): 0.654;
  - Gaussian mixture: 0.583;
  - autoencoder: 0.528.
  - So OSDE is as good as or better than modern density estimators, with about 330 interpretable coefficients.
- **Supervised** (tt̄ vs hh labels):
  - angles only: BDT 0.881, MLP 0.887, FT-Transformer 0.884;
  - adding y = f_tt̄ (leave-one-out for tt̄ training events): no gain (Δ ≤ 0.002);
  - angles + momenta: 0.9995+ (the masses give hh away);
  - particle transformer on momenta: about 1.0.
- **Controls:** shuffled labels 0.49; tt̄-vs-tt̄ 0.50.
- Paired bootstrap errors on the AUC differences were added; the parton run is being redone with them.

### Legibility (user request)
- Shared plotting style (`plotting/style.py`).
- Multi-panel grids split into two figures (production + leptonic top, hadronic top). Score plots are 2×2, ROC curves one panel per slide, validation plots at most two per slide.
- Long tables continue on further slides.
- The deck is reviewed page by page.

### Readable document
- Claude Doc "tt̄ vs hh → bb̄ℓνqq̄′: angles, OSDE density and ML comparison": https://claude.ai/code/artifact/9b8d4480-1510-428c-8332-45c9a6454023

## 2026-10-07 — Legibility pass 2; why the separation washes out at gen level

- Second slide review: outline duplicate removed, MG5 syntax replaced by physics notation (tt̄ and hh),
  caption-free angle diagram for the slide, Δp_z split into 3 figures, lep-vs-had per case 1×2,
  scores split into two wide figures per level, CV curves one per slide, larger legends/fonts in
  acceptance, truncation, basis comparison; "parton_trig" no longer appears in figures.
- Question: parton-level f_tt̄ separates hh/tt̄ (AUC 0.65) but gen gives 0.525. Two studies
  (study_washout/mechanism, study_washout/decomposition):
  * Parton: hh "tops" are W(h1)+b(h2), m_Wb median 135 GeV, one W off-shell (median 32 GeV) ->
    cosθ, cosθ̃ peak at −1, cosΘ isotropic (no t-channel), no V−A zero in cosθ'.
  * Step losses in f_tt̄ AUC (LOO): selection 0, ISR recoil −0.007, FSR −0.033, χ² jet choice −0.063
    (largest; χ² picks true jets in 8% of hh vs 34% tt̄), ν from m_W constraint −0.028 (−0.05 on matched
    events; mostly the p_z-solution choice). Supervised BDT on angles: 0.88 -> 0.66.
  * Information is not gone: BDT on hypothesis-free gen observables (m_bb, ΔR_bb, MET, m_T, …) 0.87.
  * Conclusion: the tt̄-hypothesis reconstruction moves hh into the tt̄-dense region of angle space.
- Slides: new section "Why the separation washes out at gen level".
