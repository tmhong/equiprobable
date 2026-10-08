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
- ML (parton and parton+trig) complete; paired OSDE − flow +0.011 ± 0.003 / +0.008 ± 0.003; y does not help
  supervised ML (±0.002). Shuffled-label control at parton+trig 0.517 ± 0.006 (3σ; one of ~30 AUCs).
  ROC legends enlarged; ML conclusion slides carry the paired difference and the controls.
- Claude Doc: ML, washout and summary sections filled (all sections now complete).

## 2026-10-08 — Salvaging y: momentum through the off-shell directions (study_salvage/salvage.py)

- User: the method fails because hh has an off-shell W, not the same final state as on-shell tt̄.
- Idea: the ten angles are coordinates of the on-shell (narrow-width) manifold; the missing directions
  are the resonance masses. Extend f_tt̄(Ω) → f(Ω)·g(m), g an OSDE in Cauchy-mapped masses
  (u = 1/2 + arctan((m−M)/W)/π, Legendre in u, Jacobian du/dm); score = summed surprisal.
- Feasibility (AUC hh vs tt̄): parton masses 1.000 (trivial); gen (χ², 3 masses since m_W,lep is fixed by
  the ν constraint) angles 0.528 → masses 0.692, product 0.658; supervised BDT Ω+m 0.744 (+m_bb 0.848).
  With m_bb the product form fails (tt̄ angle–mass dependence AUC 0.63).
- Next steps proposed: ν from the m_t,lep constraint (frees m_W,lep); sum f over jet assignments and ν
  solutions instead of one χ² hypothesis; joint OSDE in (Ω, m).
- Simplest version (study_salvage/simplest.py): y_ext = y(Ω) × Π_k BW(m_k); surprisals add.
  Parton: PDG M, Γ, nothing fitted: AUC 0.648 → 1.000. Gen (χ²): Cauchy with tt̄ median, half-IQR
  (2 numbers per mass): 0.528 → 0.689, i.e. gen with masses exceeds the parton-level angles (0.648).
  Fitted Legendre correction adds only +0.004. Figures masses_{parton,gen}.png.
- Slides: momentum section rewritten (motivation by 18-component counting, the one-line extension,
  result table); plain-language ML explainer (unsupervised incl. normalizing flow; supervised).
- Angular terms needed with the BW factors (study_salvage/angle_terms.py): parton angles+BW = 1.000 and
  gen = 0.689 for every truncation incl. flat angles. Angles alone: 23 terms (|c|/σ>10) give 0.665 vs
  0.648 for all 14,400 at parton. Top terms: P2,P4(cosΘ), H2,H4(Y), W helicity (j1=0, j2=1,2) and products.

## 2026-10-08 — BSM benchmark, angles vs BW, internal note, pileup and L1 studies started

- Angles on top of BW (study_salvage/angles_vs_bw.py): product model +0.000 (gen) / +0.002 (reco);
  supervised BDT angles+masses − masses: +0.029 ± 0.004 (gen), +0.037 ± 0.015 (reco).
- BSM benchmark H(500 GeV)→tt̄, Γ_H = 25 GeV set by hand (MG5 auto width gave 4e5 GeV; first sample
  run_seed3001 discarded; run_seed3002 used; σ×B = 0.35 fb LO heft). study_bsm/htt_test.py:
  parton H→tt̄ vs tt̄: f_tt̄ angles 0.554, BW 0.506, +m_tt density 0.49 (dilution), hh-trained BDT 0.52
  (angles) / 0.43 (with masses: below random), dedicated BDT 0.89. Reco: f_tt̄ angles 0.565, hh-BDT 0.51.
  Lesson: anomaly scores find signals where background is rare, not excesses where it is common.
- Internal note: note/make_note.py → note/note.pdf (34 pages), numbers from the same JSONs as the slides.
- Pileup (μ = 50, 200) infrastructure agent and L1-trigger LUT study agent running (Opus, local).

## 2026-10-08 — Study B (tt̄ vs H→tt̄), pileup production, L1 study; usage limit reached mid-run

- Deck reorganized: Study A (tt̄ vs hh, kept in full) / Study B (tt̄ vs H→tt̄, m_H = 500 and 350 GeV, Γ set by hand) / Study C (pileup, L1).
- Study B (study_bsm/htt_study.py): anomaly score weak (angles alone AUC 0.55–0.57 at all levels; masses nothing;
  m_tt density does not help: the resonance sits on the bulk). Coefficient test (c_data − c_tt, 14,400 real dofs,
  calibrated: pull std 0.99 at ε=0): only P2(cosΘ) moves, pull ≈ 3.8–4.0 at ε = 10% in 10k events (noise ceiling 3.9);
  pull ∝ ε√N → 5σ needs ε√N ≈ 150; m_tt bump hunt ≈ 7× more sensitive. Decay-vertex (F_R) benchmark dropped
  (already constrained by single-top / W-helicity measurements; study_bsm/wtb_reweight.py kept).
- Pileup: infrastructure done (pileup/infra_report.md); shared analysis pileup/analyze_pileup.py; productions running
  (μ=0, 50: queue in work/pu_queue_pu0_pu50.*; μ=200: work/pu200_queue.sh, log work/pu200_queue.log). After "ALL DONE":
  merge parts, run analyze_pileup.py per μ (commands in the agent prompts / infra report §7), write pileup/results.json
  {"mu": {"0": {...}, "50": {...}, "200": {...}}} for note/make_note.py, then rebuild slides + note.
- L1 trigger LUT study (study_l1/) in progress by an agent.

## 2026-10-08 — Study A′ (hh at m_H = 170), B2 (right-handed W), pileup at μ = 0/50/200, L1 headline

- **Study A′** (`study_bsm/hh170_study.py` → `hh170_study.json`): hh with m_H = 170 GeV, so both W bosons are on shell, compared with hh(125). Same chain, nothing retrained.
  - Parton-level W Breit–Wigner factor: AUC 0.99 → 0.50. It goes blind, as expected.
  - Reco level, angles + masses: AUC 0.602 → 0.606.
  - Reco selection efficiency: 9.4% → 20%.
- **B2, right-handed W** (`study_bsm/wtb_study.py` → `wtb_study.json`): the independent tt̄ sample is reweighted to a W helicity fraction F_R (F_0 kept, F_L → F_L − F_R). The coefficients are fitted with event weights.
  - As an anomaly score it is weak: AUC 0.60 at F_R = 0.3 at parton level (4.9% of the signal in the 1% tt̄ tail), and nothing at reco level.
  - The coefficient test is the sensitive one. The cos θ′ term (M^01_00, ⟨cos θ′⟩ = (F_R − F_L)/2) has pull 8.6 at F_R = 0.05 and 15 at F_R = 0.1 with 10k parton events. 5σ at F_R = 0.02 needs ≈ 23k parton-level events.
- **Pileup** (`pileup/results.json`, review in `pileup/summary.md`): the consistency, independent-recomputation and physics checks pass.

  | ⟨μ⟩ | 0 | 50 | 200 |
  |---|---|---|---|
  | reco efficiency tt̄ / hh | 19.7% / 7.6% | 27.6% / 13.4% | 32.2% / 19.5% |
  | all four χ² jets correct (tt̄) | 17.5% | 5.8% | 1.1% |
  | AUC masses (BW) | 0.616 | 0.540 | 0.499 |
  | AUC angles (>10σ) + masses | 0.617 | 0.538 | 0.497 |
  | gen check (angles + masses) | 0.698 | 0.698 | 0.698 |

  - The loss comes from jet combinatorics, not resolution. The card has no pileup-jet rejection, and with ~7.8 jets per event the χ² builds W and top masses in any event.
  - The gen check does not change with ⟨μ⟩, so all of the loss happens at reconstruction.
- **L1** (`study_l1/design.md`): the masses-only (χ² + BW) design on a Versal Premium VP1802 uses 0.83% of the LUTs and 1.45% of the DSPs, with 0.43 µs latency at 240 MHz. Its AUC is 0.582 against 0.602 offline; the loss is the missing b-tag at L1.
- **Slides and note:**
  - B2 section added.
  - Legibility pass: fig_wtb shown as two crops; the L1 figures shown row by row with legible captions; long titles shortened to one line; the pileup table fits on one slide.
  - 0 overfull boxes.
  - The review corrected the "Pileup: what it shows" slide: only W(m_t,lep) grows with pileup, and the angle-only AUC falls below 0.5 at μ = 200 (0.476).

## 2026-10-08 — HH(170) as the main study

- **Main signal is now HH(170):** m_H = 170 GeV and Γ_H = 0.38 GeV, so both W bosons are on shell. The sample is `samples/hh_bblvqq/run_m170_seed2170` (20k events).
  - The full Study A chain was rerun on it with the same tt̄ samples. hh(125) becomes the off-shell variant (Study A′).
  - `make_slides.py` and `note/make_note.py` take `--signal h125|h170` (default h125). With h170 they read the `*_h170` directories and files (`sdir`/`sfile` helpers; the signal-specific slide text is in `SIGTEXT`).
  - A missing `_h170` directory skips its section; it never falls back to the hh(125) outputs.
- **f_tt̄ (OSDE, paper basis):** AUC of HH(170) vs the independent tt̄ sample, hh(125) in brackets.

  | level | all terms | CV-truncated |
  |---|---|---|
  | parton | 0.638 [0.653] | 0.652 [0.665] |
  | parton+trig | 0.652 [0.669] | 0.669 [0.685] |
  | gen | 0.547 [0.525] | 0.553 [0.525] |
  | reco | 0.534 [0.507] | 0.540 [0.505] |

- **Acceptance (parton+trig / gen / reco):** 73.3% / 44.4% / 20.0%, against 50.7% / 22.1% / 9.4% for hh(125). The leptons from on-shell W bosons are harder.
- **Washout chain:** f_tt̄ is refit at each step (LOO AUC).
  - S0 parton 0.636 → S1 selected 0.645 → S1c post-ISR 0.644 → S1d pre-hadronization (FSR) 0.614 → S2χ (χ² jets + gen lepton + true ν) 0.573 → S4 gen χ² reconstruction 0.550.
  - The χ² picks the right four jets in 16% of HH(170) events, against 8% for hh(125).
- **Mass extension at reco:** BW masses alone give 0.600; angles (>10σ) + BW give 0.606.
- **ML at parton level:**
  - OSDE (CV-truncated) 0.649 against 0.643 for the normalizing flow; the paired difference is +0.006 ± 0.003.
  - Supervised on the angles: 0.916. With the momenta added: 0.996.
- **Outputs:** `osde_h170/`, `plots_hh_h170/`, `acceptance_h170/`, `ml_h170/`, `study_washout_h170/` and `study_salvage/*_h170.{json,png}`.
  - Deck: `slides_h170/slides.pdf` (199 pages).
  - Note: `note/note_h170.pdf` (45 pages).
- **Build both versions:**
  ```
  # hh(125), the default
  .venv/bin/python make_slides.py --sample samples/ttbar_bblvqq/run_seed12345 --hh samples/hh_bblvqq/run_seed2001 --osde osde --acceptance acceptance --no-tests
  .venv/bin/python note/make_note.py
  # HH(170)
  .venv/bin/python make_slides.py --signal h170 --sample samples/ttbar_bblvqq/run_seed12345 --hh samples/hh_bblvqq/run_m170_seed2170 --osde osde_h170 --acceptance acceptance_h170 --no-tests --out slides_h170
  .venv/bin/python note/make_note.py --signal h170
  ```
- **Known caveat:** some parts of the HH(170) versions still refer to hh(125), because their tt̄-only fits were reused rather than refit:
  - the extended-basis and spherical-harmonic sections, and the AUC column of the basis comparison;
  - the tt̄-only inputs copied into `osde_h170` (`truncation_gen.*`, `order_scan.json`).
- **Final check:**
  - Both decks build with 0 overfull boxes and no missing figures.
  - The pages that differ from the hh(125) deck were read at 75 dpi. One layout fix: the Study A divider title is split into two balanced lines.
  - The note has 5 overfull hboxes (4.8–34 pt) in the outlook and appendix, from long `\texttt` file names. The hh(125) note has the same ones.
- **Open: wording left unchanged** (physics text, for review):
  - The slide "Answer, parton+trig level" says that adding y changes the supervised AUC "within the ±0.002 uncertainty". For HH(170) the paired gains are +0.009 ± 0.001 (MLP) and +0.007 ± 0.001 (FT-Transformer).
  - The cutflow note ("softer (from W*)") and the prompt-lepton bullet (h → WW*) still describe hh(125).
  - In `note_h170`, parts of the abstract and of the introduction, sample, ML and washout sections still describe the off-shell W*. The numbers in them are the HH(170) ones.

## 2026-10-08 — B2 (right-handed W): ML comparison

- `study_bsm/ml_wtb.py` → `study_bsm/ml_wtb.json` (+ `ml_wtb.log`). Same signal as `wtb_study.py`: the independent tt̄ sample reweighted to F_R. Compared:
  - the normalizing flow of `ml_compare.py`, trained on SM tt̄ only;
  - f_tt̄ angles (>10σ), recomputed here and identical to `wtb_study.json`;
  - a supervised BDT that knows w(F_R), the reachable optimum when the signal is known;
  - the oracle w (parton truth).
- Weighted AUC; the signal efficiency at 1% tt̄ efficiency is in brackets.

  | level | F_R | f_tt̄ angles | flow | BDT (knows w) | oracle w |
  |---|---|---|---|---|---|
  | parton | 0.1 | 0.534 (2.2%) | 0.528 (1.6%) | 0.579 (3.5%) | 0.578 (3.4%) |
  | parton | 0.3 | 0.600 (4.9%) | 0.583 (3.3%) | 0.731 (9.2%) | 0.729 (9.1%) |
  | gen | 0.3 | 0.522 (1.3%) | 0.523 (1.0%) | 0.603 (1.9%) | 0.714 (8.9%) |
  | reco | 0.3 | 0.505 (1.6%) | 0.508 (0.9%) | 0.579 (1.4%) | 0.705 (7.8%) |

  - At parton level, f_tt̄ beats the flow (paired +0.017 ± 0.001 at F_R = 0.3). At gen and reco the two are equal within errors.
  - The BDT reaches the oracle at parton level. At gen and reco it stays far below it, because the parton-truth weight is lost in the smearing.
  - So the anomaly scores are weak for this signal at every level, whichever method is used. The coefficient test (`wtb_study.py`) is still the sensitive probe.
- Checks pass:
  - F_R = 0 gives AUC 0.500 for every score.
  - BDTs trained on shuffled weights give 0.495–0.500 ± 0.02.
  - n_eff and the f_tt̄ AUC equal `wtb_study.json`.
  - The flow's hh-vs-tt̄ AUC is 0.654, equal to `ml_compare`.
  - The flow's NLL on the independent sample is within 0.04–0.15 of its training NLL.
- Run time: 7.7 min with 3 threads.
- In the deck: two B2 slides, a table and a summary. In the note: a paragraph and Table 35 in the B2 section.

## 2026-10-08 — HH(2m_W): full Study A chain

- **Sample:** `samples/hh_bblvqq/run_m2mw_seed2161`, 20k events, m_H = 2m_W = 160.8 GeV, Γ_H = 0.083 GeV. The scalar sits exactly at the WW threshold.
- **Chain:** `work/h2mw_chain.sh` (logged to `work/h2mw_chain.log`; about 95 min with 3 threads). It is the HH(170) chain with `OUT_SUFFIX=_h2mw`. Same tt̄ samples and fits, nothing retrained.
  - The tt̄-only `order_scan.json` and `truncation_gen.*` were copied into `osde_h2mw`, as for `osde_h170`.
  - `paper_style_plots.py` was missing from the first pass, so the deck lacked 5 slides. It was added and the deck rebuilt.
- **Outputs:** `osde_h2mw/`, `study_washout_h2mw/`, `study_salvage/*_h2mw.*`, `plots_hh_h2mw/`, `acceptance_h2mw/`, `ml_h2mw/`, `study_bsm/hh2mw_study.json` and `fig_hh2mw.png`.
- **Results.** HH(2m_W) first, with HH(170) in brackets.

  | quantity | HH(2m_W) [HH(170)] |
  |---|---|
  | f_tt̄ all terms, parton / parton+trig / gen / reco | 0.636 / 0.655 / 0.541 / 0.534 [0.638 / 0.652 / 0.547 / 0.534] |
  | f_tt̄ CV-truncated, parton / parton+trig / gen / reco | 0.649 / 0.670 / 0.544 / 0.538 [0.652 / 0.669 / 0.553 / 0.540] |
  | trigger / gen / reco efficiency | 70.5% / 40.8% / 17.0% [73.3% / 44.4% / 20.0%] |
  | washout, S0 → S1 → S1d → S2χ → S4 (LOO) | 0.634 → 0.643 → 0.609 → 0.567 → 0.544 [0.636 → 0.645 → 0.614 → 0.573 → 0.550] |
  | χ² picks the right 4 jets | 14.7% [16.1%] |
  | reco, BW masses / angles (>10σ) + BW | 0.593 / 0.602 [0.600 / 0.606] |
  | ML parton: flow / BDT, MLP, FT-Transformer on the angles / MLP on angles + momenta | 0.643 / 0.921, 0.923, 0.927 / 0.998 [0.643 / 0.912, 0.916, 0.916 / 0.996] |

  - At parton level, m_W,had still gives a 1-D AUC of 0.70, against 0.80 for hh(125). At threshold one W sits slightly below the pole.
  - Otherwise HH(2m_W) and HH(170) agree within the errors almost everywhere.
- **Control:** the "BDT, shuffled labels" control at parton+trig gives 0.535 ± 0.006.
  - The other runs give 0.517 (hh125), 0.489 (h170) and 0.512 (h2mw parton).
  - The spread across runs is larger than the quoted error, so the quoted error underestimates this control's spread. This holds for all three signals. Not investigated.
- **Documents:**
  - New: `slides_h2mw/slides.pdf` (199 pages) and `note/note_h2mw.pdf` (46 pages).
  - The hh(125) deck and note now take Study A′ from HH(2m_W). `cmp_variant` prefers h2mw when `hh2mw_study.json` exists.
  - All six documents rebuilt: slides 202 / 199 / 199 pages, notes 46 / 46 / 46; 0 overfull boxes in the decks.
  - Build commands: as for HH(170), with `--signal h2mw --hh samples/hh_bblvqq/run_m2mw_seed2161 --osde osde_h2mw --acceptance acceptance_h2mw --out slides_h2mw`, then `note/make_note.py --signal h2mw`. All six are in `work/build_all.sh`.
- **Caveats:** the same as for HH(170).
  - Some tt̄-only sections still describe hh(125).
  - Some HH-specific wording describes off-shell W*.

## 2026-10-08 — HH(170) is the reference on-shell variant again; HH(2m_W) wording corrected

- **Which variant isolates the angles.** At parton level the f_tt̄ angles are the same for both (0.633 vs 0.631). The masses differ:
  - HH(170): the W masses are tt̄-like (W-factor AUC 0.50).
  - HH(2m_W): the lighter W sits below the pole (median 77 GeV, 10% below 59 GeV; both W within 5 GeV of the pole in 58% of events), so the W factors still see it (AUC 0.59).
  - The top factors see both at 0.97. The pseudo-top m_Wb has a broad spread: IQR 153–217 GeV for HH(170), 145–211 GeV for HH(2m_W), against 172–174 GeV for tt̄.
- **Change:** `cmp_variant` (deck) and `variant()` (note) now prefer h170. HH(2m_W) stays as a threshold cross-check (`--signal h2mw`).
- **Wording:** the W-state, pseudo-top and neutrino texts are now per-variant keys in `VARIANTS` (`wstate`, `wshort`, `samew`, `tops`, `nu`). Corrected:
  - HH(2m_W) no longer says "effectively on shell" or "same W masses as tt̄".
  - The pseudo-top spread is no longer "180 GeV with a 100 GeV spread" for either variant.
- `samples/Htt_bblvqq/run_m350_seed3003/metadata.json` (gitignored): process label corrected to m_H = 350 GeV, Γ_H = 10 GeV.
- **Rebuilt all six documents:**
  - Slides: 202 / 199 / 199 pages, 0 overfull boxes.
  - Notes: 46 pages each.
