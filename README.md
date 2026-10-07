# ttbar → bbWW → b b ℓν qq′ with MadGraph5 + Pythia8 + Delphes (ATLAS, no pileup)

```
setup_tools.sh        one-time install: MG5_aMC 3.6.7, Pythia8, Delphes 3.5.1, python venv
generate_samples.py   MG5 -> Pythia8 -> Delphes(ATLAS card) sample production
make_plots.py         validation plots (parton / particle / reco level overlays)
osde_fit.py           f_tt̄ on the ten angles by OSDE; applies it to tt̄ and hh
make_slides.py        Beamer slides; every number is recomputed from the samples
tests/                unit tests of the basis functions and OSDE (pytest)
plot_config.py        <- the plot definitions: edit this to add/change plots
plotting/             io.py (LHE + Delphes readers), physics.py (objects, selection,
                      chi2 ttbar reconstruction), angles.py (the paper's ten angles),
                      osde.py (basis + OSDE), summary.py (numbers), histo.py (drawing)
```

## 1. Setup (once)

Needs ROOT (`root-config` in PATH), gfortran, cmake, and curl. The setup was already run in this directory.

```bash
./setup_tools.sh             # installs into ./tools and ./.venv; re-running skips finished steps
source .venv/bin/activate    # or call .venv/bin/python explicitly
```

## 2. Generate

```bash
python generate_samples.py --nevents 10000                    # full chain, 13 TeV, l = e, mu
python generate_samples.py --nevents 10000 --seed 2           # another statistically independent run
python generate_samples.py --stage parton --nevents 50000     # LHE only (fast)
python generate_samples.py --leptons e,mu,ta --sqrts 13600 --name ttbar_bblvqq_13p6_tau
python generate_samples.py --set "run_card ptj 20"            # any MG5 card setting (repeatable)
python generate_samples.py --dry-run                          # print MG5 commands only
python generate_samples.py --process hh --nevents 20000       # hh -> bb WW* -> bb l nu qq (heft)
```

Process (LO, `sm` model, 4-flavour proton, spin correlations kept through the decay chain):

```
generate    p p > t t~, (t > b w+, w+ > lep+ nul), (t~ > b~ w-, w- > q q)
add process p p > t t~, (t > b w+, w+ > q q),     (t~ > b~ w-, w- > lep- nul~)
```

No generator-level cuts are applied (`cut_decays = False`). Expect σ ≈ 136 pb at 13 TeV for l = e, μ. That equals σ(tt̄, LO) ≈ 460 pb × 2 · BR(W→ℓν) · BR(W→qq′).

The output goes to `samples/<name>/<run>/`:

| file | content |
|---|---|
| `parton_events.lhe.gz` | MG5 parton-level events (tops, Ws, b's, lepton, ν, quarks) |
| `delphes_events.root` | Delphes ATLAS reco (`Jet`, `Electron`, `Muon`, `Photon`, `MissingET`, `ScalarHT`) **and** generator level (`Particle` = full Pythia8 record, `GenJet`, `GenMissingET`) |
| `pythia8_events.hepmc[.gz]` | only with `--keep-hepmc` (~90 kB/event); otherwise deleted after Delphes |
| `metadata.json` | cross section, seed, settings, paths |

Notes:
- **Pileup:** the standard `delphes_card_ATLAS.tcl` has no pileup. For pileup later, pass `--delphes-card <MG5>/Delphes/cards/delphes_card_ATLAS_PileUp.tcl` (it needs a minimum-bias file).
- **File size:** expect ~70 kB/event (10k events ≈ 0.7 GB). By default the Delphes output drops the `Track`, `Tower` and `EFlow*` branches, which saves ~30%; the detector simulation is unchanged. Use `--full-delphes-output` to keep them. The remaining size is almost entirely the `Particle` branch (the full Pythia8 record, ~1500 particles/event). The particle-level code needs it to find prompt leptons through their mother links.
- **Event order:** event *i* in the LHE file is event *i* in the Delphes file. This was checked: the signed lepton PID agrees in 100% of events.
- **MG5 working area:** `work/<name>/`. A new `--run`/`--seed` reuses the same process directory.

## 3. Plot / validate

```bash
python make_plots.py --sample nominal=samples/ttbar_bblvqq/run_seed12345
python make_plots.py --sample s1=samples/ttbar_bblvqq/run_seed12345 --sample s2=samples/ttbar_bblvqq/run_seed2 \
                     --levels reco --norm xsec --lumi 140
python make_plots.py --sample nominal=samples/ttbar_bblvqq/run_seed12345 --only top_had_mass,lep_pt --format png,pdf
python make_plots.py --sample some/delphes_events.root      # single files work too (.lhe(.gz) or .root)
```

- With **one sample**, the parton, particle (gen) and reco levels are overlaid, and the ratio panel is relative to the first level drawn.
- With **several samples**, each sample is compared with the first sample at the same level (gen↔gen, reco↔reco). The line style marks the level (`:` parton, `--` gen, `-` reco).
- `--norm unit` (default) normalizes each histogram to unit area. `--norm xsec --lumi L` gives events for L fb⁻¹.
- The script also prints a cutflow (1 lepton / ≥4 jets / ≥1 b / ≥2 b) for the gen and reco levels.

### Adding or changing a plot

Edit `plot_config.py`. Each plot is one dict:

```python
dict(name="lead_bjet_eta", levels=["gen", "reco"],
     select=lambda ev: ev.nbjets >= 1,           # or "presel", "reco_ok", or None
     expr=lambda ev: ev.bjets[:, 0].eta,         # one entry (or a list) per event
     bins=(30, -2.5, 2.5), xlabel=r"Leading b-jet $\eta$"),
```

`expr` and `select` can be `{level: function}` dicts when the levels need different code. The top of `plot_config.py` lists everything available on `ev`. Object cuts (jet/lepton p_T and η, χ² resolutions) are in `CUTS` in `plotting/physics.py`; override them from `plot_config.py`.

### Object and reconstruction definitions

| level | jets | b-jets | leptons | MET |
|---|---|---|---|---|
| parton | MG5 quarks | MG5 b, b̄ | MG5 lepton | ν p_T |
| gen | Delphes `GenJet` (anti-kT 0.4), p_T>25, \|η\|<2.5 | GenJet within ΔR<0.4 of a b quark (p_T>5) | status-1 e/μ from a W (or τ←W) | `GenMissingET` |
| reco | Delphes `Jet`, p_T>25, \|η\|<2.5 | `BTag` bit 0 (ATLAS card efficiency map) | `Electron` (\|η\|<2.47), `Muon` (\|η\|<2.5), p_T>25 | `MissingET` |

Preselection (`ev.presel`): exactly 1 lepton, ≥4 jets, ≥1 b-jet.

The tt̄ reconstruction (gen and reco) uses a χ² built from m_jj = m_W, m_jjb = m_t and m_ℓνb = m_t. It considers all assignments of the 6 leading jets in which the two b slots hold min(n_b, 2) b-tagged jets, and both m_W-constrained ν p_z solutions. The m_ℓνb = m_t term therefore resolves the two-fold ν ambiguity, as in the paper. On reco, jets truth-matched to the hadronic-top quarks give m_t ≈ 172 GeV and m_W ≈ 82 GeV. The χ² peaks are broader because the best χ² picks the fully correct assignment in only ~35% of events, and one or more quarks is often out of acceptance or merged.

### The ten angles (`plotting/angles.py`, as defined in `main.tex`)

`x = {Y, Θ, θ, φ, θ′, φ′, θ̃, φ̃, θ̃′, φ̃′}` are computed at all three levels. They are stored on `ev` as `Y, cos_Theta, cos_theta, phi, cos_theta_prime, phi_prime, cos_theta_tilde, phi_tilde, cos_theta_prime_tilde, phi_prime_tilde`; the plot files are named `angle_*.png`.

- **Y**: rapidity of the tt̄ system in the lab. **Θ**: polar angle, relative to the beam, of the leptonic top in the tt̄ rest frame.
- **θ, φ**: direction of the W in the top rest frame. z is the top's lab direction, y = beam × z (normal to the beam–top plane), and x = y × z.
- **θ′, φ′**: direction of the fermion in the W rest frame, reached by the standard boost from the top frame. z′ = Ŵ, y′ = z × z′ (in the xy plane) and x′ = y′ × z′, i.e. the D(φ, θ, 0) rotation.
- Unadorned angles are for the leptonic top, with the charged lepton as the fermion. Tilded angles are for the hadronic top, with the **down-type** quark (the analogue of ℓ) as the fermion.
- At gen/reco, the down-type W_had jet is chosen by **truth matching** to the LHE quarks, which matches the paper's assumption of up/down tagging. Events are aligned, so this needs the LHE file in the sample directory.
- Options are set by `ANGLES` in `plot_config.py`: `had_fermion: "down" | "up"` and `jet_assignment: "truth" | "softer" | "harder"`.

Validation (10k events):
- A synthetic round trip, building momenta from chosen angles and recovering them, agrees to 1e-14.
- At parton level, the helicity fractions from cos θ′ and cos θ̃′ are F₀ = 0.715 ± 0.013 and 0.705 ± 0.013, with F_L ≈ 0.29 and F_R ≈ 0. The SM values are F₀ = 0.70, F_L = 0.30, F_R = 0. Using the up-type quark instead swaps F_L ↔ F_R, as it should.
- In events where the χ² picked the correct jets (~20%), the gen/reco angles correlate with the parton angles at 0.84–0.98, except φ′ at reco (0.47), which depends on the ν reconstruction.
- ν p_z resolution relative to the truth (68% half-width, events with a reconstructed tt̄): 54 GeV at gen level and 72 GeV at reco level, with median bias < 1 GeV.

## hh → bb̄ WW* → bb̄ ℓν qq̄′

`--process hh` generates `g g > h h, (h > b b~), (h > lep+ nul q q)` plus the charge conjugate, using MG5's built-in `heft` model as in the paper. Because m_h < 2m_W, h → ℓνqq̄′ is a four-body decay through WW*.

- The cross section (0.11 fb at 13 TeV) is LO HEFT with MG5's default parameters; its h width is not the sum of its own partial widths. It is **not** an SM prediction, so compare hh shapes with unit-area normalization.
- For tt̄-hypothesis quantities (the angles), hh uses the same χ² at gen/reco level. At parton level, hh uses the b pairing that best matches m_t.

## f_tt̄ by OSDE (`osde_fit.py`, `plotting/osde.py`)

```bash
python osde_fit.py --train samples/ttbar_bblvqq/run_seed1001 --test samples/ttbar_bblvqq/run_seed12345 \
                   --hh samples/hh_bblvqq/run_seed2001          # all levels; outputs in osde/
python -m pytest -q tests                                       # basis and OSDE checks
python make_slides.py --sample samples/ttbar_bblvqq/run_seed12345 --hh samples/hh_bblvqq/run_seed2001 --osde osde
```

**Basis** (as in the paper):

- M^{j₁j₂}_{m′m} = (1/4π)·√(2j₁+1)·√(2j₂+1)·D^{j₁}_{m′m}(φ,θ,0)·D^{j₂}_{m0}(φ′,θ′,0), with j₁ ≤ 1 and j₂ ≤ 2. That gives 24 functions per top.
- cos Θ: normalized Legendre polynomials √((2l+1)/2)·P_l.
- Y: normalized Hermite *functions* (2ⁿn!√π)^{−1/2}·H_n(Y)·e^{−Y²/2}, with scale `--y-scale` (paper: 1).
- Total: 24 × 24 × 5 × 5 = 14,400 coefficients, estimated as ĉ = mean(B*) together with their standard errors. The basis factorizes, so the fit uses einsum and never builds an N × 14,400 array.

**Outputs** (in `osde/`):

- `f_ttbar_<level>.npz`: coefficients and their errors; reload with `TTbarDensity.load`.
- `scores_<level>.npz`: f evaluated on every event.
- `projections*_<level>.png`: the ten projections of f against the data.
- `scores_<level>.png`: distributions of y and s = −ln y, and the hh-vs-tt̄ efficiency curve.
- `summary.json`: all numbers, including the coefficient-truncation and Hermite-scale diagnostics.

**Checks** (`tests/`, 7 tests):

- Wigner d against closed forms and against sympy.
- Exact orthonormality of every basis factor by quadrature.
- Closure of the basis under complex conjugation, so f is real.
- A toy with known coefficients: all 14,400 are recovered with unit-width pulls, and the projections and ∫f = 1 are reproduced.

**Results** (tt̄ trained on 100k events and tested on an independent 10k; hh 20k): see `slides/slides.pdf`.

- AUC(hh vs tt̄) is 0.65 at parton level and 0.53 / 0.51 at gen / reco.
- Only 1–2% of the coefficients are significant at 3σ, so keeping all 14,400 overfits (the training sample looks more probable than the test sample). Truncating removes the overfitting but does not restore separation at gen/reco.

Angles are cached as `<sample>/angles_<options>.npz`. Delete the cache, or use `--rebuild-cache`, after changing object definitions.

## Other scripts

| script | what it makes |
|---|---|
| `acceptance_plots.py` | parton-level angles before and after the gen/reco selections, plus the efficiency vs angle cut by cut (`acceptance/`) |
| `nu_pz_plot.py` | Δp_z of the reconstructed neutrino: gen vs reco, effect of the jet assignment, choice of the m_W solution (`plots/nu_pz.png`) |
| `hh_plots.py` | hh overview and the hadronic-W mass split into on-shell W and W* (`plots_hh/`) |
| `osde_simplify.py` | how many coefficients matter: Kronmal–Tarter rule, 5-fold cross-validated threshold, physics labels of the kept terms (`osde/simplify_*`, `osde/terms_*.txt`) |
| `figures/make_angle_diagram.py` | TikZ diagram of (θ, φ, θ′, φ′) in the top and W rest frames (`figures/angle_diagram.pdf`) |

Full deck: `python make_slides.py --sample samples/ttbar_bblvqq/run_seed12345 --hh samples/hh_bblvqq/run_seed2001 --osde osde --acceptance acceptance`.
It uses a University of Pittsburgh color theme, includes an outline, and runs the unit tests. `LOG.md` is the running log of requests, decisions and results.
