# Go/no-go verdict: NO-GO (2026-10-13)

Rule: `CRITERIA.md` (committed in 2f0f6f5 before any result). Code: `analyze.py`; results: `results.json`, `results.md`,
`fig_enrichment.png`. Independent checks: `verify/` (recompute, robust_met, physics_skeptic, critic).

## Outcome
No benchmark passes, at reconstruction level with χ² jets or with truth-matched jets: 0 of 12 cells (η_t, stop and
H(350) at ε_tt̄ = 1% and 10%, at both levels). Every R was reproduced by an independent implementation to within 0.12σ.
Leave-one-out was verified exactly: refitting gives the same coefficients, and a brute-force leave-one-out agrees with
the formula.

R = ε_S/ε_tt̄ at reco level (χ² jets), at 1% (10%) of tt̄:

| | f_tt̄ (CV) | flow | BW masses | best m_tt̄ window | supervised BDT, angles |
|---|---|---|---|---|---|
| η_t(343) | 1.31 (1.27) | 1.42 (1.20) | 0.71 (0.88) | **6.40 (3.46)** | 1.57 (1.48) |
| stop (300, 120) | 1.75 (1.13) | 1.07 (1.01) | 1.47 (1.26) | 1.31 (1.09) | 2.60 (1.78) |
| H(350) | 1.21 (1.22) | 1.23 (1.11) | 1.04 (1.14) | **2.40 (2.10)** | 1.09 (1.18) |
| tt̄, no spin corr. | 0.98 (1.06) | 1.20 (1.03) | 0.87 (1.01) | 1.09 (1.04) | 1.09 (1.00) |
| tt̄ independent (null) | 1.27 (1.18) | 1.15 (1.06) | 0.90 (1.01) | 0.94 (1.21) | 1.27 (1.20) |

## What the enrichments actually are (verified)
- **Even perfect density estimation could not pass.** The supervised classifier on the same ten angles reaches only
  1.57 (η_t; 1.70 in a 5-fold variant), 1.09 (H350) and 1.09 (no spin) at 1%. Per event, the angles carry too little
  spin information at reco level.
- **SM-size spin correlations are invisible per event, even at parton level.** tt̄ with against tt̄ without spin
  correlations gives a supervised angle BDT of R = 1.02 (AUC 0.507). Spin information in tt̄ is an ensemble property,
  reachable through fitted coefficients, not through a per-event score.
- **η_t: the angular enrichment tracks m_tt̄, not spin.** At fixed m_tt̄ (threshold set per tt̄ m_tt̄ decile), R@10%
  is 0.89 ± 0.03 at parton level and 1.09 ± 0.04 at reco, no larger than the null.
- **Stop: the reco-level excess is a missing-energy artifact.** f_tt̄ gives R = 0.76 with the true neutrino (parton +
  trigger), 1.43 at generator level and 1.75 at reco. Subtracting the final-state neutralino momenta from the Delphes
  E_T^miss on the same events brings it to 0.96 (paired −0.78 ± 0.11), and the angle BDT drops from 2.60 to 1.23.
  (A second verifier subtracted the LHE-level neutralino pT, which misses the ISR recoil, and got 1.55; the critic
  reproduced both and traced the difference.) Standard stop handles do far better: m_T(ℓ, E_T^miss) R@1% = 5.96 ± 0.44,
  E_T^miss 4.80 ± 0.29. These handles were not in the rule, an omission for a benchmark with invisible particles.
- **f_tt̄ is at least as good as the flow** on the same angles at parton level (η_t 1.45 vs 1.06, H350 1.46 vs 0.91 at
  1%). At reco the two are comparable (η_t 1.31 vs 1.42, stop 1.75 vs 1.07).

## Caveats (all make the NO-GO more conservative or do not affect it)
- **The setup favours f_tt̄:** truth up/down tagging of the W jets, no pileup, and leave-one-out scores after choosing
  terms on the full fit sample (about +0.05 at 10%).
- **Near-misses are not real passes.** Off-registration variants pass the literal rule for the stop (f with
  |c|/σ > 4 at 0.5%: 2.20 ± 0.19; f + BW at 1%: 2.11). All of them fall to about 1 once the neutralinos are removed
  from E_T^miss.
- **The null control is high at 10%.** tt̄ independent sits at R@10% = 1.18 ± 0.07 at reco, probably a fluctuation
  of 2440 events. The η_t and H350 R@10% values are not significant against it.
- **The m_tt̄ window is chosen on half the signal and evaluated on the other half** (the rule says "with knowledge of
  the signal"). Choosing it on the full signal changes no outcome.

## Corrections to the benchmark description (from the physics check)
- **η_t width:** the experiments generate η_t with Γ = 2.8 GeV (about 2Γ_t), not 7 GeV (the Fuks et al. 2021 value).
  The spin state is the same, so the angular verdict is unaffected; only the parton m_tt̄ peak is wider here.
- **η_t cross section:** in lepton+jets, CMS measures 5.1 ± 0.9 pb (6.1σ, arXiv:2608.19546, checked); the 8.8 pb
  used for the yields is the dilepton value. R does not depend on the normalization.
- **Stop point: already excluded.** (300, 120) lies just outside the CMS corridor exclusion (145-295 GeV for
  m_χ ≤ 100 GeV), but ATLAS lepton+jets with 139 fb⁻¹ (arXiv:2012.03799, checked) states: "In the diagonal region,
  where the mass difference between the stop and the neutralino coincides with the mass of the top quark, stop masses
  up to 600 GeV are excluded". The point is a kinematic benchmark only, not "at the edge" of the limits as
  CRITERIA.md says. An unexcluded diagonal point needs m_t̃ > 600 GeV (σ about 0.2 pb), where E_T^miss is stronger.

## Consequence for the paper
No spin-structure niche for a per-event angular anomaly score, so no PRL claim on that basis. The honest content is
for PRD:
- a symmetry-defined, closed-form density that matches or beats a flow on the same inputs;
- the information-loss accounting through the levels;
- this benchmark study, showing that per-event spin information in tt̄ is bounded by the decay analyzing powers and
  that mass, m_tt̄ and E_T^miss handles dominate where new physics changes the kinematics;
- the coefficient-level (ensemble) test as the right tool for spin structure.
