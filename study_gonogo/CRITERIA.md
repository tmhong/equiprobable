# Go/no-go test for a PRL: decision rule (fixed before any result was computed)

Written 2026-10-13, after the benchmark samples were validated at parton level and before any score was evaluated
on them. The rule follows the PRL review (`~/Claude-code/review/PRL_review_2026-10-13.md`, Section 2, item 1).

## Question
Does the unsupervised tt̄ angular density f_tt̄ flag a new-physics signal that **differs from tt̄ in spin structure
rather than in mass**, at reconstruction level, better than the mass-based handles and at least as well as a
generic density estimator trained on the same inputs?

## Benchmarks (same b b̄ ℓν q q̄′ final state, MG5 → Pythia8 → Delphes ATLAS, no pileup)
| key | process | why | normalization |
|---|---|---|---|
| `etat` | η_t: CP-odd colour-singlet spin-0 state, m = 343 GeV, Γ = 7 GeV (HC model, cos α = 0) | the threshold excess observed by CMS and ATLAS; spin singlet, sits where tt̄ is densest in m_tt̄ | 8.8 pb (CMS 2025) |
| `stop` | t̃₁t̃₁ → tt̄ χ̃₁⁰χ̃₁⁰, m_t̃ = 300, m_χ̃ = 120 GeV | stealth stop at the edge of the top-corridor exclusion; no tt̄ spin correlation, polarized tops | 10.57 pb (NNLO+NNLL) |
| `H350` | g g → H(350, CP-even, Γ = 10 GeV) → tt̄ (existing sample) | ³P₀ spin structure at threshold | none (LO heft, arbitrary) |
| `nospin` | QCD tt̄ with the spin correlations switched off (MadSpin spinmode none) | control: identical masses and kinematics, only the decay-angle correlations differ | none (control) |
| `tt_indep` | the independent tt̄ sample | null control: every score must give AUC 0.5 and R = 1 | — |

Parton-level validation (`validate_parton.py`, ⟨cos φ⟩ of ℓ and d in their top rest frames): tt̄ +0.079,
η_t +0.330 (spin singlet), no-spin +0.006, H350 −0.120 (³P₀), stop +0.009.

## Scores (all trained or fitted on tt̄ only, except the supervised ceiling)
1. **f_tt̄, CV-truncated** (|ĉ|/σ > 2.5, the cross-validated choice on tt̄ alone): **the primary detector**.
2. f_tt̄, all 14,400 terms (the paper's estimator): secondary.
3. Breit-Wigner mass score: product of Cauchy factors for m_W,had, m_t,lep and m_t,had (tt̄ median, half the IQR).
4. 1-D tt̄ density of m_tt̄ (unsupervised).
5. **Best m_tt̄ window** at the same tt̄ efficiency, chosen with knowledge of the signal: an optimistic bump hunt.
6. Normalizing flow (RealNVP, the paper's baseline) on the same ten angles, tt̄ only, out-of-fold.
7. Angles (1) combined with the masses (3): the sum of −ln of their tt̄ tail fractions.
8. Supervised BDT ceiling: angles; angles + masses + m_tt̄ (reported, not part of the rule).

tt̄ scores are leave-one-out (OSDE) or out-of-fold (flow) on the tt̄ fit sample.

## Metric
Tail enrichment R = ε_S / ε_tt̄ at ε_tt̄ = 1% and 10% (the gain in S/B from cutting on the score), with bootstrap
uncertainties over both samples. Paired-bootstrap σ for differences between scores.

## Rule
Primary level: **reconstruction with the χ² jet assignment** (what an analysis can do today).

**GO** if, for at least one physical benchmark (`etat`, `stop`, `H350`) at ε_tt̄ = 1% or 10%, all three hold for
the primary detector (1):
- (a) R ≥ 2 and R − 1 > 3σ (a real enrichment);
- (b) R exceeds R of the mass score (3), of the m_tt̄ density (4) and of the best m_tt̄ window (5), each by > 2σ
  (paired);
- (c) R is not below the flow's (6) by more than 2σ (paired).

Otherwise **NO-GO**: the Letter cannot claim a spin-structure niche, and the paper goes to PRD with an honest
account of where the information is lost.

The same rule is evaluated with ΔR-matched jets (`study_truthmatch`): a pass there but not with χ² jets is a
"conditional GO" (the niche exists if jet assignment is solved), reported but not a GO.

Expected yields at 140 fb⁻¹ (σ_tt̄ = 832 pb, BR(e/μ+jets) = 0.291) are reported for context: S/B in the tail and
Z = S/√(B + (δB)²) with δ = 0, 5% and 10%.
