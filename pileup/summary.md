# Pileup study: review summary

2026-10-08. Review of `pileup/results_pu{0,50,200}.json`, `pileup/scores_pu*.npz` and the figures from `pileup/analyze_pileup.py` (one script, identical for every ⟨μ⟩) and `pileup/compile_results.py`. Setup: Delphes ATLAS *pileup* card at ⟨μ⟩ = 0 (control, same card), 50 and 200. The hard events are the same at every ⟨μ⟩ (tt̄ fit: seeds 5001–5004, 4 × 12.5k; tt̄ test: seed 12345, 10k; hh: seeds 6001–6002, 2 × 10k). The μ = 200 analysis ran without errors (09:22–09:25).

## Key numbers versus ⟨μ⟩

| quantity | ⟨μ⟩ = 0 | ⟨μ⟩ = 50 | ⟨μ⟩ = 200 |
|---|---|---|---|
| generated: tt̄ fit / tt̄ test / hh | 50k / 10k / 20k | 50k / 10k / 20k | 50k / 10k / 20k |
| selected (reco_ok, finite angles): tt̄ fit / test / hh | 10217 / 1965 / 1515 | 13763 / 2755 / 2677 | 15973 / 3221 / 3906 |
| reco efficiency tt̄ (test) [%] | 19.7 ± 0.4 | 27.6 ± 0.4 | 32.2 ± 0.5 |
| reco efficiency hh [%] | 7.6 ± 0.2 | 13.4 ± 0.2 | 19.5 ± 0.3 |
| jets / event (p_T > 25 GeV, all events), tt̄ / hh | 3.73 / 3.45 | 4.43 / 4.21 | 7.75 / 7.55 |
| b-tagged jets / event, tt̄ | 1.155 ± 0.008 | 1.087 ± 0.008 | 1.013 ± 0.008 |
| all 4 χ² jets correct, of selected tt̄ [%] | 17.5 ± 0.9 | 5.8 ± 0.4 | 1.1 ± 0.2 |
| ν p_z 68% half-width, selected tt̄ [GeV] | 65.2 ± 3.0 | 91.9 ± 2.9 | 124.4 ± 3.6 |
| MET − p_T(ν) 68% half-width, selected tt̄ [GeV] | 18.9 ± 0.5 | 33.2 ± 0.5 | 47.8 ± 0.9 |
| same, all tt̄ events (validation definition) [GeV] | 19.9 ± 0.3 | 34.3 ± 0.3 | 51.6 ± 0.5 |
| MET − p_T(ν) median, all tt̄ events [GeV] | +1.9 | +14.0 | +36.0 |
| mass median M, tt̄ fit: m_W,had / m_t,lep / m_t,had [GeV] | 82.8 / 173.9 / 176.6 | 83.2 / 179.8 / 180.8 | 81.9 / 189.5 / 181.2 |
| mass width W (half-IQR), tt̄ fit [GeV] | 13.5 / 14.9 / 26.8 | 14.4 / 22.5 / 27.9 | 12.9 / 32.6 / 26.7 |
| mass width W (half-IQR), hh [GeV] | 21.1 / 24.7 / 40.0 | 17.0 / 24.3 / 33.2 | 12.4 / 29.5 / 27.3 |
| f terms kept: all / >10σ / >5σ / >3σ | 14400 / 17 / 62 / 150 | 14400 / 47 / 152 / 334 | 14400 / 112 / 345 / 757 |
| AUC masses (BW) | 0.616 ± 0.009 | 0.540 ± 0.008 | 0.499 ± 0.007 |
| AUC angles (>10σ) | 0.524 ± 0.009 | 0.497 ± 0.007 | 0.476 ± 0.006 |
| AUC angles, all terms | 0.494 ± 0.010 | 0.482 ± 0.007 | 0.474 ± 0.006 |
| AUC angles (>10σ) + masses | 0.617 ± 0.009 | 0.538 ± 0.008 | 0.497 ± 0.007 |
| ε_hh at 1% / 10% tt̄, masses [%] | 1.4 ± 0.4 / 15.8 ± 1.3 | 1.0 ± 0.3 / 12.4 ± 1.0 | 1.1 ± 0.3 / 10.6 ± 0.7 |
| ε_hh at 1% / 10% tt̄, angles + masses [%] | 1.4 ± 0.4 / 16.5 ± 1.4 | 1.0 ± 0.3 / 12.3 ± 1.1 | 1.1 ± 0.3 / 9.8 ± 0.7 |
| gen check AUC (angles + masses, gen level) [n tt̄ / hh] | 0.6982 ± 0.0056 [4103 / 4607] | same | same |

Errors: efficiencies binomial; half-widths 200-replica bootstrap; AUC the script's 100-replica bootstrap (agrees with Hanley–McNeil within 5%); ε 500-replica bootstrap resampling tt̄ and hh separately with the threshold re-derived each time. W is the half-IQR of the selected events.

## Checks and outcome

1. **Consistency: PASS.** Event counts 50000 / 10000 / 20000 at every ⟨μ⟩; Delphes entries = LHE events in every merged sample; all 21 production jobs OK. Pairing: the md5 of all LHE `<event>` blocks is identical across ⟨μ⟩; GenJet multiplicity and Σp_T identical event by event (80k events per ⟨μ⟩). Gen check 0.69823 ± 0.0056 at all three ⟨μ⟩ on the same events; compatible with the no-pileup gen value 0.692. μ = 200 vs the 1000-event validation: ⟨n_jets⟩ 7.752 ± 0.021 vs 7.695 ± 0.068 (0.8σ), ⟨n_b⟩ 1.013 vs 1.018 (0.2σ), reco_ok 0.3228 vs 0.332 (0.6σ), all four jets correct 0.0115 vs 0.006 (1.2σ), MET half-width 51.6 vs 50.4 GeV (0.7σ); μ = 0 and 50 within 1.3σ for every quantity.
2. **Independent recomputation: PASS.** All 9 scores per ⟨μ⟩ recomputed from `scores_pu<mu>.npz` with sklearn and with the Mann–Whitney rank sum: largest difference from the JSON 1.1e-16. eff1/eff10 identical. The JSON validation fields recomputed from the samples with separate code: identical (see open issue 6).
3. **Physics: PASS**, two features explained. (a) The m_W,had and m_t,had widths do not grow and the hh widths shrink (hh W(m_W,had) 21.1 → 17.0 → 12.4 GeV): with ~7.5 jets the χ² always finds a combination near the tt̄ mass hypotheses, so the masses become tt̄-like by construction, which is why the mass AUC goes to 0.5. (b) The angle-only AUC falls below 0.5 at μ = 200 (0.476 ± 0.006): against tt̄ with ≤ 1 χ² jet matched to a parton (73% of events) hh gives 0.501, against ≥ 2 matched 0.408, against the 37 fully matched events 0.328. The reco f is dominated by wrong-jet combinations (>10σ terms 17 → 47 → 112), so the genuine tt̄ structure is the atypical part and hh, which never has it, looks more tt̄-like than tt̄.
4. **Figures.** `compile_results.py` plotting fixed (literal `\%`, mixed units in panel 3, 13 pt legends, a curve through a legend); `summary_vs_mu.png` regenerated: one unit per panel, 16 pt legends in headroom, jets per event as a top tick row. `masses_*` and `scores_*` have no overlaps or clipping (their 13 pt legends are small on a slide; drawn by `analyze_pileup.py`, not changed).

## Physics reading

Pileup degrades the method through combinatorics, not resolution. Without pileup-jet rejection, pileup jets pass the ≥ 4-jet selection: efficiencies rise while the χ² picks the right four jets in only 1% of tt̄ at μ = 200; with that many jets the χ² builds W and top masses for any event, tt̄ or hh. The mass discrimination is 0.616 at μ = 0, half lost at μ = 50 (0.540) and gone at μ = 200 (0.499 ± 0.007); the angles carry nothing at reco level beyond μ = 0. The gen check is unchanged, so the whole loss happens at reconstruction. To survive HL-LHC pileup the method needs pileup-jet rejection (JVT-like), or a jet choice that does not use the mass hypotheses it then tests.

## Open issues

1. No pileup-jet rejection in the ATLAS pileup card (only ρ·A and track subtraction): a worst case. A `PileUpJetID`-like or JVT-like cut is the next step.
2. Jet radius: both ATLAS cards use anti-k_T R = 0.6 (README says 0.4); at μ = 200 about 215 GeV is subtracted per R = 0.6 jet against 95 GeV for R = 0.4, so the pileup-jet rate is larger: pessimistic.
3. The μ = 0 control is not the no-pileup card (JES, calorimeter, MET, tracking thresholds differ): AUC 0.617 ± 0.009 vs 0.602; quote pileup effects only within this set.
4. MET includes pileup: median bias +14 / +36 GeV, which moves the m_t,lep median from 174 to 190 GeV; a vertex-associated soft term would reduce it.
5. Two MET definitions: `met_res_half_width` is for selected events; the infra validation used all events.
6. `analyze_pileup.py` `load()` modifies `R.reco_ok` in place (`ok &= finite`), so the validation fields are for "reco_ok and finite angles"; at μ = 200 this drops 7 tt̄ and 10 hh events (0.2%); negligible; use `.copy()` if touched again.
7. Slide wording "mass widths W grow accordingly" holds only for m_t,lep.
8. Errors are statistical only; the events are paired, so differences between ⟨μ⟩ values have smaller errors than the quadrature sum.
9. Min-bias reuse at μ = 200: about 160 uses per library event; 18% of events contain one library event twice; small.
