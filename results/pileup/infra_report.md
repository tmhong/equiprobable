# Pileup infrastructure: MG5 → Pythia8 → Delphes at ⟨μ⟩ = 50 and 200

2026-10-08. Validated with 1000 tt̄ events at μ = 0 (control), 50 and 200 (seed 777, identical hard events).

## 1. What was created or changed

| path | what |
|---|---|
| `cards/delphes_card_ATLAS_PU50.tcl`, `cards/delphes_card_ATLAS_PU200.tcl` | Delphes 3.5.1 `delphes_card_ATLAS_PileUp.tcl` with three changes (below) |
| `cards/delphes_card_ATLAS_PU0.tcl` | the same card with `MeanPileUp 0`: a μ = 0 control with the *same* detector card |
| `generate_samples.py` | one regex: the slimming also drops `ParticleFlowCandidate` and `Vertex` (diff below) |
| `pileup/MinBias_13TeV.pileup` | min-bias library, 100,000 events, 682 MB |
| `pileup/minbias_13TeV.cmnd`, `pileup/src/minbias2pileup.cc`, `pileup/bin/minbias2pileup`, `pileup/make_minbias_library.sh` | how the library was made (rerun: `./pileup/make_minbias_library.sh`, ~35 s on 4 cores) |
| `pileup/merge_pileup.py` | merge or inspect `.pileup` files (`--info`) |
| `pileup/parts/`, `pileup/tmp/` | the four 25k-event inputs to the library (a 682 MB duplicate that can be deleted) and a 500-event speed test |
| `pileup/validate_pileup.py`, `pileup/validation.json` | the validation below (uses `make_plots.load_sample` and `summary.correct_assignment` unchanged) |
| `pileup/merge_samples.sh` | merges sample directories from parallel jobs into one (hadd + LHE concatenation; tested) |
| `samples/ttbar_bblvqq/test_pu0`, `test_pu50`, `test_pu200` | validation samples, 1000 events each |
| `pileup/failed_attempts/test_pu50_source_path_bug/` | the first, failed test_pu50 attempt (kept, moved out of `samples/` and `work/`) |
| `work/ttbar_bblvqq/Events/test_pu{0,50,200,200_fulloutput}` | MG5 run directories of the tests (logs only; the HepMC files are deleted) |

**Card changes** (header comment in each card):
1. `set PileUpFile` is the absolute path of `pileup/MinBias_13TeV.pileup`. MG5 runs Delphes from `<proc>/Events`, so the relative `MinBias.pileup` is not found.
2. `set MeanPileUp 50 / 200 / 0`.
3. `source trackResolutionATLAS.tcl` is replaced by the content of that file (five `set ...ResolutionFormula` lines). Delphes resolves `source` relative to the directory of the top card. Under MG5 that is `<proc>/Cards`, so the line fails; with an absolute path Delphes builds `../Cards//Users/...`, which also fails. The original card therefore cannot run through MG5 as the README suggests.

Nothing else is changed. As required, the card has `JetFlavorAssociation` and `BTagging` on `JetEnergyScale/jets` → `UniqueObjectFinder/jets` → `Jet` (`BTag`, `Flavor`). Compared with the no-pileup ATLAS card, the only modules it lacks are `ECal`/`HCal` (replaced by the combined `Calorimeter`) and `ChargedHadronFilter` (its output is unused), so nothing needed was added. The ATLAS PileUp card has **no** `PileUpJetID` module (only the CMS cards do). Its pileup handling is `TrackPileUpSubtractor`, `Rho` (FastJet grid median) and `JetPileUpSubtractor` (ρ·A).

**generate_samples.py** (`install_delphes_card`, the only change):
```
-        text, n = re.subn(r"^([ \t]*add Branch \S+ (Track|Tower|EFlowTrack|EFlowPhoton|EFlowNeutralHadron) )",
+        # ParticleFlowCandidate and Vertex exist only in pileup cards (huge at high <mu>).
+        text, n = re.subn(r"^([ \t]*add Branch \S+ (Track|Tower|EFlowTrack|EFlowPhoton|EFlowNeutralHadron"
+                          r"|ParticleFlowCandidate|Vertex) )",
```
The installed default ATLAS card (and the CMS card) is byte-identical before and after this change, with and without `--full-delphes-output`. For the pileup cards, `Tower`, `ParticleFlowCandidate` and `Vertex` are dropped (at μ = 200: 718 of 810 kB/event, see section 5). `Rho` (0.02 MB / 1000 events) is kept as a per-event pileup-density diagnostic.

## 2. Min-bias library

| | shipped `Delphes/MinBias.pileup` | new `pileup/MinBias_13TeV.pileup` |
|---|---|---|
| events | **1,000** | **100,000** |
| generator | Pythia8, presumably Delphes' `examples/Pythia8/generatePileUp.cmnd` (`SoftQCD:all`, Tune:pp 5) | Pythia 8.317 (MG5 HEPTools), `SoftQCD:inelastic`, default tune (Monash, as for the hard events), 13 TeV |
| elastic events (≤ 2 particles) | 22% → MeanPileUp X means about 0.78·X inelastic interactions | 0%: MeanPileUp = mean number of inelastic interactions |
| particles / event (all η) | 143 | 189 (141 within \|η\| < 5) |
| charged dN/dη, \|η\| < 0.5, p_T > 0.1 GeV | 5.3 | 6.3 (ATLAS 13 TeV, events with ≥ 2 charged particles: 6.4) |
| size | 5.2 MB | 682 MB |

`DelphesPythia8` cannot be built here: ROOT and Delphes are x86_64 builds (Rosetta), while the HEPTools Pythia8 is arm64. `minbias2pileup` therefore links Pythia8 with Delphes' own ROOT-free `DelphesPileUpWriter`/`DelphesXDRWriter` and writes, for every final-state particle, exactly what `hepmc2pileup` writes. The merged file was read back with Delphes' `pileup2root` (100,000 events, 189.5 particles/event) and is used by `PileUpMerger` in the runs below.

**Re-use:** a 100k-event production at μ = 200 draws 2×10⁷ min-bias events, so each library event is used about 200 times (each with a new random φ rotation and z/t vertex). Within one event, the same library event appears twice with probability 1 − e^(−μ²/2N) = 18% at μ = 200 (1.2% at μ = 50). With the shipped 1000-event file there would be about 20 repeated pairs in *every* μ = 200 event.

## 3. Timing (this Mac, load ≈ 5 from other jobs; Delphes is single-threaded)

Measured on 1000 events with `--cores 4`, from the MG5 log and Delphes log timestamps:

| | MG5 parton | Pythia8 | **Delphes** | **Delphes / event** | Delphes peak RSS | MG5 gzip of HepMC |
|---|---|---|---|---|---|---|
| no-PU ATLAS card (100k run `run_seed1001`, 10 cores) | 45 s | 248 s (25 ms/event·core) | 741 s | 7.4 ms | – | 941 s |
| μ = 0, PU card | 16 s | 5 s | 16.9 s | 16.9 ms | 455 MB | none (`autoremove`) |
| μ = 50 | 15 s | 5 s | 74.9 s | **75 ms** | 651 MB | skipped (tag_2 rerun, pitfall 2) |
| μ = 200 | 16 s | 4 s | 564.4 s | **564 ms** | 712 MB | 21 s |

Fit through the three points: t_Delphes ≈ 17 + 0.64·μ + 0.0105·μ² ms/event. The cost grows faster than linearly in μ.

**Extrapolation to 100k events (one `generate_samples.py` job, `--cores 4`, with `autoremove`):**

| | MG5 + Pythia8 | Delphes | total |
|---|---|---|---|
| μ = 50 | ≈ 11 min | 7,500 s = 2.1 h | **≈ 2.3 h** |
| μ = 200 | ≈ 11 min | 56,400 s = 15.7 h | **≈ 16 h** |
| μ = 0 control | ≈ 11 min | 28 min | ≈ 40 min |

Without `autoremove`, MG5 spends another 16–35 min per 100k events gzipping a HepMC file that generate_samples.py then deletes. Since Delphes uses one core, **parallel jobs are the way to speed this up** (section 7). With 4 jobs × 25k events (4 cores): μ = 200 ≈ 4.2 h, μ = 50 ≈ 45 min. With 8 jobs × 12.5k: μ = 200 ≈ 2.2 h. Add ~20% for contention between parallel jobs; this was not measured.

## 4. Validation (1000 events each)

`nopu` is the first 1000 events of `run_seed12345` (no-PU ATLAS card, different hard events). `pu0`, `pu50` and `pu200` share identical hard events (seed 777), so their differences are due to pileup only. Statistical errors are binomial or standard errors of the mean.

| quantity | nopu | pu0 (PU card) | pu50 | pu200 |
|---|---|---|---|---|
| ⟨ρ⟩ (\|η\| < 2.5) [GeV] | – | 0.7 | 44.0 | 191.7 |
| reco ⟨n_jets⟩ (p_T > 25, \|η\| < 2.5) | 3.97 ± 0.04 | 3.70 ± 0.04 | 4.45 ± 0.05 | 7.70 ± 0.07 |
| reco ⟨n_b-tag⟩ | 1.22 ± 0.03 | 1.15 ± 0.02 | 1.09 ± 0.03 | 1.02 ± 0.02 |
| fraction with exactly 1 lepton | 0.47 | 0.45 | 0.46 | 0.48 |
| fraction `reco_ok` | 0.239 ± 0.013 | 0.196 ± 0.013 | 0.258 ± 0.014 | 0.332 ± 0.015 |
| MET − p_T(ν): std / 68% half-width [GeV] | 26.1 / 20.9 | 28.4 / 20.2 | 39.0 / 34.1 | 52.2 / 50.4 |
| MET − p_T(ν): median [GeV] | −0.3 | 2.6 | 13.2 | 33.3 |
| χ² jets all correct / `reco_ok` | 0.180 ± 0.025 | 0.214 ± 0.029 | 0.058 ± 0.015 | 0.006 ± 0.004 |
| χ² jets all correct / all events | 0.043 | 0.042 | 0.015 | 0.002 |
| median m_W,had / m_t,had (`reco_ok`) [GeV] | 83.9 / 175.0 | 82.6 / 177.2 | 84.7 / 182.0 | 84.8 / 182.1 |
| gen ⟨n_jets⟩ | 3.88 ± 0.04 | 3.82 ± 0.04 | 3.82 ± 0.04 | 3.82 ± 0.04 |
| GenJet multiplicity (all GenJets) | 5.80 | 5.83 | 5.83 | 5.83 |
| gen MET − p_T(ν) 68% half-width [GeV] | 12.6 | 12.7 | 12.7 | 12.7 |

- All branches that `plotting/io.py` reads are present in every file. `make_plots.load_sample(..., ['parton','gen','reco'], ...)` runs on all samples.
- **Gen level is unaffected by pileup:** GenJets are identical *event by event* in pu0, pu50 and pu200, and agree with the no-pileup sample within statistics. `Particle` holds only the hard event (69 MB per 1000 events in every file).
- The ≥ 4-jet selection becomes pileup-dominated: `reco_ok` *rises* with μ, while the correct χ² assignment falls from 21% to 6% (μ = 50) and 0.6% (μ = 200). The card has no pileup-jet rejection.
- nopu → pu0 shows the effect of the card itself, at μ = 0: fewer jets (−0.27) and lower `reco_ok` (0.24 → 0.20). The no-pileup card's jet energy scale √((3−0.2|η|)²/p_T + 1) (+17% at 25 GeV) becomes 1.0, the calorimeter and MET definitions change, and tracking thresholds go from 0.1 to 0.5 GeV. **Compare pileup samples with a μ = 0 sample made with the PU card, not with the existing no-pileup samples.**

## 5. Output size

| | kB / event | per 100k events | largest branches (MB / 1000 events) |
|---|---|---|---|
| nopu | 69.6 | 7.0 GB | Particle 67.8, GenJet 1.0, Jet 0.7 |
| pu0 | 71.1 | 7.1 GB | Particle 69.0, GenJet 1.0, Jet 0.8 |
| pu50 | 73.8 | 7.4 GB | Particle 69.0, Jet 3.5, GenJet 1.0 |
| pu200 | 92.7 | 9.3 GB | Particle 68.9, Jet 22.4, GenJet 1.0 |

Every other branch is ≤ 0.05 MB / 1000 events. Measured with `--full-delphes-output` (100 events at μ = 200, written to /tmp): 810 kB/event, of which ParticleFlowCandidate is 512, Tower 166 and Vertex 40 kB/event. With the old slimming (Tower only) a 100k μ = 200 sample would be about 64 GB; with the new one it is 9.3 GB.

## 6. Pitfalls

1. **The `source` path made Delphes fail. MG5 and generate_samples.py did not notice:** both exited 0 and left a 1.6 kB `delphes_events.root`. Check every production: `.venv/bin/python -c "import uproot,sys; print(uproot.open(sys.argv[1])['Delphes'].num_entries)" <sample>/delphes_events.root` must equal `--nevents`. The Delphes log is `work/<name>/Events/<run>/tag_*_delphes.log`.
2. **Re-running a failed `--run` name:** generate_samples.py refuses while `work/<name>/Events/<run>` exists. After that directory was moved away, MG5 still knew the run and wrote `tag_2_*` files. That works, but its post-run HepMC gzip was skipped. Use fresh run names.
3. **Disk:** during a run the uncompressed HepMC takes 252 MB per 1000 events (25 GB per 100k events in total, however the jobs are split) in `work/<name>/Events/<run>/`. About 227 GB are free now.
4. **Memory:** Delphes peaks at ≤ 0.72 GB at μ = 200, independent of the number of events, so 8 parallel jobs are fine.
5. **Physics content of the card** (kept as instructed): no PU-jet ID or JVT, and MET is built from all tracks plus neutral energy flow, so pileup tracks enter it (median MET bias +13 / +33 GeV at μ = 50 / 200). Both the reco jets and the GenJets are anti-k_T **R = 0.6**, in both cards; the README says 0.4.
6. `pileup2root` takes `output input`, in that order. A wrong order is refused (TFile CREATE), so no harm was done.
7. The README pileup note (`--delphes-card .../delphes_card_ATLAS_PileUp.tcl`) does not work through MG5 (pitfall 1 plus the relative `MinBias.pileup`). Use the cards above.
8. `.gitignore` does not cover `pileup/*.pileup` or `pileup/parts/` (1.4 GB). Do not `git add` them.
9. MG5 keeps `--set pythia8_card ...` in `work/<name>/Cards/pythia8_card.dat`. That is harmless, because generate_samples.py sets `HEPMCoutput:file auto` on every launch before any `--set`.

## 7. Commands

All commands work in zsh and bash.

Recommended settings: the same seed at every μ, so the hard events are identical and the comparison is paired; an absolute card path (metadata.json records it verbatim); `autoremove`; jobs in one `--name` run one after another (one MG5 process directory supports one launch at a time).

**Single job per μ (100k events; 2.3 h at μ = 50, 16 h at μ = 200, 40 min for the μ = 0 control):**
```bash
cd /Users/TMHONG/Claude-code/ttbar_bblvqq
C=/Users/TMHONG/Claude-code/ttbar_bblvqq/cards
for MU in 0 50 200; do
  .venv/bin/python generate_samples.py --process ttbar --nevents 100000 --seed 5000 --run pu${MU}_seed5000 \
      --delphes-card $C/delphes_card_ATLAS_PU${MU}.tcl --cores 4 \
      --set "pythia8_card HEPMCoutput:file autoremove" > work/pu${MU}_seed5000.out 2>&1
done
# -> samples/ttbar_bblvqq/pu{0,50,200}_seed5000/
```

**Parallel at μ = 200 (4 jobs × 25k events, about 4.2 h on 4 cores; each `--name` gets its own MG5 directory, created in 5 s), then merge:**
```bash
cd /Users/TMHONG/Claude-code/ttbar_bblvqq
C=/Users/TMHONG/Claude-code/ttbar_bblvqq/cards
for j in 1 2 3 4; do
  .venv/bin/python generate_samples.py --process ttbar --name ttbar_bblvqq_pu200_j$j --nevents 25000 \
      --seed $((5000+j)) --run pu200_seed$((5000+j)) --delphes-card $C/delphes_card_ATLAS_PU200.tcl \
      --cores 1 --set "pythia8_card HEPMCoutput:file autoremove" > work/pu200_j$j.out 2>&1 &
done; wait
pileup/merge_samples.sh samples/ttbar_bblvqq/pu200_seed5001-5004 \
    samples/ttbar_bblvqq_pu200_j1/pu200_seed5001 samples/ttbar_bblvqq_pu200_j2/pu200_seed5002 \
    samples/ttbar_bblvqq_pu200_j3/pu200_seed5003 samples/ttbar_bblvqq_pu200_j4/pu200_seed5004
```
This recipe was tested end to end with 2 concurrent jobs × 200 events at μ = 50 (`--workdir`/`--outdir` in /tmp). Each job has about 1.5 min of fixed overhead (new process directory and first compilation); with 2 jobs running together, Delphes took 90 ms/event (75 ms alone). `merge_samples.sh` gave one 400-event sample.
For μ = 50 and the μ = 0 control, replace `pu200`/`PU200` with `pu50`/`PU50` or `pu0`/`PU0` and keep the seeds. For hh, use `--process hh` (default name `hh_bblvqq`). Validate with:
`.venv/bin/python pileup/validate_pileup.py pu0=<dir> pu50=<dir> pu200=<dir>` (append `:N` to a path to read only N events). It also reports whether the GenJets are identical event by event.
