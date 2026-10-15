#!/bin/bash
# 80k more HH(170) events (4 x 20k, seeds 2171-2174), same settings as run_m170_seed2170, then Delta-R matching.
cd "$(dirname "$0")/../.."                   # the repo top
for seed in 2171 2172 2173 2174; do
  echo "=== seed $seed $(date +%H:%M:%S)"
  .venv/bin/python code/generation/generate_samples.py --process hh --run run_m170_seed$seed --nevents 20000 --seed $seed --cores 8 \
    --set "param_card mass 25 170" --set "param_card decay 25 0.38" --set "pythia8_card HEPMCoutput:file autoremove" \
    > work/hh170_seed$seed.out 2>&1 || { echo "generation failed for $seed"; tail -5 work/hh170_seed$seed.out; exit 1; }
  tail -3 work/hh170_seed$seed.out
  OMP_NUM_THREADS=3 .venv/bin/python code/analysis/study_truthmatch/build_matched.py samples/hh_bblvqq/run_m170_seed$seed \
    > results/analysis/study_truthmatch/build_matched_seed$seed.log 2>&1 && grep -E 'n_all"|n_matched_presel' results/analysis/study_truthmatch/build_matched_seed$seed.log
done
echo "=== ALL DONE $(date +%H:%M:%S)"
