#!/usr/bin/env bash
# Go/no-go benchmark samples (MG5 -> Pythia8 -> Delphes ATLAS, no pileup), run one after another on all cores.
set -u
cd "$(dirname "$0")"                         # code/generation/ (generate_samples.py, cards/)
PY=../../.venv/bin/python
run() { echo "=== $(date '+%H:%M:%S') $*"; "$PY" generate_samples.py "$@" || echo "FAILED: $*"; }
run --process etat --nevents 80000 --seed 4001 --run run_seed4001
run --process stop --nevents 60000 --seed 4002 --run run_seed4002 --delphes-card cards/delphes_card_ATLAS_lsp.tcl
run --process ttnospin_lt --nevents 30000 --seed 4003 --run run_seed4003
run --process ttnospin_lb --nevents 30000 --seed 4004 --run run_seed4004
echo "=== $(date '+%H:%M:%S') ALL DONE"
