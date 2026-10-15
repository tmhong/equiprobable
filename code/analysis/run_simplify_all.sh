#!/usr/bin/env bash
# re-run the truncation for every basis and level (the saved _simplified f = CV-best truncation)
cd "$(dirname "$0")/../.."                   # repository top: code/, results/, samples/
S="--train samples/ttbar_bblvqq/run_seed1001 --test samples/ttbar_bblvqq/run_seed12345 --hh samples/hh_bblvqq/run_seed2001"
for d in osde osde_sh osde_ext; do
  .venv/bin/python code/analysis/osde_simplify.py $S --osde results/$d --levels parton,parton_trig,gen,reco | grep simplify
done
.venv/bin/python code/analysis/basis_compare.py
.venv/bin/python code/analysis/osde_compare.py --train samples/ttbar_bblvqq/run_seed1001 --test samples/ttbar_bblvqq/run_seed12345 | tail -16
echo SIMPLIFY_DONE
