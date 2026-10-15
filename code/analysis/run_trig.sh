#!/usr/bin/env bash
# fit the parton+trigger level for all three bases (angle caches are rebuilt with the new level)
set -e
cd "$(dirname "$0")/../.."                   # repository top: code/, results/, samples/
PY=.venv/bin/python
S="--train samples/ttbar_bblvqq/run_seed1001 --test samples/ttbar_bblvqq/run_seed12345 --hh samples/hh_bblvqq/run_seed2001"
$PY code/analysis/osde_fit.py $S --levels parton,parton_trig,gen,reco --rebuild-cache --out results/analysis/osde_tmp_cache_build
rm -rf results/analysis/osde_tmp_cache_build
$PY code/analysis/osde_order_scan.py --train samples/ttbar_bblvqq/run_seed1001
for b in paper sh extended; do
  out=results/analysis/osde; [ $b = sh ] && out=results/analysis/osde_sh; [ $b = extended ] && out=results/analysis/osde_ext
  $PY code/analysis/osde_fit.py $S --basis $b --levels parton_trig --out $out
  $PY code/analysis/osde_simplify.py $S --osde $out --levels parton_trig | grep simplify
done
echo TRIG_DONE
