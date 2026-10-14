#!/usr/bin/env bash
# fit the parton+trigger level for all three bases (angle caches are rebuilt with the new level)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
S="--train samples/ttbar_bblvqq/run_seed1001 --test samples/ttbar_bblvqq/run_seed12345 --hh samples/hh_bblvqq/run_seed2001"
$PY osde_fit.py $S --levels parton,parton_trig,gen,reco --rebuild-cache --out osde_tmp_cache_build
rm -rf osde_tmp_cache_build
$PY osde_order_scan.py --train samples/ttbar_bblvqq/run_seed1001
for b in paper sh extended; do
  out=osde; [ $b = sh ] && out=osde_sh; [ $b = extended ] && out=osde_ext
  $PY osde_fit.py $S --basis $b --levels parton_trig --out $out
  $PY osde_simplify.py $S --osde $out --levels parton_trig | grep simplify
done
echo TRIG_DONE
