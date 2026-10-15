#!/usr/bin/env bash
# regenerate every figure used by the slides (slide style; split multi-panel figures)
cd "$(dirname "$0")/../.."                   # repository top: code/, results/, samples/
PY=.venv/bin/python
TT=samples/ttbar_bblvqq/run_seed1001; TI=samples/ttbar_bblvqq/run_seed12345; HH=samples/hh_bblvqq/run_seed2001
S="--train $TT --test $TI --hh $HH"
set -x
rm -rf results/plots results/plots_hh results/plots_tt_vs_hh
$PY code/plotting/make_plots.py --sample nominal=$TI > /dev/null
$PY code/plotting/make_plots.py --sample hh=$HH --out results/plots_hh > /dev/null
$PY code/plotting/make_plots.py --sample tt=$TI --sample hh=$HH --out results/plots_tt_vs_hh > /dev/null
$PY code/plotting/nu_pz_plot.py --sample $TI > /dev/null
$PY code/plotting/hh_plots.py --sample $HH > /dev/null
$PY code/plotting/lep_had_compare.py --sample $TT | tail -4
rm -f results/plots/nu_pz.png results/plots/lep_vs_had.png results/plots/lep_vs_had_parton.png results/plots/lep_vs_had_gen.png results/plots/lep_vs_had_reco.png results/plots/lep_vs_had_reco_correct_jets.png
for d in osde osde_sh osde_ext; do $PY code/plotting/osde_scores_plot.py --dir results/$d; $PY code/analysis/osde_simplify.py $S --osde results/$d --replot; done
$PY code/analysis/osde_truncation_check.py --train $TT > /dev/null
$PY code/plotting/paper_style_plots.py --dir results/osde > /dev/null 2>&1
$PY code/plotting/paper_style_plots.py --dir results/osde_ext > /dev/null 2>&1
$PY code/plotting/acceptance_plots.py --sample tt=$TT --sample hh=$HH | tail -2
$PY code/plotting/angle_diagram/make_angle_diagram.py
$PY code/analysis/basis_compare.py | tail -1
$PY code/analysis/osde_compare.py --train $TT --test $TI | tail -1
echo REGEN_DONE
