#!/usr/bin/env bash
# regenerate every figure used by the slides (slide style; split multi-panel figures)
cd "$(dirname "$0")"
PY=.venv/bin/python
TT=samples/ttbar_bblvqq/run_seed1001; TI=samples/ttbar_bblvqq/run_seed12345; HH=samples/hh_bblvqq/run_seed2001
S="--train $TT --test $TI --hh $HH"
set -x
rm -rf plots plots_hh plots_tt_vs_hh
$PY make_plots.py --sample nominal=$TI > /dev/null
$PY make_plots.py --sample hh=$HH --out plots_hh > /dev/null
$PY make_plots.py --sample tt=$TI --sample hh=$HH --out plots_tt_vs_hh > /dev/null
$PY nu_pz_plot.py --sample $TI > /dev/null
$PY hh_plots.py --sample $HH > /dev/null
$PY lep_had_compare.py --sample $TT | tail -4
rm -f plots/nu_pz.png plots/lep_vs_had.png plots/lep_vs_had_parton.png plots/lep_vs_had_gen.png plots/lep_vs_had_reco.png plots/lep_vs_had_reco_correct_jets.png
for d in osde osde_sh osde_ext; do $PY osde_scores_plot.py --dir $d; $PY osde_simplify.py $S --osde $d --replot; done
$PY osde_truncation_check.py --train $TT > /dev/null
$PY paper_style_plots.py --dir osde > /dev/null 2>&1
$PY paper_style_plots.py --dir osde_ext > /dev/null 2>&1
$PY acceptance_plots.py --sample tt=$TT --sample hh=$HH | tail -2
$PY figures/make_angle_diagram.py
$PY basis_compare.py | tail -1
$PY osde_compare.py --train $TT --test $TI | tail -1
echo REGEN_DONE
