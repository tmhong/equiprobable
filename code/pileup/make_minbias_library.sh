#!/bin/bash
# Build the 13 TeV minimum-bias pile-up library for Delphes (results/pileup/MinBias_13TeV.pileup).
#   ./code/pileup/make_minbias_library.sh [jobs=4] [events_per_job=25000] [first_seed=90001]
# 4 x 25k events: ~35 s wall on 4 cores, 682 MB.  Pythia 8.317 (MG5 HEPTools), SoftQCD:inelastic.
# The jobs write results/pileup/parts/*.pileup; merge_pileup.py concatenates them (byte copy + new index).
# The generator source src/minbias2pileup.cc and the Pythia card minbias_13TeV.cmnd are taken from code/pileup/
# if they are there, else from results/pileup/; the binary goes to code/pileup/bin/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)          # code/pileup
REPO=$(cd "$HERE/../.." && pwd)
OUT=$REPO/results/pileup
SRC=$HERE/src/minbias2pileup.cc; [ -e "$SRC" ] || SRC=$OUT/src/minbias2pileup.cc
CMND=$HERE/minbias_13TeV.cmnd; [ -e "$CMND" ] || CMND=$OUT/minbias_13TeV.cmnd
MG5=$REPO/tools/MG5_aMC_v3_6_7
D=$MG5/Delphes
P=$MG5/HEPTools/pythia8
JOBS=${1:-4}; NEV=${2:-25000}; SEED0=${3:-90001}
export PYTHIA8DATA=$P/share/Pythia8/xmldoc

# Native (arm64) Pythia8 + Delphes' ROOT-free pile-up writer.  DelphesPythia8 cannot be
# built here: ROOT/Delphes are x86_64 builds and the HEPTools Pythia8 is arm64.
mkdir -p "$HERE/bin" "$OUT/parts"
c++ -O2 -std=c++17 -I"$D" -I"$P/include" "$SRC" \
    "$D/classes/DelphesPileUpWriter.cc" "$D/classes/DelphesXDRWriter.cc" \
    -o "$HERE/bin/minbias2pileup" -L"$P/lib" -lpythia8 -Wl,-rpath,"$P/lib" -ldl

parts=()
for ((j = 0; j < JOBS; j++)); do
  s=$((SEED0 + j)); f=$OUT/parts/minbias_13TeV_seed$s.pileup; parts+=("$f")
  "$HERE/bin/minbias2pileup" "$CMND" "$NEV" "$s" "$f" > "${f%.pileup}.log" 2>&1 &
done
wait
"$REPO/.venv/bin/python" "$HERE/merge_pileup.py" -o "$OUT/MinBias_13TeV.pileup" "${parts[@]}"
"$REPO/.venv/bin/python" "$HERE/merge_pileup.py" --info "$OUT/MinBias_13TeV.pileup"
