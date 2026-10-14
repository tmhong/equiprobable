#!/usr/bin/env bash
# Install everything needed by generate_samples.py and make_plots.py:
#   - python venv (<repo>/.venv) with uproot, awkward, vector, numpy, matplotlib
#   - MadGraph5_aMC@NLO, Pythia8 (+ MG5 interface, via MG5's installer)
#   - Delphes (built with CMake against your ROOT; MG5's own Makefile build
#     fails to link with recent ROOT versions)
# Safe to re-run: finished steps are skipped.
#
# Requirements: ROOT (root-config in PATH), gfortran, a C++ compiler, cmake, curl.
#
# Usage: code/setup_tools.sh [TOOLS_DIR]      (default: <repo>/tools)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # code/
REPO="$(dirname "$HERE")"                                  # repository top: tools/, .venv, requirements.txt
TOOLS="${1:-$REPO/tools}"
MG_VERSION=3.6.7
DELPHES_VERSION=3.5.1
PYTHON="${PYTHON:-python3}"
NCORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

MG_DIR="$TOOLS/MG5_aMC_v${MG_VERSION//./_}"
VENV="$REPO/.venv"

for exe in root-config gfortran cmake curl "$PYTHON"; do
  command -v "$exe" >/dev/null || { echo "ERROR: '$exe' not found in PATH"; exit 1; }
done
# shellcheck disable=SC1091
source "$(root-config --prefix)/bin/thisroot.sh"
mkdir -p "$TOOLS"

echo "=== 1/4 python environment ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$REPO/requirements.txt"

echo "=== 2/4 MadGraph5_aMC@NLO $MG_VERSION ($MG_DIR)"
if [ ! -x "$MG_DIR/bin/mg5_aMC" ]; then
  curl -sSL -o "$TOOLS/mg5.tgz" \
    "https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v${MG_VERSION}.tar.gz"
  tar xzf "$TOOLS/mg5.tgz" -C "$TOOLS" && rm "$TOOLS/mg5.tgz"
fi
CFG="$MG_DIR/input/mg5_configuration.txt"
set_cfg() {  # set_cfg key value : set (and uncomment) an option in mg5_configuration.txt
  if grep -qE "^#? *$1 *=" "$CFG"; then
    sed -i.bak -E "s|^#? *$1 *=.*|$1 = $2|" "$CFG" && rm -f "$CFG.bak"
  else
    echo "$1 = $2" >> "$CFG"
  fi
}
set_cfg auto_update 0
set_cfg automatic_html_opening False
set_cfg notification_center False

echo "=== 3/4 Pythia8 + MG5 interface (takes ~10-20 min the first time)"
if [ ! -d "$MG_DIR/HEPTools/MG5aMC_PY8_interface" ]; then
  printf "install pythia8\nexit\n" > "$TOOLS/install_pythia8.mg5"
  (cd "$MG_DIR" && "$VENV/bin/python" bin/mg5_aMC "$TOOLS/install_pythia8.mg5" > "$TOOLS/install_pythia8.log" 2>&1) || true
  [ -d "$MG_DIR/HEPTools/MG5aMC_PY8_interface" ] || { echo "Pythia8 install failed, see $TOOLS/install_pythia8.log"; exit 1; }
fi

echo "=== 4/4 Delphes $DELPHES_VERSION"
DEL="$MG_DIR/Delphes"
if [ ! -x "$DEL/DelphesHepMC2" ]; then
  if [ ! -f "$DEL/CMakeLists.txt" ]; then
    curl -sSL -o "$TOOLS/delphes.tgz" "http://cp3.irmp.ucl.ac.be/downloads/Delphes-${DELPHES_VERSION}.tar.gz"
    mkdir -p "$DEL" && tar xzf "$TOOLS/delphes.tgz" -C "$DEL" --strip-components 1 && rm "$TOOLS/delphes.tgz"
  fi
  mkdir -p "$DEL/build"
  (cd "$DEL/build" && cmake .. -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 && make -j"$NCORES" > make.log 2>&1) \
    || { echo "Delphes build failed, see $DEL/build/{cmake,make}.log"; exit 1; }
  # MG5 expects the executables directly in the Delphes directory
  for e in DelphesHepMC2 DelphesHepMC3 DelphesLHEF DelphesSTDHEP DelphesROOT; do
    ln -sf "build/readers/$e" "$DEL/$e"
  done
fi
set_cfg delphes_path "$DEL"

echo
echo "Done.  MG5:     $MG_DIR"
echo "       Delphes: $DEL"
echo "Next:  $VENV/bin/python $HERE/generate_samples.py --nevents 10000"
