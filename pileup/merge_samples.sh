#!/bin/bash
# Merge sample directories from parallel generate_samples.py jobs into one sample directory
# that make_plots.load_sample reads like any other (event i of the LHE = event i of Delphes).
#   pileup/merge_samples.sh <out_dir> <sample_dir_1> <sample_dir_2> ...
# delphes_events.root: hadd (keeps the input order); parton_events.lhe.gz: header of the first
# file + the <event> blocks of all files in the same order; metadata.json: summed nevents,
# mean cross section, list of inputs.
set -euo pipefail
out=$1; shift
[ -e "$out/delphes_events.root" ] && { echo "$out already has delphes_events.root"; exit 1; }
mkdir -p "$out"
roots=(); for d in "$@"; do roots+=("$d/delphes_events.root"); done
hadd "$out/delphes_events.root" "${roots[@]}" > "$out/hadd.log" 2>&1
{ gunzip -c "$1/parton_events.lhe.gz" | sed '/<\/LesHouchesEvents>/d'
  for d in "${@:2}"; do gunzip -c "$d/parton_events.lhe.gz" | sed -n '/<event>/,/<\/event>/p'; done
  echo '</LesHouchesEvents>'; } | gzip > "$out/parton_events.lhe.gz"
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/../.venv/bin/python" - "$out" "$@" <<'EOF'
import json, sys, gzip, uproot
out, ins = sys.argv[1], sys.argv[2:]
metas = [json.load(open(f"{d}/metadata.json")) for d in ins]
m = dict(metas[0])
m["nevents"] = sum(x["nevents"] for x in metas)
m["cross_section_pb"] = sum(x["cross_section_pb"] * x["nevents"] for x in metas) / m["nevents"]
m["seed"] = [x["seed"] for x in metas]
m["merged_from"] = ins
m["files"] = {"parton_events.lhe.gz": f"{out}/parton_events.lhe.gz", "delphes_events.root": f"{out}/delphes_events.root"}
json.dump(m, open(f"{out}/metadata.json", "w"), indent=2)
n_lhe = sum(1 for l in gzip.open(f"{out}/parton_events.lhe.gz", "rt") if l.startswith("<event>"))
n_root = uproot.open(f"{out}/delphes_events.root")["Delphes"].num_entries
print(f"merged {len(ins)} samples into {out}: {n_lhe} LHE events, {n_root} Delphes events")
assert n_lhe == n_root == m["nevents"], "event counts differ"
EOF
