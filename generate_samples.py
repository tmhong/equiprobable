#!/usr/bin/env python3
"""
Generate pp -> ttbar -> b b W W -> b b l nu q q' samples with MadGraph5_aMC@NLO,
shower with Pythia8, and run Delphes with the ATLAS card (no pileup).

Outputs (copied to <outdir>/<name>/<run>/):
  parton_events.lhe.gz   MG5 parton-level (generator-level) events
  delphes_events.root    Delphes reco (Jet, Electron, Muon, MissingET, ...) plus
                         generator-level GenJet, GenMissingET and Particle branches
  pythia8_events.hepmc[.gz] (only with --keep-hepmc) full Pythia8 record
  metadata.json          cross section, settings, file paths

Examples:
  # full chain, 10k events at 13 TeV
  python generate_samples.py --nevents 10000

  # parton level only (fast), different seed and run name
  python generate_samples.py --stage parton --nevents 5000 --seed 7 --run run_seed7

  # include taus in the W -> l nu decay, 13.6 TeV
  python generate_samples.py --leptons e,mu,ta --sqrts 13600 --name ttbar_bblvqq_tau

  # just write the MG5 command files and show them
  python generate_samples.py --dry-run
"""
import argparse
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MG5 = os.path.join(HERE, "tools", "MG5_aMC_v3_6_7")

LEPTON_DEFS = {
    # name : (charged lepton+, neutrino)
    "e": ("e+", "ve"),
    "mu": ("mu+", "vm"),
    "ta": ("ta+", "vt"),
}


def anti(p):
    """Antiparticle name in MG5 sm syntax for the leptons used here."""
    if p.endswith("+"):
        return p[:-1] + "-"
    if p.endswith("-"):
        return p[:-1] + "+"
    return p + "~"


def process_commands(args, proc_dir):
    """MG5 commands that define the process and write the process directory."""
    flavours = [f.strip() for f in args.leptons.split(",") if f.strip()]
    bad = [f for f in flavours if f not in LEPTON_DEFS]
    if bad or not flavours:
        sys.exit(f"--leptons must be a comma list of {list(LEPTON_DEFS)}; got {args.leptons}")
    lp = " ".join(LEPTON_DEFS[f][0] for f in flavours)
    vl = " ".join(LEPTON_DEFS[f][1] for f in flavours)
    lm = " ".join(anti(LEPTON_DEFS[f][0]) for f in flavours)
    vlb = " ".join(anti(LEPTON_DEFS[f][1]) for f in flavours)

    model = args.model or PROCESSES[args.process]["model"]
    return f"""\
set automatic_html_opening False
set notification_center False
import model {model}
define lep+ = {lp}
define lep- = {lm}
define nul = {vl}
define nul~ = {vlb}
define q = u c d s u~ c~ d~ s~
{PROCESSES[args.process]["generate"]}
output {proc_dir} -nojpeg
"""


# Process definitions (MG5 syntax).  Both use the lep+/lep-/nul/nul~/q multiparticles.
PROCESSES = {
    "ttbar": dict(
        model="sm",
        label="p p > t t~ > b b~ l nu q q'",
        generate="""\
# t -> b W+(-> l+ nu), t~ -> b~ W-(-> q q')
generate p p > t t~, (t > b w+, w+ > lep+ nul), (t~ > b~ w-, w- > q q)
# t -> b W+(-> q q'), t~ -> b~ W-(-> l- nu~)
add process p p > t t~, (t > b w+, w+ > q q), (t~ > b~ w-, w- > lep- nul~)"""),
    # Higgs-pair production via the effective ggh/gghh couplings (heft, LO, m_t -> inf).
    # h -> W W* has one off-shell W, so h -> l nu q q is generated as a 4-body decay.
    "hh": dict(
        model="heft",
        label="g g > h h > b b~ l nu q q'",
        generate="""\
generate g g > h h, (h > b b~), (h > lep+ nul q q)
add process g g > h h, (h > b b~), (h > lep- nul~ q q)"""),
}


def launch_commands(args, run_name):
    """madevent commands that launch event generation (+ shower + detector)."""
    ebeam = args.sqrts / 2.0
    shower = "Pythia8" if args.stage in ("shower", "delphes") else "OFF"
    detector = "Delphes" if args.stage == "delphes" else "OFF"
    lines = [
        "set automatic_html_opening False",
        "set notification_center False",
        "set run_mode 2",
        f"set nb_core {args.cores}",
        f"launch {run_name}",
        f"shower={shower}",
        f"detector={detector}",
        "analysis=OFF",
        "madspin=OFF",
        "done",
        f"set run_card nevents {args.nevents}",
        f"set run_card iseed {args.seed}",
        f"set run_card ebeam1 {ebeam}",
        f"set run_card ebeam2 {ebeam}",
        # Everything is a decay product here; keep generator-level cuts off
        # so the samples are unbiased for validation.
        "set run_card cut_decays False",
        f"set run_card use_syst {args.use_syst}",
    ]
    if args.stage in ("shower", "delphes"):
        lines.append(f"set pythia8_card Main:numberOfEvents {args.nevents}")
        # MG5 needs the HepMC file on disk to run Delphes (its "fifo" mode does not
        # feed Delphes); it is deleted afterwards unless --keep-hepmc is given.
        lines.append("set pythia8_card HEPMCoutput:file auto")
    for extra in args.set:
        lines.append(f"set {extra}")
    lines += ["done", "exit"]
    return "\n".join(lines) + "\n"


def run_mg5(args, exe, cmd_text, cmd_file, log_file):
    """Run an MG5 executable (bin/mg5_aMC or <proc>/bin/madevent) on a command file."""
    with open(cmd_file, "w") as f:
        f.write(cmd_text)
    cmd = [args.python, exe, cmd_file]
    print(f"[generate] running: {' '.join(cmd)}\n[generate] log: {log_file}")
    t0 = time.time()
    with open(log_file, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, cwd=os.path.dirname(cmd_file))
    print(f"[generate] finished in {time.time() - t0:.0f}s (exit code {proc.returncode})")
    # MG5 often returns 0 even on failure; also scan the log for errors.
    with open(log_file) as f:
        text = f.read()
    if proc.returncode != 0 or re.search(r"Command \".*\" interrupted with error|Error detected in", text):
        print("".join(text.splitlines(True)[-40:]))
        sys.exit(f"[generate] MG5 reported a problem, see {log_file}")


def install_delphes_card(args, proc_dir):
    card = args.delphes_card or os.path.join(args.mg5, "Delphes", "cards", "delphes_card_ATLAS.tcl")
    if not os.path.isfile(card):
        sys.exit(f"[generate] Delphes card not found: {card}\n"
                 "Install Delphes (./setup_tools.sh) or pass --delphes-card.")
    with open(card) as f:
        text = f.read()
    if re.search(r"^\s*module\s+PileUpMerger", text, re.M):
        print("[generate] WARNING: the Delphes card contains pileup merging")
    if not args.full_delphes_output:
        # Same detector simulation, but don't store the bulky low-level
        # collections (tracks, towers, energy-flow objects): ~30% smaller files.
        text, n = re.subn(r"^([ \t]*add Branch \S+ (Track|Tower|EFlowTrack|EFlowPhoton|EFlowNeutralHadron) )",
                          r"#\1", text, flags=re.M)
        print(f"[generate] slimmed Delphes output ({n} low-level branches dropped)")
    with open(os.path.join(proc_dir, "Cards", "delphes_card.dat"), "w") as f:
        f.write(text)
    print(f"[generate] Delphes card: {card}")


def read_cross_section(run_dir):
    """Cross section (pb) and error from the MG5 run banner."""
    for banner in sorted(glob.glob(os.path.join(run_dir, "*banner.txt"))):
        with open(banner) as f:
            text = f.read()
        m = re.search(r"Integrated weight \(pb\)\s*:\s*([0-9.eE+-]+)", text)
        if m:
            return float(m.group(1))
    lhe = os.path.join(run_dir, "unweighted_events.lhe.gz")
    if os.path.exists(lhe):
        with gzip.open(lhe, "rt") as f:
            for line in f:
                if line.strip().startswith("<init>"):
                    next(f)
                    return float(next(f).split()[0])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--process", choices=sorted(PROCESSES), default="ttbar")
    p.add_argument("--name", default=None, help="process / sample name (default <process>_bblvqq)")
    p.add_argument("--run", default=None, help="run name (default run_seed<seed>)")
    p.add_argument("--nevents", type=int, default=10000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sqrts", type=float, default=13000.0, help="centre-of-mass energy in GeV")
    p.add_argument("--leptons", default="e,mu", help="W -> l nu flavours, comma list of e,mu,ta")
    p.add_argument("--model", default=None, help="UFO model (default: sm for ttbar, heft for hh)")
    p.add_argument("--stage", choices=["parton", "shower", "delphes"], default="delphes",
                   help="stop after MG5 (parton), Pythia8 (shower) or Delphes (default)")
    p.add_argument("--delphes-card", default=None, help="default: Delphes ATLAS card (no pileup)")
    p.add_argument("--full-delphes-output", action="store_true",
                   help="keep Track/Tower/EFlow branches in the Delphes output (large files)")
    p.add_argument("--cores", type=int, default=os.cpu_count() or 1)
    p.add_argument("--use-syst", default="False", choices=["True", "False"],
                   help="compute scale/PDF weights (slower)")
    p.add_argument("--set", action="append", default=[],
                   help="extra MG5 'set' command, e.g. --set 'run_card ptj 20' (repeatable)")
    p.add_argument("--keep-hepmc", action="store_true", help="copy the Pythia8 HepMC file to outdir")
    p.add_argument("--mg5", default=DEFAULT_MG5, help="MG5_aMC installation directory")
    p.add_argument("--python", default=sys.executable, help="python used to run MG5")
    p.add_argument("--workdir", default=os.path.join(HERE, "work"))
    p.add_argument("--outdir", default=os.path.join(HERE, "samples"))
    p.add_argument("--dry-run", action="store_true", help="write and print MG5 commands only")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)    # progress visible when logging to a file

    args.mg5 = os.path.abspath(args.mg5)
    args.name = args.name or f"{args.process}_bblvqq"
    run_name = args.run or f"run_seed{args.seed}"
    os.makedirs(args.workdir, exist_ok=True)
    proc_dir = os.path.join(os.path.abspath(args.workdir), args.name)

    if not os.path.isfile(os.path.join(args.mg5, "bin", "mg5_aMC")):
        sys.exit(f"[generate] MG5 not found at {args.mg5}. Run ./setup_tools.sh or pass --mg5.")

    proc_cmds = process_commands(args, proc_dir)
    launch_cmds = launch_commands(args, run_name)
    if args.dry_run:
        print("# ---- process definition ----\n" + proc_cmds)
        print(f"# ---- launch (run with {proc_dir}/bin/madevent) ----\n" + launch_cmds)
        return

    # 1) Write the process directory (only once per --name)
    if not os.path.isdir(proc_dir):
        run_mg5(args, os.path.join(args.mg5, "bin", "mg5_aMC"), proc_cmds, os.path.join(args.workdir, f"{args.name}_proc.mg5"),
                os.path.join(args.workdir, f"{args.name}_proc.log"))
    else:
        print(f"[generate] reusing existing process directory {proc_dir}")

    run_dir = os.path.join(proc_dir, "Events", run_name)
    if os.path.exists(run_dir):
        sys.exit(f"[generate] {run_dir} already exists; choose another --run or --seed")

    # 2) Detector card, then launch
    if args.stage == "delphes":
        install_delphes_card(args, proc_dir)
    # Launch through the process directory's own madevent interface; this avoids
    # the MG5 master interface's version check on exit.
    run_mg5(args, os.path.join(proc_dir, "bin", "madevent"), launch_cmds, os.path.join(args.workdir, f"{args.name}_{run_name}.mg5"),
            os.path.join(args.workdir, f"{args.name}_{run_name}.log"))

    # 3) Collect outputs
    out = os.path.join(os.path.abspath(args.outdir), args.name, run_name)
    os.makedirs(out, exist_ok=True)
    files = {}

    def collect(pattern, dest, move=False):
        """Copy (or move, for big files) the newest MG5 output matching pattern."""
        hits = sorted(glob.glob(os.path.join(run_dir, pattern)), key=os.path.getmtime)
        if not hits:
            sys.exit(f"[generate] expected output {pattern} not found in {run_dir}")
        src = hits[-1]
        if dest.endswith("*"):         # keep the source's extension(s), e.g. .hepmc.gz
            base = os.path.basename(src)
            dest = dest[:-1] + base[base.index("."):]
        (shutil.move if move else shutil.copy)(src, os.path.join(out, dest))
        files[dest] = os.path.join(out, dest)

    collect("unweighted_events.lhe.gz", "parton_events.lhe.gz")
    if args.stage == "delphes":
        collect("*delphes_events.root", "delphes_events.root", move=True)
    if args.stage in ("shower", "delphes"):
        if args.keep_hepmc or args.stage == "shower":
            collect("*pythia8_events.hepmc*", "pythia8_events*", move=True)   # keeps .hepmc / .hepmc.gz
        for h in glob.glob(os.path.join(run_dir, "*pythia8_events.hepmc*")):
            os.remove(h)               # large (~90 kB/event) and no longer needed

    xsec = read_cross_section(run_dir)
    meta = {
        "process": PROCESSES[args.process]["label"],
        "leptons": args.leptons,
        "sqrts_GeV": args.sqrts,
        "nevents": args.nevents,
        "seed": args.seed,
        "cross_section_pb": xsec,
        "stage": args.stage,
        "delphes_card": args.delphes_card or "delphes_card_ATLAS.tcl (no pileup)",
        "mg5_run_dir": run_dir,
        "files": files,
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[generate] cross section = {xsec} pb")
    print(f"[generate] outputs in {out}:")
    for k, v in files.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
