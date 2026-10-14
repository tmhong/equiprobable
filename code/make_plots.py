#!/usr/bin/env python3
"""
Validation plots for the ttbar -> bblvqq samples.

A "sample" is a directory written by generate_samples.py (containing
parton_events.lhe.gz and/or delphes_events.root), or a single .lhe(.gz) or
.root file.  Give one or more as LABEL=PATH; all are overlaid, with a ratio
to the first one.

Examples:
  python make_plots.py --sample nominal=samples/ttbar_bblvqq/run_seed12345
  python make_plots.py --sample s1=samples/ttbar_bblvqq/run_seed1 --sample s2=samples/ttbar_bblvqq/run_seed2 \\
                       --levels reco --norm xsec --lumi 140
  python make_plots.py --sample nominal=samples/ttbar_bblvqq/run_seed12345 --only top_had_mass,lep_pt
"""
import argparse
import json
import os
import runpy
import sys

import awkward as ak
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plotting import angles, histo, io, physics  # noqa: E402

CHUNK = 10000           # Delphes events read at a time
LEVEL_STYLE = {"parton": ":", "parton_trig": "-.", "gen": "--", "reco": "-"}


def load_sample(label, path, levels, max_events, angle_opts):
    """Return {level: Event} (with the ten paper angles added) and the cross section."""
    lhe = root = None
    if os.path.isdir(path):
        lhe = next((os.path.join(path, f) for f in ("parton_events.lhe.gz", "unweighted_events.lhe.gz",
                                                     "parton_events.lhe") if os.path.exists(os.path.join(path, f))), None)
        root = next((os.path.join(path, f) for f in ("delphes_events.root", "tag_1_delphes_events.root")
                     if os.path.exists(os.path.join(path, f))), None)
    elif path.endswith((".lhe", ".lhe.gz")):
        lhe = path
    elif path.endswith(".root"):
        root = path
    else:
        sys.exit(f"don't know how to read {path}")

    xsec = None
    meta = os.path.join(path, "metadata.json") if os.path.isdir(path) else None
    if meta and os.path.exists(meta):
        xsec = json.load(open(meta)).get("cross_section_pb")

    events = {}
    # the parton record is also needed to truth-tag the W_had jets (up vs down type)
    need_parton = "parton" in levels or "parton_trig" in levels or (angle_opts["jet_assignment"] == "truth" and
                                         ("gen" in levels or "reco" in levels))
    if need_parton and lhe:
        print(f"[{label}] reading {lhe}")
        parts, info = io.read_lhe(lhe, max_events)
        xsec = xsec or info["xsec_pb"]
        events["parton"] = physics.build_parton(parts, np.ones(len(parts)))
    if ("gen" in levels or "reco" in levels) and root:
        print(f"[{label}] reading {root}")
        n_tot = io.delphes_entries(root) if max_events is None else min(max_events, io.delphes_entries(root))
        parts = {"gen": [], "reco": []}
        for lo in range(0, n_tot, CHUNK):          # chunks keep memory bounded (Particle is big)
            d = io.read_delphes(root, entry_start=lo, max_events=min(lo + CHUNK, n_tot))
            if xsec is None and "Event" in d and "weight" in d["Event"].fields:
                # MG5+Pythia8 store the cross section (pb) as the event weight
                xsec = float(np.mean(ak.to_numpy(ak.flatten(d["Event"].weight))))
            if "gen" in levels:
                parts["gen"].append(physics.build_gen(d))
            if "reco" in levels:
                parts["reco"].append(physics.build_reco(d))
            del d
        for lvl, chunks in parts.items():
            if chunks:
                events[lvl] = physics.Event.concat(chunks)
    parton = events.get("parton")
    for lvl, ev in events.items():
        if lvl != "parton":
            angles.assign_had_quarks(ev, parton, angle_opts["jet_assignment"])
        angles.add_angles(ev, angle_opts["had_fermion"])
    if "parton_trig" in levels and "parton" in events:
        # parton level + the offline single-lepton (trigger) cuts on the charged lepton
        events["parton_trig"] = events["parton"].mask(events["parton"].lep_trigger)
    if "parton" not in levels:
        events.pop("parton", None)
    missing = [l for l in levels if l not in events]
    if missing:
        print(f"[{label}] note: no input for level(s) {missing}")
    for lvl, ev in events.items():
        ev.weight = np.ones(len(ev))          # unit weights; normalisation applied later
    return events, xsec


def per_level(x, level):
    return x.get(level) if isinstance(x, dict) else x


def evaluate(plot, ev, level):
    sel = per_level(plot.get("select"), level)
    if sel is not None:
        mask = getattr(ev, sel) if isinstance(sel, str) else sel(ev)
        ev = ev.mask(mask)
    expr = per_level(plot["expr"], level)
    if expr is None:
        return None
    vals = expr(ev)
    if len(vals) != len(ev):
        raise ValueError(f"plot {plot['name']} ({level}): expr must return one entry per event "
                         f"(got {len(vals)} for {len(ev)} events); use select= to drop events")
    return histo.flatten_values(vals, ev.weight)


def cutflow(label, events):
    for lvl in ("gen", "reco"):
        ev = events.get(lvl)
        if ev is None:
            continue
        n = len(ev)
        steps = [("all", np.ones(n, bool)),
                 ("== 1 lepton", ev.nleptons == 1),
                 (">= 4 jets", ev.njets >= 4),
                 (">= 1 b-jet", ev.nbjets >= 1),
                 (">= 2 b-jets", ev.nbjets >= 2)]
        print(f"\nCutflow [{label}, {lvl}]")
        m = np.ones(n, bool)
        for name, cut in steps:
            m = m & np.asarray(cut)
            print(f"  {name:<14s} {m.sum():>9d}  {m.sum() / n:7.2%}")
        print(f"  chi2 reco ok   {ev.reco_ok.sum():>9d}  {ev.reco_ok.sum() / n:7.2%}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--config", default=os.path.join(HERE, "plot_config.py"))
    p.add_argument("--levels", default="parton,parton_trig,gen,reco",
                   help="subset of parton,parton_trig,gen,reco to draw")
    p.add_argument("--only", default=None, help="comma list of plot names to make")
    p.add_argument("--norm", choices=["unit", "xsec", "none"], default="unit")
    p.add_argument("--lumi", type=float, default=140.0, help="fb^-1, used with --norm xsec")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--out", default=os.path.join(HERE, "plots"))
    p.add_argument("--format", default="png", help="png, pdf, ... (comma list allowed)")
    args = p.parse_args()

    cfg = runpy.run_path(args.config)
    physics.CUTS.update(cfg.get("CUTS", {}))
    plots = cfg["PLOTS"]
    lower = [pl["name"].lower() for pl in plots]
    dupes = sorted({n for n in lower if lower.count(n) > 1})
    if dupes:   # file names must also be unique on case-insensitive file systems (macOS)
        sys.exit(f"plot names must be unique (ignoring case): {dupes}")
    angle_opts = {"had_fermion": "down", "jet_assignment": "truth", **cfg.get("ANGLES", {})}
    if args.only:
        wanted = set(args.only.split(","))
        plots = [pl for pl in plots if pl["name"] in wanted]
        unknown = wanted - {pl["name"] for pl in plots}
        if unknown:
            sys.exit(f"unknown plot name(s): {sorted(unknown)}")
    levels = [l for l in args.levels.split(",") if l]

    samples = []
    for s in args.sample:
        label, _, path = s.partition("=")
        if not path:
            label, path = os.path.basename(os.path.normpath(s)), s
        events, xsec = load_sample(label, path, levels, args.max_events, angle_opts)
        samples.append((label, events, xsec))
        cutflow(label, events)

    os.makedirs(args.out, exist_ok=True)
    multi = len(samples) > 1
    for plot in plots:
        entries = []
        for si, (label, events, xsec) in enumerate(samples):
            for lvl in plot.get("levels", levels):
                if lvl not in levels or lvl not in events:
                    continue
                res = evaluate(plot, events[lvl], lvl)
                if res is None:
                    continue
                vals, w = res
                if args.norm == "xsec":
                    if not xsec:
                        sys.exit(f"--norm xsec needs a cross section for sample {label}")
                    w = w * xsec * 1e3 * args.lumi / len(events[lvl])
                shown = lvl.replace("parton_trig", "parton+trig")
                name = f"{label} ({shown})" if multi else shown
                # ratio reference: with several samples, the first sample at the same
                # level; with one sample, the first level drawn
                ref = next((i for i, e in enumerate(entries) if e["level"] == lvl), len(entries)) \
                    if multi else 0
                entries.append(dict(label=name, values=vals, weights=w, level=lvl, ref=ref,
                                    ref_label=samples[0][0] if multi else None,
                                    color=histo.COLORS[(si if multi else levels.index(lvl)) % len(histo.COLORS)],
                                    linestyle=LEVEL_STYLE[lvl] if multi else "-"))
        if not entries:
            print(f"[plots] skipping {plot['name']}: nothing to draw")
            continue
        for fmt in args.format.split(","):
            out = os.path.join(args.out, f"{plot['name']}.{fmt}")
            histo.draw(plot, entries, args.norm, out, lumi_fb=args.lumi)
        print(f"[plots] wrote {plot['name']} ({len(entries)} histograms)")
    print(f"\nPlots in {args.out}")


if __name__ == "__main__":
    main()
