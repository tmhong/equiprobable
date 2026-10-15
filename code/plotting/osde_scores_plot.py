#!/usr/bin/env python3
"""Redraw the score plots (scores_<level>.png) from the saved per-event scores, without refitting.

  python code/osde_scores_plot.py --dir results/osde        (or results/osde_ext)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import repo_paths  # noqa: E402
from osde_fit import plot_scores  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=os.path.join(repo_paths.RESULTS, "osde"))
    args = p.parse_args()
    for lvl in ("parton", "parton_trig", "gen", "reco"):
        path = os.path.join(args.dir, f"scores_{lvl}.npz")
        if os.path.exists(path):
            z = np.load(path)
            auc = plot_scores({k: z[k] for k in z.files}, lvl, os.path.join(args.dir, f"scores_{lvl}.png"))
            print(f"{args.dir} {lvl}: AUC {auc:.3f}")


if __name__ == "__main__":
    main()
