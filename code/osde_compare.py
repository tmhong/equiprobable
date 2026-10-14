#!/usr/bin/env python3
"""
Compare the paper's basis (results/osde/) with the extended basis (results/osde_ext/): projection χ²/ndf of
every angle against the tt̄ fit sample, for the full and the cross-validation-truncated
("simplified") functions, at each level.  Also writes the projection plots of the
simplified functions.

Output: results/osde_ext/projection_chi2_comparison.json, <dir>/projections_ttbar_simplified_<level>.png
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import osde_fit  # noqa: E402
import repo_paths  # noqa: E402
from plotting import osde  # noqa: E402
from plotting.angles import ANGLE_NAMES  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--paper", default=os.path.join(repo_paths.RESULTS, "osde"))
    p.add_argument("--extended", default=os.path.join(repo_paths.RESULTS, "osde_ext"))
    p.add_argument("--tag", default="down_truth")
    args = p.parse_args()

    tr = np.load(os.path.join(args.train, f"angles_{args.tag}.npz"))
    te = np.load(os.path.join(args.test, f"angles_{args.tag}.npz"))

    def get(z, lvl):
        return {a: z[f"{lvl}/{a}"] for a in ANGLE_NAMES}

    res = {}
    for d, lab in ((args.paper, "paper"), (args.extended, "extended")):
        for lvl in ("parton", "parton_trig", "gen", "reco"):
            for kind in ("", "_simplified"):
                path = os.path.join(d, f"f_ttbar_{lvl}{kind}.npz")
                if not os.path.exists(path):
                    continue
                f = osde.load_density(path)
                c2 = osde_fit.plot_projections(f, {"train": get(tr, lvl), "test": get(te, lvl)}, lvl,
                                               os.path.join(d, f"projections_ttbar{kind}_{lvl}.png"))
                res[f"{lab}{kind}/{lvl}"] = {a: [float(v[0]), int(v[1])] for a, v in c2.items()}
    out = os.path.join(args.extended, "projection_chi2_comparison.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{'basis':22s} {'level':7s} " + " ".join(f"{a[:10]:>10s}" for a in ANGLE_NAMES))
    for k, v in res.items():
        b, lvl = k.split("/")
        print(f"{b:22s} {lvl:7s} " + " ".join(f"{v[a][0] / v[a][1]:10.1f}" for a in ANGLE_NAMES))


if __name__ == "__main__":
    main()
