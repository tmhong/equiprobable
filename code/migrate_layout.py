#!/usr/bin/env python3
"""
Move the files that git does not track (fits, caches, .npz, ROOT files, plots) from the old flat layout into the
code/ + results/ layout, after pulling the reorganization.  git moves the tracked files itself; this script moves the
rest.  Code-like files (.py .sh .C .tcl .tex) go to code/<dir>/, everything else to results/<dir>/; files at the top
level that are not code go to results/figures/ (pdf, png) or results/logs/.  samples/, tools/, work/, .venv/, recon_cicero/, code/, results/ and .git
are left alone.

Dry run by default (prints what it would do).  Run from anywhere:
  python code/migrate_layout.py            # show
  python code/migrate_layout.py --do       # move
  python code/migrate_layout.py --from OLD_CHECKOUT --link --do   # hard-link from another (old-layout) checkout
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SKIP = {"samples", "tools", "work", ".venv", "recon_cicero", "code", "results", ".git", "__pycache__", ".claude"}
CODE_EXT = {".py", ".sh", ".C", ".cc", ".cmnd", ".tcl", ".tex"}
CODE_NAMES = {"Makefile", "README", "minbias2pileup"}
GENERATED = ("note/", "slides", "figures/")       # .tex there is written by make_note.py / make_slides.py: a result


def is_code(rel):
    base, ext = os.path.basename(rel), os.path.splitext(rel)[1]
    if "failed_attempts" in rel:
        return False
    if rel.startswith("pileup/bin/"):
        return True                                  # compiled helper programs live with the pileup code
    if ext == ".tex" and rel.startswith(GENERATED) and not rel.startswith("figures/angle_diagram"):
        return False
    return ext in CODE_EXT or base in CODE_NAMES
TOP_KEEP = {"README.md", "LOG.md", "LICENSE", "requirements.txt", ".gitignore", ".DS_Store",
            "py.py"}                         # py.py: MG5's parser table, rewritten wherever MG5 is started (ignored)


def plan(src_root):
    out = []
    for top in sorted(os.listdir(src_root)):
        p = os.path.join(src_root, top)
        if top in SKIP or top.startswith("."):
            continue
        if os.path.isfile(p):
            if top in TOP_KEEP:
                continue
            ext = os.path.splitext(top)[1]
            kind = ("code" if ext in CODE_EXT else "results/figures" if ext in (".pdf", ".png")
                    else "results/logs")
            out.append((p, os.path.join(REPO, kind, top)))
            continue
        for dp, dn, fn in os.walk(p):
            dn[:] = [d for d in dn if d != "__pycache__"]
            for f in fn:
                rel = os.path.relpath(os.path.join(dp, f), src_root)
                kind = "code" if is_code(rel) else "results"
                out.append((os.path.join(dp, f), os.path.join(REPO, kind, rel)))
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--from", dest="src", default=REPO, help="old-layout root to take the files from (default: this repo)")
    a.add_argument("--link", action="store_true", help="hard-link instead of moving (with --from another checkout)")
    a.add_argument("--do", action="store_true", help="actually move (default: dry run)")
    args = a.parse_args()
    src = os.path.abspath(args.src)
    moves = plan(src)
    n_skip = 0
    for s, d in moves:
        if os.path.exists(d):
            n_skip += 1
            continue
        if not args.do:
            print(f"{os.path.relpath(s, src)} -> {os.path.relpath(d, REPO)}")
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if args.link:
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
        else:
            shutil.move(s, d)
    print(f"{len(moves) - n_skip} files {'moved' if args.do else 'to move'} ({n_skip} already in place)")
    if args.do and not args.link:                      # remove the emptied old directories
        for top in os.listdir(src):
            p = os.path.join(src, top)
            if top in SKIP or not os.path.isdir(p):
                continue
            for dp, dn, fn in sorted(os.walk(p, topdown=False), key=lambda t: -len(t[0])):
                if not os.listdir(dp):
                    os.rmdir(dp)


if __name__ == "__main__":
    main()
