#!/usr/bin/env python3
"""
Move the files that git does not track (fits, caches, .npz, ROOT files, plots, scripts never committed) from an older
layout into the current one, after pulling the reorganization.  git moves the tracked files itself; this script moves
the rest.  Two older layouts are known:

  flat    everything at the top (osde_fit.py, plotting/, paper_figs/, study_cv/, ...).  Code-like files (.py .sh .C
          .cc .cmnd .tcl .tex, Makefile, README) go next to the moved code in code/<category>/ (see below), everything
          else to its results/ group (see below); files at the top level that are not code go to results/plotting/ (pdf,
          png) or results/logs/.
  code/   code/ + results/, with code/ not yet sorted into categories (code/osde_fit.py, code/paper_figs/,
          code/study_cv/, ...).  What is left in the old code/ directories after the pull goes the same way.
And results/ itself: untracked data left in an ungrouped results/<dir>/ (results/osde/, results/study_cv/, ...) moves to
its group, results/analysis/, plotting/ or docs/ (repo_paths.LAYOUT); the Fig. 2, 3, 6 and 7 outputs that
study_truthmatch/ used to hold (fig2_osde_projections_recomatched.*, ...) move to results/plotting/paper/.

Where the code goes now (old directory or file -> new place under code/):
  plotting/ (flat layout: the shared library) -> lib/            cards/      -> generation/cards/
  figures/    -> plotting/angle_diagram/                          note/       -> docs/
  paper_figs/ -> plotting/paper/                                  study_*/    -> analysis/study_*/
  pileup/     -> generation/pileup/ (src/, bin/, shell scripts, merge_*/make_*) or analysis/pileup/ (other .py)
  study_gonogo/generate_gonogo.sh, study_truthmatch/generate_hh170_more.sh -> generation/;
  study_truthmatch/paper_figs_matched.py, fig6_corner_matched.py -> plotting/paper/
  top-level scripts -> generation/, analysis/, plotting/ or docs/ (TOP_SCRIPTS below); unknown ones stay in code/.
code/plotting/ of the code/ layout (the shared library, now code/lib/) is left alone: after the pull it holds the
plotting scripts.  samples/, tools/, work/, .venv/, recon_cicero/, results/ and .git are left alone.

Dry run by default (prints what it would do).  Run from anywhere:
  python code/migrate_layout.py            # show
  python code/migrate_layout.py --do       # move
  python code/migrate_layout.py --from OLD_CHECKOUT --link --do   # hard-link from another (old-layout) checkout
"""
import argparse
import os
import shutil

import repo_paths

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PAPER_OUTPUTS = ("fig2_osde_projections_recomatched.", "fig3_discriminator_recomatched.", "figA_levels_hh170.",
                 "fig6_corner_recomatched.", "paper_figs_matched.json")   # written to study_truthmatch/ before


def new_results_path(rel):
    """Where an output at results/<rel> of an ungrouped layout goes now, relative to the repository top."""
    top, _, rest = rel.partition("/")
    if top == "study_truthmatch" and "/" not in rest and rest.startswith(PAPER_OUTPUTS):
        rel = "paper_figs/" + rest
    return os.path.relpath(repo_paths.results(rel), REPO)
SKIP = {"samples", "tools", "work", ".venv", "recon_cicero", "code", "results", ".git", "__pycache__", ".claude"}
CODE_EXT = {".py", ".sh", ".C", ".cc", ".cmnd", ".tcl", ".tex"}
CODE_NAMES = {"Makefile", "README", "minbias2pileup"}
GENERATED = ("note/", "slides", "figures/")       # .tex there is written by make_note.py / make_slides.py: a result
CODE_NOW = {"generation", "analysis", "plotting", "docs", "lib", "tests", "repo_paths.py", "migrate_layout.py",
            "__init__.py", "__pycache__"}            # code/ as it is now: not an old directory
NEW_DIRS = {"cards": "generation/cards", "figures": "plotting/angle_diagram", "note": "docs",
            "paper_figs": "plotting/paper", "tests": "tests"}
MOVED_FILES = {"study_gonogo/generate_gonogo.sh": "generation/generate_gonogo.sh",
               "study_truthmatch/generate_hh170_more.sh": "generation/generate_hh170_more.sh",
               "study_truthmatch/paper_figs_matched.py": "plotting/paper/paper_figs_matched.py",
               "study_truthmatch/fig6_corner_matched.py": "plotting/paper/fig6_corner_matched.py"}
TOP_SCRIPTS = {
    "generation": ("generate_samples.py", "setup_tools.sh"),
    "analysis": ("osde_fit.py", "osde_simplify.py", "osde_order_scan.py", "osde_truncation_check.py", "osde_compare.py",
                 "basis_compare.py", "ml_compare.py", "run_simplify_all.sh", "run_trig.sh"),
    "plotting": ("make_plots.py", "plot_config.py", "hh_plots.py", "acceptance_plots.py", "lep_had_compare.py",
                 "nu_pz_plot.py", "osde_scores_plot.py", "paper_style_plots.py", "regen_figures.sh"),
    "docs": ("make_slides.py",),
}


def new_code_path(rel, flat=True):
    """Where the code file rel (relative to the old top, or to the old code/) goes now, relative to code/."""
    if rel in MOVED_FILES:
        return MOVED_FILES[rel]
    top, _, rest = rel.partition("/")
    if not rest:                                     # a script at the top
        return next((f"{cat}/{rel}" for cat, names in TOP_SCRIPTS.items() if rel in names), rel)
    if top == "plotting" and flat:
        return "lib/" + rest
    if top in NEW_DIRS:
        return f"{NEW_DIRS[top]}/{rest}"
    if top.startswith("study_"):
        return "analysis/" + rel
    if top == "pileup":                              # production (minimum bias, merging) vs analysis of the samples
        production = "/" in rest or not rest.endswith(".py") or rest.startswith(("merge_", "make_"))
        return ("generation/pileup/" if production else "analysis/pileup/") + rest
    return rel


def old_code_entries(src_root):
    """The code/ layout: what is in code/ but not part of code/ now (old directories, loose scripts)."""
    d = os.path.join(src_root, "code")
    if not os.path.isdir(d):
        return []
    return [e for e in sorted(os.listdir(d)) if e not in CODE_NOW and not e.startswith(".")]


def is_code(rel):
    base, ext = os.path.basename(rel), os.path.splitext(rel)[1]
    if "failed_attempts" in rel or "/build/" in rel:  # records of a failed run; LaTeX build directories
        return False
    if rel.startswith("pileup/bin/"):
        return True                                  # compiled helper programs live with the pileup code
    if ext == ".tex" and rel.startswith(GENERATED) and not rel.startswith("figures/angle_diagram"):
        return False
    return ext in CODE_EXT or base in CODE_NAMES
TOP_KEEP = {"README.md", "LOG.md", "LICENSE", "requirements.txt", ".gitignore", ".DS_Store",
            "py.py"}                         # py.py: MG5's parser table, rewritten wherever MG5 is started (ignored)


def tracked_tops(src_root):
    """Top-level entries that git tracks (co-authors' directories such as cicero/ or whitten/): never moved."""
    import subprocess
    r = subprocess.run(["git", "-C", src_root, "ls-files"], capture_output=True, text=True)
    return {f.split("/")[0] for f in r.stdout.split()} if r.returncode == 0 else set()


def plan(src_root):
    out = []
    keep = tracked_tops(src_root)
    for top in sorted(os.listdir(src_root)):
        p = os.path.join(src_root, top)
        if top in SKIP or top in keep or top.startswith("."):
            continue
        if os.path.isfile(p):
            if top in TOP_KEEP:
                continue
            ext = os.path.splitext(top)[1]
            dest = (os.path.join("code", new_code_path(top)) if ext in CODE_EXT
                    else os.path.join("results/plotting" if ext in (".pdf", ".png") else "results/logs", top))
            out.append((p, os.path.join(REPO, dest)))
            continue
        for dp, dn, fn in os.walk(p):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".pytest_cache")]
            for f in fn:
                rel = os.path.relpath(os.path.join(dp, f), src_root)
                dest = os.path.join("code", new_code_path(rel)) if is_code(rel) else new_results_path(rel)
                out.append((os.path.join(dp, f), os.path.join(REPO, dest)))
    code = os.path.join(src_root, "code")              # the code/ layout: what is left in the old code/ directories
    for top in old_code_entries(src_root):
        p = os.path.join(code, top)
        if os.path.isfile(p):
            new = new_code_path(top, flat=False)
            if is_code(top) and new != top:
                out.append((p, os.path.join(REPO, "code", new)))
            continue
        for dp, dn, fn in os.walk(p):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".pytest_cache")]
            for f in fn:
                rel = os.path.relpath(os.path.join(dp, f), code)
                dest = (os.path.join("code", new_code_path(rel, flat=False)) if is_code(rel)
                        else new_results_path(rel))
                out.append((os.path.join(dp, f), os.path.join(REPO, dest)))
    for top in old_results_entries(src_root):          # ungrouped results/<dir>/ -> results/<group>/<dir>/
        for dp, dn, fn in os.walk(os.path.join(src_root, "results", top)):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".pytest_cache")]
            for f in fn:
                rel = os.path.relpath(os.path.join(dp, f), os.path.join(src_root, "results"))
                out.append((os.path.join(dp, f), os.path.join(REPO, new_results_path(rel))))
    tm = os.path.join(src_root, "results", "analysis", "study_truthmatch")   # paper outputs already grouped there
    if os.path.isdir(tm):
        for f in sorted(os.listdir(tm)):
            if f.startswith(PAPER_OUTPUTS) and os.path.isfile(os.path.join(tm, f)):
                out.append((os.path.join(tm, f), repo_paths.results("paper_figs", f)))
    return out


def old_results_entries(src_root):
    """The ungrouped results/ layout: directories in results/ that LAYOUT now places in a group."""
    d = os.path.join(src_root, "results")
    if not os.path.isdir(d):
        return []
    return [e for e in sorted(os.listdir(d)) if e in repo_paths.LAYOUT and os.path.isdir(os.path.join(d, e))]


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
        olds = [os.path.join(src, t) for t in os.listdir(src) if t not in SKIP and t not in tracked_tops(src)]
        olds += [os.path.join(src, "code", t) for t in old_code_entries(src)]
        olds += [os.path.join(src, "results", t) for t in old_results_entries(src)]
        for p in olds:
            if not os.path.isdir(p):
                continue
            for dp, dn, fn in os.walk(p):                  # byte-code caches are rebuilt on import
                for d in [d for d in dn if d in ("__pycache__", ".pytest_cache")]:
                    shutil.rmtree(os.path.join(dp, d))
            for dp, dn, fn in sorted(os.walk(p, topdown=False), key=lambda t: -len(t[0])):
                if not os.listdir(dp):
                    os.rmdir(dp)


if __name__ == "__main__":
    main()
