"""
Where things live in this repository (one place for every script):

  code/       all code (this file is code/repo_paths.py), by what it does:
                generation/  samples: MG5 -> Pythia8 -> Delphes, the cards, pileup production, tool setup
                analysis/    OSDE fits, the ML comparison, the pileup analysis and the study_* directories
                plotting/    figures: validation plots and plot_config.py, paper/ (paper figures), angle_diagram/
                docs/        the slides and the internal note
                lib/         the shared library (readers, physics objects, the ten angles, OSDE, drawing)
                tests/       unit tests
              Scripts put code/ first on sys.path and import from there: import repo_paths, from lib import osde,
              from plotting import make_plots, from analysis import osde_fit.
  results/    every output, grouped like code/:
                analysis/    fits (osde*/), ML (ml*/), pileup/, and one study_*/ per study
                plotting/    validation plots (plots*/, acceptance*/), paper/ (every paper figure), angle_diagram/
                docs/        note/, slides*/
                logs/
              Scripts name an output directory by its short name (results_dir("study_cv"), results("osde", ...)), and
              LAYOUT below places it: results/analysis/study_cv, results/analysis/osde/...
  samples/    generated event samples (large, not in git)
  tools/      MG5_aMC, Pythia8, Delphes (not in git)
  work/       MG5 process and scratch directories (not in git)
  recon_cicero/   J. Cicero's reconstruction code (kept as is)

Use results_dir("study_cv") for a study's output directory (created on first use) and results("osde", "f.npz") for a
path in one.
"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RESULTS = os.path.join(REPO, "results")
SAMPLES = os.path.join(REPO, "samples")
TOOLS = os.path.join(REPO, "tools")
WORK = os.path.join(REPO, "work")


_ANALYSIS = ("osde", "osde_ext", "osde_sh", "osde_h170", "osde_h2mw", "ml", "ml_h170", "ml_h2mw", "ml_quick", "pileup",
             "study_bsm", "study_cv", "study_gonogo", "study_l1", "study_salvage", "study_truthmatch",
             "study_washout", "study_washout_h170", "study_washout_h2mw")
_PLOTTING = ("plots", "plots_hh", "plots_hh_h170", "plots_hh_h2mw", "plots_tt_vs_hh",
             "acceptance", "acceptance_h170", "acceptance_h2mw")
_DOCS = ("note", "slides", "slides_h170", "slides_h2mw")
LAYOUT = {**{d: "analysis/" + d for d in _ANALYSIS}, **{d: "plotting/" + d for d in _PLOTTING},
          "paper_figs": "plotting/paper", "figures": "plotting/angle_diagram", **{d: "docs/" + d for d in _DOCS}}


def results(*parts):
    """The path results/<parts...>, its first directory placed by LAYOUT: results("osde", "f.npz") ->
    results/analysis/osde/f.npz.  Paths already in the grouped layout (results("analysis", ...)) and absolute paths
    pass through unchanged."""
    if not parts:
        return RESULTS
    p = os.path.join(*parts)
    if os.path.isabs(p):
        return p
    head, _, rest = os.path.normpath(p).partition(os.sep)
    group = LAYOUT.get(head) or ("analysis/" + head if head.startswith("study_") else None)   # new studies too
    if group:
        p = os.path.join(group, rest) if rest else group
    return os.path.join(RESULTS, p)


def results_dir(*parts, create=True):
    """results/<parts...> as placed by results(), created if needed."""
    d = results(*parts)
    if create:
        os.makedirs(d, exist_ok=True)
    return d
