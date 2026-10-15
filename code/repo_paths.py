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
  results/    every output: fits, figures, JSON, logs, reports, the note and the slides
  samples/    generated event samples (large, not in git)
  tools/      MG5_aMC, Pythia8, Delphes (not in git)
  work/       MG5 process and scratch directories (not in git)
  recon_cicero/   J. Cicero's reconstruction code (kept as is)

Use results_dir("study_cv") for a study's output directory (created on first use).
"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
RESULTS = os.path.join(REPO, "results")
SAMPLES = os.path.join(REPO, "samples")
TOOLS = os.path.join(REPO, "tools")
WORK = os.path.join(REPO, "work")


def results_dir(*parts, create=True):
    """results/<parts...>, created if needed."""
    d = os.path.join(RESULTS, *parts)
    if create:
        os.makedirs(d, exist_ok=True)
    return d
