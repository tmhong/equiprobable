"""
The same events binned two ways, in a 2x2 figure:

  top left      the type-i (ttbar) events in the plane of two angles, x1 = cos theta and x2 = phi,
                as (1/N) d^2N/dOmega (colour); the type-j (hh) events as dots; and contours of
                y = f_i(x) at the bin edges of the bottom-right panel, each with its y value
  top right     (1/N) dN/dx2: the top-left panel integrated over x1
  bottom left   (1/N) dN/dx1: the top-left panel integrated over x2
  bottom right  (1/N) dN/dy:  the events binned in their density value y = f_i(x), or in -ln y

Each bin of the last panel collects the events of one band of the top-left panel, between two
contours. One band (an annulus between two contours, HIGHLIGHT_RANGE) and its bins are hatched.

Usage
  python compare_y.py                       the toy example         -> compare_y.pdf / .png
  python compare_y.py lny                   ... binned in -ln y     -> compare_lny.pdf / .png
  python compare_y.py ttbar.txt             your type-i events      -> ttbar_y.pdf / .png
  python compare_y.py ttbar.txt hh.txt      type i and type j       -> ttbar_y.pdf / .png
  python compare_y.py ttbar.txt hh.txt lny  ... binned in -ln y     -> ttbar_lny.pdf / .png
  python compare_y.py --write-toy           the toy events as text files, toy_i.txt and toy_j.txt

Input files
  Plain text, one event per line, columns separated by spaces, tabs or commas; '#' starts a
  comment. The first line may name the columns (x1 x2 y), or a '#' line just above the numbers can.
  COLUMNS (below) gives each column's position (0 = first) or name:
    x1, x2   the two variables of the top-left panel
    y        each event's density value under type i, y = f_i(x). Optional: if the file has no
             such column, f_i is estimated from the type-i events in (x1, x2).
    weight   event weights (optional)
    type     which events are type i and which are type j, if both are in one file (optional)
  The first file holds the type-i events, the process whose density defines y. The second file,
  if given, holds the type-j events; their y must be the same f_i, evaluated at them.
  If y depends on more variables than x1 and x2, it has no contours in the (x1, x2) plane. The
  script checks for this and then marks the type-i events of the hatched bin as dots instead.
  With SOLID_ANGLE = True (the default), x1 is a polar angle and y is a density per solid angle,
  like f_ttbar = (1/Gamma) dGamma/dOmega. With X1_COS = True (the default), the plots use
  cos(x1) instead of x1; since dOmega = d(cos theta) dphi, y is then a plain density in the
  (cos theta, phi) plane. Files give theta (X1_INPUT = "theta") or cos theta (X1_INPUT = "cos").

Needs numpy and matplotlib.
"""
import io
import os
import sys

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import (AutoMinorLocator, FixedFormatter, FixedLocator, FuncFormatter, LogFormatterMathtext,
                               LogLocator, MaxNLocator, MultipleLocator, ScalarFormatter,
                               NullFormatter, NullLocator)

# ==========================================================================================
# Things you are most likely to tweak
# ==========================================================================================
# input columns: position (0 = first column) or name from the header line; None = not in the file
COLUMNS = {"x1": 0, "x2": 1, "y": 2, "weight": None, "type": None}
NPZ_KEYS = {"x1": ("theta", "x1"), "x2": ("phi", "x2"), "y": ("f", "y"),   # array names looked for in a
            "weight": ("weight", "w"), "type": ("type",)}                 # .npz file (COLUMNS names win)
TYPE_I, TYPE_J = "0", "1"          # values in the "type" column that mark type i and type j

# symbols (math mode, without the $ signs) and names used in the labels
X1, X2, Y, S = r"x", r"\phi", r"y", r"s"     # x is the plotted first variable, x = cos theta
X1_AXIS = r"$x = \cos\theta$"                                 # title of the bottom-left x axis ...
X1_AXIS_TOP = r"$x = \cos\theta$"               # ... and of the top-left one, where x is defined
Y_AXIS = r"$y = f_{t\bar{t}}(x,\phi)$"         # x-axis title of the last panel, binned in y ...
S_AXIS = r"$s = -\ln y$"                        # ... and binned in -ln y
NAME_I, NAME_J = r"$t\bar{t}$", r"$HH_{170}$"
SHORT_I, SHORT_J = None, None      # shorter names for the ratio and the efficiencies (None: the names)
TOY_NAMES = (r"$\mathrm{Toy}\ \frac{1}{2} \to 1, \frac{1}{2}$", r"$\mathrm{Toy}\ 0 \to \frac{1}{2}, \frac{1}{2}$",
             r"N_{\frac{1}{2}}", r"N_0")   # legends, then the ratio's names
TOY_SUBS = (r"\frac{1}{2}", r"0")   # the toy's types as subscripts (f_1/2) and on the efficiency axes (Toy 1/2)   # the toy's

# binning
X1_COS = True                      # plot x1 = cos(theta), not theta; dOmega = d(cos theta) dphi, so y per solid
                                   # angle is a plain density in the (cos theta, phi) plane
X1_INPUT = "theta"                 # what the x1 column of the input files holds: "theta" or "cos"
X1_RANGE, X2_RANGE = (-1, 1), (0, 2 * np.pi)      # (low, high) of x1 (cos theta) and phi; None = range of the data
SOLID_ANGLE = True                 # x1 is a polar angle and y is per solid angle, like f_ttbar = (1/Gamma) dGamma/dOmega;
                                   # with X1_COS = False the plane is (theta, phi), dOmega = sin(theta) dtheta dphi.
                                   # False: y is per dx1 dx2
PI_TICKS = True                    # ticks at multiples of pi/2 when the range ends are such multiples
PI_HALVES = False                  # also label pi/2 and 3pi/2; False: values at 0, pi, 2pi only, with tick
                                   # marks every pi/4
BINS_X = 24                        # bins per axis in the top-left panel and in the projections
N_BINS_Y = 12                      # about this many round-numbered bins in the last panel ...
N_BINS_Y_DATA = 40                 # ... or this many when y is not a function of (x1, x2), as for real events
                                   # (no contours to match, so finer bins show the two types' shapes better)
Y_EDGES = S_EDGES = None           # ... unless you give the bin edges (in y, and in -ln y)
HIGHLIGHT_RANGE = (0.10, 0.12)     # the hatched band in y, between these two contours, and its bins in the
                                   # last panel (snapped to the nearest bin edges); None = HIGHLIGHT_AT
HIGHLIGHT_AT = None                # with HIGHLIGHT_RANGE = None: a value of y (or -ln y) whose one bin gets
                                   # hatched; None = the middle: the bin of the median type-i event
CONTOURS = "auto"                  # contours of y at the bin edges: True, False or "auto"
SMOOTH = None                      # only when y is estimated: kernel width / range (None = Scott's rule)
COLOR_SCALE = "linear"                # colour scale of the top-left panel: "log" or "linear"
SHOW_VOLUME = True                 # label the hatched band dV in the top-left panel, and add dV/d(-ln y),
                                   # from the areas between the contours, to the last panel
VOLUME_CURVE = True                # dV/d(-ln y) as a smooth curve: from bins 20 times narrower than the last
CURVE_SMOOTH = 0.3                 # panel's, smoothed by a Gaussian this many of its bins wide (which also
                                   # tames the spike at a saddle of f); VOLUME_CURVE = False: one dot per bin
DENSITY_VALUES = False             # tick values on the (1/N) dN/dx and dN/dphi axes (only their shapes matter);
                                   # their tick marks are drawn either way
DNDY_VALUES = False                # tick values on the (1/N) dN/dy axis
TITLE_TOP = 0.96                   # top edge of the dN/dx and dN/dy titles and of the tt/hh legend (the same
                                   # height for all three), in axes fraction
TITLE_LEFT = -0.35                 # left edge of the dN/dx and dN/dy titles, left of their plots, in axes fraction
LEGEND_LEFT = 0.04                 # left edge of the tt/hh legend, inside the dN/dx plot, in axes fraction
DNDX_TITLE = "upright"             # the (1/N) dN/dx title: "upright" (left of its axis, at mid-height) or
                                   # "outside" (left of the plot, at the top)
DNDY_TITLE = "upright"             # the (1/N) dN/dy title: "upright", "outside" or "inside" (a free spot in
                                   # its plot)
GRADIENT_FILL = True               # fill the ttbar histograms with the top-left panel's colours. Where y is
                                   # that panel's density (the toy): a dN/dy bin in its y's colour, a dN/dx or
                                   # dN/dphi bin in the mean y of its ttbar events. Otherwise (real events):
                                   # a dN/dx or dN/dphi bin in the mean colour of the strip it sums, and dN/dy
                                   # in one light shade. False: one light shade everywhere
CBAR_TITLE = r"$t\bar{t}\ \mathrm{probability\ density}\ (y)$"   # above the colour bar
CURVE_COLOR = "#00b7eb"            # the dV/d(-ln y) curve (or dots) in the last panel
DV_OUTSIDE = True                  # Delta V written outside the top-left plot, in the gap on its right, with a
DV_AT = (1.25, 0.76)               # line to the hatched band; DV_AT: where, in axes fraction of that plot.
                                   # False: inside the band, on a small navy box
MARK_COLOR = "white"               # colour of the contours, their values, the hatching and the Delta V label
                                   # in the top-left panel
MARK_HALO = 2.0                    # a thin navy outline (points) around those values and Delta V, so they read
                                   # where they cross the white lines and hatching (0 = none)
MINOR_TICKS = True                 # minor tick marks on every axis
MIRROR_TICKS = False               # tick marks also on the opposite side of each plot (no values there)
MEAN_CONTOURS = True               # without contours of y: draw contours of its smoothed average over the
MEAN_SMOOTH = 1.5                  # type-i events instead (MEAN_SMOOTH: kernel width, times Scott's rule).
                                   # They show the trend of y over (x1, x2), but a band between two of them
                                   # does not hold the events of a y bin, so nothing is hatched
SPLIT_LEFT = 1                     # bins left of the hatched band (toy) split into this many, for detail;
                                   # the contours stay at the coarse edges
Y_SHIFT_ROOM = 1.0                 # y_min = this times the lowest y's size, rounded up to one figure, so the
                                   # lowest y' is above 0 and the s' plot ends with room to spare
T_SCALE = True                     # with -ln y and the shift: bin in t = (s'max - s')/(s'max - s'min), with
T_ENDS = (0.001, 0.995)            # s' = -ln(y + y_min), from 0 to 1; s'min and s'max are these quantiles of
                                   # all events (one extreme event would otherwise squeeze the rest), and t is
                                   # clipped to [0, 1] beyond them
Z_STEP = 0.125                      # the toy's bins (and axis values) in z
Y_RESCALE = True                   # the toy's y shown as y' = (y - y0)/(y1 - y0), from 0 to 1, with y0 = 0 and
HIGHLIGHT_RANGE_SCALED = (0.625, 0.75)   # y1 its largest value; then the hatched band is given in y'
Y_SHIFT = True                     # if some y are negative (a truncated series), use y' = y + y_min instead,
                                   # y_min = -(the lowest y of both types), so y' starts at 0 and -ln y' exists
MAP_Y_WITHOUT_CONTOURS = True      # without contours (real events): colour the top-left plot by the mean y of
                                   # the type-i events in each cell, so every panel shares one colour scale of
                                   # y; False: by how densely the events fall there, (1/N) d^2N/dOmega
HATCH_WITHOUT_CONTOURS = False     # without contours (y depends on more than x1 and x2, as for real events):
                                   # still hatch a bin and show its type-i events as white dots
RATIO_OF = "j/i"                   # that ratio: "j/i" = hh/ttbar, or "i/j" = ttbar/hh
RATIO_PROJECTIONS = True           # the same ratio for dN/dx (axis on its right) and dN/dphi (axis below it);
RATIO_GAP = 0.3                    # RATIO_GAP: inches added between the columns for the dN/dx ratio's values
INF_MIN_SHARE = 0.001              # ... and at least this share of them (not just the sample's last few)
INF_MIN_EVENTS = 5                 # past the last event of the bottom type, this many of the top type make the
                                   # ratio infinite there: drawn flat along the top of its axis, marked infinity
RATIO_COLOR = "#d62728"            # the hh/ttbar curves and their axes
RATIO_LW = 1.0                     # width of the ratio curves
RATIO_SMOOTH = 1.5                 # it is smooth: both y distributions smoothed by one Gaussian this many
                                   # times Scott's width, then divided
RATIO_CURVE = True                 # the bin-by-bin ratio of the last panel's two histograms, as a curve with its own
RATIO_MIN_EVENTS = 50              # axis on the right: "data" = only without contours (real events), True =
                                   # always, False = never; bins with fewer events of either type are left out
MAKE_LR = True                     # also draw <name>_lr (ROC curves of the likelihood ratio of each observable)
LR_MIN_EVENTS = 20                 # the ROC plot's ttbar axis stops where fewer ttbar events than this are left
LR_BINS = 100                      # and <name>_lr3d (the ratio over the (x1, x2) plane, in 3-D); needs both
LR3D_RANGE = (-4.0, 3.0)           # types. LR_BINS: bins per observable, each with equal numbers of events;
LR_FIGSIZE, LR3D_FIGSIZE = (4.6, 3.9), (5.4, 4.5)   # LR3D_RANGE: the log10 range shown in 3-D
LR3D_BINS = 12                     # bins per axis for the 3-D ratio from events (the toy uses its exact densities)
TICKS_OUT = True                   # tick marks point out of the plots
TICK_LEN = 4.5                     # length of the major tick marks, in points (minor: 2.5)

# drawing
N_SCATTER = 250                    # type-j events drawn as dots in the top-left panel
PANEL_GROW = 0.25                  # for event files: this much bigger plots ...
GAP_SHRINK = (0.12, 0.12)          # ... and this much less space between the columns and the rows
PANEL = 1.8                        # side of each square plot area, in inches
# one font size, in points, for all the text: axis titles, tick values, legends, arrow labels, the dV
# label and the contour values. The figure is printed at 0.95 of a PRD column (3.2 in), about 0.55 of
# its size here, so 14 pt here is about 7.5 pt on paper
FONT_SIZE = 14
TITLE_SIZE = TICK_SIZE = LEGEND_SIZE = ARROW_SIZE = NOTE_SIZE = FONT_SIZE
CONTOUR_LABEL_SIZE = FONT_SIZE     # the y values written on the contours (0 = no values)
CONTOUR_VALUES = "band"            # which contours get their y value: "all", or "band" = the hatched band's
                                   # two edges and the highest contour (fewer, so they fit at FONT_SIZE)
CONTOUR_NAME = True                # the hatched band's lower edge reads "y = 0.1", the others just their value
NAME_LEVEL = True                  # write that "y = 0.1" level (horizontal), where its contour runs level
GAP_X, GAP_Y = 0.72, 0.76          # inches between the columns and between the rows (the arrows sit here)


def d(sym):
    """The differential of a symbol, with a thin space before an operator such as cos."""
    return r"\mathrm{d}" + (r"\," if sym.startswith(("\\cos", "\\sin", "\\ln")) else "") + sym


ARROW_TO_X2 = r"$%s$" % d(X1)              # top left -> top right: integrate over x1 (the integral sign, text-
ARROW_TO_X1 = r"$%s$" % d(X2)              # top left -> bottom left: integrate over x2   style, is put in front)
INT_GAP = 3.5                      # points between the integral sign and the differential
ARROW_TO_Y = {"y": r"$\mathrm{bin\ in}\ %s$" % Y, "-ln y": r"$\mathrm{in\ bins\ of}\ {-\ln %s}$" % Y}

# colours of Fig. 4 (zeroprob.pdf): navy and amber, with lighter shades blended with white
NAVY, AMBER, GRAY, INK = "#003094", "#f4bf57", "#7f7f7f", "#0b0b0b"
# type i: the top-left panel runs from white to COLOR_I; its histograms are FILL_I outlined in LINE_I
COLOR_I, FILL_I, LINE_I = NAVY, "#ccd5e9", NAVY            # FILL_I: navy at 20%
# type j: dots (COLOR_J with a DOT_EDGE_J rim, so they show on white and on navy) and lines (LINE_J)
COLOR_J, DOT_EDGE_J, LINE_J = AMBER, "#9c6b10", "#e3a43a"   # LINE_J: deeper amber, so a thin line shows on white
# to swap: COLOR_I, FILL_I, LINE_I = AMBER, "#fbe7bf", "#e3a43a"
#          COLOR_J, DOT_EDGE_J, LINE_J = NAVY, "white", NAVY
halo = [patheffects.withStroke(linewidth=MARK_HALO, foreground=COLOR_I)] if MARK_HALO else []

# the toy example, used when no file is given
N_TOY = 1_000_000
TOY_CACHE = True                   # draw the toy's events once, keep them in toy_events_<key>.npz, reuse them
TOY_SEED = 11                      # the seed they are drawn with                   # events per type in the figure
N_TOY_FILE = 50_000                # events per type written by --write-toy


TOY_ALPHA_P = 1.0               # toy type i: analysing power times polarisation, |alpha P| <= 1 (the W in a top
TOY_POL = (120.0, 180.0)        # decay has alpha = 0.39); the polarisation's direction, polar and azimuthal, degrees


def toy_f_i(theta, phi):
    """Type i, per solid angle: a spin-1/2 particle decaying to a spin-1 and a spin-1/2 one (like a top quark
    to a W and a b), with theta and phi the direction of one daughter in its rest frame. For spin 1/2 the
    decay can only depend on that direction through its angle to the polarisation P:
        (1 + alpha P cos(angle to P)) / 4pi,
    a dipole on the sphere, with alpha the decay's analysing power. Its contours are circles around P,
    and the area between two of them is the same for equal steps in y (Archimedes' hat box), so the
    distribution of y is a straight line. Normalized over the sphere, never negative."""
    tp, pp = np.radians(TOY_POL[0]), np.radians(TOY_POL[1])
    cos_to_p = (np.sin(tp) * np.cos(pp) * np.sin(theta) * np.cos(phi)
                + np.sin(tp) * np.sin(pp) * np.sin(theta) * np.sin(phi) + np.cos(tp) * np.cos(theta))
    return (1 + TOY_ALPHA_P * cos_to_p) / (4 * np.pi)


def toy_f_j(theta, phi):
    """Type j, per solid angle: a spin-0 particle decaying to two spin-1/2 ones (like a Higgs boson to b bbar):
    with no spin there is no direction to prefer, whatever the daughters' spins (they can only be
    correlated with each other), so the decay is isotropic, 1/4pi, and flat in y too."""
    return np.full(np.broadcast(theta, phi).shape, 1 / (4 * np.pi))
# ==========================================================================================

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": FONT_SIZE,
                     "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "hatch.color": INK, "hatch.linewidth": 0.5,
                     "xtick.direction": "out" if TICKS_OUT else "in",
                     "ytick.direction": "out" if TICKS_OUT else "in",
                     "xtick.major.pad": 2.5 if TICKS_OUT else 6.0,      # for outward ticks, measured from
                     "ytick.major.pad": 2.5 if TICKS_OUT else 3.5,      # the tick tips
                     "axes.labelpad": 2,
                     "xtick.major.size": TICK_LEN, "ytick.major.size": TICK_LEN,
                     "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
                     "legend.frameon": False, "legend.fontsize": LEGEND_SIZE})
rng = np.random.default_rng(3)
PER_SIN = SOLID_ANGLE and not X1_COS   # the plane is (theta, phi) and y is per solid angle: sin(theta) factors


# ------------------------------------------------------------------------------------------
# getting the events
# ------------------------------------------------------------------------------------------
def plane_f(f):
    """A toy density, written as a function of (theta, phi), as a function of the plane's (x1, x2)."""
    return (lambda x1, x2: f(np.arccos(np.clip(x1, -1, 1)), x2)) if X1_COS else f


def toy_events():
    """The toy's events of both types: read from toy_events_<key>.npz if it is there, else drawn once and
    written there. The key changes with the toy's densities, their settings and N_TOY, so a changed toy
    is drawn anew."""
    import hashlib
    import inspect
    recipe = (inspect.getsource(toy_f_i) + inspect.getsource(toy_f_j)
              + repr((N_TOY, TOY_ALPHA_P, TOY_POL, X1_COS, PER_SIN, X1_RANGE, X2_RANGE, SOLID_ANGLE)))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "toy_events_%s.npz" % hashlib.sha1(recipe.encode()).hexdigest()[:10])
    if TOY_CACHE and os.path.exists(path):
        with np.load(path) as kept:
            print(f"toy events read from {os.path.basename(path)}")
            return kept["xi"], kept["xj"]
    gen = np.random.default_rng(TOY_SEED)                 # its own generator, so the rest is the same
    xi, xj = sample(toy_f_i, N_TOY, gen), sample(toy_f_j, N_TOY, gen)   # either way
    if TOY_CACHE:
        np.savez_compressed(path, xi=xi, xj=xj)
        print(f"toy events drawn and kept in {os.path.basename(path)}")
    return xi, xj


def sample(f, n, gen=None):
    """Accept-reject sampling of n events (x1, x2) in the plane from the toy density f(theta, phi),
    per solid angle if SOLID_ANGLE (drawn uniformly in cos theta) or per dx1 dx2."""
    r1 = X1_RANGE or ((-1, 1) if X1_COS else (0, np.pi))
    r2 = X2_RANGE or (0, 2 * np.pi)
    lo, hi = (np.cos(r1[1]), np.cos(r1[0])) if PER_SIN else r1   # drawn uniformly in this
    to_x1 = np.arccos if PER_SIN else (lambda u: u)              # ... and turned into x1
    fp = plane_f(f)
    g1, g2 = np.meshgrid(to_x1(np.linspace(lo, hi, 201)), np.linspace(*r2, 201))
    fmax = 1.02 * fp(g1, g2).max()
    out, have = [], 0
    while have < n:
        gen = gen or rng
        x1, x2 = to_x1(gen.uniform(lo, hi, 2 * n)), gen.uniform(*r2, 2 * n)
        keep = gen.uniform(0, fmax, 2 * n) < fp(x1, x2)
        out.append(np.column_stack([x1[keep], x2[keep]]))
        have += keep.sum()
    return np.vstack(out)[:n]


def density_coords(p):
    """Events in the coordinates in which y is a plain density: (cos x1, x2) for a (theta, phi) plane
    with y per solid angle; the plane itself otherwise, e.g. (cos theta, phi)."""
    return np.column_stack([np.cos(p[:, 0]), p[:, 1]]) if PER_SIN else p


def is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def subset(ev, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in ev.items()}


def read_npz(path):
    """The events in a numpy .npz file: arrays named as in COLUMNS (where it gives names) or NPZ_KEYS."""
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as err:
        sys.exit(f"cannot read {path}: {err}")

    def array(role):
        c = COLUMNS.get(role)
        for key in ((c,) if isinstance(c, str) else NPZ_KEYS[role]):
            if key in data.files:
                a = np.asarray(data[key])
                return a.astype(str) if role == "type" else a.astype(float)
        return None
    x1, x2 = array("x1"), array("x2")
    if x1 is None or x2 is None:
        sys.exit(f"{path}: no arrays for x1 and x2 (looked for {NPZ_KEYS['x1']} and {NPZ_KEYS['x2']}); "
                 f"it has {data.files}")
    return ({"x": np.column_stack([x1, x2]), "y": array("y"), "w": array("weight"), "type": array("type")},
            f"arrays {data.files}")


def read_events(path):
    """The events in a text or .npz file, as {"x": (N, 2) array, "y", "w", "type": arrays or None}."""
    if path.endswith(".npz"):
        ev, described = read_npz(path)
    else:
        ev, described = read_text(path)
    if X1_COS and X1_INPUT == "theta":                         # the plots use cos theta
        theta = ev["x"][:, 0]
        if np.nanmax(np.abs(theta)) <= 1:
            print(f"{path}: x1 is within [-1, 1]; if it is already cos theta, set X1_INPUT = \"cos\"")
        ev["x"][:, 0] = np.cos(theta)
    if X2_RANGE is not None and np.isclose(X2_RANGE[1] - X2_RANGE[0], 2 * np.pi):   # an angle: wrap it
        x2 = ev["x"][:, 1]                                                         # into X2_RANGE
        outside = (x2 < X2_RANGE[0]) | (x2 > X2_RANGE[1])
        if outside.any():
            ev["x"][:, 1] = X2_RANGE[0] + np.mod(x2 - X2_RANGE[0], 2 * np.pi)
            print(f"{path}: {outside.sum():,} values of x2 wrapped into the range of X2_RANGE")
    keep = np.isfinite(ev["x"]).all(axis=1)
    if ev["y"] is not None:
        keep &= ~np.isnan(ev["y"])
    if (~keep).any():
        print(f"{path}: skipping {(~keep).sum():,} events with a missing (nan) value")
    print(f"{path}: {keep.sum():,} events, {described}" + ("" if ev["y"] is not None else "; no y column"))
    return subset(ev, keep)


def read_text(path):
    """The columns of a text file, as {"x": (N, 2) array, "y", "w", "type": arrays or None}."""
    try:
        with open(path) as fh:
            lines = fh.read().replace(",", " ").splitlines()
    except OSError as err:
        sys.exit(f"cannot read {path}: {err}")
    names, comment, start = None, None, len(lines)
    for k, line in enumerate(lines):
        body, _, note = line.partition("#")
        tokens = body.split()
        if not tokens:
            comment = note.split() or comment
            continue
        if not any(is_number(t) for t in tokens):              # a line of column names
            names, start = tokens, k + 1
        else:                                                  # the first line of numbers
            names, start = (comment if comment and len(comment) == len(tokens) else None), k
        break
    try:
        table = np.loadtxt(io.StringIO("\n".join(lines[start:])), dtype=str, comments="#", ndmin=2)
    except ValueError as err:
        sys.exit(f"{path}: {err}")
    if table.size == 0:
        sys.exit(f"{path}: no events found")

    def column(key, numeric=True):
        c = COLUMNS.get(key)
        if c is None:
            return None
        if isinstance(c, str):
            if not names or c not in names:
                sys.exit(f"{path}: COLUMNS gives {key} as the column named '{c}', "
                         f"but the column names are {names}")
            c = names.index(c)
        if c >= table.shape[1]:
            if key == "y":
                return None                                    # no y column: y gets estimated
            sys.exit(f"{path}: COLUMNS gives {key} as column {c} (0 = first), "
                     f"but the file has {table.shape[1]} columns")
        if not numeric:
            return table[:, c]
        try:
            return table[:, c].astype(float)
        except ValueError:
            sys.exit(f"{path}: column {c} ({key}) is not all numbers")

    ev = {"x": np.column_stack([column("x1"), column("x2")]), "y": column("y"),
          "w": column("weight"), "type": column("type", numeric=False)}
    return ev, f"columns {names if names else table.shape[1]}"


def same(values, wanted):
    """Which entries of a column equal `wanted` (as numbers if both are numbers)."""
    try:
        return values.astype(float) == float(wanted)
    except ValueError:
        return values == str(wanted)


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def quantile(v, w, q):
    """Weighted quantile."""
    o = np.argsort(v)
    c = np.cumsum(np.abs(w[o]))
    return v[o][min(np.searchsorted(c, q * c[-1]), len(v) - 1)]


def round_edges(lo, hi, n):
    """About n bins covering [lo, hi], with a round bin width and round edges."""
    raw = (hi - lo) / n
    mag = 10.0 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 4, 5, 10) if m * mag >= 0.999 * raw)
    first, last = np.floor(lo / step + 1e-9), np.ceil(hi / step - 1e-9)
    return np.round(step * np.arange(first, last + 1), 12)


def round_range(v):
    """The range of v, widened to round numbers."""
    lo, hi = v.min(), v.max()
    step = 10.0 ** (np.floor(np.log10(hi - lo)) - 1)
    return np.floor(lo / step + 1e-9) * step, np.ceil(hi / step - 1e-9) * step


def estimate_density(x, w, r1, r2, n=128):
    """f_i estimated from the type-i events: a fine 2-D histogram smoothed with a Gaussian
    kernel, mirrored at the edges of the range. Returns the grid, the values, and an evaluator."""
    e1, e2 = np.linspace(*r1, n + 1), np.linspace(*r2, n + 1)
    grid = np.histogram2d(x[:, 0], x[:, 1], bins=[e1, e2], weights=w)[0]
    if SMOOTH is None:                                         # Scott's rule, per axis
        n_eff = w.sum() ** 2 / (w ** 2).sum()
        width = x.std(axis=0) * n_eff ** (-1 / 6) / [r1[1] - r1[0], r2[1] - r2[0]]
    else:
        width = np.array([SMOOTH, SMOOTH])
    for axis in (0, 1):
        sigma = max(width[axis] * n, 0.5)                      # in grid cells
        m = int(np.ceil(3 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-m, m + 1) / sigma) ** 2)
        pad = [(0, 0), (0, 0)]
        pad[axis] = (m, m)
        grid = np.apply_along_axis(np.convolve, axis, np.pad(grid, pad, mode="symmetric"),
                                   kernel / kernel.sum(), mode="valid")
    grid /= w.sum() * (e1[1] - e1[0]) * (e2[1] - e2[0])
    c1, c2 = (e1[1:] + e1[:-1]) / 2, (e2[1:] + e2[:-1]) / 2

    def at(p):                                                 # bilinear interpolation
        u = np.clip((p[:, 0] - c1[0]) / (c1[1] - c1[0]), 0, n - 1)
        v = np.clip((p[:, 1] - c2[0]) / (c2[1] - c2[0]), 0, n - 1)
        i, j = np.minimum(u.astype(int), n - 2), np.minimum(v.astype(int), n - 2)
        u, v = u - i, v - j
        val = (grid[i, j] * (1 - u) * (1 - v) + grid[i + 1, j] * u * (1 - v)
               + grid[i, j + 1] * (1 - u) * v + grid[i + 1, j + 1] * u * v)
        outside = (p[:, 0] < r1[0]) | (p[:, 0] > r1[1]) | (p[:, 1] < r2[0]) | (p[:, 1] > r2[1])
        return np.where(outside, 0.0, val)
    return c1, c2, grid.T, at, width


def unexplained(x, y, r1, r2, n=2000):
    """Share of the variance of y left over once x1 and x2 are known, from each event's nearest
    neighbour in the (x1, x2) plane: near 0 if y is a smooth function of x1 and x2 alone."""
    k = rng.choice(len(y), min(n, len(y)), replace=False)
    p = (x[k] - [r1[0], r2[0]]) / [r1[1] - r1[0], r2[1] - r2[0]]
    sq = (p ** 2).sum(axis=1)
    d = sq[:, None] + sq[None, :] - 2 * p @ p.T
    np.fill_diagonal(d, np.inf)
    yk = y[k]
    return np.mean((yk - yk[d.argmin(axis=1)]) ** 2) / (2 * yk.var())


def density(v, edges, w):
    """(1/N) dN/dv in the given bins, with N the total weight."""
    return np.histogram(v, bins=edges, weights=w)[0] / (w.sum() * np.diff(edges))


# ------------------------------------------------------------------------------------------
# the events
# ------------------------------------------------------------------------------------------
args = sys.argv[1:]
if {"-h", "--help"} & set(args):
    print(__doc__)
    sys.exit()
LOG = bool({"lny", "--lny", "s", "--s", "t", "--t"} & set(args))   # bin in s' = -ln(y - y0), or in t
if {"s", "--s"} & set(args):
    T_SCALE = False                          # "s": s' itself; "t" (or "lny"): t, from s'
elif {"t", "--t", "lny", "--lny"} & set(args):
    T_SCALE = True
BIN_IN = "-ln y" if LOG else "y"

if "--write-toy" in args:
    for fname, f in (("toy_i.txt", toy_f_i), ("toy_j.txt", toy_f_j)):
        x = sample(f, N_TOY_FILE)
        theta = np.arccos(np.clip(x[:, 0], -1, 1)) if X1_COS else x[:, 0]
        np.savetxt(fname, np.column_stack([theta, x[:, 1], toy_f_i(theta, x[:, 1])]), fmt="%.6f",
                   header="theta phi y")
        print(f"wrote {fname}: {N_TOY_FILE:,} events, columns theta phi y (y = f_ttbar, also for hh)")
    sys.exit()

IN_Y = "y" in args or "--y" in args          # the toy in y itself: (1/N) dN/dy against dV/d(-ln y)
files = [a for a in args if a not in ("lny", "--lny", "y", "--y", "s", "--s", "t", "--t")]
if len(files) > 2:
    sys.exit("give at most two files: the type-i events, then the type-j events")
if files:
    A = read_events(files[0])
    B = read_events(files[1]) if len(files) == 2 else None
    if B is None and A["type"] is not None:                    # both types in one file
        t = A["type"]
        A, B = subset(A, same(t, TYPE_I)), subset(A, same(t, TYPE_J))
        print(f"type column: {len(A['y'] if A['y'] is not None else A['x']):,} events of type i "
              f"('{TYPE_I}'), {len(B['x']):,} of type j ('{TYPE_J}')")
        if len(B["x"]) == 0:
            B = None
    name = os.path.splitext(os.path.basename(files[0]))[0] + (("_t" if T_SCALE else "_s") if LOG else "_y")
    is_toy = False
else:
    xi, xj = toy_events()
    f_toy = plane_f(toy_f_i)                                    # y = f_ttbar, in the plane's coordinates
    A = {"x": xi, "y": f_toy(xi[:, 0], xi[:, 1]), "w": None}
    B = {"x": xj, "y": f_toy(xj[:, 0], xj[:, 1]), "w": None}
    name = "compare_s" if LOG else ("compare_lny" if IN_Y else "compare_lnz")   # in y: dV/d(-ln y); in z
    is_toy = True
    NAME_I, NAME_J, SHORT_I, SHORT_J = TOY_NAMES
    Y_AXIS = r"$y$"                                         # the toy's y (defined at the top: f_1/2)
    CBAR_TITLE = r"$y = f_{%s}(x,\phi)$" % TOY_SUBS[0]
if len(A["x"]) == 0:
    sys.exit("no type-i events")

xi, wi = A["x"], (A["w"] if A["w"] is not None else np.ones(len(A["x"])))
if B is not None:
    xj, wj = B["x"], (B["w"] if B["w"] is not None else np.ones(len(B["x"])))
both_x = np.vstack([xi, xj]) if B is not None else xi
r1 = X1_RANGE if X1_RANGE is not None else round_range(both_x[:, 0])
r2 = X2_RANGE if X2_RANGE is not None else round_range(both_x[:, 1])

# y = f_i(x) for every event: from the file, or estimated from the type-i events
estimated = A["y"] is None
if estimated:
    if B is not None and B["y"] is not None:
        sys.exit("the type-i events have no y column but the type-j events do: give y for both or neither")
    dr1 = (np.cos(r1[1]), np.cos(r1[0])) if PER_SIN else r1       # the x1 range in density coordinates
    _, _, _, f_at, width = estimate_density(density_coords(xi), wi, dr1, r2)
    yi = f_at(density_coords(xi))
    yj = f_at(density_coords(xj)) if B is not None else None
    gx, gy = np.linspace(*r1, 241), np.linspace(*r2, 241)         # the estimate on a grid in (x1, x2)
    GX, GY = np.meshgrid(gx, gy)
    f_grid = f_at(density_coords(np.column_stack([GX.ravel(), GY.ravel()]))).reshape(GX.shape)
    print("y estimated from the type-i events in (x1, x2): smoothed histogram, kernel width "
          f"{width[0]:.3f} and {width[1]:.3f} of the x1 and x2 ranges")
else:
    if B is not None and B["y"] is None:
        sys.exit("the type-j events have no y column: give y = f_i(x) for them too")
    yi = A["y"]
    yj = B["y"] if B is not None else None

# y' = y + y_min, starting at 0, when some y are negative; then it is used everywhere
lowest = np.nanmin(yi) if yj is None else min(np.nanmin(yi), np.nanmin(yj))
y_min = 0.0
if Y_SHIFT and lowest < 0:
    raw = Y_SHIFT_ROOM * -lowest
    tenth = 10.0 ** (np.floor(np.log10(raw)) - 1)      # two figures
    y_min = float(np.ceil(raw / tenth) * tenth)       # e.g. 1.486e-5 -> 1.5e-5
if y_min:
    yi = yi + y_min
    yj = yj + y_min if yj is not None else None
    Y = Y + "'"
    Y_AXIS = r"$y' = y - y_0$"
    S = "s'"
    S_AXIS = r"$s'$"
    CBAR_TITLE = r"$y' = y - y_0$"
    ARROW_TO_Y = {"y": r"$\mathrm{bin\ in}\ %s$" % Y, "-ln y": r"$\mathrm{bin\ in}\ %s$" % S}
    if LOG:                                 # the scale shows s'
        CBAR_TITLE = r"$%s = -\ln(y - y_0)$" % S
        if T_SCALE:                         # t = (s'max - s')/(s'max - s'min), from 0 to 1
            S = r"\upsilon"                  # called t before; the letter calls it upsilon
            S_AXIS = r"$\upsilon$"
            CBAR_TITLE = r"$\upsilon = \frac{s^{\prime}_1 - s^{\prime}}{s^{\prime}_1 - s^{\prime}_0}$"
            ARROW_TO_Y["-ln y"] = r"$\mathrm{bin\ in}\ \upsilon$"
    print(f"some y are negative (lowest {lowest:.4g}): using y' = y + y_min with y_min = {y_min:.4g}")


rescale = bool(Y_RESCALE and is_toy and not LOG and not IN_Y)
ry0, ry1 = 0.0, 1.0
if rescale:                                 # y0 = 0, y1 = the toy's largest density
    gx_r, gy_r = np.meshgrid(np.linspace(-1, 1, 801), np.linspace(0, 2 * np.pi, 801))
    f_grid_r = f_toy(gx_r, gy_r)
    ry0, ry1 = float(f_grid_r.min()), float(f_grid_r.max())      # its lowest and largest values
    Y = "z"
    Y_AXIS = r"$z$"
    CBAR_TITLE = r"$z = (y - y_0)/\Delta y,\ \ \Delta y = y_1 - y_0$"
    ARROW_TO_Y = {"y": r"$\mathrm{bin\ in}\ z$", "-ln y": ARROW_TO_Y["-ln y"]}
    HIGHLIGHT_RANGE = tuple(ry0 + (ry1 - ry0) * np.asarray(HIGHLIGHT_RANGE_SCALED))
    print(f"z = (y - y0)/(y1 - y0) with y0 = {ry0:g} and y1 = {ry1:.4g}")


def to_yp(v):
    return (np.asarray(v) - ry0) / (ry1 - ry0)


def sci(v):
    """A number as m x 10^e, in math text."""
    e = int(np.floor(np.log10(abs(v))))
    return r"%.2f\times10^{%d}" % (v / 10.0 ** e, e)


# the variable of the last panel, b = y or -ln y (y <= 0 has no -ln y)
with np.errstate(divide="ignore", invalid="ignore"):
    to_b = (lambda y: -np.log(y)) if LOG else (lambda y: y)
    bi = to_b(yi)
    bj = to_b(yj) if B is not None else None
use_t = bool(LOG and y_min and T_SCALE)
if use_t:                                   # s' -> t = (s'max - s')/(s'max - s'min), clipped to [0, 1]
    pooled_s = np.concatenate([bi[np.isfinite(bi)]] + ([bj[np.isfinite(bj)]] if B is not None else []))
    s_lo, s_hi = np.quantile(pooled_s, T_ENDS)
    bi = np.clip((s_hi - bi) / (s_hi - s_lo), 0.0, 1.0)
    bj = np.clip((s_hi - bj) / (s_hi - s_lo), 0.0, 1.0) if B is not None else None
    print(f"upsilon = (s'1 - s')/(s'1 - s'0) with s'0 = {s_lo:.4g} and s'1 = {s_hi:.4g}")
for label, y, w in (("type i", yi, wi),) + ((("type j", yj, wj),) if B is not None else ()):
    share = w[y <= 0].sum() / w.sum()
    if share > 0:
        print(f"{label}: {100 * share:.2f}% of the events have y <= 0"
              + (" and are not in the -ln y panel" if LOG else ""))

# bin edges of the last panel
okj = np.isfinite(bj) if B is not None else None
edges_b = S_EDGES if LOG else Y_EDGES
if edges_b is None:
    oki = np.isfinite(bi)
    if LOG:
        lo = quantile(bi[oki], wi[oki], 0.001)
        hi = quantile(bi[oki], wi[oki], 0.995)
        if B is not None:
            hi = max(hi, quantile(bj[okj], wj[okj], 0.98))
        if y_min:                           # shifted: every s' is finite, so show them all, with room
            top_s = max(quantile(bi[oki], wi[oki], 0.995),
                        quantile(bj[okj], wj[okj], 0.995) if B is not None else -np.inf)
            hi = top_s + 0.05 * (top_s - lo)          # nearly all events, then a little room
        if use_t:                           # t runs from 0 to 1
            lo, hi = 0.0, 1.0
    else:
        lo = min(0.0, quantile(yi, wi, 0.001), quantile(yj, wj, 0.001) if B is not None else 0.0)
        hi = quantile(yi, wi, 0.999)
    edges_b = round_edges(lo, hi, N_BINS_Y)
    if rescale:                             # round bins in y', from 0 to 1
        edges_b = ry0 + (ry1 - ry0) * np.arange(0.0, 1.0 + 1e-9, Z_STEP)   # 0, 0.2, ..., 1
edges_b = np.asarray(edges_b, dtype=float)
use_range = HIGHLIGHT_RANGE is not None
if use_range:                              # the bins from one edge to another: HL0 up to (not incl.) HL1
    with np.errstate(divide="ignore", invalid="ignore"):
        ends = sorted(to_b(np.asarray(HIGHLIGHT_RANGE, dtype=float)))
    if not (np.isfinite(ends).all() and ends[0] < edges_b[-1] and ends[1] > edges_b[0]):
        print(f"HIGHLIGHT_RANGE {HIGHLIGHT_RANGE} is outside the last panel's range, so the hatched bin is "
              f"the middle one instead")
        use_range = False
if use_range:
    HL0, HL1 = (int(np.argmin(np.abs(edges_b - e))) for e in ends)
    HL0 = min(HL0, len(edges_b) - 2)
    HL1 = min(max(HL1, HL0 + 1), len(edges_b) - 1)
else:
    if HIGHLIGHT_AT is not None:
        at = HIGHLIGHT_AT
    else:                                  # the middle: where the median type-i event is
        at = quantile(bi[np.isfinite(bi)], wi[np.isfinite(bi)], 0.5)
    HL0 = int(np.clip(np.searchsorted(edges_b, at, side="right") - 1, 0, len(edges_b) - 2))
    HL1 = HL0 + 1
band = (np.exp(-edges_b[HL1]), np.exp(-edges_b[HL0])) if LOG else (edges_b[HL0], edges_b[HL1])
coarse_b = edges_b.copy()                                   # the contours stay at these edges
if use_range and SPLIT_LEFT > 1 and not LOG and Y_EDGES is None and HL0 > 0:
    width_hl = HL1 - HL0
    fine_left = np.concatenate([np.linspace(a, b, SPLIT_LEFT + 1)[:-1]
                                for a, b in zip(edges_b[:HL0], edges_b[1:HL0 + 1])])
    edges_b = np.concatenate([fine_left, edges_b[HL0:]])
    HL0 = len(fine_left)
    HL1 = HL0 + width_hl
y_levels = np.sort(np.exp(-edges_b)) if LOG else edges_b     # every bin edge, in y: a band for each bin

# contours of y in the (x1, x2) plane: exact for an estimated y; for a y from the file, only if
# y is a function of x1 and x2 alone
all_x = np.vstack([xi, xj]) if B is not None else xi
all_y = np.concatenate([yi, yj]) if B is not None else yi
if is_toy:                                  # the toy's exact density, on a grid
    gx, gy = np.linspace(*r1, 241), np.linspace(*r2, 241)
    GX, GY = np.meshgrid(gx, gy)
    f_grid = f_toy(GX, GY)
    contour_mode = "grid" if CONTOURS else None
elif estimated:
    contour_mode = "grid" if CONTOURS else None
else:
    if CONTOURS == "auto":
        left = unexplained(all_x, all_y, r1, r2)
        contour_mode = "tri" if left < 0.1 else None
        print(f"y vs (x1, x2): {100 * left:.1f}% of its variance is not explained by x1 and x2, so "
              + ("y is a function of x1 and x2: drawing its contours" if contour_mode else
                 "y depends on more variables: no contours of y"))
    else:
        contour_mode = "tri" if CONTOURS else None
    if not contour_mode and N_BINS_Y_DATA and (S_EDGES if LOG else Y_EDGES) is None:   # finer bins of y
        edges_b = round_edges(edges_b[0], edges_b[-1], N_BINS_Y_DATA)
        at = quantile(bi[np.isfinite(bi)], wi[np.isfinite(bi)], 0.5)
        HL0 = int(np.clip(np.searchsorted(edges_b, at, side="right") - 1, 0, len(edges_b) - 2))
        HL1 = HL0 + 1
        band = (np.exp(-edges_b[HL1]), np.exp(-edges_b[HL0])) if LOG else (edges_b[HL0], edges_b[HL1])
    if contour_mode:
        k = rng.choice(len(all_y), min(40_000, len(all_y)), replace=False)
        _, uniq = np.unique(all_x[k], axis=0, return_index=True)
        k = k[uniq]
        tri = mtri.Triangulation(all_x[k, 0], all_x[k, 1])
        tri.set_mask(mtri.TriAnalyzer(tri).get_flat_tri_mask(0.02))
        tri_y = all_y[k]

# ------------------------------------------------------------------------------------------
# layout, in inches, so the four plot areas are the same size
# ------------------------------------------------------------------------------------------
LEFT, RIGHT, BOTTOM, TOP = 0.85, 0.75, 0.45, 0.7
CB_PAD, CB_H = 0.07, 0.09                                      # colour bar above the top-left panel
show_ratio = B is not None and (RATIO_CURVE is True or (RATIO_CURVE == "data" and not contour_mode))
if show_ratio and RATIO_PROJECTIONS:
    GAP_X += RATIO_GAP                       # room for the dN/dx ratio's values between the columns
color_r = RATIO_COLOR                       # the same everywhere, clear of the red dV curve
short_i = SHORT_I if SHORT_I else NAME_I.strip("$")
short_j = SHORT_J if SHORT_J else NAME_J.strip("$")
names_r = ((short_j, short_i) if RATIO_OF == "j/i" else (short_i, short_j))
ratio_name = r"$%s/%s$" % (names_r[0].strip("$"), names_r[1].strip("$"))
ratio_frac = r"$\dfrac{%s}{%s}$" % (names_r[0].strip("$"), names_r[1].strip("$"))
ratio_small = r"$\frac{%s}{%s}$" % (names_r[0].strip("$"), names_r[1].strip("$"))   # the smaller one


def smoothed_ratio(v_i, v_j, w_i, w_j, lo, hi, n=401, trim=False):
    """The ratio hh/ttbar (or ttbar/hh, RATIO_OF) of the two types' distributions of v, both smoothed by
    the same Gaussian (RATIO_SMOOTH times Scott's width), on n - 1 points from lo to hi; nan wherever
    either type has fewer than RATIO_MIN_EVENTS events under the kernel."""
    ok_i, ok_j = np.isfinite(v_i), np.isfinite(v_j)
    v_i, w_i, v_j, w_j = v_i[ok_i], w_i[ok_i], v_j[ok_j], w_j[ok_j]
    fine = np.linspace(lo, hi, n)
    step = fine[1] - fine[0]

    def n_eff(w):
        return w.sum() ** 2 / (w ** 2).sum()

    def spread(v, w):
        return np.sqrt(np.average((v - np.average(v, weights=w)) ** 2, weights=w))
    bw = (RATIO_SMOOTH * 1.06 * min(spread(v_i, w_i), spread(v_j, w_j))
          * min(n_eff(w_i), n_eff(w_j)) ** -0.2)                         # Scott's rule
    sig = max(bw / step, 0.5)                                            # in fine bins
    m = int(np.ceil(3 * sig))
    kern = np.exp(-0.5 * (np.arange(-m, m + 1) / sig) ** 2)
    kern /= kern.sum()
    inside = np.convolve(np.ones(n - 1), kern, mode="same")             # renormalised at the ends

    def smooth(v, w):
        return np.convolve(np.histogram(v, bins=fine, weights=w)[0], kern, mode="same") / inside
    d_i, d_j = smooth(v_i, w_i) / (w_i.sum() * step), smooth(v_j, w_j) / (w_j.sum() * step)
    under = np.sqrt(2 * np.pi) * sig                                     # fine bins under the kernel
    enough = (smooth(v_i, None) * under >= RATIO_MIN_EVENTS) & (smooth(v_j, None) * under >= RATIO_MIN_EVENTS)
    top_d, bottom_d = (d_j, d_i) if RATIO_OF == "j/i" else (d_i, d_j)
    enough &= bottom_d > 0
    if trim:                                # past the last events the smoothing has nothing to
        mid = (fine[1:] + fine[:-1]) / 2    # average and bends: stop a kernel width before them
        bw_v = sig * step
        first = max(lo, min(v_i.min(), v_j.min()))          # the data, or the plot's end if the data
        last = min(hi, max(v_i.max(), v_j.max()))           # runs past it (cut off there, too)
        enough &= (mid > first + bw_v) & (mid < last - bw_v)
    v_top, v_bottom = (v_j, v_i) if RATIO_OF == "j/i" else (v_i, v_j)
    infinite = []                           # where the bottom type has no events but the top type does
    for beyond in (v_top[v_top < v_bottom.min()], v_top[v_top > v_bottom.max()]):
        if len(beyond) >= max(INF_MIN_EVENTS, INF_MIN_SHARE * len(v_top)):
            edge = v_bottom.min() if beyond.max() < v_bottom.min() else v_bottom.max()
            a, b = sorted((edge, beyond.min() if edge == v_bottom.min() else beyond.max()))
            if b > lo and a < hi:
                infinite.append((max(a, lo), min(b, hi), edge))
    return ((fine[1:] + fine[:-1]) / 2, np.where(enough, top_d / np.where(bottom_d > 0, bottom_d, 1.0), np.nan),
            infinite)


def style_ratio(ax, axis, values):
    """A ratio's own axis (axis = ax.yaxis or ax.xaxis) in its colour: linear from 0, or log over a wide
    range."""
    lo_v, hi_v = np.nanmin(values), np.nanmax(values)
    set_scale, set_lim = (ax.set_yscale, ax.set_ylim) if axis is ax.yaxis else (ax.set_xscale, ax.set_xlim)
    if hi_v / lo_v > 20:
        set_scale("log")
        axis.set_major_formatter(FuncFormatter(lambda v, _: "$%g$" % v))   # 0.1, 1, 10
        set_lim(lo_v / 1.5, hi_v * 1.5)
    else:
        set_lim(0, 1.15 * hi_v)
        axis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
        axis.set_major_formatter(FuncFormatter(lambda v, _: "$%g$" % v))
        if MINOR_TICKS:
            axis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="y" if axis is ax.yaxis else "x", which="both", colors=color_r, labelcolor=color_r,
                   labelsize=TICK_SIZE)                  # values and marks in red


if not is_toy:                              # event files: bigger plots, tighter gaps
    PANEL += PANEL_GROW
    GAP_X -= GAP_SHRINK[0]
    GAP_Y -= GAP_SHRINK[1]
W = LEFT + 2 * PANEL + GAP_X + RIGHT
H = BOTTOM + 2 * PANEL + GAP_Y + TOP
fig = plt.figure(figsize=(W, H))


def panel(col, row, **kw):
    """Axes in column col (0 = left) and row row (0 = top)."""
    left = LEFT + col * (PANEL + GAP_X)
    bottom = BOTTOM + (1 - row) * (PANEL + GAP_Y)
    return fig.add_axes([left / W, bottom / H, PANEL / W, PANEL / H], **kw)


def upright_ylabel(ax, text, size=TITLE_SIZE):
    """Horizontal axis title left of the y axis; placed next to the tick labels at the end."""
    ax.set_ylabel(text, rotation=0, ha="right", va="center", fontsize=size)


ax2d = panel(0, 0)
axx2 = panel(1, 0)
axx1 = panel(0, 1, sharex=ax2d)
axy = panel(1, 1)
cax = fig.add_axes([LEFT / W, (BOTTOM + 2 * PANEL + GAP_Y + CB_PAD) / H, PANEL / W, CB_H / H])
def frac(sym):
    """(1/N) dN/d(sym), with full-size numerators and denominators like the rest of the text."""
    return r"$\dfrac{1}{N}\,\dfrac{\mathrm{d}N}{%s}$" % d(sym)

# ---- top left: (1/N) d^2N/dx1 dx2 of type i, contours of y at the bin edges, type j as dots ----
e1, e2 = np.linspace(*r1, BINS_X + 1), np.linspace(*r2, BINS_X + 1)
h2 = np.histogram2d(xi[:, 0], xi[:, 1], bins=[e1, e2], weights=wi)[0]
width1 = (np.cos(e1[:-1]) - np.cos(e1[1:])) if PER_SIN else np.diff(e1)       # bin sizes: solid angle ...
h2 = (h2 / (wi.sum() * width1[:, None] * np.diff(e2)[None, :])).T            # ... or dx1 dx2
map_is_y = bool(contour_mode) or MAP_Y_WITHOUT_CONTOURS    # does the top-left colour scale show y?
map_in_s = LOG and not contour_mode                      # ... as y, or as s' = -ln y' with LOG (events)
v_map = bi if map_in_s else yi
if not contour_mode and MAP_Y_WITHOUT_CONTOURS:            # y depends on more than (x1, x2): show its
    has_y = np.isfinite(v_map)                             # mean over the type-i events in each cell
    num = np.histogram2d(xi[has_y, 0], xi[has_y, 1], bins=[e1, e2], weights=(wi * v_map)[has_y])[0]
    den = np.histogram2d(xi[has_y, 0], xi[has_y, 1], bins=[e1, e2], weights=wi[has_y])[0]
    h2 = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan).T   # empty cells stay white
if COLOR_SCALE == "log":                    # empty bins stay white
    positive = h2[h2 > 0]
    norm = LogNorm(vmin=max(positive.min(), positive.max() / 100), vmax=positive.max())
else:
    if contour_mode or not MAP_Y_WITHOUT_CONTOURS:
        norm = Normalize(vmin=0, vmax=np.nanmax(h2))
    else:                                   # mean y per cell: from its 1st to its 99th percentile, so
        norm = Normalize(vmin=np.nanpercentile(h2, 1) if y_min else 0,     # a shift by y_min or one noisy
                         vmax=np.nanpercentile(h2, 99))                    # cell keeps the contrast
cmap_i = LinearSegmentedColormap.from_list("type_i", ["#ffffff", COLOR_I])
if map_in_s and not use_t:                  # s' = -ln y' runs the other way: dark at low s' (high y'),
    cmap_i = cmap_i.reversed()              # as dark is at high y' in the y' version
im = ax2d.pcolormesh(e1, e2, h2, cmap=cmap_i, norm=norm, rasterized=True)


def ttbar_fill(ax, values, edges, levels, horizontal=False, lines=None):
    """The filled part of a ttbar histogram: each bar in the top-left panel's colour for the density value
    `levels` (one per bin), or all in FILL_I without GRADIENT_FILL. `lines`: colour of the lines between bars
    (None = none)."""
    if not GRADIENT_FILL or levels is None:       # no colours to match: one light shade
        ax.stairs(values, edges, orientation="horizontal" if horizontal else "vertical", fill=True, color=FILL_I)
        return
    lv = np.asarray(levels, dtype=float)
    if isinstance(norm, LogNorm):
        lv = np.maximum(lv, norm.vmin)
    colors = cmap_i(norm(lv))
    edge, lw = (lines, 0.8) if lines else (colors, 0.3)      # bars outlined in their own colour: no seams
    mid, size = (edges[1:] + edges[:-1]) / 2, np.diff(edges)
    if horizontal:
        ax.barh(mid, values, height=size, color=colors, edgecolor=edge, linewidth=lw, zorder=1)
    else:
        ax.bar(mid, values, width=size, color=colors, edgecolor=edge, linewidth=lw, zorder=1)


def strip_mean(values, edges, along):
    """The mean of the top-left plot's density over each strip a projection sums: (1/N) dN/dx1 (along="x1")
    or dN/dx2 (along="x2") divided by the strip's extent in the other variable (its solid angle for a
    (theta, phi) plane), so a bar gets the average colour of its strip."""
    if along == "x1":
        return values * np.diff(edges) / (np.abs(np.diff(np.cos(edges))) * (r2[1] - r2[0])) if PER_SIN \
            else values / (r2[1] - r2[0])
    return values / abs(np.cos(r1[0]) - np.cos(r1[1])) if PER_SIN else values / (r1[1] - r1[0])


def mean_y(values, edges):
    """The mean y of the type-i events in each bin of `values` (0 where a bin is empty)."""
    ok = np.isfinite(v_map)
    num = np.histogram(values[ok], bins=edges, weights=(wi * v_map)[ok])[0]
    den = np.histogram(values[ok], bins=edges, weights=wi[ok])[0]
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
if contour_mode:
    if contour_mode == "grid":
        draw, drawf, where, values = ax2d.contour, ax2d.contourf, (gx, gy), f_grid
    else:
        draw, drawf, where, values = ax2d.tricontour, ax2d.tricontourf, (tri,), tri_y
    levels = [v for v in y_levels if values.min() < v < values.max()]
    others = [v for v in levels if not np.isclose(v, band).any()       # the band's edges are drawn bold below,
              and not band[0] < v < band[1]]                           # and nothing inside it
    if others:                              # thin contours
        thin = draw(*where, values, levels=others, colors=MARK_COLOR, linewidths=0.5, alpha=0.8)
    with plt.rc_context({"hatch.color": MARK_COLOR}):    # the hatching in MARK_COLOR (the last panel keeps INK)
        drawf(*where, values, levels=list(band), colors="none", hatches=["////"])
    edges = draw(*where, values, levels=list(band), colors=MARK_COLOR, linewidths=1.0)
    if CONTOUR_LABEL_SIZE:                  # the y value of each contour, written on it
        value_text = ((lambda v: "$%g$" % round(float(to_yp(v)), 3)) if rescale   # in y' for the toy
                      else (lambda v: "$%g$" % float("%.2g" % v)))                # 0.04, 0.12, ...
        placed = []                         # label positions so far, as fractions of the panel

        def spots(cs, wanted):
            """For each wanted level, the point of its longest line farthest from the panel's edges
            and from the labels already placed."""
            out = []
            for level, segs in zip(cs.levels, cs.allsegs):
                if not segs or not np.isclose(level, wanted).any():
                    continue
                line = max(segs, key=len)
                if len(line) == 0:
                    continue
                u = (line[:, 0] - r1[0]) / (r1[1] - r1[0])
                v = (line[:, 1] - r2[0]) / (r2[1] - r2[0])
                room = np.minimum.reduce([u, 1 - u, v, 1 - v])
                for pu, pv in placed:
                    room = np.minimum(room, 0.6 * np.hypot(u - pu, v - pv))
                k = int(np.argmax(room))
                placed.append((u[k], v[k]))
                out.append((line[k, 0], line[k, 1]))
            return out
        band_names = {v: value_text(v) for v in band}     # with CONTOUR_NAME, the lower edge says "y = ..."
        if CONTOUR_NAME:
            band_names[band[0]] = "$%s = %s$" % (Y, value_text(band[0]).strip("$"))
        def level_spot(cs, wanted):
            """The point of the wanted level's longest line where it runs most nearly level, away from
            the panel's edges and from the labels already placed (None if it never runs level)."""
            for level, segs in zip(cs.levels, cs.allsegs):
                if not segs or not np.isclose(level, wanted):
                    continue
                line = max(segs, key=len)
                u = (line[:, 0] - r1[0]) / (r1[1] - r1[0])
                v = (line[:, 1] - r2[0]) / (r2[1] - r2[0])
                tilt = np.abs(np.arctan2(np.gradient(v), np.gradient(u)))
                tilt = np.minimum(tilt, np.pi - tilt)                    # 0 where the line is level
                room = np.minimum.reduce([u, 1 - u, v, 1 - v])
                for pu, pv in placed:
                    room = np.minimum(room, 0.6 * np.hypot(u - pu, v - pv))
                good = (tilt < np.radians(8)) & (room > 0.08)
                if not good.any():
                    return None
                k = int(np.argmax(np.where(good, room, -1.0)))
                placed.append((u[k], v[k]))
                return line[k, 0], line[k, 1]
            return None
        flat = level_spot(edges, band[0]) if CONTOUR_NAME and NAME_LEVEL else None
        where = [flat] + spots(edges, [band[1]]) if flat is not None else spots(edges, list(band))
        band_labels = ax2d.clabel(edges, manual=where, fmt=band_names, colors=MARK_COLOR,
                                  fontsize=CONTOUR_LABEL_SIZE, inline_spacing=2)
        if flat is not None:                # the named one written level, where its line runs level
            for label in band_labels:
                if label.get_text() == band_names[band[0]]:
                    label.set_rotation(0)
        for label in band_labels:
            label.set_bbox(dict(facecolor="none", edgecolor="none", pad=0.4, alpha=0.85))
            label.set_path_effects(halo)
        if others:                          # plain values
            names = {v: value_text(v) for v in others}
            labelled = others if CONTOUR_VALUES == "all" else [v for v in [max(others)] if v > max(band)]
            if labelled:
                for label in ax2d.clabel(thin, manual=spots(thin, labelled), fmt=names, colors=MARK_COLOR,
                                         fontsize=CONTOUR_LABEL_SIZE, inline_spacing=2):
                    label.set_path_effects(halo)
elif HATCH_WITHOUT_CONTOURS:                # no contours: the type-i events of the hatched bin, as dots
    inside = np.flatnonzero((bi >= edges_b[HL0]) & (bi < edges_b[HL1]))
    pick = rng.choice(inside, min(N_SCATTER, len(inside)), replace=False)
    ax2d.scatter(xi[pick, 0], xi[pick, 1], s=5, color="white", edgecolors=INK, linewidths=0.4)
mean_labels = []
if not contour_mode and MEAN_CONTOURS:     # contours of the smoothed average of y, or of -ln y with LOG
    v_m = bi if LOG else yi                 # (not of y itself)
    has_y = np.isfinite(v_m)
    pm = density_coords(xi[has_y])
    dm1 = (np.cos(r1[1]), np.cos(r1[0])) if PER_SIN else r1
    n_m = 128
    n_eff = wi[has_y].sum() ** 2 / (wi[has_y] ** 2).sum()
    kw = MEAN_SMOOTH * pm.std(axis=0) * n_eff ** (-1 / 6) / [dm1[1] - dm1[0], r2[1] - r2[0]]   # Scott's rule

    def smoothed(weights):
        """A fine 2-D histogram of the type-i events with these weights, smoothed with the same Gaussian
        kernel for both sums (mirrored at the edges of the range)."""
        grid = np.histogram2d(pm[:, 0], pm[:, 1], bins=[np.linspace(*dm1, n_m + 1), np.linspace(*r2, n_m + 1)],
                              weights=weights)[0]
        for axis in (0, 1):
            sigma = max(kw[axis] * n_m, 0.5)
            m = int(np.ceil(3 * sigma))
            kernel = np.exp(-0.5 * (np.arange(-m, m + 1) / sigma) ** 2)
            pad = [(0, 0), (0, 0)]
            pad[axis] = (m, m)
            grid = np.apply_along_axis(np.convolve, axis, np.pad(grid, pad, mode="symmetric"),
                                       kernel / kernel.sum(), mode="valid")
        return grid
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_grid = (smoothed((wi * v_m)[has_y]) / smoothed(wi[has_y])).T   # sum of w v over sum of w
    c_m1 = np.linspace(*dm1, n_m + 1)
    c_m1 = (c_m1[1:] + c_m1[:-1]) / 2
    c_m2 = np.linspace(*r2, n_m + 1)
    c_m2 = (c_m2[1:] + c_m2[:-1]) / 2
    if PER_SIN:                                    # back from cos(x1) to x1 for drawing
        c_m1 = np.arccos(np.clip(c_m1, -1, 1))
    lo_m, hi_m = np.nanpercentile(mean_grid, [2, 98])
    mean_levels = MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo_m, hi_m)
    mean_levels = mean_levels[(mean_levels > lo_m) & (mean_levels < hi_m)]
    if len(mean_levels):
        cs_mean = ax2d.contour(c_m1, c_m2, mean_grid, levels=mean_levels, colors=MARK_COLOR, linewidths=0.9)
        mpow = int(np.floor(np.log10(norm.vmax))) if norm.vmax < 1e-2 and not LOG else 0   # the bar's power
        mean_names = {v: "$%g$" % round(v / 10.0 ** mpow, 6) for v in mean_levels}
        top_level = mean_levels[len(mean_levels) // 2]              # one of them says what they are
        mean_names[top_level] = r"$\langle %s\rangle = %g$" % (S if LOG else Y,
                                                                round(top_level / 10.0 ** mpow, 6))
        pieces = []                         # every long piece of every contour, left to right
        for level, segs in zip(cs_mean.levels, cs_mean.allsegs):
            for seg in segs:
                v = (seg[:, 1] - r2[0]) / (r2[1] - r2[0])
                if len(seg) > 1 and v.max() - v.min() > 0.2:
                    pieces.append((np.mean(seg[:, 0]), seg))
        pieces.sort(key=lambda p: p[0])
        where_m = []                        # one value per piece, low and high in turn so they stay apart
        for k, (_, seg) in enumerate(pieces):
            v = (seg[:, 1] - r2[0]) / (r2[1] - r2[0])
            j = int(np.argmin(np.abs(v - (0.25 if k % 2 == 0 else 0.75))))
            where_m.append((seg[j, 0], seg[j, 1]))
        named_done = False
        mean_labels = ax2d.clabel(cs_mean, manual=where_m, fmt=mean_names, colors=MARK_COLOR,
                                  fontsize=CONTOUR_LABEL_SIZE, inline=False)   # written level across
        for label in mean_labels:
            if label.get_text() == mean_names[top_level] and named_done:     # the name only once
                label.set_text(r"$%g$" % round(top_level / 10.0 ** mpow, 6))
            named = label.get_text() == mean_names[top_level]
            named_done = named_done or named
            label.set_rotation(90)                                          # along the upright lines
                                                                # along its upright line, the rest level
            label.set_multialignment("center")
            label.set_path_effects(halo)
        print("contours of the smoothed average y at " + ", ".join("%.3g" % v for v in mean_levels)
              + " (y depends on more than (x1, x2), so these are of its average, not of y)")
if B is not None:
    p = np.abs(wj) / np.abs(wj).sum() if B["w"] is not None else None
    pick = rng.choice(len(xj), min(N_SCATTER, len(xj)), replace=p is not None, p=p)
    ax2d.scatter(xj[pick, 0], xj[pick, 1], s=7, color=COLOR_J, edgecolors=DOT_EDGE_J, linewidths=0.35)
ax2d.set(xlim=r1, ylim=r2)
ax2d.set_xlabel(X1_AXIS_TOP, fontsize=TITLE_SIZE)
upright_ylabel(ax2d, "$%s$" % X2)
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cax.xaxis.set_ticks_position("top")
cax.xaxis.set_label_position("top")
cax.set_xlabel(CBAR_TITLE if map_is_y else                     # f_ttbar where the colours show y; otherwise
               r"$\dfrac{1}{N}\,\dfrac{\mathrm{d}^2N}{\mathrm{d}\Omega}$",   # the events' 2-D density
               fontsize=TITLE_SIZE)                       # on top, above the tick values,
cax.xaxis.label.set_verticalalignment("bottom")              # like the dN/dphi title
cb.ax.tick_params(which="major", labelsize=TICK_SIZE, length=0.75 * TICK_LEN, width=0.6)
cb.ax.tick_params(which="minor", length=0.4 * TICK_LEN, width=0.5)   # shorter than the majors
if COLOR_SCALE == "log":
    cax.xaxis.set_major_locator(LogLocator(subs=(1.0, 3.0)))
    cax.xaxis.set_minor_formatter(NullFormatter())
else:
    cax.xaxis.set_major_locator(MaxNLocator(4))
    if norm.vmax < 1e-2:                    # tiny values: 1, 2, 3 on the bar, the power of ten in its title
        cpow = int(np.floor(np.log10(norm.vmax)))      # (the format itself is set with the others below)
        cax.set_xlabel(cax.get_xlabel() + r"$\ \ [10^{%d}]$" % cpow, fontsize=TITLE_SIZE)
if y_min and not LOG:                       # the shift, at the top, where y' is defined
    cax.set_xlabel(cax.get_xlabel() + r"$,\ \ y_0 = %s$" % sci(-y_min), fontsize=TITLE_SIZE,
                   loc="left")                      # one line, from the bar's left end
cb.outline.set_linewidth(0.6)

# ---- the "volume": V(y) = area where f_i > y; the hatched band between two contours is a shell dV ----
volume = SHOW_VOLUME and bool(contour_mode)
dv_text = None                                   # the Delta V label, if there is one
if volume:
    g1, g2 = np.linspace(*r1, 401), np.linspace(*r2, 401)
    G1, G2 = np.meshgrid((g1[1:] + g1[:-1]) / 2, (g2[1:] + g2[:-1]) / 2)   # centres of a fine grid
    if is_toy:
        FG = f_toy(G1, G2)
    elif contour_mode == "grid":
        FG = f_at(density_coords(np.column_stack([G1.ravel(), G2.ravel()]))).reshape(G1.shape)
    else:
        FG = np.ma.filled(mtri.LinearTriInterpolator(tri, tri_y)(G1, G2), np.nan)
    cell = np.broadcast_to((g1[1] - g1[0]) * (g2[1] - g2[0]) * (np.sin(G1) if PER_SIN else 1.0), G1.shape)
    yb = np.exp(-edges_b) if LOG else edges_b                     # bin edges, in y
    in_shell = [(FG >= min(a, b)) & (FG < max(a, b)) for a, b in zip(yb[:-1], yb[1:])]
    shell = np.array([cell[s].sum() for s in in_shell])                      # dV of each bin
    y_shell = np.array([np.average(FG[s], weights=cell[s]) if cell[s].sum() > 0 else 0.0
                        for s in in_shell])                                     # y in that shell
    from_volume = y_shell * shell / np.diff(edges_b)            # (1/N) dN/dy = dV/d(-ln y), bin by bin
    if VOLUME_CURVE:                    # the same on bins 20 times narrower, over the whole panel (zero where
        has = np.isfinite(FG) & (FG > 0)   # f has no values), then smoothed with a Gaussian
        fb = -np.log(FG[has]) if LOG else FG[has]                      # each grid cell's y, or -ln y
        width_b = np.diff(edges_b).min()
        fine = np.linspace(edges_b[0], edges_b[-1], int(round((edges_b[-1] - edges_b[0]) / width_b)) * 20 + 1)
        raw = np.histogram(fb, bins=fine, weights=(FG * cell)[has])[0] / np.diff(fine)
        mid_fine = (fine[1:] + fine[:-1]) / 2
        sigma = CURVE_SMOOTH * 20                                        # in fine bins
        m = int(np.ceil(3 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-m, m + 1) / sigma) ** 2)
        curve = (np.convolve(raw, kernel, mode="same")                   # renormalised where the kernel
                 / np.convolve(np.ones_like(raw), kernel, mode="same"))  # runs off the panel
    # Delta V: inside the hatched band where it is widest (most band around it); moved further below if it
    # would sit on a contour value
    in_band = (FG >= band[0]) & (FG < band[1])
    box = np.ones(25) / 25                                         # about 6% of the panel
    around = np.apply_along_axis(np.convolve, 0, in_band.astype(float), box, mode="same")
    around = np.apply_along_axis(np.convolve, 1, around, box, mode="same")
    U, V = (G1 - r1[0]) / (r1[1] - r1[0]), (G2 - r2[0]) / (r2[1] - r2[0])
    ok = in_band & (U > 0.06) & (U < 0.94) & (V > 0.06) & (V < 0.94)
    if ok.any():
        score = np.where(ok, around - 0.5 * ((U - 0.5) ** 2 + (V - 0.5) ** 2), -9.0)  # wide, and not in a corner
        dv_spots = np.argsort(score, axis=None)[::-1][:int(ok.sum())]               # best first
        k = dv_spots[0]
        if DV_OUTSIDE:                  # off the plot, with a line into the band (its end is chosen below)
            dv_value = float((cell * in_band).sum())   # the band's area in the plane, written with it
            dv_text = ax2d.annotate(r"$\Delta V = %.1f$" % dv_value, xy=(G1.flat[k], G2.flat[k]), xytext=DV_AT,
                                    textcoords="axes fraction", ha="center", va="center", fontsize=NOTE_SIZE,
                                    color=INK, annotation_clip=False,
                                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkA=2, shrinkB=0))
            dv_dot, = ax2d.plot([G1.flat[k]], [G2.flat[k]], "o", ms=3.2, color=INK, zorder=6)
        else:                           # inside the band, on a navy box, so it reads over the hatching
            dv_text = ax2d.text(G1.flat[k], G2.flat[k], r"$\Delta V$", ha="center", va="center",
                                fontsize=NOTE_SIZE, color=MARK_COLOR,
                                bbox=dict(boxstyle="round,pad=0.12", facecolor=COLOR_I, edgecolor="none", alpha=0.9))

# ---- top right: (1/N) dN/dx2, drawn like the bottom plots ----
pi2 = density(xi[:, 1], e2, wi)
ttbar_fill(axx2, pi2, e2, mean_y(xi[:, 1], e2) if map_is_y else strip_mean(pi2, e2, "x2"))
                                     # each bin in its mean y's colour, or in its strip's mean colour
axx2.stairs(pi2, e2, color=LINE_I, lw=1.2, baseline=None, zorder=2)
top2 = pi2.max()
if B is not None:
    pj2 = density(xj[:, 1], e2, wj)
    axx2.stairs(pj2, e2, color=LINE_J, lw=1.6, baseline=None, zorder=2.5)
    top2 = max(top2, pj2.max())
axx2.set_xlim(*r2)
axx2.set_ylim(0, 1.15 * top2)
axx2.set_xlabel("$%s$" % X2, fontsize=TITLE_SIZE)
upright_ylabel(axx2, frac(X2))

# ---- bottom left: (1/N) dN/dx1, sharing the x1 axis ----
pi1 = density(xi[:, 0], e1, wi)
ttbar_fill(axx1, pi1, e1, mean_y(xi[:, 0], e1) if map_is_y else strip_mean(pi1, e1, "x1"))
                                     # each bin in its mean y's colour, or in its strip's mean colour
axx1.stairs(pi1, e1, color=LINE_I, lw=1.2, baseline=None, zorder=2)
top1 = pi1.max()
if B is not None:
    pj1 = density(xj[:, 0], e1, wj)
    axx1.stairs(pj1, e1, color=LINE_J, lw=1.6, baseline=None, zorder=2.5)
    top1 = max(top1, pj1.max())
axx1.set_ylim(0, (1.5 if not (show_ratio and not volume) else 1.15) * top1)   # room for its legend on top,
                                                                          # when it has one
axx1.set_xlabel(X1_AXIS, fontsize=TITLE_SIZE)
handles = [Patch(facecolor=cmap_i(0.55) if GRADIENT_FILL else FILL_I, edgecolor=LINE_I, lw=1.0,
                 label=NAME_I)]
if B is not None:
    handles.append(Line2D([], [], color=LINE_J, lw=1.6, marker="o", ms=3.5, mfc=COLOR_J,
                          mec=DOT_EDGE_J, mew=0.4, label=NAME_J))
if show_ratio and RATIO_PROJECTIONS:        # the ratio over x, on an axis right of the dN/dx plot
    g1r, rat1, inf1 = smoothed_ratio(xi[:, 0], xj[:, 0], wi, wj, *r1)
    ax1r = axx1.twinx()
    ax1r.plot(g1r, rat1, color=color_r, lw=RATIO_LW, zorder=6, solid_capstyle="round")
    style_ratio(ax1r, ax1r.yaxis, rat1)
    ax1r.set_ylabel(ratio_frac, rotation=0, ha="left", va="center", color=color_r, fontsize=TITLE_SIZE)
    ax1r.yaxis.set_label_position("right")
    ax1r.spines["right"].set_color(color_r)
    g2r, rat2, inf2 = smoothed_ratio(xi[:, 1], xj[:, 1], wi, wj, *r2)  # ... and over x2, right of dN/dx2
    ax2r = axx2.twinx()
    ax2r.plot(g2r, rat2, color=color_r, lw=RATIO_LW, zorder=6, solid_capstyle="round")
    style_ratio(ax2r, ax2r.yaxis, rat2)
    ax2r.set_ylabel(ratio_frac, rotation=0, ha="left", va="center", color=color_r, fontsize=TITLE_SIZE)
    ax2r.yaxis.set_label_position("right")
    ax2r.spines["right"].set_color(color_r)
if not contour_mode and HATCH_WITHOUT_CONTOURS:
    handles.append(Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec=INK, mew=0.5,
                          label=NAME_I + r" $\mathrm{in\ the}$" + "\n" + r"$\mathrm{hatched\ bin}$"))
if DNDX_TITLE == "outside":                  # the (1/N) dN/dx title: left of the plot at the top ...
    dndx_title = axx1.text(TITLE_LEFT, TITLE_TOP, frac(X1), transform=axx1.transAxes, ha="left", va="top",
                           fontsize=TITLE_SIZE)
else:                                        # ... or upright, left of its axis at mid-height
    upright_ylabel(axx1, frac(X1))
legend_x1 = None if show_ratio and not volume else axx1.legend(   # (none when the dN/dy legend names them)
                        handles=handles, loc="upper left" if contour_mode else "best",
                        bbox_to_anchor=(LEGEND_LEFT, TITLE_TOP) if contour_mode else None,
                        borderaxespad=0 if contour_mode else 0.5, borderpad=0, labelspacing=0.1,
                        handlelength=1.4, handletextpad=0.5)    # ... and the legend inside, level
                                                                          # with it, its entries close together

# ---- bottom right: (1/N) dN/dy, or (1/N) dN/ds with s = -ln y ----
oki = np.isfinite(bi)
ci = np.histogram(bi[oki], bins=edges_b, weights=wi[oki])[0] / (wi.sum() * np.diff(edges_b))
mid_y = (edges_b[1:] + edges_b[:-1]) / 2
if use_t and GRADIENT_FILL:              # upsilon panel: the same white-to-navy gradient as Fig. 3 of the letter,
    lo_u, hi_u = np.percentile(bi[oki], [5, 60])   # linear in upsilon between its 5% and 60% ttbar quantiles
    shade_u = np.clip((mid_y - lo_u) / (hi_u - lo_u), 0, 1)
    cols_u = cmap_i(0.08 + 0.92 * shade_u)
    axy.bar(mid_y, ci, width=np.diff(edges_b), color=cols_u, edgecolor=cols_u, linewidth=0.3, zorder=1)
else:
    ttbar_fill(axy, ci, edges_b, (mid_y if map_in_s or not LOG else np.exp(-mid_y)) if map_is_y else None)
                                     # each bin in its y's colour, where y is the top-left density
axy.stairs(ci, edges_b, color=LINE_I, lw=1.2, baseline=None, zorder=2)
tallest = ci.max()
if B is not None:
    cj = np.histogram(bj[okj], bins=edges_b, weights=wj[okj])[0] / (wj.sum() * np.diff(edges_b))
    axy.stairs(cj, edges_b, color=LINE_J, lw=1.6, baseline=None, zorder=2.5)
    tallest = max(tallest, cj.max())
top = 1.12 * tallest
if contour_mode or HATCH_WITHOUT_CONTOURS:   # the hatched bins: the events of the hatched band on top left
  axy.stairs(ci[HL0:HL1], edges_b[HL0:HL1 + 1], fill=True, facecolor="none", hatch="////", zorder=1.5,
             edgecolor=MARK_COLOR if GRADIENT_FILL and contour_mode else INK,
             lw=0 if GRADIENT_FILL and contour_mode else 0.8)                                 # the hatching,
                                                                    # as tall as the histogram
if GRADIENT_FILL and contour_mode:        # and its two sides in white, like the band's edges in the top left
    axy.vlines([edges_b[HL0], edges_b[HL1]], 0, [ci[HL0], ci[HL1 - 1]], colors=MARK_COLOR, lw=1.0, zorder=1.8)
axy.set(xlim=(edges_b[0], edges_b[-1]), ylim=(0, top))
if show_ratio:                             # hh/ttbar (or ttbar/hh), smoothed, on its own axis on the right
    mid_r, ratio, inf_y = smoothed_ratio(bi, bj, wi, wj, edges_b[0], edges_b[-1], trim=not use_t)
    axr = axy.twinx()
    axr.plot(mid_r, ratio, color=color_r, lw=RATIO_LW, zorder=6, solid_capstyle="round")   # gaps: few events
    style_ratio(axr, axr.yaxis, ratio)
    if RATIO_PROJECTIONS:                   # the two projections: linear, on one scale, to compare
        top_p = 1.15 * max(np.nanmax(rat1), np.nanmax(rat2))
        for ax_p in (ax1r, ax2r):
            ax_p.set_yscale("linear")
            ax_p.set_ylim(0, top_p)
            ax_p.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
            ax_p.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "$%g$" % v))
            if MINOR_TICKS:
                ax_p.yaxis.set_minor_locator(AutoMinorLocator())
    if RATIO_PROJECTIONS and axr.get_yscale() == ax1r.get_yscale() == ax2r.get_yscale() == "linear":
        top_r = 1.15 * max(np.nanmax(ratio), np.nanmax(rat1), np.nanmax(rat2))   # one scale for all three
        axr.set_ylim(0, top_r)
        ax1r.set_ylim(0, top_r)
        ax2r.set_ylim(0, top_r)

    def draw_infinite(ax_r, grid, values, infinite):
        """Where the ratio is infinite: from the curve's last value, a dotted rise to the top of the axis,
        a flat line along it, and the top marked with infinity."""
        if not infinite:
            return
        top = ax_r.get_ylim()[1]
        for a, b, edge in infinite:
            ax_r.plot([a, b], [top, top], color=color_r, lw=RATIO_LW + 0.6, zorder=6, clip_on=False,
                      solid_capstyle="butt")
            known = np.flatnonzero(np.isfinite(values))
            if len(known):                  # from the nearest end of the finite curve
                k = known[np.argmin(np.abs(grid[known] - edge))]
                ax_r.plot([grid[k], edge], [values[k], top], color=color_r, lw=RATIO_LW, ls=":", zorder=6,
                          clip_on=False)
        ticks = [t for t in ax_r.get_yticks() if ax_r.get_ylim()[0] <= t < 0.9 * top]
        form = ax_r.yaxis.get_major_formatter()
        ax_r.yaxis.set_major_locator(FixedLocator(ticks + [top]))
        ax_r.yaxis.set_major_formatter(FixedFormatter([form(t) for t in ticks] + [r"$\infty$"]))
    draw_infinite(axr, mid_r, ratio, inf_y)
    if RATIO_PROJECTIONS:
        draw_infinite(ax1r, g1r, rat1, inf1)
        draw_infinite(ax2r, g2r, rat2, inf2)
    axr.set_ylabel(ratio_frac, rotation=0, ha="left", va="center", color=color_r, fontsize=TITLE_SIZE)
    axr.yaxis.set_label_position("right")
    axr.spines["right"].set_color(color_r)
    if not volume:                          # a legend inside: the two types and the ratio
        handles_y = [Patch(facecolor=cmap_i(0.55) if GRADIENT_FILL else FILL_I, edgecolor=LINE_I, lw=1.0,
                           label=NAME_I),
                     Line2D([], [], color=LINE_J, lw=1.6, marker="o", ms=3.5, mfc=COLOR_J, mec=DOT_EDGE_J,
                            mew=0.4, label=NAME_J),
                     Line2D([], [], color=color_r, lw=RATIO_LW, label=ratio_small)]   # compact fraction
        leg_r = axr.legend(handles=handles_y, loc="upper left", bbox_to_anchor=(0.5, TITLE_TOP),
                           borderaxespad=0, borderpad=0, labelspacing=0.1, handlelength=1.4,
                           handletextpad=0.5)             # moved to a free spot, clear of the curves, below
        leg_r.get_texts()[-1].set_color(color_r)          # the ratio's name in red, like its axis
axy.set_xlabel(S_AXIS if LOG else (Y_AXIS if contour_mode else Y_AXIS.replace(r"(x,\phi)", "")),
               fontsize=TITLE_SIZE)                  # without contours, y depends on more than (x, phi)
if DNDY_TITLE == "outside":                 # left of the plot at the top, level with the dN/dx title
    dndy_title = axy.text(TITLE_LEFT, TITLE_TOP, frac(S if LOG else Y), transform=axy.transAxes,
                          ha="left", va="top", fontsize=TITLE_SIZE)
elif DNDY_TITLE == "inside":                # inside, near the top; moved to a free spot further below
    dndy_title = axy.text(0.97, TITLE_TOP, frac(S if LOG else Y), transform=axy.transAxes,
                          ha="right", va="top", fontsize=TITLE_SIZE)
else:
    upright_ylabel(axy, frac(S if LOG else Y))
if volume:
    mid_b = (edges_b[1:] + edges_b[:-1]) / 2
    name_v = (r"$y\,\dfrac{\mathrm{d}V}{\mathrm{d}(-\ln y)}$" if LOG else
              r"$-\Delta y\,(z + z_0)\,\frac{\mathrm{d}V}{\mathrm{d}z}$" if rescale
              else                                                    # = (1/N) dN/dz, on two lines
              r"$\dfrac{\mathrm{d}V}{\mathrm{d}(-\ln y)}$")
    if VOLUME_CURVE:
        axy.plot(mid_fine, curve, color=CURVE_COLOR, lw=1.0, zorder=5, solid_capstyle="round", label=name_v)
    else:
        axy.plot(mid_b, from_volume, "o", color=CURVE_COLOR, ms=3.2, zorder=5, label=name_v)
    legend_y = axy.legend(loc="upper left", bbox_to_anchor=(0.5, TITLE_TOP), borderaxespad=0, borderpad=0,
                          handletextpad=0.4, handlelength=1.6 if VOLUME_CURVE else 1.0,  # inside, level with
                          fontsize=LEGEND_SIZE - (2 if rescale else 0))
                                                                      # dN/dy; moved to a free spot below

# tick labels without trailing zeros, and the upright axis titles just left of the tick labels
plain = FuncFormatter(lambda v, _: "$%g$" % v)              # in the math font, like all the other text
for ax in (ax2d, axx1, axx2, axy, cax):
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
if COLOR_SCALE != "log" and norm.vmax < 1e-2:                   # the colour bar's tiny values, over 10^cpow
    cax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "$%g$" % round(v / 10.0 ** cpow, 6)))
def pi_ticks(axis, lo, hi):
    """Ticks at multiples of pi/2, labelled with pi, if lo and hi are such multiples."""
    q = np.array([lo, hi]) / (np.pi / 2)
    if not PI_TICKS or not np.allclose(q, np.round(q)):
        return False
    k = np.arange(int(round(q[0])), int(round(q[1])) + 1)
    if len(k) > 5 or not PI_HALVES:
        k = k[k % 2 == 0]
    names = {0: "$0$", 1: r"$\pi/2$", 2: r"$\pi$", 3: r"$3\pi/2$", 4: r"$2\pi$", -1: r"$-\pi/2$", -2: r"$-\pi$"}
    axis.set_ticks(k * np.pi / 2, [names.get(m, r"$%d\pi/2$" % m) for m in k])
    if MINOR_TICKS:
        axis.set_minor_locator(MultipleLocator(np.pi / 8 if PI_HALVES else np.pi / 4))
    return True


for axis, (lo, hi) in ((ax2d.xaxis, r1), (ax2d.yaxis, r2), (axx2.xaxis, r2)):
    if not pi_ticks(axis, lo, hi):
        axis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        if MINOR_TICKS:
            axis.set_minor_locator(AutoMinorLocator())
for axis in (axx1.yaxis, axx2.yaxis, axy.yaxis):            # the (1/N) dN/d... axes: tick marks, but values
    axis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))   # only if DENSITY_VALUES (or DNDY_VALUES)
    if not (DENSITY_VALUES or (axis is axy.yaxis and DNDY_VALUES)):        # no values, except the zero
        axis.set_major_formatter(FuncFormatter(lambda v, _p: "$0$" if abs(v) < 1e-12 else ""))
    if MINOR_TICKS:
        axis.set_minor_locator(AutoMinorLocator())
if is_toy and not rescale and not LOG:      # dN/dy (the toy in y): two values, 0 and 12 per unit y, the 12
    axy.yaxis.set_major_locator(FixedLocator([0.0, 12.0]))      # high enough to leave its title beside the axis
    axy.yaxis.set_major_formatter(FixedFormatter(["$0$", "$12$"]))
    if MINOR_TICKS:
        axy.yaxis.set_minor_locator(MultipleLocator(3.0))
if rescale:                                 # dN/dz: two values, 0 and 1 per unit z, to read heights off
    per_z = 1.0 / (ry1 - ry0)                # 1 per unit z, in the per-unit-y units it is drawn in
    axy.yaxis.set_major_locator(FixedLocator([0.0, 2 * per_z]))
    axy.yaxis.set_major_formatter(FixedFormatter(["$0$", "$2$"]))
    if MINOR_TICKS:
        axy.yaxis.set_minor_locator(MultipleLocator(per_z / 2))
axy.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))   # 0, 0.05, 0.1, 0.15
if rescale:                                 # the toy: values of y' on the axis and on the colour bar
    yp_ticks = ry0 + (ry1 - ry0) * np.array([0, 0.25, 0.5, 0.75, 1.0])   # every other bin edge
    yp_form = FuncFormatter(lambda v, _: "$%g$" % round(float(to_yp(v)), 3))
    axy.xaxis.set_major_locator(FixedLocator(yp_ticks))
    axy.xaxis.set_major_formatter(yp_form)
    cax.xaxis.set_major_locator(FixedLocator(yp_ticks[yp_ticks <= norm.vmax * 1.0001]))
    cax.xaxis.set_major_formatter(yp_form)
if np.abs(edges_b).max() < 1e-2:                      # tiny values (e.g. a density in many angles):
    power = int(np.floor(np.log10(np.abs(edges_b).max())))  # 1, 2, 3 on the axis, the power of ten in
    axy.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "$%g$" % round(v / 10.0 ** power, 6)))  # the title
    axy.set_xlabel(axy.get_xlabel() + r"$\ \ [10^{%d}]$" % power, fontsize=TITLE_SIZE)
if MINOR_TICKS:
    axy.xaxis.set_minor_locator(AutoMinorLocator())
# tick marks on all four sides of each plot; values only on the sides chosen above
for ax, x_values, y_values in ((ax2d, "bottom", "left"), (axx1, "bottom", "left"),
                               (axy, "bottom", "left"), (axx2, "bottom", "left")):
    ax.tick_params(axis="x", which="both", bottom=MIRROR_TICKS or x_values == "bottom",
                   top=MIRROR_TICKS or x_values == "top",
                   labelbottom=x_values == "bottom", labeltop=x_values == "top")
    ax.tick_params(axis="y", which="both", left=MIRROR_TICKS or y_values == "left",
                   right=MIRROR_TICKS or y_values == "right",
                   labelleft=y_values == "left", labelright=y_values == "right")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

frame_2d = ax2d.get_window_extent(renderer)
for label in mean_labels:                   # contour values kept inside the plot's edges
    b = label.get_window_extent(renderer)
    dx = max(frame_2d.x0 + 3 - b.x0, 0) + min(frame_2d.x1 - 3 - b.x1, 0)
    dy = max(frame_2d.y0 + 3 - b.y0, 0) + min(frame_2d.y1 - 3 - b.y1, 0)
    if dx or dy:
        px, py = ax2d.transData.transform(label.get_position())
        label.set_position(ax2d.transData.inverted().transform((px + dx, py + dy)))

if dv_text is not None:
    values_boxes = [t.get_window_extent(renderer).padded(3) for t in ax2d.texts     # the contour values
                    if t is not dv_text and t.get_visible() and t.get_text().strip()]
    frame = ax2d.get_window_extent(renderer)
    if DV_OUTSIDE:              # the line's end: the nearest wide part of the band, the line clear of the values
        start = ax2d.transAxes.transform(DV_AT)
        wide = dv_spots[around.flat[dv_spots] >= 0.6 * around.flat[dv_spots].max()]
        ends = ax2d.transData.transform(np.column_stack([G1.flat[wide], G2.flat[wide]]))
        t = np.linspace(0.0, 1.0, 40)[:, None]
        for j in np.argsort(np.hypot(*(ends - start).T)):
            path = start + t * (ends[j] - start)                     # points along the line
            if not any(o.contains(px, py) for o in values_boxes for px, py in path):
                dv_text.xy = (G1.flat[wide[j]], G2.flat[wide[j]])
                dv_dot.set_data([G1.flat[wide[j]]], [G2.flat[wide[j]]])
                break
        else:
            print("Delta V: every line to the band crosses a contour value")
    else:                       # inside: the best spot in the band where it touches none of the values
        for k in dv_spots[::4]:
            dv_text.set_position((G1.flat[k], G2.flat[k]))
            b = dv_text.get_window_extent(renderer)
            if (frame.x0 <= b.x0 and b.x1 <= frame.x1 and frame.y0 <= b.y0 and b.y1 <= frame.y1
                    and not any(b.overlaps(o) for o in values_boxes)):
                break
        else:
            dv_text.set_position((G1.flat[dv_spots[0]], G2.flat[dv_spots[0]]))
            print("Delta V found no spot in the band clear of the contour values")


def free_spot(ax, artist, place, edges, heights, avoid=None, top=TITLE_TOP, margin=0.02, from_left=False):
    """Move a text or legend whose top is at `top` (axes fraction) to the right-most place where it is clear
    of the histograms (heights per bin of `edges`, in data units, with the axis starting at 0) and of the x
    range `avoid`; place(u) puts its left edge at u. Returns False, leaving it where it is, if there is none."""
    box = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())
    lo, hi = ax.get_xlim()
    y_top = ax.get_ylim()[1]
    tries = np.linspace(1 - margin - box.width, margin, 80)       # left edge, from the right ...
    for u in (tries[::-1] if from_left else tries):               # ... or from the left
        a = lo + (u - margin) * (hi - lo)
        b = lo + (u + box.width + margin) * (hi - lo)
        if avoid is not None and b > avoid[0] and a < avoid[1]:
            continue
        over = (edges[1:] > a) & (edges[:-1] < b)
        if over.any() and heights[over].max() > (top - box.height - margin) * y_top:
            continue
        place(u)
        return True
    return False


heights_y = np.maximum(ci, cj) if B is not None else ci.copy()   # what is drawn in each bin of the last panel
if volume and VOLUME_CURVE:                                        # ... including the dV/d(-ln y) curve
    for kb in range(len(heights_y)):
        here = (mid_fine >= edges_b[kb]) & (mid_fine < edges_b[kb + 1])
        if here.any():
            heights_y[kb] = max(heights_y[kb], curve[here].max())
if DNDY_TITLE == "inside":
    def put_title(u):
        dndy_title.set_position((u, TITLE_TOP))
        dndy_title.set_horizontalalignment("left")
    if not free_spot(axy, dndy_title, put_title, edges_b, heights_y):
        print("the (1/N) dN/dy title found no free spot inside its plot; try DNDY_TITLE = \"outside\"")
    fig.canvas.draw()
if volume:
    put_legend = lambda u: legend_y.set_bbox_to_anchor((u, TITLE_TOP), transform=axy.transAxes)
    found = free_spot(axy, legend_y, put_legend, edges_b, heights_y)
    for _ in range(4):                      # no room: raise the plot's top a little and look again
        if found:
            break
        axy.set_ylim(0, axy.get_ylim()[1] * 1.2)
        found = free_spot(axy, legend_y, put_legend, edges_b, heights_y)
    if not found:                           # still none: just above the plot instead
        try:
            legend_y.set_loc("lower right")
        except AttributeError:                                  # matplotlib before 3.8
            legend_y._loc = 4
        legend_y.set_bbox_to_anchor((1.0, 1.02), transform=axy.transAxes)
        print("the dV/d(-ln y) legend found no free spot inside its plot, so it sits just above it")
if show_ratio and not volume:               # the data legend: at the upper right, the histograms kept tall
    tall = np.maximum(ci, cj if cj is not None else 0)  # in the corner over the lower histograms:
    third = max(len(tall) // 3, 1)                      # upper right, or upper left (as for s')
    on_left = tall[-third:].max() > tall[:third].max()
    try:
        leg_r.set_loc("upper left" if on_left else "upper right")
    except AttributeError:                  # matplotlib before 3.8
        leg_r._loc = 2 if on_left else 1
    leg_r.set_bbox_to_anchor((0.02 if on_left else 0.98, TITLE_TOP), transform=axr.transAxes)
    for _ in range(0):                      # (the old search for a free spot, which raised the plot)
        heights_r = heights_y.copy()
        for kb in range(len(heights_r)):   # the ratio curve, in the units of the left axis
            here = (mid_r >= edges_b[kb]) & (mid_r < edges_b[kb + 1]) & np.isfinite(ratio)
            if here.any():
                heights_r[kb] = max(heights_r[kb], ratio[here].max() / axr.get_ylim()[1] * axy.get_ylim()[1])
        if free_spot(axy, leg_r, lambda u: leg_r.set_bbox_to_anchor((u, TITLE_TOP), transform=axr.transAxes),
                     edges_b, heights_r, margin=0.05, from_left=LOG):   # top left for -ln y
            break
        axy.set_ylim(0, 1.15 * axy.get_ylim()[1])


def title_at_plot_top(ax):
    """A right-hand axis title with its top level with the top of the plot, where the values start;
    the values it would cover are left out."""
    shown = [t for t in ax.yaxis.get_majorticklabels() if t.get_visible() and t.get_text()]
    frame = ax.get_window_extent(renderer)
    left = min((t.get_window_extent(renderer).x0 for t in shown), default=frame.x1 + 0.1 * fig.dpi)
    ax.yaxis.label.set_horizontalalignment("left")
    ax.yaxis.label.set_verticalalignment("top")
    ax.yaxis.set_label_coords(ax.transAxes.inverted().transform((left, frame.y1))[0], 1.0)
    bottom = frame.y1 - ax.yaxis.label.get_window_extent(renderer).height - 0.02 * fig.dpi
    lo_v, hi_v = ax.get_ylim()
    ticks = [v for v in ax.get_yticks() if lo_v <= v <= hi_v + 1e-9]
    form = ax.yaxis.get_major_formatter()
    labels = ["" if ax.transData.transform((0, v))[1] > bottom else form(v) for v in ticks]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))


def title_at_top_value(ax):
    """On a linear ratio axis: the title, as a small fraction, written in place of the highest value."""
    lo_v, hi_v = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo_v <= t <= hi_v + 1e-9]
    form = ax.yaxis.get_major_formatter()
    labels = [form(t) for t in ticks]
    finite = [k for k, lab in enumerate(labels) if "infty" not in lab]
    if not finite:
        return
    labels[finite[-1]] = ratio_frac                    # e.g. 3 -> hh/ttbar, full size
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))
    ax.set_ylabel("")


def title_above_top_value(ax):
    """A right-hand axis title just above its highest value, starting where the values start."""
    lo_v, hi_v = ax.get_ylim()
    shown = [t for t in ax.yaxis.get_majorticklabels() if t.get_visible() and t.get_text()
             and lo_v <= t.get_position()[1] <= hi_v + 1e-9]
    if not shown:
        return title_on_top(ax)
    b = max((t.get_window_extent(renderer) for t in shown), key=lambda bb: bb.y1)
    ax.yaxis.label.set_horizontalalignment("left")
    ax.yaxis.label.set_verticalalignment("bottom")
    u, v = ax.transAxes.inverted().transform((b.x0, b.y1 + 0.03 * fig.dpi))
    ax.yaxis.set_label_coords(u, v)


def title_on_top(ax):
    """A right-hand axis title just above the top of its axis, centred over its tick values."""
    frame = ax.get_window_extent(renderer)
    right = [t.get_window_extent(renderer).x1 for t in ax.yaxis.get_majorticklabels()
             if t.get_visible() and t.get_text()]
    middle = (frame.x1 + max(right, default=frame.x1)) / 2
    ax.yaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("bottom")
    ax.yaxis.set_label_coords(ax.transAxes.inverted().transform((middle, frame.y1))[0], 1.03)


def tuck_title(ax, side, height=0.75, beside_at=0.5):
    """Put an upright y-axis title just outside the tick labels beside it (at mid-height), so it
    sits close to the axis wherever those labels are short, e.g. next to "0"."""
    title = ax.yaxis.label.get_window_extent(renderer)
    middle = ax.transAxes.transform((0, beside_at))[1]               # the labels beside this height count
    beside = [b for b in (t.get_window_extent(renderer) for t in ax.yaxis.get_majorticklabels()
                          if t.get_visible() and t.get_text())
              if b.y1 > middle - title.height / 2 - 2 and b.y0 < middle + title.height / 2 + 2]
    frame = ax.get_window_extent(renderer)
    pad = 0.06 * fig.dpi                                            # 0.06 inch, in pixels
    tick = (TICK_LEN / 72 * fig.dpi) if TICKS_OUT else 0.0          # outward tick marks, in pixels
    if side == "left":
        x = min([b.x0 for b in beside], default=frame.x0 - tick) - pad
    else:
        x = max([b.x1 for b in beside], default=frame.x1 + tick) + pad
    ax.yaxis.set_label_coords(ax.transAxes.inverted().transform((x, middle))[0], height)


tuck_title(ax2d, "left")                                    # phi, at three-quarter height
if DNDX_TITLE != "outside":
    tuck_title(axx1, "left", height=0.5)                    # dN/dx and dN/dy at mid-height
if DNDY_TITLE == "upright":
    tuck_title(axy, "left", height=0.5)
tuck_title(axx2, "left", height=0.5)                        # dN/dx2 at mid-height, like the bottom plots
if show_ratio:                              # the ratio axes' titles near their tops, clear of dN/dy
    for ax_r in [axr] + ([ax1r, ax2r] if RATIO_PROJECTIONS else []):
        if RATIO_PROJECTIONS and ax_r is ax2r:
            (title_at_plot_top if is_toy else title_above_top_value)(ax_r)    # top right: level with the
                                            # top value (toy), or just above it (events)
        else:
            ax_r.set_ylabel("")             # the bottom plots: no title, all their values kept

# ---- arrows between the panels (figure coordinates, from the inch layout) ----
def at_inch(x, y):
    return x / W, y / H


def arrow(p, q):
    fig.add_artist(FancyArrowPatch(at_inch(*p), at_inch(*q), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=12, lw=1.0, color=INK,
                                   shrinkA=0, shrinkB=0))


def inches(artist):
    """The box (x0, y0, x1, y1) of a drawn artist, in inches from the figure's lower left."""
    b = artist.get_window_extent(renderer)
    return np.array([b.x0, b.y0, b.x1, b.y1]) / fig.dpi


gap_left, gap_right = LEFT + PANEL, LEFT + PANEL + GAP_X      # the gap between the columns
gap_top, gap_bottom = BOTTOM + PANEL + GAP_Y, BOTTOM + PANEL  # the gap between the rows
# top left -> top right: across the column gap at the top, its label just above
CMEX = FontProperties(fname=os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "cmex10.ttf"),
                      size=ARROW_SIZE)         # Computer Modern's extension font, for the integral sign


SIGN_INK = TextPath((0, 0), chr(0x52), prop=CMEX).get_extents()   # the small integral's outline, in points
AXIS = 0.25                     # height of the math axis above the baseline, in em; TeX centres integrals on it


def integral_label(x, y, differential, ha="center", va="center"):
    """A text-style integral sign (as TeX's \\textstyle\\int), INT_GAP points of space, then the differential,
    at (x, y) in inches; ha ("left", "center", "right") and va ("center": the math axis at y, "bottom": the
    lowest ink at y) place the pair. The sign is centred on the math axis by its outline, like TeX does; its
    text box is much taller than the sign. Two plain texts, so the tight crop of the saved figure moves them."""
    em = ARROW_SIZE / 72 * fig.dpi                                   # 1 em, in pixels
    sign = fig.text(0, 0, chr(0x52), fontproperties=CMEX, ha="left", va="baseline")   # cmex10 slot 0x52
    rest = fig.text(0, 0, differential, fontsize=ARROW_SIZE, ha="left", va="baseline")
    ws, wr = sign.get_window_extent(renderer).width, rest.get_window_extent(renderer).width
    gap = INT_GAP / 72 * fig.dpi
    total = ws + gap + wr
    x0 = x * fig.dpi - {"left": 0.0, "center": total / 2, "right": total}[ha]
    lift = (AXIS - (SIGN_INK.y0 + SIGN_INK.y1) / 2 / ARROW_SIZE) * em    # sign's baseline above the text's
    sign_low = (SIGN_INK.y0 / ARROW_SIZE) * em + lift                    # its lowest ink, from the baseline
    base = y * fig.dpi - (sign_low if va == "bottom" else AXIS * em)
    sign.set_position((x0 / (W * fig.dpi), (base + lift) / (H * fig.dpi)))
    rest.set_position(((x0 + ws + gap) / (W * fig.dpi), base / (H * fig.dpi)))


row_top = gap_top + PANEL                                       # along the top edge of the top row,
arrow((gap_left + 0.06, row_top), (gap_right - 0.06, row_top))  # its label above it
integral_label((gap_left + gap_right) / 2, row_top + 0.05, ARROW_TO_X2, ha="center", va="bottom")
# top left -> bottom left: down the left edge of the two plots, from just under the tick values to the
# panel below, its label outside, on its left
values = [inches(t) for t in ax2d.xaxis.get_majorticklabels() if t.get_visible() and t.get_text()]
under_values = min(v[1] for v in values)
x_down = LEFT
arrow((x_down, under_values - 0.04), (x_down, gap_bottom + 0.05))
integral_label(x_down - 0.06, (under_values - 0.04 + gap_bottom + 0.05) / 2, ARROW_TO_X1, ha="right")
# top left -> bottom right: across the middle of the figure, its label along it
p0, p1 = np.array([gap_left + 0.1, gap_top - 0.1]), np.array([gap_right - 0.04, gap_bottom + 0.04])
arrow(p0, p1)
dv = p1 - p0
normal = np.array([-dv[1], dv[0]]) / np.hypot(*dv)
bin_label = fig.text(*at_inch(*(p0 + 0.45 * (p1 - p0) + 0.04 * normal)), ARROW_TO_Y[BIN_IN],   # on top of
                     rotation=np.degrees(np.arctan2(dv[1], dv[0])), rotation_mode="anchor",  # the arrow ...
                     ha="center", va="bottom", fontsize=ARROW_SIZE)
values_near = [t.get_window_extent(renderer).padded(2) for t in axx2.xaxis.get_majorticklabels()
               if t.get_visible() and t.get_text()]
corner = axx2.transData.transform((r2[0], 0))[0]           # (only the top-right plot's first values)
if any(b.overlaps(bin_label.get_window_extent(renderer)) for b in values_near if b.x0 < corner + 0.3 * fig.dpi):
    bin_label.set_position(at_inch(*(p0 + 0.55 * (p1 - p0) - 0.04 * normal)))   # ... or, if it would touch
    bin_label.set_verticalalignment("top")                                      # them, below it


fig.savefig(name + ".pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(name + ".png", dpi=220, bbox_inches="tight", pad_inches=0.03)
print("last panel: bins in %s with edges %s; hatched bin %g to %g"
      % (BIN_IN, np.round(edges_b, 6).tolist(), edges_b[HL0], edges_b[HL1]))
print(f"wrote {name}.pdf and {name}.png")


# ==========================================================================================
# likelihood ratios of the two types, lambda = f_j / f_i, in two more figures:
#   <name>_lr    ROC curves for cuts on the likelihood ratio of each observable on its own (x, x2 and y),
#                with that of x and x2 together as the reference: Neyman-Pearson's best possible
#   <name>_lr3d  log10 lambda over the (x, x2) plane, in 3-D
# Each likelihood ratio comes from histograms of the two types' events (for the toy, the 3-D one comes
# from the exact densities), so it works for event files too.
# ==========================================================================================
if MAKE_LR and B is not None:
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import LogLocator as _LogLocator, LogFormatterMathtext

    def roc_from(p_i, p_j):
        """Efficiencies of both types for cuts on lambda = p_j/p_i, most j-like bins first, and the area
        under the curve (the chance that a j event looks more j-like than an i event)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(p_i > 0, p_j / np.where(p_i > 0, p_i, 1.0), np.inf)
        lam[(p_i == 0) & (p_j == 0)] = 0.0
        o = np.argsort(-lam, kind="stable")
        eff_j = np.concatenate([[0.0], np.cumsum(p_j[o])])
        eff_i = np.concatenate([[0.0], np.cumsum(p_i[o])])
        return eff_j, eff_i, float(np.trapezoid(eff_j, eff_i))

    def roc(v_i, v_j, n=LR_BINS):
        """The ROC curve of one observable's likelihood ratio, from histograms with n bins that each hold
        the same number of events of both types together."""
        ok_i, ok_j = np.isfinite(v_i), np.isfinite(v_j)
        pooled = np.concatenate([v_i[ok_i], v_j[ok_j]])
        edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n + 1)))
        edges[-1] = np.nextafter(edges[-1], np.inf)                  # keep the largest value in the last bin
        p_i = np.histogram(v_i[ok_i], bins=edges, weights=wi[ok_i])[0] / wi[ok_i].sum()
        p_j = np.histogram(v_j[ok_j], bins=edges, weights=wj[ok_j])[0] / wj[ok_j].sum()
        return roc_from(p_i, p_j)

    plain_i = short_i
    curves = [(r"$x=\cos\theta$" if X1_COS else "$%s$" % X1, roc(xi[:, 0], xj[:, 0]), LINE_I, "-"),
              ("$%s$" % X2, roc(xi[:, 1], xj[:, 1]), LINE_J, "-."),        # the figure's navy and amber,
              (("$%s$" % S) if use_t else (r"$%s$" % Y) if (y_min or rescale) else (r"$y=f_{%s}$" % plain_i),
               roc(yi, yj), INK, "-")]                                    # no red against green
    h_i = np.histogram2d(xi[:, 0], xi[:, 1], bins=[e1, e2], weights=wi)[0].ravel() / wi.sum()
    h_j = np.histogram2d(xj[:, 0], xj[:, 1], bins=[e1, e2], weights=wj)[0].ravel() / wj.sum()
    curves.append((r"$%s\ \mathrm{and}\ %s\ \mathrm{jointly}$" % (X1, X2), roc_from(h_i, h_j), "#8c8c8c", "--"))

    fig_r, axr = plt.subplots(figsize=LR_FIGSIZE)
    floor_i = 1.0
    for label, (eff_j, eff_i, auc), color, style in curves:
        axr.plot(eff_j, eff_i, color=color, ls=style, lw=2.2 if color == INK else 1.6,
                 label=label)
        if (eff_i > 0).any():                # the first step: below it the binned ratio says nothing
            floor_i = min(floor_i, eff_i[eff_i > 0].min())
    axr.set_yscale("log")
    axr.set_xlim(0, 1)
    n_i_eff = wi.sum() ** 2 / (wi ** 2).sum()
    axr.set_ylim(max(LR_MIN_EVENTS / n_i_eff, floor_i), 1.0)   # below this, fewer than LR_MIN_EVENTS ttbar
                                                               # events are left: no statistics
    axr.annotate("", xy=(0.10, 0.90), xytext=(0.02, 0.98), xycoords="axes fraction",   # from the corner,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=0, shrinkB=0))
    axr.text(0.115, 0.905, r"$\mathrm{better}$", transform=axr.transAxes, fontsize=LEGEND_SIZE, ha="left",
             va="bottom")                                        # beside the arrow
    eff_i, eff_j = ((r"\mathrm{Toy}\ " + TOY_SUBS[0], r"\mathrm{Toy}\ " + TOY_SUBS[1]) if is_toy
                    else (plain_i, short_j))
    axr.set_xlabel(r"$%s\ \mathrm{efficiency}$" % eff_j, fontsize=TITLE_SIZE)
    axr.set_ylabel(r"$%s\ \mathrm{efficiency}$" % eff_i, fontsize=TITLE_SIZE)
    axr.xaxis.set_major_formatter(plain)
    axr.xaxis.set_minor_locator(AutoMinorLocator())
    axr.yaxis.set_major_locator(_LogLocator(base=10))
    axr.yaxis.set_major_formatter(LogFormatterMathtext())
    axr.tick_params(which="both", top=False, right=False)
    lines_r, names_r2 = axr.get_legend_handles_labels()       # a second column with the areas, row by row
    blanks = [Line2D([], [], alpha=0) for _ in curves]
    axr.legend(lines_r + blanks, names_r2 + [r"$(%.3f)$" % c[1][2] for c in curves], ncol=2,
               loc="lower right", title=r"$\mathrm{likelihood\ ratio\ of}$" + "\n"
               + r"$(\mathrm{area\ under\ the\ curve})$", title_fontsize=LEGEND_SIZE, alignment="left",
               handlelength=1.6, borderaxespad=0.5, frameon=False, columnspacing=-0.6)   # inside
    fig_r.savefig(name + "_lr.pdf", bbox_inches="tight", pad_inches=0.03)
    fig_r.savefig(name + "_lr.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    print("likelihood ratios, area under each ROC curve: "
          + "; ".join("%s %.3f" % (label.replace("$", ""), c[2]) for label, c, _, _ in curves))
    print(f"wrote {name}_lr.pdf and {name}_lr.png")

    # ---- lambda over the (x, x2) plane, in 3-D ----
    if is_toy and not PER_SIN:                       # the exact densities, on a fine grid
        g1, g2 = np.linspace(*r1, 121), np.linspace(*r2, 121)
        C1, C2 = np.meshgrid((g1[1:] + g1[:-1]) / 2, (g2[1:] + g2[:-1]) / 2)
        f_i3 = f_toy(C1, C2)
        f_j3 = plane_f(toy_f_j)(C1, C2)
        f_j3 = f_j3 / (f_j3.sum() * (g1[1] - g1[0]) * (g2[1] - g2[0]))     # normalised like f_i
    else:                                            # from the events, on the top-left panel's bins
        b1, b2 = np.linspace(*r1, LR3D_BINS + 1), np.linspace(*r2, LR3D_BINS + 1)    # coarse, against noise
        C1, C2 = np.meshgrid((b1[1:] + b1[:-1]) / 2, (b2[1:] + b2[:-1]) / 2)
        f_i3 = np.histogram2d(xi[:, 0], xi[:, 1], bins=[b1, b2], weights=wi)[0].T / wi.sum()
        f_j3 = np.histogram2d(xj[:, 0], xj[:, 1], bins=[b1, b2], weights=wj)[0].T / wj.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.log10(f_j3 / f_i3)
    Z = np.where(np.isnan(Z), np.nan, np.clip(Z, *LR3D_RANGE))
    fig3 = plt.figure(figsize=LR3D_FIGSIZE)
    ax3 = fig3.add_subplot(projection="3d")
    lr_map = LinearSegmentedColormap.from_list("lr", [COLOR_I, "#ffffff", LINE_J])
    z_lo, z_hi = np.nanmin(Z), np.nanmax(Z)
    ax3.plot_surface(C1, C2, Z, cmap=lr_map, norm=TwoSlopeNorm(0.0, min(z_lo, -1e-6), max(z_hi, 1e-6)),
                     rstride=1, cstride=1, linewidth=0, antialiased=False, rasterized=True)
    ax3.set_xlabel(X1_AXIS_TOP if X1_COS else "$%s$" % X1, fontsize=TITLE_SIZE, labelpad=6)
    ax3.set_ylabel("$%s$" % X2, fontsize=TITLE_SIZE, labelpad=6)
    ax3.set_zlabel(r"$\log_{10}\,(f_{%s}/f_{%s})$" % ((TOY_SUBS[1], TOY_SUBS[0]) if is_toy else (short_j, plain_i)),
                   fontsize=TITLE_SIZE,
                   labelpad=8)
    ax3.set_xlim(*r1)
    ax3.set_ylim(*r2)
    ax3.set_zlim(*LR3D_RANGE)
    q = np.array(r2) / np.pi
    if np.allclose(q, np.round(q)):                  # phi ticks at the multiples of pi
        k = np.arange(int(round(q[0])), int(round(q[1])) + 1)
        ax3.set_yticks(k * np.pi, [{0: "$0$", 1: r"$\pi$", -1: r"$-\pi$"}.get(m, r"$%d\pi$" % m) for m in k])
    ax3.xaxis.set_major_formatter(plain)
    ax3.zaxis.set_major_formatter(plain)
    ax3.view_init(elev=24, azim=-122)
    fig3.savefig(name + "_lr3d.pdf", bbox_inches="tight", pad_inches=0.15)
    fig3.savefig(name + "_lr3d.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {name}_lr3d.pdf and {name}_lr3d.png (log10 likelihood ratio shown from {LR3D_RANGE[0]:g} "
          f"to {LR3D_RANGE[1]:g}; the ratio grows without limit where f_i -> 0)")
