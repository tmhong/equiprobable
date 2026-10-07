"""Shared matplotlib style.  A 16:9 slide has ~150 x 65 mm for a figure; a figure of W x H inches
is scaled by min(150/(25.4 W), 65/(25.4 H)), so text needs to be large in figure units to stay
>= ~7 pt on the slide (e.g. 19 pt at a 13 x 7 in figure -> ~7 pt)."""
import matplotlib

SLIDE = {
    "font.size": 19, "axes.labelsize": 20, "axes.titlesize": 17, "xtick.labelsize": 16,
    "ytick.labelsize": 16, "legend.fontsize": 17, "figure.titlesize": 20, "lines.linewidth": 2.0,
}


def use():
    matplotlib.rcParams.update(SLIDE)
