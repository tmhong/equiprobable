"""Histogram filling and drawing (matplotlib) with an optional ratio panel."""
import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#1f4e9c", "#d1495b", "#2a9d8f", "#e9a03b", "#6a4c93", "#4d4d4d", "#8ab17d", "#c06c84"]
LINESTYLES = ["-", "--", ":", "-."]


def flatten_values(values, weight):
    """Turn whatever a plot expression returned into flat (values, weights), dropping None/NaN."""
    if isinstance(values, (ak.Array, ak.Record)) or hasattr(values, "layout"):
        values = ak.Array(values)
        w = ak.Array(weight)
        if values.ndim > 1:
            w = w[(slice(None),) + (np.newaxis,) * (values.ndim - 1)]   # (nev, 1, ...)
            w = ak.broadcast_arrays(w, values)[0]
        values, w = ak.flatten(values, axis=None), ak.flatten(w, axis=None)
        mask = ~ak.is_none(values)
        values = ak.to_numpy(values[mask]).astype(float)
        w = ak.to_numpy(w[mask]).astype(float)
    else:
        values = np.asarray(values, dtype=float)
        w = np.asarray(weight, dtype=float).reshape((-1,) + (1,) * (values.ndim - 1))
        w = np.broadcast_to(w, values.shape)
        values, w = values.ravel(), w.ravel()
    good = np.isfinite(values)
    return values[good], w[good]


def make_hist(values, weights, bins, overflow=True):
    edges = bins if not isinstance(bins, tuple) else np.linspace(*bins[1:], bins[0] + 1)
    edges = np.asarray(edges, dtype=float)
    v = np.clip(values, edges[0], edges[-1] - 1e-9 * (edges[-1] - edges[0])) if overflow else values
    h, _ = np.histogram(v, edges, weights=weights)
    e2, _ = np.histogram(v, edges, weights=weights**2)
    return h, np.sqrt(e2), edges


def draw(plot, entries, norm, outpath, lumi_fb=None):
    """
    plot:    dict from plot_config (name, bins, xlabel, ylabel, logy, ratio, ...)
    entries: list of dicts {label, values, weights}
    norm:    'unit' (area 1), 'xsec' (weights already in events/lumi) or 'none'
    """
    from . import style
    style.use()
    ratio = plot.get("ratio", True) and len(entries) > 1
    if ratio:
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(7.6, 7.0), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    else:
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        rax = None

    hists = []
    for i, ent in enumerate(entries):
        h, err, edges = make_hist(ent["values"], ent["weights"], plot["bins"], plot.get("overflow", True))
        if norm == "unit" and h.sum() > 0:
            scale = 1.0 / h.sum()
            if plot.get("density", False):
                scale /= np.diff(edges)
            h, err = h * scale, err * scale
        color = ent.get("color", COLORS[i % len(COLORS)])
        ls = ent.get("linestyle", LINESTYLES[(i // len(COLORS)) % len(LINESTYLES)])
        mean = np.average(ent["values"], weights=ent["weights"]) if len(ent["values"]) else np.nan
        label = ent["label"] + (f"  (mean {mean:.3g})" if plot.get("show_mean", True) else "")
        ax.stairs(h, edges, color=color, linestyle=ls, linewidth=1.6, label=label)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ax.errorbar(centers, h, yerr=err, fmt="none", ecolor=color, elinewidth=0.8, alpha=0.7)
        hists.append((h, err, color, ls))

    ylabel = plot.get("ylabel") or {"unit": "Normalized entries", "xsec": "Events",
                                     "none": "Entries"}[norm]
    ax.set_ylabel(ylabel)
    if plot.get("logy"):
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.margins(x=0)
    # legend above the axes so it never covers the data
    ncol = 2 if len(entries) > 2 else len(entries)
    ax.legend(frameon=False, fontsize=15, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=ncol,
              borderaxespad=0.2, handlelength=1.6, columnspacing=1.0)
    title = plot.get("title", "")
    if norm == "xsec" and lumi_fb:
        title = (title + "   " if title else "") + f"$\\mathcal{{L}}$ = {lumi_fb:g} fb$^{{-1}}$"
    if title:
        ax.text(1.0, 1.0, title, transform=ax.transAxes, ha="right", va="bottom", fontsize=14)

    if rax is not None:
        # each entry is compared with entries[ent["ref"]] (default: the first entry)
        refs = sorted({ent.get("ref", 0) for ent in entries})
        for i, (ent, (h, err, color, ls)) in enumerate(zip(entries, hists)):
            ref = ent.get("ref", 0)
            if i == ref:
                continue
            h0 = hists[ref][0]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(h0 > 0, h / h0, np.nan)
                rerr = np.where(h0 > 0, err / h0, np.nan)
            rax.stairs(r, edges, color=color, linestyle=ls, linewidth=1.4)
            rax.errorbar(0.5 * (edges[1:] + edges[:-1]), r, yerr=rerr, fmt="none", ecolor=color, elinewidth=0.8)
        h0, e0 = hists[refs[0]][0], hists[refs[0]][1]
        with np.errstate(divide="ignore", invalid="ignore"):
            band = np.where(h0 > 0, e0 / h0, 0)
        rax.stairs(1 + band, edges, baseline=1 - band, fill=True, color="0.85", zorder=0)
        rax.axhline(1, color="0.3", linewidth=0.8)
        rax.set_ylim(*plot.get("ratio_range", (0.5, 1.5)))
        rax.set_yticks([0.75, 1.0, 1.25])                  # no tick at the panel edges (no collisions)
        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
        rax.set_ylabel(plot.get("ratio_label") or f"Ratio to\n{entries[refs[0]].get('ref_label') or entries[refs[0]]['label']}",
                       fontsize=14)
        rax.set_xlabel(plot.get("xlabel", plot["name"]))
    else:
        ax.set_xlabel(plot.get("xlabel", plot["name"]))

    fig.savefig(outpath, bbox_inches="tight", dpi=130)
    plt.close(fig)
