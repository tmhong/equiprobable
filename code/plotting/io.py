"""Readers for MG5 LHE files and Delphes ROOT files into awkward arrays."""
import gzip

import awkward as ak
import numpy as np
import uproot


def read_lhe(path, max_events=None):
    """
    Read an (optionally gzipped) LHE file.

    Returns (particles, info):
      particles: ak.Array [event][particle] with fields
                 pid, status, mother1, mother2, px, py, pz, e, m
                 (mother indices are 0-based, -1 = none)
      info:      dict with 'weight' (np.ndarray per event) and 'xsec_pb'
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    counts, rows, weights = [], [], []
    xsec = None
    with opener(path, "rt") as f:
        in_event = in_init = False
        block = []
        for line in f:
            s = line.strip()
            if s.startswith("<init>"):
                in_init, block = True, []
                continue
            if in_init:
                if s.startswith("</init>"):
                    in_init = False
                    # line 0: beam info, following lines: xsec xerr xmax lprup
                    xsec = sum(float(l.split()[0]) for l in block[1:] if l and not l.startswith("<"))
                else:
                    block.append(s)
                continue
            if s.startswith("<event"):
                in_event, block = True, []
                continue
            if in_event:
                if s.startswith("<") or s.startswith("#"):
                    # end of particle list (rwgt, mgrwt, scales, </event>)
                    if block:
                        _add_event(block, counts, rows, weights)
                        block = []
                    if s.startswith("</event>"):
                        in_event = False
                        if max_events and len(counts) >= max_events:
                            break
                    continue
                block.append(s)
    data = np.array(rows, dtype=np.float64).reshape(-1, 13)
    cnt = np.array(counts, dtype=np.int64)

    def col(i, dtype=np.float64):
        return ak.unflatten(data[:, i].astype(dtype), cnt)

    particles = ak.zip({
        "pid": col(0, np.int64),
        "status": col(1, np.int64),
        "mother1": col(2, np.int64) - 1,
        "mother2": col(3, np.int64) - 1,
        "px": col(6), "py": col(7), "pz": col(8), "e": col(9), "m": col(10),
    })
    return particles, {"weight": np.array(weights), "xsec_pb": xsec}


def _add_event(block, counts, rows, weights):
    header = block[0].split()
    n = int(header[0])
    weights.append(float(header[2]))
    counts.append(n)
    for l in block[1:1 + n]:
        rows.append([float(x) for x in l.split()])


# Branches read from Delphes for each collection (Delphes class member names).
DELPHES_BRANCHES = {
    "Jet": ["PT", "Eta", "Phi", "Mass", "BTag", "TauTag", "Flavor"],
    "GenJet": ["PT", "Eta", "Phi", "Mass", "Flavor"],
    "Electron": ["PT", "Eta", "Phi", "Charge"],
    "Muon": ["PT", "Eta", "Phi", "Charge"],
    "Photon": ["PT", "Eta", "Phi"],
    "MissingET": ["MET", "Phi"],
    "GenMissingET": ["MET", "Phi"],
    "Particle": ["PID", "Status", "M1", "M2", "PT", "Eta", "Phi", "Mass", "Charge"],
    "Event": ["Weight"],
}


def read_delphes(path, collections=None, max_events=None, tree="Delphes", entry_start=None):
    """
    Read a Delphes ROOT file.

    Returns dict {collection: ak.Array [event][object]} with lower-case fields,
    e.g. out["Jet"].pt, out["Jet"].btag, out["MissingET"].met.
    entry_start/max_events select the entry range [entry_start, max_events).
    """
    collections = collections or list(DELPHES_BRANCHES)
    t = uproot.open(path)[tree]
    available = set(t.keys())
    out = {}
    for coll in collections:
        fields = DELPHES_BRANCHES.get(coll)
        if fields is None:
            raise KeyError(f"no branch list for {coll}; add it to DELPHES_BRANCHES")
        names = [f"{coll}/{coll}.{f}" for f in fields if f"{coll}/{coll}.{f}" in available]
        if not names:
            print(f"[io] warning: collection {coll} not in {path}")
            continue
        out[coll] = ak.zip({n.split(".")[-1].lower(): t[n].array(entry_start=entry_start, entry_stop=max_events, library="ak")
                            for n in names})
    return out


def delphes_entries(path, tree="Delphes"):
    return uproot.open(path)[tree].num_entries
