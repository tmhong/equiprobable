#!/usr/bin/env python3
"""
Merge / inspect Delphes pile-up libraries (.pileup, written by DelphesPileUpWriter).

File layout (XDR = big-endian), from Delphes classes/DelphesPileUpWriter.cc:
  for each event : int32 n, then n records of {int32 PID; float x, y, z, t, px, py, pz, E}
  index          : int64 byte offset of every event
  trailer        : int64 number of events

  python code/pileup/merge_pileup.py --info lib.pileup [...]          # events, particles/event
  python code/pileup/merge_pileup.py -o merged.pileup a.pileup b.pileup  # exact byte copy + new index
"""
import argparse
import os
import struct
import sys

import numpy as np

REC = np.dtype([("pid", ">i4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4"), ("t", ">f4"),
                ("px", ">f4"), ("py", ">f4"), ("pz", ">f4"), ("e", ">f4")])


def read_index(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(size - 8)
        n = struct.unpack(">q", f.read(8))[0]
        f.seek(size - 8 - 8 * n)
        offsets = np.frombuffer(f.read(8 * n), dtype=">i8").astype(np.int64)
    data_end = size - 8 - 8 * n
    if n and (offsets[0] != 0 or offsets[-1] >= data_end or np.any(np.diff(offsets) <= 0)):
        sys.exit(f"{path}: corrupt index")
    return offsets, data_end


def info(path):
    offsets, data_end = read_index(path)
    with open(path, "rb") as f:
        counts = []
        for o in offsets:
            f.seek(o)
            counts.append(struct.unpack(">i", f.read(4))[0])
    counts = np.array(counts)
    sizes = np.diff(np.append(offsets, data_end))
    assert np.all(sizes == 4 + 36 * counts), "record sizes inconsistent with particle counts"
    print(f"{path}: {len(offsets)} events, {os.path.getsize(path) / 1e6:.1f} MB, "
          f"particles/event mean {counts.mean():.1f} (min {counts.min()}, max {counts.max()}), "
          f"events with <= 2 particles (elastic-like): {np.mean(counts <= 2):.1%}")
    return counts


def merge(out, inputs):
    all_offsets, pos = [], 0
    with open(out, "wb") as fo:
        for path in inputs:
            offsets, data_end = read_index(path)
            with open(path, "rb") as fi:
                remaining = data_end
                while remaining:                       # events are contiguous: copy the block
                    chunk = fi.read(min(remaining, 64 << 20))
                    fo.write(chunk)
                    remaining -= len(chunk)
            all_offsets.append(offsets + pos)
            pos += data_end
        offsets = np.concatenate(all_offsets)
        fo.write(offsets.astype(">i8").tobytes())
        fo.write(struct.pack(">q", len(offsets)))
    print(f"wrote {out}: {len(offsets)} events from {len(inputs)} files")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output")
    p.add_argument("--info", action="store_true")
    a = p.parse_args()
    if a.output:
        merge(a.output, a.inputs)
        # check: every event of the merged file is byte-identical to its source event
        mo, _ = read_index(a.output)
        k = 0
        with open(a.output, "rb") as fm:
            for path in a.inputs:
                so, se = read_index(path)
                with open(path, "rb") as fs:
                    for i in np.linspace(0, len(so) - 1, 50).astype(int):
                        n = 4 + 36 * struct.unpack(">i", (fs.seek(so[i]), fs.read(4))[1])[0]
                        fs.seek(so[i]); fm.seek(mo[k + i])
                        assert fs.read(n) == fm.read(n), f"mismatch in {path} event {i}"
                k += len(so)
        print("spot check of 50 events per input: identical")
    if a.info or not a.output:
        for path in ([a.output] if a.output else a.inputs):
            info(path)


if __name__ == "__main__":
    main()
