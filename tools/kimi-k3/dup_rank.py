#!/usr/bin/env python3
"""Rank the experts that live only in the second expert directory (dir_b, the slower tier) by how often recorded K3.1 runs
routed to them, and write the names of the top N (candidates for a copy in the primary directory). Prints how much of the
observed dir_b-only traffic the list covers. Traces are logs of runs made with DS4_K3_TRACE_ROUTES=1.

usage: dup_rank.py PRIMARY_DIR DIR_B N out.lst trace.log [trace.log ...]"""
import os
import re
import sys
from collections import Counter

RAID, SAM = sys.argv[1], sys.argv[2]
del sys.argv[1:3]
n_want, out = int(sys.argv[1]), sys.argv[2]
raid = {e.name for e in os.scandir(RAID) if e.name.endswith(".bin")}
sam = {e.name for e in os.scandir(SAM) if e.name.endswith(".bin")}
only = sam - raid
uses, uses_only, total = Counter(), 0, 0
for path in sys.argv[3:]:
    for line in open(path, errors="replace"):
        m = re.match(r"route layer (\d+) pos (\d+):(.*)", line)
        if not m:
            continue
        layer = int(m[1])
        for x in m[3].split():
            name = f"L{layer}-E{x.split(':')[0]}.bin"
            total += 1
            if name in only:
                uses[name] += 1
                uses_only += 1
ranked = [n for n, _ in uses.most_common()]
rest = sorted(only - set(ranked))          # never seen in the traces: after the ranked ones, in name order
pick = (ranked + rest)[:n_want]
covered = sum(uses[n] for n in pick)
print(f"primary {len(raid)} files, dir_b {len(sam)}, in dir_b only {len(only)}")
print(f"traces: {total} expert uses, {uses_only} ({100.0 * uses_only / max(total, 1):.1f} %) went to dir_b-only experts, "
      f"{len(ranked)} of them seen at least once")
print(f"top {len(pick)}: {covered} of those uses ({100.0 * covered / max(uses_only, 1):.1f} %)")
open(out, "w").write("".join(n + "\n" for n in pick))
