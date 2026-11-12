#!/usr/bin/env python3
"""Build a prefill read-ahead ranking from routed-expert dumps.

Each input is a file written by DS4_ARGODRIVE_PREFILL_IDS_DUMP: one line per layer of one prompt pass,
"layer n id:count ...". The output has one line per layer, "layer n id:count ...", with counts summed over the
inputs; the engine ranks each layer's experts by count when it loads the file (DS4_ARGODRIVE_HOTLIST).
The ranking only orders reads. It never changes routing, weights or output.
usage: make_hotlist.py <out file> <dump>..."""
import sys
from collections import defaultdict
out, dumps = sys.argv[1], sys.argv[2:]
counts = defaultdict(lambda: defaultdict(int)); passes = defaultdict(int)
for d in dumps:
    for line in open(d):
        p = line.split()
        if len(p) < 3: continue
        il = int(p[0]); passes[il] += 1
        for tok in p[2:]:
            e, _, c = tok.partition(':')
            counts[il][int(e)] += int(c)
with open(out, 'w') as f:
    for il in sorted(counts):
        ranked = sorted(counts[il].items(), key=lambda kv: (-kv[1], kv[0]))
        f.write('%d %d %s\n' % (il, len(ranked), ' '.join('%d:%d' % kv for kv in ranked)))
print('%d layers from %d dumps (%d prompt passes at layer 0) -> %s' % (len(counts), len(dumps), passes.get(0, 0), out))
