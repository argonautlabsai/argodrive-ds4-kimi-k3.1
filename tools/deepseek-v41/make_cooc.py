#!/usr/bin/env python3
"""Build the conditional read-ahead prior for DS4_ARGODRIVE_COOC from per-token routing dumps.

Each input is a file written by DS4_ARGODRIVE_PREFILL_TOKENS_DUMP during one prompt pass: lines "layer t id0..id5".
For every pair of consecutive layers the table counts how often an expert of layer L+1 was routed by a token that
routed an expert of layer L (uint16, saturating), followed by each layer's expert frequencies (uint32).
File: "ARCOOC01", layers and experts as two uint32, the counts, the frequencies. Like the ranking file it only orders
the read-ahead; routing and output do not depend on it.
usage: make_cooc.py <out file> <layers> <experts> <dump>..."""
import sys
import numpy as np
out, nl, ne, dumps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
C = np.zeros((nl - 1, ne, ne), dtype=np.uint32)
M = np.zeros((nl, ne), dtype=np.uint32)
tokens = 0
for d in dumps:
    rows = {}
    for line in open(d):
        p = line.split()
        if len(p) < 3: continue
        rows.setdefault(int(p[0]), {})[int(p[1])] = [int(x) for x in p[2:]]
    for il in range(nl):
        for t, ids in rows.get(il, {}).items():
            for e in ids: M[il, e] += 1
            if il + 1 < nl:
                nxt = rows.get(il + 1, {}).get(t)
                if nxt is None: continue
                for e in ids:
                    for e2 in nxt: C[il, e, e2] += 1
    tokens += len(rows.get(0, {}))
with open(out, 'wb') as f:
    f.write(b'ARCOOC01'); f.write(np.array([nl, ne], dtype='<u4').tobytes())
    f.write(np.minimum(C, 65535).astype('<u2').tobytes()); f.write(M.astype('<u4').tobytes())
print('%d prompts, %d tokens -> %s (%d saturated cells)' % (len(dumps), tokens, out, int((C > 65535).sum())))
