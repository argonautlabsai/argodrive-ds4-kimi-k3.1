#!/usr/bin/env python3
"""Replay recorded DeepSeek V4.1 routes through a model of the Metal expert cache.

Record a run with
    DS4_ARGODRIVE_PREFILL_TOKENS_DUMP=<prefill dump>   every prompt token's six routed experts per layer
    DS4_MOE_RECORD_SELECTED_IDS=<decode trace>          int32[6] per decode layer call
and replay it here to see how many experts a decode step has to load at a given cache size, without running the
engine again. Policies:
  engine  the engine's rule: evict the lowest route hotness, oldest first (hotness +1 per selection, the prompt's
          seeds add 1..32 by rank, everything halves every 16 decode tokens)
  lru     least recently used
  opt     Belady's bound: evict the entry whose next use is farthest away (needs the future; a floor, not a rule)
The model reproduced the engine's measured misses within 1 % at 2,400, 3,000 and 3,600 experts on an M5 Max
(12 November 2026).

usage: route_cache_sim.py <prefill dump> <decode trace> <budget>[,<budget>...] [--seed-budget N] [--policy engine,lru,opt]
"""
import struct, sys
from collections import defaultdict
L, E, K = 40, 384, 6

def load(pre, dec, limit=None):
    prompt = defaultdict(list)                      # layer -> list of 6-tuples in token order
    for line in open(pre):
        p = line.split()
        prompt[int(p[0])].append(tuple(int(x) for x in p[2:8]))
    raw = open(dec, 'rb').read()
    n = len(raw) // 24
    recs = [struct.unpack_from('<6i', raw, 24 * i) for i in range(n)]
    tokens = [recs[t * L:(t + 1) * L] for t in range(n // L)]
    return prompt, tokens[:limit] if limit else tokens

def seeds(prompt, budget):
    """The engine's prefill seed: per layer the top budget/L experts by prompt frequency with a recency bonus."""
    target = min(budget // L, E)
    out = []                                         # (layer, expert, priority) in install order
    for il in range(L):
        rows = prompt[il]; count = len(rows); recent = min(count, 32)
        freq = [0] * E
        for t, row in enumerate(rows):
            for e in row:
                freq[e] += 1 + (count * K if t >= count - recent else 0)
        order = sorted(range(E), key=lambda e: (-freq[e], e))
        picked = [e for e in order[:target] if freq[e] > 0]
        pr = {e: 1 + (31 * (target - n)) // target for n, e in enumerate(picked)}
        # the seed loop installs from the end of the list in groups of six
        rem = len(picked); seq = []
        while rem:
            b = min(6, rem); rem -= b
            seq += picked[rem:rem + b]
        out += [(il, e, pr[e]) for e in seq]
    return out

def run(prompt, tokens, budget, policy='engine', decay=16, seed_budget=None, hot_bonus=1, seed=True):
    cache = {}                                       # (layer, expert) -> last_used
    hot = defaultdict(int)
    clock = 0
    if seed:
        for il, e, pr in seeds(prompt, seed_budget or budget):
            hot[(il, e)] += pr
            clock += 1; cache[(il, e)] = clock
    nxt = None
    if policy == 'opt':
        # next-use index per access position
        seq = [(il, e) for tok in tokens for il, row in enumerate(tok) for e in row]
        nxt = [0] * len(seq); last = {}
        for i in range(len(seq) - 1, -1, -1):
            nxt[i] = last.get(seq[i], 1 << 60); last[seq[i]] = i
        nextuse = {}
        for i in range(len(seq) - 1, -1, -1):
            nextuse[seq[i]] = i                      # first use of each key
    misses = []; pos = 0
    layer_miss_calls = 0
    for t, tok in enumerate(tokens):
        if policy == 'engine' and decay and t and t % decay == 0:
            for k in list(hot):
                hot[k] >>= 1
                if not hot[k]: del hot[k]
        m = 0
        for il, row in enumerate(tok):
            keys = [(il, e) for e in row]
            for k in keys: hot[k] += hot_bonus
            protected = set(keys)
            had = 0
            for j, k in enumerate(keys):
                if policy == 'opt': nextuse[k] = nxt[pos + j]
                if k in cache:
                    clock += 1; cache[k] = clock; continue
                m += 1; had = 1
                if len(cache) >= budget:
                    cand = (x for x in cache if x not in protected)
                    if policy == 'engine': v = min(cand, key=lambda x: (hot.get(x, 0), cache[x]))
                    elif policy == 'lru': v = min(cand, key=lambda x: cache[x])
                    else: v = max(cand, key=lambda x: nextuse.get(x, 1 << 60))
                    del cache[v]
                clock += 1; cache[k] = clock
            layer_miss_calls += had
            pos += K
        misses.append(m)
    return misses, layer_miss_calls

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(zip(sys.argv[1:], sys.argv[2:]))
    if len(args) < 3:
        sys.exit(__doc__)
    prompt, tokens = load(args[0], args[1])
    seed_budget = int(opts['--seed-budget']) if '--seed-budget' in opts else None
    policies = opts.get('--policy', 'engine').split(',')
    print('prompt rows per layer %d, decode tokens %d' % (len(prompt[0]), len(tokens)))
    for budget in (int(b) for b in args[2].split(',')):
        for policy in policies:
            misses, calls = run(prompt, tokens, budget, policy, seed_budget=seed_budget)
            print('budget %5d %-6s loads per token %5.1f   layer calls with a load %4.1f per token' % (
                budget, policy, sum(misses) / len(misses), calls / len(misses)))
