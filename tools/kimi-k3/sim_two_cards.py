#!/usr/bin/env python3
"""Two-card expert cache replay for Kimi K3.1 (18 Nov 2026).

Replays recorded routes (DS4_K3_TRACE_ROUTES lines, "route layer L pos P: e:w ...") through the engine's cache
hierarchy: a device cache of D slabs (LRU, decode inserts at the hot end, a prompt pass at the cold end and
transient), and the pinned host pool of H slabs that holds device evictions (victim cache). With in-place compute an
expert is used on the card that caches it, so two cards are one device cache of D0 + D1 slabs as far as disk reads
go; where a fill lands only decides which PCIe link carries it. Prints, per configuration, the share of decode
expert uses served by the device, the host pool and the drives, and reads per decode token for the whole trace and
for its second half (the first half warms the caches).

usage: sim_two_cards.py [--configs "D:H,D:H,..."] trace.log [trace.log ...]   (several traces = one session, in order)
"""
import re
import sys
from collections import OrderedDict

LAYERS, USED = 92, 16


def load(path):
    n_prompt, rows = None, []
    for line in open(path, errors="replace"):
        m = re.match(r"ds4: K3 prefill (\d+) tokens in", line)
        if m and n_prompt is None:
            n_prompt = int(m[1])
        m = re.match(r"route layer (\d+) pos (\d+):(.*)", line)
        if m:
            rows.append((int(m[2]), int(m[1]), [int(x.split(":")[0]) for x in m[3].split()]))
    rows.sort()
    n_prompt = n_prompt or 0
    pre = [(l, e) for pos, l, ex in rows if pos < n_prompt for e in ex]
    dec = [(pos, l, e) for pos, l, ex in rows if pos >= n_prompt for e in ex]
    return n_prompt, pre, dec


def run(reqs, D, H):
    dev, host, transient = OrderedDict(), OrderedDict(), set()
    stats = []   # per decode use: 0 device, 1 host, 2 disk

    def host_put(k):
        if k in host:
            host.move_to_end(k)
            return
        host[k] = 1
        if len(host) > H:
            host.popitem(last=False)

    def dev_put(k, hot):
        if k in dev:
            dev.move_to_end(k, last=hot)
            return
        dev[k] = 1
        if not hot:
            dev.move_to_end(k, last=False)
            transient.add(k)
        if len(dev) > D:
            victim, _ = dev.popitem(last=False)
            if victim in transient:
                transient.discard(victim)
            elif H:
                host_put(victim)

    for _, pre, dec in reqs:
        for k in pre:
            if k not in dev:
                dev_put(k, False)
        for pos, l, e in dec:
            k = (l, e)
            if k in dev:
                stats.append(0)
                dev.move_to_end(k)
                transient.discard(k)
                continue
            if k in host:
                stats.append(1)
                del host[k]          # exclusive: the pool holds only what the device evicted
            else:
                stats.append(2)
            dev_put(k, True)
    return stats


def main():
    args = sys.argv[1:]
    configs = [(1548, 1468), (1536, 3059), (1548 + 5200, 1468), (1548 + 5200, 3059), (1548 + 5200, 4200)]
    if args and args[0] == "--configs":
        configs = [tuple(int(x) for x in c.split(":")) for c in args[1].split(",")]
        args = args[2:]
    reqs = [load(p) for p in args]
    n_use = sum(len(d) for _, _, d in reqs)
    n_tok = n_use // (LAYERS * USED)
    uniq = len({(l, e) for _, _, d in reqs for _, l, e in d})
    print(f"{len(reqs)} request(s), {sum(r[0] for r in reqs)} prompt tokens, {n_tok} decode tokens, {n_use} expert uses, "
          f"{uniq} distinct experts in decode")
    print(f"{'device':>7} {'host':>6} | {'device %':>8} {'host %':>7} {'disk %':>7} {'reads/tok':>9} {'GB/tok':>7} | second half: "
          f"{'disk %':>7} {'reads/tok':>9} {'GB/tok':>7}")
    for D, H in configs:
        s = run(reqs, D, H)
        half = s[len(s) // 2:]
        d, h, m = s.count(0), s.count(1), s.count(2)
        m2 = half.count(2)
        rt, rt2 = m / n_tok, m2 / (len(half) / (LAYERS * USED))
        print(f"{D:7d} {H:6d} | {100 * d / n_use:8.1f} {100 * h / n_use:7.1f} {100 * m / n_use:7.1f} {rt:9.0f} {rt * 17.547264 / 1e3:7.2f} | "
              f"{'':13}{100 * m2 / len(half):7.1f} {rt2:9.0f} {rt2 * 17.547264 / 1e3:7.2f}")


if __name__ == "__main__":
    main()
