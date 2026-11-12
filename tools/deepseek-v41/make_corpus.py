#!/usr/bin/env python3
"""Cut prompt slices for the read-ahead ranking from public texts in the engine tree.
None overlaps a measured prompt: the benchmark reads the head of speed-bench/promessi_sposi.txt, and the two
held-out test prompts are the heads of README.md and ds4_engram.c, so every slice of those files starts far in.
usage: make_corpus.py <engine dir> <out dir>"""
import sys
from pathlib import Path
eng, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
PLAN = [  # (file, tag, [byte offsets])
    ('speed-bench/promessi_sposi.txt', 'it-book', [100000, 250000, 400000, 550000, 700000, 850000, 1000000, 1150000, 1300000]),
    ('tests/long_context_story_prompt.txt', 'en-story', [8000, 38000, 68000, 98000, 126000]),
    ('tests/long_context_security_prompt.txt', 'en-security', [4000, 30000, 60000]),
    ('README.md', 'en-readme', [40000, 80000]),
    ('AGENT.md', 'en-agent', [0]),
    ('CONTRIBUTING.md', 'en-contrib', [0]),
    ('ds4_engram.c', 'c-engram', [40000]),
    ('ds4_ssd.c', 'c-ssd', [0]),
    ('ds4_server.c', 'c-server', [200000]),
    ('ds4_kvstore.c', 'c-kvstore', [0]),
    ('Makefile', 'make', [2000]),
]
n = 0
for rel, tag, offsets in PLAN:
    p = eng / rel
    if not p.is_file():
        print('missing', rel); continue
    raw = p.read_bytes()
    for off in offsets:
        if off >= len(raw) - 6000:
            print('skip %s@%d (file has %d bytes)' % (rel, off, len(raw))); continue
        start = raw.find(b'\n', off) + 1 if off else 0
        piece = raw[start:start + 12000]            # 512 tokens need about 2-3 KB; keep a margin
        (out / ('%s-%07d.txt' % (tag, off))).write_bytes(piece)
        n += 1
for extra in sorted(eng.glob('metal/*.metal'))[:2] + sorted(eng.glob('tools/*.py'))[:2] + sorted(eng.glob('docs/*.md'))[:3]:
    raw = extra.read_bytes()
    if len(raw) < 8000: continue
    (out / ('%s-%s.txt' % ('src' if extra.suffix != '.md' else 'en-docs', extra.stem[:24]))).write_bytes(raw[:12000])
    n += 1
print('%d slices in %s' % (n, out))
