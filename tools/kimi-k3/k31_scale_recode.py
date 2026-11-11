#!/usr/bin/env python3
"""Lossless recode of a K3.1 raw-v1 expert file's scale planes (optional; the engine reads both forms and tells them apart by size).

raw-v1 (17,547,264 bytes): three sections [packed nibbles 5,505,024 | one scale byte per 32-weight block 344,064], for the gate
(3,072 rows x 112 blocks), down (3,584 x 96) and up (3,072 x 112) matrices.
v2 (16,785,408 bytes, 4.34 % smaller): three sections [packed nibbles 5,505,024 | compact scales 90,112]; a compact block is
  rows base bytes (the row's largest scale), then 2 bits per block (base - scale: 0, 1 or 2; 3 = escape), LSB first, then at
  offset 89,600 an escape table: u16 count, then count x (u32 block index, u8 scale); the rest zero. At most 100 escapes per plane,
  else the file is left as it is.
Every conversion is checked by decoding the result and comparing all 17,547,264 bytes with the source.

  k31_scale_recode.py check <file> [...]          encode + decode in memory, report sizes (reads only)
  k31_scale_recode.py encode <v1 file> <v2 file>  write the v2 form of one file (never in place)
  k31_scale_recode.py decode <v2 file> <v1 file>  write back the raw-v1 form
  k31_scale_recode.py convert <dir> [--reverse] [--jobs N] [--limit N] [--dry-run]
        every L*-E*.bin of the directory in place, one file at a time: the new form is written beside the file, decoded again and
        compared byte for byte with the source, synced, and renamed over it. --reverse turns v2 files back into raw-v1. A file that
        already has the target size is skipped, so an interrupted run is simply started again. Symbolic links are left alone.
        ONLY for a store whose every reader understands v2 (this engine from 11 November 2026 on): the releases up to
        10 November 2026 do not, and would stop on the first converted file.
"""
import os, re, struct, sys, time
import numpy as np
PACKED = 5505024; SCALE = 344064; SEC1 = PACKED + SCALE; V1 = 3 * SEC1
COMPACT = 90112; ESC_OFF = 89600; SEC2 = PACKED + COMPACT; V2 = 3 * SEC2; MAX_ESC = 100
PLANES = [(3072, 112), (3584, 96), (3072, 112)]
def enc_plane(scale, rows, blocks):
    s = np.frombuffer(scale, dtype=np.uint8).reshape(rows, blocks)
    base = s.max(axis=1)
    d = (base[:, None].astype(np.int16) - s.astype(np.int16))
    esc = np.argwhere(d > 2)
    if len(esc) > MAX_ESC: return None
    code = np.where(d > 2, 3, d).astype(np.uint8).reshape(-1, 4)
    packed = (code[:, 0] | (code[:, 1] << 2) | (code[:, 2] << 4) | (code[:, 3] << 6)).astype(np.uint8)
    out = bytearray(COMPACT)
    out[:rows] = base.tobytes(); out[rows:rows + len(packed)] = packed.tobytes()
    assert rows + len(packed) <= ESC_OFF
    struct.pack_into("<H", out, ESC_OFF, len(esc))
    for i, (r, b) in enumerate(esc): struct.pack_into("<IB", out, ESC_OFF + 2 + 5 * i, int(r) * blocks + int(b), int(s[r, b]))
    return bytes(out)
def dec_plane(comp, rows, blocks):
    base = np.frombuffer(comp[:rows], dtype=np.uint8)
    packed = np.frombuffer(comp[rows:rows + rows * blocks // 4], dtype=np.uint8)
    code = np.stack([packed & 3, (packed >> 2) & 3, (packed >> 4) & 3, (packed >> 6) & 3], axis=1).reshape(rows, blocks)
    s = (base[:, None].astype(np.int16) - code.astype(np.int16)).astype(np.uint8).reshape(-1).copy()
    n = struct.unpack_from("<H", comp, ESC_OFF)[0]
    for i in range(n):
        idx, v = struct.unpack_from("<IB", comp, ESC_OFF + 2 + 5 * i); s[idx] = v
    return s.tobytes()
def encode(buf):
    if len(buf) != V1: raise ValueError(f"not a raw-v1 expert file ({len(buf)} bytes)")
    out = []
    for i, (rows, blocks) in enumerate(PLANES):
        o = i * SEC1; c = enc_plane(buf[o + PACKED:o + SEC1], rows, blocks)
        if c is None: return None
        out.append(buf[o:o + PACKED]); out.append(c)
    return b"".join(out)
def decode(buf):
    if len(buf) != V2: raise ValueError(f"not a v2 expert file ({len(buf)} bytes)")
    out = []
    for i, (rows, blocks) in enumerate(PLANES):
        o = i * SEC2; out.append(buf[o:o + PACKED]); out.append(dec_plane(buf[o + PACKED:o + SEC2], rows, blocks))
    return b"".join(out)
def convert_one(args):
    path, reverse, dry = args
    try:
        if os.path.islink(path): return ("link", 0)
        size = os.path.getsize(path)
        if size == (V1 if reverse else V2): return ("done", 0)
        if size != (V2 if reverse else V1): return ("odd", 0)
        with open(path, "rb") as f: src = f.read()
        new = decode(src) if reverse else encode(src)
        if new is None: return ("escapes", 0)
        back = encode(new) if reverse else decode(new)       # the round trip must give the source back
        if back != src: return ("MISMATCH", 0)
        if dry: return ("would", len(src) - len(new))
        tmp = path + ".recode-tmp"
        with open(tmp, "wb") as f:
            f.write(new); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        return ("converted", len(src) - len(new))
    except Exception as e:
        return (f"ERROR {type(e).__name__}: {e}", 0)
def convert(argv):
    d = argv[0]; reverse = "--reverse" in argv; dry = "--dry-run" in argv
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 4
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    names = sorted(n for n in os.listdir(d) if re.fullmatch(r"L\d+-E\d+\.bin", n))
    if limit is not None: names = names[:limit]
    for n in os.listdir(d):
        if n.endswith(".recode-tmp"): os.unlink(os.path.join(d, n))     # left by an interrupted run; the source file is intact
    from multiprocessing import Pool
    t0 = time.time(); counts = {}; saved = 0
    with Pool(jobs) as pool:
        for i, (what, delta) in enumerate(pool.imap_unordered(convert_one, [(os.path.join(d, n), reverse, dry) for n in names], chunksize=8), 1):
            counts[what] = counts.get(what, 0) + 1; saved += delta
            if what == "MISMATCH" or what.startswith("ERROR"): print(f"  {what}", flush=True)
            if i % 2000 == 0: print(f"  {i}/{len(names)} files, {time.time() - t0:.0f} s", flush=True)
    print(f"{d}: {len(names)} files in {time.time() - t0:.0f} s: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"; {saved / 1e9:+.2f} GB freed" + (" (dry run)" if dry else ""))
    return 1 if any(k == "MISMATCH" or k.startswith("ERROR") for k in counts) else 0
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "convert": sys.exit(convert(sys.argv[2:]))
    if cmd == "check":
        n = ok = esc_files = 0
        for f in sys.argv[2:]:
            b = open(f, "rb").read(); n += 1; v2 = encode(b)
            if v2 is None: esc_files += 1; continue
            assert len(v2) == V2 and decode(v2) == b, f
            ok += 1
        print(f"{n} files: {ok} round-trip exactly at {V2} bytes ({100 * (1 - V2 / V1):.2f} % smaller), {esc_files} with too many escapes (kept raw-v1)")
    elif cmd == "encode":
        b = open(sys.argv[2], "rb").read(); v2 = encode(b)
        if v2 is None: sys.exit("too many escapes: left as raw-v1")
        assert decode(v2) == b
        open(sys.argv[3], "wb").write(v2)
    elif cmd == "decode":
        open(sys.argv[3], "wb").write(decode(open(sys.argv[2], "rb").read()))
    else: sys.exit(__doc__)
