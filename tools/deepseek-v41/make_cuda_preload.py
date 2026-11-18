#!/usr/bin/env python3
"""Preload lists for the CUDA expert caches of DeepSeek V4.1 Flash, from a ranking.

The CUDA backend can fill its VRAM expert cache (DS4_CUDA_VRAM_TIER_LOAD) and its pinned host-RAM tier
(DS4_CUDA_HOST_TIER_LOAD) from a list of experts before the first prompt. This writes both lists so that every
layer gets the same share: the VRAM list holds every ranked expert, the best of each layer first, and a card takes
as many of its own layers' experts as its cache has slots; the RAM list holds the experts that follow each layer's
first --vram-per-layer, in the same order, for the misses the VRAM cache will have.

The ranking is either a make_hotlist.py file ("layer n id:count ...", one line per layer) or the engine's own
dump of a session (DS4_CUDA_REQ_HIST_DUMP: "gate up down count" per line). The lists only decide what is loaded
ahead of time; they change speed, never output. Never rank on a prompt you are going to measure.

usage: make_cuda_preload.py <model.gguf> <ranking> <out prefix> [--vram-per-layer N] [--ram-gb G]
writes <out prefix>.vram.txt and <out prefix>.ram.txt"""
import struct
import sys

SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# bytes per block and weights per block of the routed-expert formats the V4.1 files use
BLOCK = {39: (17, 32), 12: (144, 256), 10: (84, 256), 16: (66, 256)}


def gguf_tensors(path):
    """name -> (type, dims, absolute offset) from the header of a GGUF file."""
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    def rstr():
        return f.read(rd("Q")[0]).decode("utf-8", "replace")

    def skip(t):
        if t in SCALAR:
            f.seek(struct.calcsize(SCALAR[t]), 1)
        elif t == 8:
            f.seek(rd("Q")[0], 1)
        elif t == 9:
            et, n = rd("I")[0], rd("Q")[0]
            if et in SCALAR:
                f.seek(n * struct.calcsize(SCALAR[et]), 1)
            else:
                for _ in range(n):
                    skip(et)
        else:
            sys.exit(f"{path}: unknown GGUF value type {t}")

    if f.read(4) != b"GGUF":
        sys.exit(f"{path}: not a GGUF file")
    rd("I")
    n_tensors, n_kv = rd("QQ")
    align = 32
    for _ in range(n_kv):
        key, t = rstr(), rd("I")[0]
        if key == "general.alignment" and t in SCALAR:
            align = int(rd(SCALAR[t])[0])
        else:
            skip(t)
    infos = []
    for _ in range(n_tensors):
        name = rstr()
        dims = rd("Q" * rd("I")[0])
        typ, off = rd("I")[0], rd("Q")[0]
        infos.append((name, typ, dims, off))
    data = (f.tell() + align - 1) // align * align
    return {name: (typ, dims, data + off) for name, typ, dims, off in infos}


def main():
    args, opt = [], {"--vram-per-layer": 220, "--ram-gb": 48.0}
    it = iter(sys.argv[1:])
    for a in it:
        if a in opt:
            opt[a] = type(opt[a])(next(it))
        else:
            args.append(a)
    if len(args) != 3:
        sys.exit(__doc__)
    model, ranking, prefix = args
    per_layer, ram_gb = opt["--vram-per-layer"], opt["--ram-gb"]

    tensors = gguf_tensors(model)
    layers = {}   # layer -> (gate offset, up offset, down offset, gate bytes per expert, down bytes per expert, experts)
    il = 0
    while f"blk.{il}.ffn_gate_exps.weight" in tensors:
        part = {}
        for k in ("gate", "up", "down"):
            typ, dims, off = tensors[f"blk.{il}.ffn_{k}_exps.weight"]
            if typ not in BLOCK or len(dims) != 3:
                sys.exit(f"blk.{il}.ffn_{k}_exps.weight: tensor type {typ} with dims {list(dims)} is not handled")
            block_bytes, block_weights = BLOCK[typ]
            part[k] = (off, dims[0] * dims[1] // block_weights * block_bytes, dims[2])
        if part["gate"][1] != part["up"][1] or not part["gate"][2] == part["up"][2] == part["down"][2]:
            sys.exit(f"layer {il}: gate, up and down experts do not match")
        layers[il] = (part["gate"][0], part["up"][0], part["down"][0], part["gate"][1], part["down"][1], part["gate"][2])
        il += 1
    if not layers:
        sys.exit(f"{model}: no routed-expert tensors (blk.N.ffn_gate_exps.weight)")

    ranked = {il: {} for il in layers}   # layer -> {expert: count}
    for line in open(ranking):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        if len(p) >= 3 and ":" in p[2]:                       # make_hotlist.py: layer n id:count ...
            if int(p[0]) not in ranked:
                sys.exit(f"{ranking}: layer {p[0]} is not in the model")
            for tok in p[2:]:
                e, _, c = tok.partition(":")
                ranked[int(p[0])][int(e)] = ranked[int(p[0])].get(int(e), 0) + int(c)
        elif len(p) >= 4:                                     # the engine's dump: gate up down count
            g, c = int(p[0]), int(p[3])
            for il, (go, _, _, gb, _, n) in layers.items():
                if go <= g < go + gb * n and (g - go) % gb == 0:
                    e = (g - go) // gb
                    ranked[il][e] = ranked[il].get(e, 0) + c
                    break
            else:
                sys.exit(f"{ranking}: offset {g} is no expert of {model}")
    order = {il: sorted(d.items(), key=lambda kv: (-kv[1], kv[0])) for il, d in ranked.items()}
    for il, (_, _, _, _, _, n) in layers.items():
        if any(e >= n for e, _ in order[il]):
            sys.exit(f"{ranking}: layer {il} names an expert beyond {n - 1}")

    def line(il, e, c):
        go, uo, do, gb, db, _ = layers[il]
        return f"{go + e * gb} {uo + e * gb} {do + e * db} {c}\n"

    depth = max(len(v) for v in order.values())
    go, _, _, gb, db, _ = layers[0]
    slot = 2 * ((gb + 4096 + 4095) // 4096 * 4096) + (db + 4096 + 4095) // 4096 * 4096   # the tier's bytes per expert
    ram_slots = int(ram_gb * 2 ** 30) // slot
    vram, ram = [], []
    for k in range(depth):   # the k-th best expert of every layer, layer by layer
        for il in sorted(order):
            if k < len(order[il]):
                vram.append(line(il, *order[il][k]))
                if k >= per_layer and len(ram) < ram_slots:
                    ram.append(line(il, *order[il][k]))
    open(prefix + ".vram.txt", "w").write("# gate up down count: every ranked expert, each layer's best first\n" + "".join(vram))
    open(prefix + ".ram.txt", "w").write(f"# gate up down count: after each layer's first {per_layer}\n" + "".join(ram))
    print(f"{len(layers)} layers of {layers[0][5]} experts, {(2 * gb + db) / 2 ** 20:.2f} MiB per expert; "
          f"ranked per layer {min(len(v) for v in order.values())}-{depth}")
    print(f"{prefix}.vram.txt: {len(vram)} experts; {prefix}.ram.txt: {len(ram)} experts "
          f"({ram_gb:g} GiB holds {ram_slots}), from rank {per_layer + 1} of each layer")


if __name__ == "__main__":
    main()
