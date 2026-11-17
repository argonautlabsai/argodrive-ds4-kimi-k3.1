#!/usr/bin/env python3
"""Copies the K3.1 vision tower (vision_tower.*, mm_projector.*: 168 BF16 tensors, 0.9 GB) out of the checkpoint's
shards into one safetensors file, byte for byte (17 Nov 2026), so the engine's DS4_K31_VISION directory can hold that
file instead of the shards. usage: k31_vision_extract.py --shards DIR --out FILE"""
import argparse
import json
import os
import struct
import sys

PREFIXES = ("vision_tower.", "mm_projector.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    found = {}   # name -> (path, absolute offset, length, dtype, shape)
    for fn in sorted(os.listdir(args.shards)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(args.shards, fn)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, info in hdr.items():
            if name != "__metadata__" and name.startswith(PREFIXES) and name not in found:
                a, b = info["data_offsets"]
                found[name] = (path, 8 + n + a, b - a, info["dtype"], info["shape"])
    if len(found) != 168:
        print(f"expected 168 vision tensors, found {len(found)}")
        return 1
    header, offset = {"__metadata__": {"format": "pt", "source": "Kimi-K3.1 checkpoint shards, vision tower and projector"}}, 0
    names = sorted(found)
    for name in names:
        _, _, length, dtype, shape = found[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)   # data starts 8-byte aligned
    tmp = args.out + ".part"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for name in names:
            path, at, length, _, _ = found[name]
            with open(path, "rb") as f:
                f.seek(at)
                data = f.read(length)
            assert len(data) == length, name
            out.write(data)
    os.replace(tmp, args.out)
    print(f"{len(names)} tensors, {offset / 1e9:.3f} GB of data -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
