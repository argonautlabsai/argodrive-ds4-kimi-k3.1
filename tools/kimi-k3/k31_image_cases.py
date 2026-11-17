#!/usr/bin/env python3
"""Writes the image cases the K3.1 preprocessing is checked on (17 Nov 2026): every PNG colour type Pillow treats
differently (palette with and without tRNS, gray with a colour key, gray + alpha, RGB with a colour key, RGBA), JPEG
variants (4:4:4, 4:2:2, 4:2:0, progressive, gray), and sizes that make Pillow 12 and 13 order the resize passes
differently. usage: k31_image_cases.py OUTDIR"""
import os
import sys

import numpy as np
from PIL import Image


def scene(w, h, seed):
    """a deterministic picture with gradients, edges and noise, so every resample tap matters"""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(1, w - 1)).astype(np.uint8)
    g = (y * 255 // max(1, h - 1)).astype(np.uint8)
    b = (((x // 17 + y // 23) % 2) * 200 + rng.integers(0, 55, (h, w))).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rgb = Image.fromarray(scene(2400, 1500, 1))
    alpha = Image.fromarray(((scene(2400, 1500, 2)[:, :, 0].astype(np.int32) * 3) % 256).astype(np.uint8))
    rgba = rgb.copy(); rgba.putalpha(alpha)
    cases = {}
    cases["pal-plain.png"] = rgb.resize((9000, 1000)).convert("P", palette=Image.Palette.ADAPTIVE, colors=200)
    p = rgb.resize((9100, 1100)).convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    p.info["transparency"] = 5
    cases["pal-index-trns.png"] = p
    cases["pal-alpha-trns.png"] = rgba.resize((9050, 1200)).quantize(colors=128, method=Image.Quantize.FASTOCTREE)
    cases["pal-small.png"] = rgb.resize((300, 200)).convert("P", palette=Image.Palette.ADAPTIVE, colors=32)
    gray = rgb.convert("L").resize((9200, 900))
    gray.info["transparency"] = 128
    cases["gray-key.png"] = gray
    cases["gray-plain.png"] = rgb.convert("L").resize((640, 480))
    la = rgba.convert("LA").resize((8000, 3000))
    cases["gray-alpha.png"] = la
    rk = rgb.resize((9000, 2000)); rk.info["transparency"] = (10, 20, 30)
    cases["rgb-key.png"] = rk
    cases["rgba-tall.png"] = rgba.resize((3000, 9000))
    cases["rgb-tall.png"] = rgb.resize((3000, 9000))
    cases["rgb-very-tall.png"] = rgb.resize((70, 9000))
    cases["rgba-very-tall.png"] = rgba.resize((80, 9500))
    for name, im in cases.items():
        kw = {}
        if "transparency" in im.info and name != "pal-alpha-trns.png":
            kw["transparency"] = im.info["transparency"]
        im.save(os.path.join(out, name), **kw)
    big = rgb.resize((9000, 5000))
    big.save(os.path.join(out, "jpg-444.jpg"), quality=95, subsampling=0)
    big.save(os.path.join(out, "jpg-422.jpg"), quality=90, subsampling=1)
    big.save(os.path.join(out, "jpg-420.jpg"), quality=85, subsampling=2)
    big.save(os.path.join(out, "jpg-prog.jpg"), quality=88, progressive=True)
    big.convert("L").save(os.path.join(out, "jpg-gray.jpg"), quality=90)
    rgb.resize((3000, 9000)).save(os.path.join(out, "jpg-tall.jpg"), quality=90)
    for name in sorted(os.listdir(out)):
        im = Image.open(os.path.join(out, name))
        print(f"{name}: {im.mode} {im.size} info={sorted(k for k in im.info if k in ('transparency', 'progressive'))}")


if __name__ == "__main__":
    main()
