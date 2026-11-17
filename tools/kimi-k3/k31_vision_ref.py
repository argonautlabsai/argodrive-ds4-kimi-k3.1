#!/usr/bin/env python3
"""Reference for the K3.1 vision path (17 Nov 2026): Moonshot's preprocessing and the vision tower + projector in plain
PyTorch, float32, written from vLLM's kimi_k25_vit.py (MoonViT3d, sd2_tpool merge, patchmergerv2 projector) and
checked against SGLang's kimi_k3_vl.py. The engine's GPU path is compared with this one image at a time.

Details that matter for matching the BF16 serving stacks:
  - encoder RMSNorms are nn.RMSNorm without eps, so torch uses finfo(bfloat16).eps = 2^-7 on BF16 activations: kept
    here as an explicit 2^-7 (the projector's post-norm has eps 1e-5 from projector_ln_eps);
  - MLP activation gelu (tanh form), projector activation exact gelu;
  - 2D rotary embedding on interleaved pairs (not neox), x and y frequencies alternating, theta 10000, per 128-dim head;
  - position table [64, 64, 1024] resized bilinearly (align_corners False) to the patch grid;
  - preprocessing: navit resize (in_patch_limit 65536, 512 patches a side), PIL bicubic, transparent pixels filled
    with Moonshot's chessboard after the resize (float32 blend, truncated), zero padding to a multiple of 28,
    (x / 255 - 0.5) * 2, 14x14 patches; no EXIF rotation (PIL's Image.open does not apply it).

usage: k31_vision_ref.py --shards DIR IMAGE --out PREFIX
writes PREFIX.npz: embeds [tokens x 7168] f32, pixel patches, grid (t, h, w), original size, a few stage checkpoints;
and PREFIX.patches.f32 / PREFIX.embeds.f32 raw, for k31_vision_gpu_test.
"""
import argparse
import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PATCH, MERGE = 14, 2
IN_PATCH_LIMIT, LIMIT_ONE_SIDE = 65536, 512
EPS_VT = 2.0 ** -7          # torch.finfo(torch.bfloat16).eps
EPS_PROJ = 1e-5
HEADS, HEAD_DIM = 12, 128
LAYERS = 27


def load_tensors(shard_dir):
    out = {}
    for fn in sorted(os.listdir(shard_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(shard_dir, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            base = 8 + n
            for name, info in hdr.items():
                if name == "__metadata__" or not (name.startswith("vision_tower.") or name.startswith("mm_projector.")):
                    continue
                assert info["dtype"] == "BF16", (name, info["dtype"])
                a, b = info["data_offsets"]
                f.seek(base + a)
                raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
                out[name] = torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"]).copy())
    return out


def navit_resize(width, height):
    s1 = math.sqrt(IN_PATCH_LIMIT / (max(1.0, width // PATCH) * max(1.0, height // PATCH)))
    s2 = LIMIT_ONE_SIDE * PATCH / width
    s3 = LIMIT_ONE_SIDE * PATCH / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w, new_h = min(new_w, LIMIT_ONE_SIDE * PATCH), min(new_h, LIMIT_ONE_SIDE * PATCH)
    factor = MERGE * PATCH
    return new_w, new_h, (factor - new_w % factor) % factor, (factor - new_h % factor) % factor


def chessboard_fill(img):
    """media_utils.fill_transparent_bg_with with Moonshot's chessboard config: squares of 8, white 255 at the top
    left, gray 180; alpha blend in float32, then truncation to uint8 (numpy astype), as their code does"""
    if img.mode == "RGB":
        return img
    if not ("A" in img.getbands() or "transparency" in img.info):
        return img.convert("RGB")
    a = np.array(img.convert("RGBA"))
    h, w = a.shape[:2]
    bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if (y // 8 + x // 8) % 2 == 1:
                bg[y:y + 8, x:x + 8] = 180
    alpha = a[:, :, 3].astype(np.float32) / 255.0
    alpha_3d = np.stack([alpha] * 3, axis=2)
    out = alpha_3d * a[:, :, :3] + (1 - alpha_3d) * bg
    return Image.fromarray(out.astype(np.uint8))


def preprocess(path, compat="moonshot"):
    """KimiK3VisionProcessor.preprocess for one image: the image as opened (no EXIF rotation), resized in its own
    mode by PIL bicubic (Pillow itself switches palette images to nearest), the transparent background filled after
    the resize, zero padding, normalisation, patches. compat "vllm": vLLM's fused Kimi processor, which converts every
    image to RGB on load (alpha dropped, no chessboard) before the same resize and normalisation."""
    img = Image.open(path)
    img.load()
    if compat == "vllm":
        img = img.convert("RGB")
    w, h = img.size
    new_w, new_h, pad_w, pad_h = navit_resize(w, h)
    img = img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    img = chessboard_fill(img)
    x = np.asarray(img)
    x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    x = (x / 255.0).astype(np.float32)
    x -= np.array([0.5, 0.5, 0.5])
    x *= 1.0 / np.array([0.5, 0.5, 0.5])
    x = x.astype(np.float32)[None]              # (t, H, W, C)
    T, H, W, C = x.shape
    patches = x.reshape(T, H // PATCH, PATCH, W // PATCH, PATCH, C).transpose(0, 1, 3, 5, 2, 4).reshape(-1, C, PATCH, PATCH)
    return np.ascontiguousarray(patches), (T, H // PATCH, W // PATCH), (w, h), x[0]


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_freqs(h, w):
    dim = HEAD_DIM
    k = torch.arange(0, dim, 4)[: dim // 4].float()
    freqs = 1.0 / (10000.0 ** (k / dim))                                  # 32
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    xf = torch.outer(xs.reshape(-1), freqs)                              # N, 32
    yf = torch.outer(ys.reshape(-1), freqs)
    ang = torch.stack([xf, yf], dim=-1).reshape(h * w, dim // 2)        # x0, y0, x1, y1, ...
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):                                            # x: N, heads, 128 (pairs 2i, 2i+1)
    xr, xi = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = torch.empty_like(x)
    out[..., 0::2] = xr * c - xi * s
    out[..., 1::2] = xr * s + xi * c
    return out


@torch.inference_mode()
def forward(W, patches, grid):
    t, h, w = grid
    assert t == 1
    x = torch.from_numpy(patches).reshape(patches.shape[0], -1) @ W["vision_tower.patch_embed.proj.weight"].reshape(1024, -1).T
    pos = W["vision_tower.patch_embed.pos_emb.weight"]                  # 64, 64, 1024
    if (h, w) != (64, 64):
        pos = F.interpolate(pos.permute(2, 0, 1)[None], size=(h, w), mode="bilinear")[0].permute(1, 2, 0)
    x = x + pos.reshape(h * w, 1024)
    ck = {"patch_embed": x.clone()}
    cos, sin = rope_freqs(h, w)
    n = x.shape[0]
    for i in range(LAYERS):
        p = f"vision_tower.encoder.blocks.{i}."
        y = rms(x, W[p + "norm0.weight"], EPS_VT)
        qkv = (y @ W[p + "wqkv.weight"].T).view(n, 3, HEADS, HEAD_DIM)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = torch.softmax((q.transpose(0, 1) @ k.transpose(0, 1).transpose(1, 2)) * HEAD_DIM ** -0.5, dim=-1)
        o = (att @ v.transpose(0, 1)).transpose(0, 1).reshape(n, HEADS * HEAD_DIM)
        x = x + o @ W[p + "wo.weight"].T
        y = rms(x, W[p + "norm1.weight"], EPS_VT)
        y = F.gelu(y @ W[p + "mlp.fc0.weight"].T, approximate="tanh")
        x = x + y @ W[p + "mlp.fc1.weight"].T
        if i == 0:
            ck["block0"] = x.clone()
    x = rms(x, W["vision_tower.encoder.final_layernorm.weight"], EPS_VT)
    ck["encoder"] = x.clone()
    nh, nw = h // MERGE, w // MERGE
    m = x.view(nh, MERGE, nw, MERGE, 1024).permute(0, 2, 1, 3, 4).reshape(nh * nw, MERGE * MERGE * 1024)
    y = F.gelu(m @ W["mm_projector.proj.0.weight"].T)
    y = y @ W["mm_projector.proj.2.weight"].T
    y = rms(y, W["mm_projector.post_norm.weight"], EPS_PROJ)
    return y, ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--preprocess-only", action="store_true", help="write PREFIX.patches.f32 and stop")
    ap.add_argument("--compat", choices=["moonshot", "vllm"], default="moonshot")
    ap.add_argument("image")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    patches, grid, size, pixels = preprocess(args.image, args.compat)
    if args.preprocess_only:
        patches.astype(np.float32).tofile(args.out + ".patches.f32")
        print(f"{args.image}: {size[0]}x{size[1]} -> grid {grid}, {grid[1] * grid[2] // 4} image tokens -> {args.out}.patches.f32")
        return
    W = load_tensors(args.shards)
    assert len(W) == 168, len(W)
    emb, ck = forward(W, patches, grid)
    np.savez(args.out + ".npz", embeds=emb.numpy(), patches=patches, grid=np.array(grid), size=np.array(size),
             **{k: v.numpy() for k, v in ck.items()})
    patches.astype(np.float32).tofile(args.out + ".patches.f32")   # for k31_vision_gpu_test
    emb.numpy().astype(np.float32).tofile(args.out + ".embeds.f32")
    print(f"{args.image}: {size[0]}x{size[1]} -> grid {grid}, {emb.shape[0]} image tokens of {emb.shape[1]}; "
          f"embeds mean |x| {emb.abs().mean():.4f}, rms {emb.pow(2).mean().sqrt():.4f} -> {args.out}.npz")


if __name__ == "__main__":
    main()
