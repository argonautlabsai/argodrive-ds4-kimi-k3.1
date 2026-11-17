#!/usr/bin/env python3
"""Checks the server's rendering of a K3.1 chat with images (17 Nov 2026). Moonshot's encoder (encoding_k3.py through
tokenization_kimi.py's TikTokenTokenizer, as published with the checkpoint) renders the same request; each image's single
<|media_pad|> is expanded to the image's token count, as vLLM and SGLang do; the ids are compared with the ones the server
wrote (DS4_K3_DUMP_PROMPT=IDS.txt).

usage: k31_prompt_check.py --tokenizer DIR --request REQ.json --ids IDS.txt [--thinking]
DIR holds tokenization_kimi.py, encoding_k3.py, tiktoken.model and tokenizer_config.json from the checkpoint. Images in
REQ.json are data: URLs; their sizes and token counts come from k31_vision_ref.py's navit_resize.
"""
import argparse
import base64
import io
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k31_vision_ref import MERGE, PATCH, navit_resize  # noqa: E402

MEDIA_PAD = "<|media_pad|>"


def image_of(part):
    v = part.get(part["type"])
    if isinstance(v, dict):
        v = v.get("url")
    data = v.split(";base64,", 1)[1] if v.startswith("data:") else v
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    img.load()
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer), trust_remote_code=True)   # Moonshot's TikTokenTokenizer
    with open(args.request) as f:
        req = json.load(f)
    messages = req["messages"]
    prompts, counts = [], []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                img = image_of(part)
                w, h = img.size
                nw, nh, pw, ph = navit_resize(w, h)
                counts.append(((nw + pw) // (PATCH * MERGE)) * ((nh + ph) // (PATCH * MERGE)))
                prompts.append(f"<|media_begin|>image {w}x{h}<|media_content|>{MEDIA_PAD}<|media_end|>")
    ids = tok.apply_chat_template(messages, tools=req.get("tools"), tokenize=True, add_generation_prompt=True,
                                  thinking=args.thinking, image_prompts=prompts or None)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    pad = tok.convert_tokens_to_ids(MEDIA_PAD)
    ref, k = [], 0
    for t in ids:
        if t == pad:
            ref.extend([pad] * counts[k])
            k += 1
        else:
            ref.append(t)
    assert k == len(counts), (k, len(counts))
    with open(args.ids) as f:
        got = [int(x) for x in f.read().split()]
    print(f"{len(prompts)} image(s), token counts {counts}; reference {len(ref)} ids, server {len(got)} ids")
    if got == ref:
        print("IDENTICAL")
        return 0
    i = next((i for i in range(min(len(got), len(ref))) if got[i] != ref[i]), min(len(got), len(ref)))
    print(f"DIFFERENT from position {i}: reference {ref[max(0, i - 3):i + 6]} server {got[max(0, i - 3):i + 6]}")
    print("reference text there:", repr(tok.decode(ref[max(0, i - 3):i + 6])), "server:", repr(tok.decode(got[max(0, i - 3):i + 6])))
    return 1


if __name__ == "__main__":
    sys.exit(main())
