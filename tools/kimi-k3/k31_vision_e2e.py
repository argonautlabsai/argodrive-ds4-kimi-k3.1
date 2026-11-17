#!/usr/bin/env python3
"""End-to-end check of K3.1 images through the server (17 Nov 2026). The server runs with DS4_K3_DUMP_PROMPT=DUMP; after
every request the prompt ids it wrote are compared with Moonshot's encoder (k31_prompt_check.py).

  1. image A: describe it (text, shapes, colours)
  2. the follow-up turn on image A: the stored session is reused, image A's rows come from the image cache
  3. the same conversation with image B, the same size as A, so the prompt's ids are those of turn 2 extended: the server
     must see the other image and start over, and the answer must be about B
  4. text only
  5. image A at twice the size (about 2,500 image tokens): timing
  6-8. three new conversations on image A with different questions: the second saves the state after the image (the
     prefix it shares with the first), the third restores it and prefills only its question
  9. the same question on image B (same size, same ids up to the question): the saved prefix holds image A, so it must
     not be restored, and the answer must be about B

usage: k31_vision_e2e.py --url URL --images DIR --tokenizer DIR --dump DUMP --out DIR
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTION = "Describe this image: what does the text say, and which shapes in which colours do you see, from left to right?"


def image_part(path):
    with open(path, "rb") as f:
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(f.read()).decode()}}


def user(path, text):
    return {"role": "user", "content": [image_part(path), {"type": "text", "text": text}]}


def ask(args, name, messages, max_tokens=120):
    req = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "thinking": False}
    rpath = os.path.join(args.out, name + ".json")
    with open(rpath, "w") as f:
        json.dump(req, f)
    if os.path.exists(args.dump):
        os.remove(args.dump)
    t0 = time.time()
    r = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=1800) as resp:
            out = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"--- {name}: HTTP {e.code}: {e.read().decode(errors='replace')}")
        return None
    secs = time.time() - t0
    answer = out["choices"][0]["message"]["content"]
    print(f"--- {name}: {secs:.1f} s, usage {out.get('usage')}")
    print("answer:", answer.strip())
    if os.path.exists(args.dump):
        ids = os.path.join(args.out, name + ".ids.txt")
        shutil.copy(args.dump, ids)
        chk = subprocess.run([sys.executable, "-I", os.path.join(HERE, "k31_prompt_check.py"), "--tokenizer", args.tokenizer,
                              "--request", rpath, "--ids", ids], capture_output=True, text=True)
        print("prompt ids:", (chk.stdout.strip().splitlines() or ["?"])[-1], chk.stderr.strip()[-300:] if chk.returncode not in (0, 1) else "")
    sys.stdout.flush()
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    a, b, big = (os.path.join(args.images, n) for n in ("img-a.png", "img-b.png", "img-a-big.png"))
    first = ask(args, "t1-image-a", [user(a, QUESTION)])
    if first is None:
        return 1
    follow = "Which shape is in the middle, and what colour is it?"
    conv = [user(a, QUESTION), {"role": "assistant", "content": first}, {"role": "user", "content": follow}]
    second = ask(args, "t2-follow-up", conv, max_tokens=40)
    if second is not None:
        conv_b = [user(b, QUESTION), {"role": "assistant", "content": first}, {"role": "user", "content": follow},
                  {"role": "assistant", "content": second}, {"role": "user", "content": "Look at the picture again: what does its text say now?"}]
        ask(args, "t3-image-b-same-ids", conv_b, max_tokens=40)
    ask(args, "t4-text-only", [{"role": "user", "content": "In one sentence: what is the capital of France?"}], max_tokens=30)
    ask(args, "t5-image-a-big", [user(big, QUESTION)])
    ask(args, "t6-new-chat-a", [user(a, QUESTION)])
    ask(args, "t7-new-chat-a-triangle", [user(a, "What colour is the triangle?")], max_tokens=30)
    ask(args, "t8-new-chat-a-count", [user(a, "How many shapes are there, and what does the text say?")], max_tokens=40)
    ask(args, "t9-new-chat-b-count", [user(b, "How many shapes are there, and what does the text say?")], max_tokens=40)
    return 0


if __name__ == "__main__":
    sys.exit(main())
