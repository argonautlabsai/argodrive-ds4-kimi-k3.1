# Kimi K3 and K3.1 Pro on ds4 with CUDA

This branch (`kimi-k3`) adds Moonshot's Kimi K3 and Kimi K3.1 Pro to Salvatore Sanfilippo's
[ds4](https://github.com/antirez/ds4) (upstream commit c97964b). The model is a 2.8T-parameter MoE with 92 MoE layers, 896
experts per layer and 16 routed per token, and it runs here on one 96 GB CUDA GPU.

Everything that isn't an expert stays in VRAM as Q8_0. Each expert is one raw MXFP4 file on disk, and they move through a
cache in VRAM and a pinned pool in host RAM as the router picks them.

The caches and read scheduling don't change what gets computed, and a given setup gives the same output every time. Different
arithmetic paths don't match each other bit for bit, though. There's a table
[further down](#what-is-identical-and-what-is-not).

## Weight layout

`tools/kimi-k3/k3_spine_to_gguf.py` turns the published checkpoint into

- a sparse spine GGUF (`k3-spine.gguf`): all non-expert tensors in Q8_0, with the expert tensors left as holes, and
  three metadata keys the engine reads: `kimi-k3.experts.primary_dir`, `kimi-k3.experts.dir_b` (optional) and
  `kimi-k3.experts.file_bytes`. `--compact` writes the spine without the sparse tail (plus `<out>.sparse.json`) so it
  can cross a filesystem that cannot hold sparse files; restore the length with `truncate -s` afterwards;
- one file per expert, `L{layer}-E{expert}.bin` (17,547,264 bytes each: gate, up and down projections in MXFP4),
  82,432 files, 1.45 TB in total (`tools/kimi-k3/k31_extract.py`).

Expert files can be split across directories: the primary directory (fastest storage, as many of the hottest experts
as fit), `dir_b` (the rest), and optionally `DS4_K3_EXPERT_DIR_C` and `DS4_K3_EXPERT_DIR_D` (further copies on any
storage). A file may exist in several directories; each read is homed on the directory expected to finish it first.
A file missing from one directory is read from the next.

### Kimi K3.1 Pro

K3.1 Pro has the same tensor set as K3 plus 120 `self_attn.indexer.*` tensors on its 24 full-attention layers (the
Gated DSA lightning indexer, top-2,048 tokens). The converter and the engine ignore them, so the pipeline runs K3.1
unchanged (`k3_spine_to_gguf.py --name "Kimi K3.1 Pro"`; `k31_extract.py --replace-in` overwrites a K3 expert layout
in place, atomically and resumably). We don't implement the indexer yet, so dense attention stands in for DSA. That
only holds while prompt plus generation stay within 2,048 tokens: below the indexer's window every token gets selected, so by
the model's definition the two compute the same attention. We haven't checked this against Moonshot's own implementation. Run
K3.1 with `-c 2048`. `k31_diff.py` compares two checkpoints tensor by tensor.

## Build and run

    git clone https://github.com/argonautlabsai/argodrive-ds4-kimi-k3.1.git
    cd argodrive-ds4-kimi-k3.1 && git checkout kimi-k3
    make ds4 CUDA_ARCH=sm_120          # Blackwell; use your GPU's arch

    # one-shot
    ./ds4 -m k3-spine.gguf --cuda --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 16 \
          --temp 0 -c 2048 -p "..."
    # OpenAI-compatible chat endpoint (chat template, thinking, tool calls, JSON mode)
    ./ds4 -m k3-spine.gguf --cuda --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 16 \
          --temp 0 -c 2048 -n 512 --k3-serve HOST:PORT

The examples use `-c 2048` for K3.1 Pro; Kimi K3 (not Pro) can run with `-c 4096`. Set `DS4_K3_EXPERT_DIR_C` to a third
expert directory if you have one. `DS4_K3_MLA_CACHE=bf16` keeps the dense attention cache in half precision. It frees 3 GB
of VRAM at `-c 2048` and changed decode by a few percent at most on our machine, but it can flip near-tie tokens, so the
numbers below don't use it.

## What the engine does per token

1. Attention, KDA and the router run on the GPU; routing (sigmoid + bias top-16, renormalised) is mirrored on the host.
2. The 16 experts of a layer are served from the device LRU cache (hit), the pinned host victim cache (upload), or
   disk (a reader pool with O_DIRECT reads into a pinned arena, split reads, per-tier in-flight caps, and a
   least-expected-completion balancer across the directories that hold the file).
3. Each expert's GEMVs launch as soon as its slab lands; the layer completes when the last one does. A shallow
   next-layer prefetch fills the gap while the GPU runs attention.
4. Prompt passes run as batched GEMMs with a prefetch of the next layer's experts and a prefix checkpoint for the
   shared system prompt; long prompts are sized to free VRAM and chunked only when they do not fit.

## Main environment knobs (the defaults are what we measured with)

| variable | default | meaning |
|---|---|---|
| `DS4_K3_CACHE_RAM_GB` | 50 | pinned host expert cache (yields to free RAM minus `DS4_K3_CACHE_RAM_RESERVE_MB`, 8192) |
| `DS4_K3_CACHE_VRAM_RESERVE_MB` | 4096 | VRAM kept free for prompt-pass buffers; the device cache takes the rest |
| `DS4_K3_PREFETCH` / `_N` / `_RANK` / `_INFLIGHT` | all / 3 / 4 / 6 | next-layer decode prefetch from the router prediction |
| `DS4_K3_SPLIT_READ`, `DS4_K3_SPLIT_DECODE_ONLY` | 2, 1 | decode fills read as two chunks; prompt-pass fills whole-file |
| `DS4_K3_EXPERT_SLOTS`, `DS4_K3_EXPERT_THREADS` | 32, slots x split + 16 | fills in flight, reader threads |
| `DS4_K3_SPLIT_ETA` | 2 | per-chunk homing across directories; mode 2 keeps each directory at its own depth for prompt passes (slow copies never end a layer) |
| `DS4_K3_SPLIT_ETA_GBPS`, `DS4_K3_SPLIT_STREAM_GBPS` | 27,6.8,3.6[,3.6] / 6,6.7,3.6[,3.6] | aggregate and per-stream GB/s the balancer assumes per directory (measure yours) |
| `DS4_K3_COLD_INFLIGHT`, `DS4_K3_GREEN_INFLIGHT`, `DS4_K3_DIRD_INFLIGHT` | 8, 16, 16 | in-flight caps for `dir_b`, tier C and tier D |
| `DS4_K3_EXPERT_DIR_C`, `DS4_K3_EXPERT_DIR_D` | unset | paths of further expert copies (tier C, tier D) |
| `DS4_K3_UPLOAD_STREAMS` | 8 | CUDA streams for slab uploads |
| `DS4_K3_POOL_FIRST` | 1 | decode: the layer's experts already on the device or in the host pool are visited before the ones needing a disk read (+2 % measured; from the argodrive V4.1 fork) |
| `DS4_K3_POOL_KEEP`, `DS4_K3_POOL_EVICT_BIAS` | 0, 0 | a host hit keeps its host copy; prefer evicting a slab the pool still holds (neutral to negative on this rig; kept for other cache geometries) |
| `DS4_K3_POOL_LAZY` | unset | create the expert pool at the first token instead of at load |
| `DS4_K3_PREFILL_SEQUENTIAL` | unset | process the prompt through the decode path one token at a time, about 0.7 s per token: 29 s vs 36 s fresh-process wall on the 5-token prompt, 192 s vs 39 s for a 286-token prompt pass. A different float path from the batched pass |
| `DS4_K3_PROMPT_COMPACT`, `DS4_K3_CPART_MB` | 1, 512 | fixed-order accumulation for long prompt passes, so they repeat exactly; the compact buffer's cap in MB (512 covers the 2,048-token context) |
| `DS4_K3_QWEN`, `DS4_K3_QWEN_PY`, `DS4_K3_QWEN_SCRIPT`, `DS4_K3_QWEN_DIR`, `DS4_K3_QWEN_META`, `DS4_K3_QWEN_CONF` | off, -, -, -, -, 0.6 | the optional Qwen drafter ([below](#the-optional-qwen-drafter)) |
| `DS4_K3_QWEN_CHAT` | unset | allow the drafter on chat prompts (slower on what we tested) |
| `DS4_K3_VERIFY_ADMIT` | 2 | in a verification block, a fill used by fewer rows stays transient: recycled first, never copied down. 0: every fill goes to the hot end |
| `DS4_K3_TRACE_HASH`, `DS4_K3_TRACE_HASH_LAYERS` | unset | FNV-1a of each position's logits, and of its attention result, routed output and hidden state per layer; compare two runs with `hash_compare.py` |
| `DS4_K3_TRACE_TIMING` | unset | per-token timing, cache and tier statistics on stderr |
| `DS4_K3_TRACE_FILLS` | unset | per-fill timeline of n decode layers (diagnostic) |
| `DS4_K3_EXPERT_SKIP` | 0 | experimental, lossy: drop low-weight experts that are disk misses. Off; never used for the reported numbers |

## What we measured (6 November 2026, Kimi K3.1 Pro)

We built this from a fresh clone with the commands above and ran it on the defaults, plus `DS4_K3_EXPERT_DIR_C` for the third
drive. That means a full-precision attention cache, the drafter off and no expert skipping. Monitoring was off. The plain path is unchanged since: the drafter commit's plain runs give the same token ids.

The machine is one RTX PRO 6000 Blackwell (96 GB) in a 70 GiB VM on a Threadripper host, with a 27.5 GB device cache and a
53.7 GB pinned host pool. The experts live in three places:

- a RAID0 of three NVMe drives (19 GB/s measured in the VM) with the 79,365 hottest,
- a second NVMe (6.2 GB/s) with 41,078, which together with the RAID covers all 82,432,
- a Thunderbolt NVMe (3.7 GB/s, limited by the link) with another 74,000 copies.

### One fresh process per run

These are the two prompts from the deltafin Kimi K3 benchmark, run raw and greedy on K3.1 Pro.

| prompt | prompt pass | generated | runs | steady tok/s | wall per run |
|---|---|---|---|---|---|
| "The capital of France is" (5 tokens) | sequential (`DS4_K3_PREFILL_SEQUENTIAL=1`) | 17 | 3 | 2.07 / 2.08 / 2.08 | 29 s |
| same | batched (default) | 17 | 3 | 1.98 / 1.97 / 1.96 | 36 s |
| "The three main financial statements are" (6 tokens) | batched | 128 | 2 | 2.04 / 2.04 | 90 s |
| same | batched | 512 | 2 | 1.94 / 1.94 | 292 s |

Steady is the tokens after the first divided by the time after the first token; the engine prints it as `K3 decode split`. Wall
is process start to exit. A fresh process spends about 21 s on the sequential path and 28 s on the batched one before steady
decode starts: loading, 7 s allocating the expert pool, the rest of the 48 GB spine upload, then the prompt. A running server
only pays for the loading once.

For comparison, deltafin on our M5 Max (Kimi K3, four drives, drafter on) did 1.13 tok/s over 128 tokens and 1.00 over 512. Its
17-token run takes about 20 s of wall against our 29 s, because on something that short the loading and the prompt dominate.

### Through the chat endpoint (`-c 2048`)

| prompt tokens | generated | prompt pass (two runs) | prefill rate | expert bytes read in the pass | decode tok/s (two runs) |
|---|---|---|---|---|---|
| 270 | 200 | 25.4 / 25.3 s | 10.6 tok/s | 925 GB | 1.96 / 1.93 |
| 618 | 200 | 31.7 / 31.5 s | 19.5 tok/s | 1,122 GB | 1.90 / 1.90 |
| 1,533 | 8 | 36.8 / 37.1 s | 41 tok/s | 1,241 GB | too few tokens to quote |

On the 270-token prompt each decode token costs about 590 device hits, 260 host hits and 750 disk reads of 17.5 MB. That's
13.1 GB at about 32 GB/s, plus about 1,000 copy-downs into the host pool. A step takes 0.51 s and 0.40 s of that is waiting
on the drives.

Long answers slow down because more experts miss the caches. In the 512-token run it's 11.8 GB per token over the first 128
tokens and 13.4 GB over the last 128, while the time outside the read wait stays at 0.11 s the whole way.

The prompt pass runs at the drives' combined rate, 37 GB/s, and reads each routed expert once per layer. A 5-token prompt reads
100 GB. A 270-token prompt routes to 588 of 896 experts per layer on average (from a route trace), and the 1,533-token prompt
reads 86 % of all expert bytes. So past a few hundred tokens the pass costs about the same whatever the length, and only more
bandwidth or a cached prefix helps.

## The optional Qwen drafter

Off by default. Raw completions and greedy decoding only, unless you set `DS4_K3_QWEN_CHAT=1`.

At each position `tools/kimi-k3/qwen_sidecar.py` runs Qwen3-0.6B-Base on the text so far, translates its continuation back to Kimi tokens and answers with 7 tokens or nothing. It answers
only if all 7 come from Qwen tokens at or above the confidence gate (`DS4_K3_QWEN_CONF`, default 0.6). With 7 tokens the engine runs one verification block of 8 rows, the current token plus
the 7 drafts, and keeps the drafts up to the first one Kimi disagrees with. With nothing it takes an ordinary plain step. So the emitted tokens are always Kimi's own choices.

    # needs torch with CUDA, transformers, tokenizers, tiktoken and regex, and the Qwen/Qwen3-0.6B-Base checkpoint (revision da87bfb6)
    DS4_K3_QWEN=1 DS4_K3_QWEN_PY=/path/to/venv/bin/python DS4_K3_QWEN_SCRIPT=tools/kimi-k3/qwen_sidecar.py \
    DS4_K3_QWEN_DIR=/path/to/qwen3-0.6b-base DS4_K3_QWEN_META=/path/to/kimi-tokenizer-files \
    ./ds4 -m k3-spine.gguf --cuda --raw --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 16 \
          --temp 0 -c 2048 -n 512 -p "..."

The recipe comes from the deltafin Kimi K3 work. Two things changed for this engine. Its width ladder loses here, because small blocks are expensive (2 rows cost 2.3 plain steps), so a block is
8 rows or nothing. And the cache needed a change: an 8-row block touches about 6,400 to 6,800 experts while the two caches hold about 4,600, and blocks walk the layers in a fixed cycle, so plain LRU got no
hits at all. Keeping only the experts that at least two rows use (`DS4_K3_VERIFY_ADMIT`) cut a block's disk reads by a quarter to a third.

Measured 7 November 2026 on the machine below, at commit c0dcf69, fresh process per run, threshold 0.6, third drive on, monitoring off:

| prompt | generated | plain tok/s | with Qwen tok/s | blocks | drafts accepted | tokens from blocks | wall per run, plain | wall per run, Qwen |
|---|---|---|---|---|---|---|---|---|
| "The three main financial statements are" | 128 | 2.00 / 2.02 | 2.34 | 8 | 54 of 56 | 49 % | 91 to 93 s | 85 s |
| same | 200 | 1.98 | 2.52 | 17 | 117 of 119 | 67 % | 129 s | 110 s |
| same | 512 | 1.92 | 2.61 | 53 | 363 of 371 | 81 % | 294 s | 227 s |
| `prompt-512-tokens.txt` from the deltafin benchmark | 200 | 1.58 | 2.29 | 24 | 168 of 168 | 96 % | 179 to 180 s | 143 to 144 s |

A full block takes about 2.60 s on the record prompt, against 3.95 s for eight plain steps. The sidecar uses about 2 GB of VRAM and 1.7 GB of host memory, starts in 2.5 s, and costs about 25 to 37 ms per call, 2 to 4 % of
decode time. The device cache gives up about 20 slabs. Plain rates here are 1 to 2 % below the 6 November table. Another VM on the host was busier that day, and these runs had logit hashing on, which costs under 1 ms per position.

What it doesn't help: on a 286-token chat prompt it made decoding slower, 1.80 tok/s with threshold 0.7 and 1.90 with threshold 0.9 (no block fired), against 1.95. Qwen is rarely right seven tokens in a row on chat
text, so we leave it off there. France doesn't gain either: no block fires in 17 tokens.

## What is identical and what is not

A verification block matches plain decoding bit for bit. Its rows run the plain path's kernels: attention and KDA launch per row, and an expert used by more than four rows runs in chunks on the same GEMV kernels. A rejection restores
the KDA snapshot and replays the kept rows through the same kernel. We checked this with `DS4_K3_TRACE_HASH`: on the record prompt every layer of 40 positions (10,842 hashes), and the logits of 5,325 positions
across the runs with the drafter, including forced rejections of the first, a middle and the last draft, show no differing bit against plain decoding. That's evidence from these runs on this GPU and driver, not a proof for every input.

The expert path doesn't touch the arithmetic. Caches, read scheduling, chunked reads, prefetch and enqueue order only change
when an expert's bytes arrive. Per-layer traces on a 5-token prompt plus 2 generated tokens match a reference run line for line,
and the 128- and 512-token id files are the same with and without pool-first.

A given setup repeats exactly. Every repeat in the tables above gave the same token ids, or the same text on the chat endpoint.
Long prompt passes keep one weighted row per (token, slot) in a compact buffer and sum the slots in the router's order.
`DS4_K3_PROMPT_COMPACT=0` goes back to the old arrival-order accumulation, which isn't repeatable and wasn't any faster.

Different arithmetic paths don't match each other. The sequential and batched prompt passes, the attention cache's precision and
the expert GEMM all use different float operations. Top logits differ between them by a median of 0.05 to 0.11 and a 99th
percentile of 0.3 to 0.7, with rare jumps up to 2.5 that we haven't explained yet. A token with a narrow margin can flip, and
the answers differ from there on.

Here's where each path first differs from the sequential prompt pass with the full-precision cache, counting generated tokens
from 0:

| path | France, 17 tokens | record prompt, 128 tokens | chat prompt of 286 tokens, 200 tokens |
|---|---|---|---|
| batched, full-precision cache (the default, used above) | token 1 | same | token 59 |
| batched, half-precision cache | token 1 | same | same |
| sequential, half-precision cache | token 1 | token 31 | not run |
| batched, expert GEMM off | token 6 | same | token 147 |
| batched, expert GEMM off, half-precision cache | token 1 | same | token 169 |

Every flip we saw was at a position with a top-two margin of 0.5 logits or less. Turning the expert GEMM off doesn't make the
batched pass match, and it adds 46 % to the prompt pass on the 286-token prompt, so we leave it on.

Two comparisons with the published Kimi K3 text from deltafin on Apple silicon. On the France prompt only the sequential path
with the half-precision cache gives the same 17 ids (" Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also").
On the record prompt all our runs differ at character 91 ("an overview" against "a snapshot"), and we don't know why yet.

So to be clear about what we're claiming: no expert skipping, and no quantisation beyond the Q8_0 spine (itself a quantisation of the BF16 spine) and the MXFP4 experts the checkpoint ships with. The
optional drafter doesn't change the output. The output isn't bit-identical across the other arithmetic paths above, and we haven't compared any of this with Moonshot's own implementation.

## Tools

`k3_spine_to_gguf.py` (checkpoint to spine), `k31_extract.py` (checkpoint to expert files, in place or fresh),
`k31_diff.py` (two checkpoints compared), `gguf_header.py` (spine header dump), `k3_tokenize.py` /
`k3_cpu_forward.c` / `k3src_test.c` (reference checks), `readbench*.c` (per-file read latency and throughput),
`cache_sim*.py`, `sim*.py`, `lru_test.cpp` (cache policy simulators on recorded routes), `serve_summary.py` /
`tier_summary.py` (log summaries), `qwen_sidecar.py` (the drafter), `hash_compare.py` (bit-level comparison of two runs).
