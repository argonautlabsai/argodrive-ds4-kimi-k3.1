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
| `DS4_K3_PREFILL_PREFETCH`, `_DIV`, `_MIN`, `_N`, `_INFLIGHT` | 2, 5, 0, 96, 96 | prompt pass: read ahead the next layer's predicted experts from any tier (1: the cold tier only, 0: off), n_tok / `_DIV` per layer but at least `_MIN` and at most `_N` |
| `DS4_K3_BULK_MIN` | 64 | a prompt of at most this many tokens reads like decode (split reads, least expected completion) |
| `DS4_K3_SPINE_LAZY`, `DS4_K3_SPINE_THREADS`, `DS4_K3_POOL_REGION_SLABS` | unset, 8, 16 | a fresh process reads the spine with 8 threads before the first forward, beside the host pool's pinned memory, which comes in regions of 16 slabs so other CUDA calls run between them; `_LAZY=1` stages the spine during the first forward instead |
| `DS4_K3_PROMPT_COMPACT`, `DS4_K3_CPART_MB` | 1, 512 | fixed-order accumulation for long prompt passes, so they repeat exactly; the compact buffer's cap in MB (512 covers the 2,048-token context) |
| `DS4_K3_QWEN`, `DS4_K3_QWEN_PY`, `DS4_K3_QWEN_SCRIPT`, `DS4_K3_QWEN_DIR`, `DS4_K3_QWEN_META`, `DS4_K3_QWEN_CONF` | off, -, -, -, -, 0.6 | the optional Qwen drafter ([below](#the-optional-qwen-drafter)) |
| `DS4_K3_QWEN_CHAT` | unset | allow the drafter on chat prompts; use it with the chat settings [below](#the-optional-qwen-drafter) |
| `DS4_K3_QWEN_ROWS`, `DS4_K3_QWEN_KMIN` | 16, 2 | the most rows a verification block may take (2 to 16), and the fewest drafts worth a block |
| `DS4_K3_QWEN_SPEC` | 17,3 | rows the sidecar prepares past the text while Kimi works, after a block and after a plain step; `0,0` turns it off |
| `DS4_K3_QWEN_ADAPT` | unset | `lo,hi,down,up`: the gate drops by `down` after a proposal whose drafts were all kept and rises by `up` after one that kept fewer than half, within [`lo`, `hi`] |
| `DS4_K3_VERIFY_ADMIT`, `DS4_K3_VERIFY_AUTO` | auto, 4,3,2 | in a verification block, a fill used by fewer rows than the threshold stays transient: recycled first, never copied down. `auto` uses 4 for blocks of 7 rows or more, 3 for 5 or 6 and 2 for 4 or fewer (`DS4_K3_VERIFY_AUTO` sets the three); a number fixes it; 0: every fill goes to the hot end |
| `DS4_K3_VERIFY_WARM` | 0 | a block's fill used by at least this many rows, but fewer than the threshold, is copied down to the host pool when it is evicted |
| `DS4_K3_ROWS_GEMV`, `DS4_K3_ROWS_GEMV_CHECK` | 1, unset | Q8_0 spine matmuls of 2 to 64 rows (verification blocks, short prompt passes) load each weight block once per 8 rows instead of once per row; every row is bit-identical to the single-row call. `_CHECK=<n>` compares the first n calls with the per-row loop on the GPU |
| `DS4_K3_TRACE_HASH`, `DS4_K3_TRACE_HASH_LAYERS` | unset | FNV-1a of each position's logits, and of its attention result, routed output and hidden state per layer; compare two runs with `hash_compare.py` |
| `DS4_K3_TRACE_TIMING` | unset | per-token timing, cache and tier statistics on stderr |
| `DS4_K3_TRACE_FILLS` | unset | per-fill timeline of n decode layers (diagnostic) |
| `DS4_K3_EXPERT_SKIP` | 0 | experimental, lossy: drop low-weight experts that are disk misses. Off; never used for the reported numbers |

## What we measured (6 November 2026, Kimi K3.1 Pro)

We built this from a fresh clone with the commands above and ran it on the defaults, plus `DS4_K3_EXPERT_DIR_C` for the third
drive. That means a full-precision attention cache, the drafter off and no expert skipping. Monitoring was off.
The plain path is unchanged since: the plain runs at both drafter commits, c0dcf69 and 6808130, give the same token ids.

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

### Start-up and prompt pass in this release (7 November 2026)

One fresh process per run, 2 generated, drafter off, the previous release (c0dcf69) and this one (6808130) alternating. The first token is timed from process
start the same way for both: every output line is stamped against a clock read just before the process starts.

| prompt | first token, previous | first token, this release | prompt pass, this release |
|---|---|---|---|
| "The three main financial statements are" (6 tokens, raw) | 25.3 / 28.0 s | 20.1 / 22.3 s | 3.0 / 3.1 s |
| a chat prompt of 286 tokens | 49.1 / 49.3 s | 42.0 / 44.5 s | 24.7 s |
| a chat prompt of 1,549 tokens | 56.9 / 57.4 s | 51.3 / 51.5 s | 34.1 / 34.2 s |

A fresh process now reads the 56.65 GiB spine with 8 threads before the first forward, beside the pool's pinned memory, which takes about 16.6 s
reading at 3.7 GB/s from the second NVMe. The previous release read most of the spine during the first prompt pass, so its prompt-pass times in a
fresh process include that read and aren't comparable. Then the prompt pass reads the next layer's predicted experts ahead from any tier, and a
prompt of up to 64 tokens reads like decode. Runs of the same prompt differ by up to 2.7 s, mostly before the prompt pass starts. The first token's
logits hash the same in every run of a prompt, in both releases.

With the drafter on, the sidecar loads during the spine read and both slow down. In the drafter runs below the spine came in at 2.6 to 3.2 GB/s
instead of 3.6 to 3.7, the sidecar was ready after 19 to 24 s instead of 2.5 to 2.7 s in the previous release, and the record prompt's first token
came after 22.8 to 27.4 s instead of 20.0 to 20.3 s with the drafter off (the release's own counter, which agrees with the stamped clock to 0.04 s).

## The optional Qwen drafter

Off by default. Greedy decoding only, and raw completions only unless you set `DS4_K3_QWEN_CHAT=1`.

At each position `tools/kimi-k3/qwen_sidecar.py` runs Qwen3-0.6B-Base on the text so far, translates its continuation back to Kimi tokens and answers with up to 15 tokens or nothing. A proposal
stops at the first Qwen token below the confidence gate (`DS4_K3_QWEN_CONF`, default 0.6), and one shorter than 2 tokens is dropped. With n tokens the engine runs one verification block of 1 + n
rows, the current token plus the drafts, and keeps the drafts up to the first one Kimi disagrees with. With nothing it takes an ordinary plain step. So the emitted tokens are always Kimi's own choices.

    # needs torch with CUDA, transformers, tokenizers, tiktoken and regex, and the Qwen/Qwen3-0.6B-Base checkpoint (revision da87bfb6)
    DS4_K3_QWEN=1 DS4_K3_QWEN_PY=/path/to/venv/bin/python DS4_K3_QWEN_SCRIPT=tools/kimi-k3/qwen_sidecar.py \
    DS4_K3_QWEN_DIR=/path/to/qwen3-0.6b-base DS4_K3_QWEN_META=/path/to/kimi-tokenizer-files \
    ./ds4 -m k3-spine.gguf --cuda --raw --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 16 \
          --temp 0 -c 2048 -n 512 -p "..."

The recipe comes from the deltafin Kimi K3 work. Three things changed for this engine.

- The cache. A block touches more experts than the two caches hold (an 8-row block about 6,400 to 6,800, against about 4,600), and blocks walk the layers
  in a fixed cycle, so plain LRU got no hits at all. Keeping only the experts that enough of the block's rows use (`DS4_K3_VERIFY_ADMIT`, which follows the
  block's size) cut a block's disk reads by a quarter to a third.
- The spine. A block's rows used to run the single-row GEMV one after another, so every Q8_0 weight was streamed once per row. One kernel now loads each
  weight block once for up to 8 rows (`DS4_K3_ROWS_GEMV`), with each row's arithmetic unchanged. With perfect drafts on the record prompt that took an
  8-row block from 2.37 s to 2.21 s, and Qwen's own drafts on 200 tokens from 2.81 to 2.97 tok/s.
- The width. Once rows got cheaper, both short and wide blocks paid. With perfect drafts, blocks of 8, 12 and 16 rows gave 3.45, 3.75 and 3.68 tok/s. With
  Qwen's drafts on the record prompt, 200 tokens, up to 8 rows gave 2.98 tok/s, up to 12 rows 2.97 and up to 16 rows 3.10, so a block may now take up to
  16 rows (`DS4_K3_QWEN_ROWS`). A full 16-row block takes about 3.8 s, against about 8 s for sixteen plain steps.

Measured 7 November 2026 on the machine above, at commit 6808130, fresh process per run, gate 0.6, third drive on, monitoring off:

| prompt | generated | plain tok/s | with Qwen tok/s | blocks | drafts accepted | tokens from blocks | wall per run, plain | wall per run, Qwen |
|---|---|---|---|---|---|---|---|---|
| "The three main financial statements are" | 128 | 2.02 | 2.90 / 2.92 | 19 | 97 of 107 | 91 % | 86 s | 70 to 73 s |
| same | 200 | 1.98 / 1.99 | 3.10 / 3.11 | 24 | 164 of 174 | 94 % | 124 s | 90 to 93 s |
| same | 512 | 1.88 | 3.11 | 46 | 443 of 459 | 96 % | 295 to 296 s | 191 to 196 s |
| `prompt-512-tokens.txt` from the deltafin benchmark | 200 | 1.52 / 1.53 | 2.64 / 2.67 | 13 | 185 of 185 | 99 % | 178 to 179 s | 126 to 130 s |

The sidecar uses about 2 GB of VRAM and 1.7 GB of host memory and costs 47 to 150 ms per call on the prompts above, 2.6 to 3.2 % of decode time. In a fresh
process it loads during the spine read ([start-up](#start-up-and-prompt-pass-in-this-release-7-november-2026)). While Kimi works on a block or a plain
step it prepares the next rows (`DS4_K3_QWEN_SPEC`) and stops as soon as the next request arrives. France, the 5-token prompt, fires 2 blocks: 2.17 / 2.19 / 2.20 tok/s against 1.95 plain.

Chat text is harder to predict. With the raw settings the drafter made chat slower (1.80 tok/s at a gate of 0.7 against 1.95 plain, on the 286-token
prompt below), so it stays off for chat unless you ask for it. With a stricter gate that adapts it gains instead, by 4 to 63 % on the three chat prompts
below: the gate starts at 0.9, drops by 0.02 after a proposal that was kept whole and rises by 0.1 after one that kept fewer than half of its
drafts, within 0.6 to 0.95.

    DS4_K3_QWEN_CHAT=1 DS4_K3_QWEN_CONF=0.9 DS4_K3_QWEN_ADAPT=0.6,0.95,0.02,0.1

| chat prompt, 200 generated | plain tok/s | with Qwen tok/s | blocks | drafts accepted |
|---|---|---|---|---|
| a 286-token prompt | 1.92 / 1.94 | 2.01 / 2.02 | 12 | 35 of 39 |
| a general question | 1.98 | 2.13 | 34 | 79 of 98 |
| another general question | 1.86 | 3.03 | 19 | 171 of 174 |

## What is identical and what is not

A verification block matches plain decoding bit for bit. Its rows run the plain path's arithmetic: attention and KDA launch per row, an expert used by more than four rows runs in chunks on the same GEMV
kernels, and the multi-row spine GEMV computes each row with the single-row kernel's arithmetic, in the same order. A rejection restores the KDA snapshot and replays the kept rows through the same kernel.
We checked the spine kernel on the GPU against the per-row loop (16,183 calls, all identical) and the whole path with `DS4_K3_TRACE_HASH`: in this release's runs with the drafter, the logits of 3,416
positions show no differing bit against plain decoding, including forced rejections of the first, a middle and the last draft, short proposals, a sidecar that dies or answers garbage, and end-of-text
inside a block. The first drafter release's checks covered every layer of 40 positions (10,842 hashes) and the logits of 5,325 positions. That's evidence from these runs on this GPU and driver, not a
proof for every input.

The expert path doesn't touch the arithmetic. Caches, read scheduling, chunked reads, prefetch and enqueue order only change
when an expert's bytes arrive. Per-layer traces on a 5-token prompt plus 2 generated tokens match a reference run line for line,
and the 128- and 512-token id files are the same with and without pool-first. This release's start-up and prompt-pass changes are of the same
kind: on prompts of 6, 286 and 1,549 tokens the first token's logits hash the same as with the previous release, and plain decoding gives the
same token ids as the previous release on the record and public prompts.

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
