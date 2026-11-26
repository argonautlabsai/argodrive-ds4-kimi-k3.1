<p align="center"><a href="https://argonautlabs.ai/"><img src="tools/kimi-k3/images/argodrive-mark.svg" width="112" alt="Argonaut Labs"></a></p>

# Argodrive DS4 — Kimi K3.1 Pro and DeepSeek V4.1 Flash 1017

Run Kimi K3.1 Pro (2.8T) and DeepSeek V4.1 Flash 1017 locally, on one or two NVIDIA RTX PRO 6000 cards or on a MacBook Pro
M5 Max, with the experts streamed from NVMe drives. On two cards Kimi decodes at 5.55 tok/s with a small drafter, and
DeepSeek writes JSON at 111 tok/s with its own. The engine is built on Salvatore Sanfilippo's
[ds4](https://github.com/antirez/ds4) and includes the CUDA and Metal ports, the expert caches and the weight-preparation
tools. It is a source release: model weights, drivers and build tools are separate.

## Results

**Kimi K3.1 Pro (2.8T, 1.45 TB of experts streamed from NVMe). Same output as plain decoding.**

| Hardware | Workload | Prefill tok/s | Decode tok/s | Date |
|---|---|---|---|---|
| **2× RTX PRO 6000** (192 GB, CUDA) | Raw text, 6 prompt / 512 output, Qwen drafter | – | **5.54–5.56** | [25 Nov](tools/kimi-k3/README.md#experts-placed-by-measured-disk-reads-25-november-2026-evening) |
| 2× RTX PRO 6000 | Same, drafts copied from context (no draft model) | – | 3.71 | [24 Nov](tools/kimi-k3/README.md#copy-drafts-images-and-both-cards-on-every-layer-24-november-2026) |
| 2× RTX PRO 6000 | Chat / code question, 256 output, Qwen drafter | – | 3.04 / 2.85–2.86 | [25 Nov](tools/kimi-k3/README.md#experts-placed-by-measured-disk-reads-25-november-2026-evening) |
| 2× RTX PRO 6000 | Long prompt, 21,873 tokens | **194** | 1.90 | [25 Nov](tools/kimi-k3/README.md#long-prompts-layer-by-layer-25-november-2026) |
| 1× RTX PRO 6000 (96 GB, CUDA) | Raw text, Qwen drafter / 6,769-token prompt | 120 | 3.87 | [19 Nov](tools/kimi-k3/README.md#two-cards-19-november-2026) / [11 Nov](tools/kimi-k3/README.md#wider-drafter-blocks-and-long-prompt-read-ahead-11-november-2026) |
| MacBook Pro M5 Max (128 GB, Metal) | Raw text, Qwen drafter / 512-token prompt | 9.0–10.4 | 2.49 | [11 Nov](tools/kimi-k3/README.md#apple-silicon-metal) |

Prefill "–" = prompt too short to measure meaningfully. All runs, settings and older results: [Kimi README](tools/kimi-k3/README.md).
Equivalence to Moonshot’s reference implementation remains unvalidated.

**DeepSeek V4.1 Flash 1017 (501.6 GB, MXFP4 experts streamed from NVMe). Same output with or without the drafter.**

| Hardware | Workload | Prefill tok/s | Decode tok/s | Date |
|---|---|---|---|---|
| **2× RTX PRO 6000** (192 GB, CUDA) | 6,051-token prompt, 600-token JSON answer, DSpark drafter | 1,688–1,701 | **111.2–111.9** | [22 Nov](tools/deepseek-v41/README.md#prompts-of-6000-tokens-22-november-afternoon) |
| 2× RTX PRO 6000 | Same, no drafter | **1,717–1,727** | 79.9–80.5 | [22 Nov](tools/deepseek-v41/README.md#prompts-of-6000-tokens-22-november-afternoon) |
| 1× RTX PRO 6000 (96 GB, CUDA) | 1,556-token prompt | 114.5 | 21.0 | [18 Nov](tools/deepseek-v41/README.md#cuda-two-gpus-the-layers-split-between-them) |
| MacBook Pro M5 Max (128 GB, Metal) | 512-token prompt, 200 output, three drives | 72.8 | 22.5 | [13 Nov](tools/deepseek-v41/README.md#the-current-numbers-13-november-2026) |

All runs, switches and older results: [DeepSeek README](tools/deepseek-v41/README.md). For readers:
[Kimi K3.1 Pro on one and two GPUs](https://argonautlabs.ai/research/kimi-k31-single-gpu.html) and
[DeepSeek V4.1 Flash at 111 tokens per second on two GPUs](https://argonautlabs.ai/research/deepseek-v41-two-gpus.html).

<p align="center"><img src="tools/kimi-k3/images/workstation.jpg" width="760" alt="Inside the workstation: two NVIDIA RTX PRO 6000 Blackwell cards above a Hyper M.2 x16 Gen5 card"></p>
<p align="center"><em>The workstation behind these numbers: two RTX PRO 6000 Blackwell cards (96 GB each) above an M.2 card for the NVMe drives.</em></p>

## Get started

**Kimi K3.1 Pro:** the [installation guide](https://argonautlabs.ai/guides/install-kimi-k31.html) covers the build, the
weight download and preparation, the first run and troubleshooting; switches and tools are in the
[Kimi README](tools/kimi-k3/README.md). **DeepSeek V4.1 Flash 1017:** the model file, the streaming switches and the
two-card setup are in the [DeepSeek README](tools/deepseek-v41/README.md). No separate ds4 install or Argodrive Mac app
is needed.

    git clone --branch main https://github.com/argonautlabsai/argodrive-ds4-kimi-k3.1.git
    cd argodrive-ds4-kimi-k3.1
    make ds4 CUDA_ARCH=sm_120      # on a Mac with Apple silicon: make ds4

What Kimi needs:

- 1.45 TB of experts (82,432 MXFP4 files, exactly as in Moonshot's checkpoint, not re-quantized) and a 62 GB Q8_0 spine
  for the other weights; fp32 activations, no expert skipping.
- About 1.5 TB of NVMe to run (one drive is enough; more drives decode faster) and about 3.5 TB free to prepare the
  weights.
- Tested on one or two RTX PRO 6000 (96 GB each) with 70–87 GiB of host RAM, and on a MacBook Pro M5 Max with 128 GB.
  These are the tested setups, not minimum requirements.

## The code

Kimi starts in [`ds4_kimi_k3.c`](ds4_kimi_k3.c) and [`ds4_kimi_k3.h`](ds4_kimi_k3.h); the [server](ds4_k3_serve.c) and
[draft verification](ds4_k3_draft.cuh) are in the `ds4_k3_*` files, and the Metal port is
[`ds4_metal_k3.inc`](ds4_metal_k3.inc) with [`metal/k3.metal`](metal/k3.metal). DeepSeek V4.1 Flash runs on ds4's own
engine with this fork's streaming switches. The default branch is `main`; for a fixed reproduction target use a release
tag, [k31-cuda-20261125](https://github.com/argonautlabsai/argodrive-ds4-kimi-k3.1/tree/k31-cuda-20261125) on CUDA or
[k31-metal-20261111](https://github.com/argonautlabsai/argodrive-ds4-kimi-k3.1/tree/k31-metal-20261111) on a Mac.
[Updating an old `kimi-k3` clone](tools/kimi-k3/BRANCH-MIGRATION.md).

## Credits

Built on [Salvatore Sanfilippo’s ds4](https://github.com/antirez/ds4); the Kimi port started from its commit c97964b. The
original README, model instructions and acknowledgements are in [UPSTREAM_README.md](UPSTREAM_README.md), and the
[license](LICENSE) is unchanged.
