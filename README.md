<p align="center"><a href="https://argonautlabs.ai/"><img src="tools/kimi-k3/images/argodrive-mark.svg" width="112" alt="Argonaut Labs"></a></p>

# Argodrive DS4 Kimi K3.1

## Run Kimi 3.1 locally on one GPU

**This repository is the inference engine.** It already includes the ds4-based runtime, the Kimi K3/K3.1 CUDA port, NVMe expert streaming, expert caches and weight-preparation tools. Clone and build this repository; **no separate upstream ds4 installation or Argodrive Mac app is needed**.

This is a **source release**. Model weights, NVIDIA drivers, CUDA build tools and Python dependencies are separate; there is currently no prebuilt installer.

### Get started

1. **Get and build this engine** using the commands below.
2. **Download and prepare the Kimi K3.1 weights** with the included [weight-preparation tools](tools/kimi-k3/README.md#weight-layout).
3. **Run Kimi locally** using the [CLI or OpenAI-compatible chat endpoint](tools/kimi-k3/README.md#build-and-run). The [Qwen drafter](tools/kimi-k3/README.md#the-optional-qwen-drafter) is optional and needs its own checkpoint and dependencies.

[Full setup and tested hardware](https://argonautlabs.ai/research/kimi-k31-single-gpu.html#reproduce) · [Configuration guide](tools/kimi-k3/README.md)

### Measured performance and scope

**argodrive-ds4-kimi-k3.1** runs Kimi K3.1 Pro (2.8T parameters, 1.45 TB of experts) on one 96 GB GPU, with the experts streamed off NVMe as the router asks
for them: 3.1 tok/s decode on raw completions with its Qwen drafter, which gives the same tokens as plain decoding, and prefill
at up to 44 tok/s.

**Start here:** [Run Kimi 3.1 locally on one GPU — hardware, build guide and benchmarks](https://argonautlabs.ai/research/kimi-k31-single-gpu.html). The tested setup uses one RTX PRO 6000 Blackwell (96 GB), 70 GiB of VM RAM and five physical NVMe devices. The 3.11 tok/s result is raw-text decode with the drafter; 44 tok/s is prefill. The published benchmark profile is limited to 2,048 tokens total, and equivalence to Moonshot’s reference implementation has not been validated.

[Argonaut Labs](https://argonautlabs.ai) built this Kimi K3 / K3.1 Pro port on top of Salvatore Sanfilippo's [ds4](https://github.com/antirez/ds4) at commit c97964b.
The spine sits in VRAM as Q8_0. The 82,432 MXFP4 expert files stay on disk and move through a VRAM cache and a pinned RAM
pool. There's no expert skipping. The drafter is off by default; `DS4_K3_QWEN=1` turns it on
([setup](tools/kimi-k3/README.md#the-optional-qwen-drafter)).

<p align="center"><img src="tools/kimi-k3/images/workstation.jpg" width="760" alt="Inside the workstation: two NVIDIA RTX PRO 6000 Blackwell cards above a Hyper M.2 x16 Gen5 card"></p>
<p align="center"><em>The workstation. The port runs on one of its two RTX PRO 6000 Blackwell cards (96 GB).</em></p>

Measured on 7 November at commit 6808130, one fresh process per run, greedy, raw prompts, with the Qwen drafter:

| test | prefill | decode |
|---|---|---|
| "The three main financial statements are" (6 tokens), 128 generated (tg128) | — | 2.91 tok/s |
| 512-token prompt, 128 generated (pp512 / tg128) | 18.6 tok/s | 2.68 tok/s |
| 1,512-token prompt, 128 generated (pp1512 / tg128) | 44 tok/s | 2.43 tok/s |
| the same 6-token prompt, 512 generated | — | 3.11 tok/s |

The drafter gives the same tokens as plain decoding: a block's rows run the same arithmetic as plain steps, and the logits of 3,416
positions showed no differing bit. Decode depends on how predictable the text is, and chat runs at about 2 tok/s. The 512-token
prompt is `prompt-512-tokens.txt` from the deltafin Kimi K3 benchmark.

The machine is one RTX PRO 6000 Blackwell (96 GB) in a 70 GiB VM. A RAID0 of three NVMe drives (19 GB/s) has the 79,365
hottest experts, a second NVMe (6.2 GB/s) has the rest plus some copies, and a Thunderbolt NVMe (3.7 GB/s) has another 74,000
copies.

The prompt pass is the interesting part. Once a prompt is a few hundred tokens long it hits most of the experts in every
layer, so the pass reads most of the model once and the time barely changes with length. For this model, prefill is a
bandwidth problem.

K3.1's DSA indexer isn't implemented yet. We use dense attention instead, which should compute the same thing as long as you
stay within 2,048 tokens, so run it with `-c 2048`.

Weight layout, knobs, how we measured and what is and isn't bit-identical are all in
[tools/kimi-k3/README.md](tools/kimi-k3/README.md).

    git clone https://github.com/argonautlabsai/argodrive-ds4-kimi-k3.1.git
    cd argodrive-ds4-kimi-k3.1 && git checkout kimi-k3
    make ds4 CUDA_ARCH=sm_120


## Upstream credits and documentation

This engine is built on [Salvatore Sanfilippo’s ds4](https://github.com/antirez/ds4). The original ds4 README, model instructions and acknowledgements are preserved in [UPSTREAM_README.md](UPSTREAM_README.md); the [license](LICENSE) is unchanged. Those upstream model instructions are reference material. For Kimi, use the setup in this README and [tools/kimi-k3/README.md](tools/kimi-k3/README.md).
