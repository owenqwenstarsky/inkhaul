# Using multiple GPUs with llama.cpp

This guide explains how to run [llama.cpp](https://github.com/ggml-org/llama.cpp) across more than one GPU. It covers the split modes, the command-line flags that control them, the limitations you need to know about, and ready-to-use recipes for `llama-cli` and `llama-server`.

The CLI arguments listed here are the same for both tools - or most llama.cpp binaries for that matter.

---

## When you need multi-GPU

Reach for multi-GPU when one of these is true:

- **The model doesn't fit in a single GPU's VRAM.** By spreading the weights across two or more GPUs the whole model can stay on accelerators. Otherwise part of the model will need to be run off of the comparatively slower system RAM.
- **You want more throughput.** By distributing the computation across multiple GPUs, each individual GPU has to do less work. This can result in better prefill and/or token generation performance, depending on the split mode and interconnect speed vs. the speed of an individual GPU.

---

## The split modes

Set with `--split-mode` / `-sm`.

| Mode | What it does | When to use |
|---|---|---|
| `none` | Use a single GPU only. Pick which one with `--main-gpu`. | You explicitly want to confine the model to one GPU even though more are visible. |
| `layer` (**default**) | Pipeline parallelism. Each GPU holds a contiguous slice of layers. The KV cache for layer *l* lives on the GPU that owns layer *l*. | Default and most compatible multi-GPU choice. You want more memory than a single GPU provides and your priority is a fast prefill. Can tolerate slow interconnect speeds between GPUs. |
| `row` | **Deprecated.** Older row-split tensor-parallel path with comparatively poor performance. Splits only dense weights across GPUs. | Avoid in new deployments. |
| `expert` | LongHaul only. The layers and the KV cache are on `--main-gpu`, the experts of every MoE layer are split over the GPUs by `--tensor-split`, and all GPUs compute their experts at the same time. `--expert-share` sets this mode with percentages. | One LongHaul model over several Macs. See [longhaul.md](longhaul.md#expert-split). |

> The experimental `tensor` split mode (tensor parallelism through a "meta device") is removed from this fork. `LLAMA_SPLIT_MODE_TENSOR` stays in `llama.h`, but loading a model with it fails.

> LongHaul (models with experts on disk) uses the `layer` split: each GPU caches the experts of its own layers, and `-lhc` is split over the GPUs like the layers (by `-ts`, otherwise by free memory). Pipeline parallelism stays off. `-lhcd` gives a device its own cache size. To run one model over several Macs connected with Thunderbolt, see [longhaul.md](longhaul.md#several-laptops), and for the `expert` split, which runs the GPUs at the same time, see [longhaul.md](longhaul.md#expert-split).

---

## Command-line arguments reference

| Short | Long | Value | Default | Notes |
|---|---|---|---|---|
| `-sm` | `--split-mode` | `none` \| `layer` \| `row` \| `expert` | `layer` | See modes above. |
| `-ts` | `--tensor-split` | comma-separated proportions, e.g. `3,1` | mode-dependent | How much of the model goes to each GPU. If omitted, `layer`/`row` use automatic splitting proportional to memory. With `3,1` on two GPUs, GPU 0 gets 75 %, GPU 1 gets 25 %. The values follow the order in `--device`. |
| `-mg` | `--main-gpu` | integer device index | `0` | The single GPU used in `--split-mode none`. |
| `-ngl` | `--n-gpu-layers` / `--gpu-layers` | integer \| `auto` \| `all` | `auto` | Maximum number of layers to keep in VRAM. Use `999` or `all` to push everything possible to the GPUs. |
| `-dev` | `--device` | comma-separated device names, or `none` | auto | Restrict which devices llama.cpp may use. See `--list-devices` for names. |
| | `--list-devices` | - | - | Print the available devices and their memory. Run this first to learn the names you'd pass to `--device`. |
| `-fa` | `--flash-attn` | `on` \| `off` \| `auto` | `auto` | Required when using quantized V cache. Supported (and therefore enabled by default) for most combinations of models and backends. |
| `-ctk` | `--cache-type-k` | `f32` \| `f16` \| `bf16` \| `q8_0` \| `q4_0` \| ... | `f16` | KV cache type for K. |
| `-ctv` | `--cache-type-v` | same as `-ctk` | `f16` | KV cache type for V. |
| `-fit` | `--fit` | `on` \| `off` | `on` | Auto-fit unset args to device memory. |

---

## Recipes

### 1. Default - pipeline parallel across all visible GPUs

```bash
llama-cli -m model.gguf
llama-server -m model.gguf
```

Easiest configuration. KV cache spreads across the GPUs along with the layers. `--fit` (on by default) sizes things automatically.

### 2. Pipeline parallel with a custom split ratio

```bash
llama-cli -m model.gguf -ts 3,1
```

Useful when GPUs have different memory: GPU 0 (3 parts) and GPU 1 (1 part). Proportions are normalized so `-ts 3,1` is the same as e.g. `-ts 75,25`.

### 3. Single-GPU mode, picking a specific GPU

```bash
llama-cli --list-devices
llama-cli -m model.gguf -dev MTL0
```

Use only the device listed as `MTL0` when calling with `--list-devices`.

---

## Troubleshooting

| Symptom | How to fix |
|---|---|
| Performance is worse with multi-GPU than single-GPU | The performance is bottlenecked by GPU interconnect speed. Try `--split-mode layer` (less communication than `row`). Increase GPU interconnect speed via more PCIe lanes or e.g. NVLink (if available). |
| GPU not used at all | `--n-gpu-layers` is `0` or too low - try explicitly setting `-ngl all`. Or your build doesn't include support for the relevant backend. |
