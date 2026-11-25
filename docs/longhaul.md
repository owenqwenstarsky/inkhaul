# LongHaul

LongHaul runs Inkling 2 and Ling-3.2-air (`bailingmoe3`) MoE models on a Mac or an NVIDIA GPU with less memory than the model needs. The MoE experts stay in the GGUF file on disk. A fixed-size cache in GPU memory (unified memory on a Mac, VRAM with CUDA) keeps the experts that were used recently and often. Everything else (attention, router, norms, embeddings, output) is loaded as usual.

LongHaul is always on for models with experts on disk. Only the cache size can be set:

| Argument | |
|---|---|
| `-lhc, --longhaul-cache SIZE` | cache size: a number in MiB, or with a K, M or G suffix, e.g. `512M`, `1G`, `2G` (default: `512M`) |

`llama-bench` takes the same option as a list (`-lhc 512M,1G,2G`) to compare sizes.

## How it works

- The expert tensors are memory mapped but never read through the mapping, so they use address space, not RAM.
- Each MoE layer looks up its selected experts in the cache. A miss is read from the file with `pread`, so that the OS file cache does not keep a second copy: on macOS the file is opened with `F_NOCACHE`, on Linux each read is dropped from the page cache with `POSIX_FADV_DONTNEED` (and readahead is off). On Windows the reads go through the file cache. The misses of a layer are read by 4 threads at once, each through its own 8 MiB staging buffer, into the cache, which is a Metal or CUDA buffer. `LLAMA_LONGHAUL_IO_THREADS` sets the number of threads (1 to 64; 1 reads on the compute thread only).
- With CUDA the staging buffers are pinned host memory, so the uploads to the GPU run at full PCIe speed.
- The cache keeps experts by recency and frequency (`src/llama-moe-cache.cpp`). In `inkling2_loop` models the passes of the loop core share the slots of their physical layer, so an expert used in several passes is cached once.
- Layers whose experts have the same types and shapes share one group of slots. When a file gives the experts of some layers other types (common in community quants), each layout is its own group. The budget is then split in two steps: first every group gets the slots for the largest ubatch `U` (`U * n_expert_used` slots, at most all its experts) that fits in half the budget, then the rest is split by the size of each group's experts. The startup log shows the slots of each group and `U`. A file with one layout gets the whole budget, as before. The smallest working cache is the one that gives every group `n_expert_used` slots; a smaller `-lhc` fails with an error that names it.
- Prompt processing also reads through the cache. If the cache cannot hold the experts that one ubatch selects in a layer, `n_ubatch` is reduced at startup (logged as a warning). A larger cache gives larger ubatches and faster prompt processing.
- A Metal or CUDA GPU is required, with all layers on it (the default `-ngl`). LongHaul supports one GPU: on a machine with several CUDA GPUs, select one with `-dev CUDA0` (or `-sm none -mg N`).
- `--fit` reduces the context size if needed, but it does not offload fewer layers or move experts to the CPU. If the model does not fit, use a smaller `-lhc` or context.

The budget covers the expert cache only. The resident weights, the KV cache, the compute buffers and the staging buffers (32 MiB) come on top.

## CUDA

On a discrete GPU the expert cache, the resident weights and the KV cache are in VRAM; the experts on disk use no RAM. A GPU with enough VRAM can hold every expert: set `-lhc` to the size of the experts (about 16 GiB for Mini Q4_K) and after the first reads every expert is a hit. Below that, the hit rate and the SSD speed set the generation speed as on a Mac.

## Sizing for Inkling 2 Mini

Inkling 2 Mini has 24 layers with 128 experts each, 8 used per token. The 12 core layers run 3 times, so a token makes 48 layer passes and up to 384 expert reads. In `Inkling-2-Mini-AD-Q4_K.gguf` the experts are about 94% of the 16.5 GB file, about 5 MB per expert. These are estimates from the model card, not measurements:

| Cache | Slots (about) | Max ubatch | Note |
|---|---|---|---|
| 512M | 100 | 12 | default |
| 1G | 200 | any | holds the 128 experts of a layer, no ubatch limit |
| 2G | 400 | any | |
| 4G | 800 | any | |

The slots are shared by all layers. Files that give the experts of some layers another type split the budget into groups, see above.

Without any hits a token reads about 384 x 5 MB, about 1.9 GB, from the SSD. Generation speed is therefore bound by the SSD speed and the hit rate. The resident part of the model is about 1 GB.

To estimate for another file or cache size, with `E` = bytes per expert (gate_up + down), `L` = MoE layer passes per token, `k` = experts per token and `h` = hit rate:

- disk read per generated token: about `L * k * (1 - h) * E`
- generated tokens per second: at most `SSD read speed / (L * k * (1 - h) * E)`
- cache slots: about `cache size / E`
- the cache holds all experts of a layer, with no `n_ubatch` limit, from `n_expert` slots (128 for Mini)

For Mini (`L` = 48, `k` = 8, `E` ~ 5 MB) at a hit rate of 50%, a token reads about 0.96 GB. The hit rate depends on the prompt and the routing, so measure it with the stats line below or replay a trace.

## Sizing for Ling-3.2-air

Ling-3.2-air (arch `bailingmoe3`) has 42 layers: layers 0-1 have a dense FFN, layers 2-41 have 512 experts each, 8 used per token, plus a shared expert. Its attention is hybrid: 35 KDA (linear attention) layers and 7 MLA layers. The file also has an MTP layer (block 42), which this fork does not load. Only text input is supported.

The numbers below are computed from the tensor types in the header of bartowski's `Ling-3.2-air-Q4_K_M` (74.9 GiB), not measured. The experts of layers 2-41 are 69.5 GiB, about 3.2 MiB each at Q4_K. The resident part (attention, shared experts, dense FFN, router, embeddings and output) is about 3.7 GiB, and the MTP layer (1.6 GiB) stays on disk. Without any hits a token reads about 40 x 8 x 3.2 MiB, about 1 GiB.

That file has 5 expert layouts (Q4_K/Q6_K/Q6_K in layers 2-10, Q5_K or Q6_K down in 3 layers, Q8_0 in layer 41, Q4_K in the other 27), so the budget is split as described above. Its smallest working cache is about 163 MiB. A file with all experts in one type (Q4_K below) gets larger ubatches from the same cache:

| Cache | Q4_K_M: slots | Q4_K_M: max ubatch | All Q4_K: slots | All Q4_K: max ubatch |
|---|---|---|---|---|
| 512M | 138 | 1 | 161 | 20 |
| 1G | 271 | 3 | 323 | 40 |
| 2G | 546 | 6 | 647 | any |
| 4G | 1094 | 13 | 1294 | any |

With one layout the ubatch is not limited once the cache holds the 512 experts of a layer. The ubatch limit slows prompt processing only. With 512 experts per layer the hit rate at these sizes is low, so a cache of 2-4 GiB is a better start than the default 512M. All experts at Q4_K are about 63 GiB.

To give every expert one type, quantize from a BF16 or F16 GGUF with `llama-quantize --tensor-type 'ffn_(gate|up|down)_exps=q4_k'` (see [tools/quantize](../tools/quantize/README.md)).

## Reading the stats

At exit the cache logs one line per batch class:

```
llama_moe_cache: ubatch <= 8: hits = ..., misses = ..., hit rate = ...%, read = ... MiB in ... s (... MiB/s)
```

The time and speed are for the disk reads only. If the read time is most of the run time, a larger cache helps. To compare I/O thread counts, run `llama-bench` with `LLAMA_LONGHAUL_IO_THREADS` set to 1, 2, 4 and 8 and compare the MiB/s.

`LLAMA_MOE_CACHE_TRACE=<file>` writes the selected experts of each lookup to a file, to replay the routing offline with other cache sizes.

## Tests

`LLAMA_LONGHAUL_RESIDENT=1` with a cache size of 0 keeps the experts in the mapped file without a RAM bound. It exists for tests that compare against the old behavior. `test-llama-archs` checks LongHaul against the same model in memory for caches from 2 slots to all experts, for Inkling 2 and for `bailingmoe3` (including files with several expert layouts, split files and a file with an MTP layer).
