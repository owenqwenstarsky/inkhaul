# LongHaul

LongHaul runs Inkling 2 MoE models on a Mac with less RAM than the model needs. The MoE experts stay in the GGUF file on disk. A fixed-size cache in RAM keeps the experts that were used recently and often. Everything else (attention, router, norms, embeddings, output) is loaded as usual.

LongHaul is always on for models with experts on disk. Only the cache size can be set:

| Argument | |
|---|---|
| `-lhc, --longhaul-cache SIZE` | cache size: a number in MiB, or with a K, M or G suffix, e.g. `512M`, `1G`, `2G` (default: `512M`) |

`llama-bench` takes the same option as a list (`-lhc 512M,1G,2G`) to compare sizes.

## How it works

- The expert tensors are memory mapped but never read through the mapping, so they use address space, not RAM.
- Each MoE layer looks up its selected experts in the cache. A miss is read from the file with `pread` and `F_NOCACHE`, so the macOS file cache does not keep a second copy. The misses of a layer are read by 4 threads at once, each through its own 8 MiB staging buffer, into the cache, which is a Metal buffer. `LLAMA_LONGHAUL_IO_THREADS` sets the number of threads (1 to 64; 1 reads on the compute thread only).
- The cache keeps experts by recency and frequency (`src/llama-moe-cache.cpp`). In `inkling2_loop` models the passes of the loop core share the slots of their physical layer, so an expert used in several passes is cached once.
- Prompt processing also reads through the cache. If the cache cannot hold the experts that one ubatch selects in a layer, `n_ubatch` is reduced at startup (logged as a warning). A larger cache gives larger ubatches and faster prompt processing.
- A Metal GPU is required, with all layers on it (the default `-ngl`).

The budget covers the expert cache only. The resident weights, the KV cache and the compute buffers come on top.

## Sizing for Inkling 2 Mini

Inkling 2 Mini has 24 layers with 128 experts each, 8 used per token. The 12 core layers run 3 times, so a token makes 48 layer passes and up to 384 expert reads. In `Inkling-2-Mini-AD-Q4_K.gguf` the experts are about 94% of the 16.5 GB file, about 5 MB per expert. These are estimates from the model card, not measurements:

| Cache | Slots (about) | Max ubatch | Note |
|---|---|---|---|
| 512M | 100 | 12 | default |
| 1G | 200 | any | holds the 128 experts of a layer, no ubatch limit |
| 2G | 400 | any | |
| 4G | 800 | any | |

The slots are shared by all layers. Files that give the experts of some layers another type split the budget into groups by size, with fewer slots each.

Without any hits a token reads about 384 x 5 MB, about 1.9 GB, from the SSD. Generation speed is therefore bound by the SSD speed and the hit rate. The resident part of the model is about 1 GB.

To estimate for another file or cache size, with `E` = bytes per expert (gate_up + down), `L` = MoE layer passes per token, `k` = experts per token and `h` = hit rate:

- disk read per generated token: about `L * k * (1 - h) * E`
- generated tokens per second: at most `SSD read speed / (L * k * (1 - h) * E)`
- cache slots: about `cache size / E`
- the cache holds all experts of a layer, with no `n_ubatch` limit, from `n_expert` slots (128 for Mini)

For Mini (`L` = 48, `k` = 8, `E` ~ 5 MB) at a hit rate of 50%, a token reads about 0.96 GB. The hit rate depends on the prompt and the routing, so measure it with the stats line below or replay a trace.

## Reading the stats

At exit the cache logs one line per batch class:

```
llama_moe_cache: ubatch <= 8: hits = ..., misses = ..., hit rate = ...%, read = ... MiB in ... s (... MiB/s)
```

The time and speed are for the disk reads only. If the read time is most of the run time, a larger cache helps. To compare I/O thread counts, run `llama-bench` with `LLAMA_LONGHAUL_IO_THREADS` set to 1, 2, 4 and 8 and compare the MiB/s.

`LLAMA_MOE_CACHE_TRACE=<file>` writes the selected experts of each lookup to a file, to replay the routing offline with other cache sizes.

## Tests

`LLAMA_LONGHAUL_RESIDENT=1` with a cache size of 0 keeps the experts in the mapped file without a RAM bound. It exists for tests that compare against the old behavior. `test-llama-archs` checks LongHaul against the same model in memory for caches from 2 slots to all experts.
