# LongHaul

LongHaul runs Inkling 2, Qwen 3.5 / 3.6 MoE (`qwen35moe`) and Ling-3.2-air (`bailingmoe3`) MoE models on a Mac or an NVIDIA GPU with less memory than the model needs. The MoE experts stay in the GGUF file on disk. A fixed-size cache in GPU memory (unified memory on a Mac, VRAM with CUDA) keeps the experts that were used recently and often. Everything else (attention, router, norms, embeddings, output) is loaded as usual.

LongHaul is always on for models with experts on disk. These options control it:

| Argument | |
|---|---|
| `-lhc, --longhaul-cache SIZE` | cache size: a number in MiB, or with a K, M or G suffix, e.g. `512M`, `1G`, `2G` (default: `512M`). With several devices, the total of the devices that `-lhcd` does not name. |
| `-lhcd, --longhaul-cache-device DEV=SIZE,...` | cache size of a device, e.g. `RPC0=12G,MTL0=4G` |
| `-es, --expert-share SHARE` | split the experts of every MoE layer over the devices, see [Expert split](#expert-split) |
| `-kvd, --kv-device DEV` | with `-es`: the device of the layers and the KV cache (default: the local GPU) |

`DEV` is a device name as `--list-devices` prints it (`MTL0`, `RPC0`, ...), a unique prefix of it, or the address of an RPC server (`10.0.0.2:50052`). The same options can be set with the env vars `LLAMA_ARG_LONGHAUL_CACHE`, `LLAMA_ARG_LONGHAUL_CACHE_DEVICE`, `LLAMA_ARG_EXPERT_SHARE` and `LLAMA_ARG_KV_DEVICE`.

`llama-bench` takes `-lhc` as a list (`-lhc 512M,1G,2G`) to compare sizes. It also takes `-sm expert` with `-ts` and `-mg` in place of `-es` and `-kvd`, and `-lhcd 12G/4G` (sizes in the order of the devices, 0 = from `-lhc`).

## How it works

- The expert tensors are memory mapped but never read through the mapping, so they use address space, not RAM.
- Each MoE layer looks up its selected experts in the cache. A miss is read from the file with `pread`, so that the OS file cache does not keep a second copy: on macOS the file is opened with `F_NOCACHE`, on Linux each read is dropped from the page cache with `POSIX_FADV_DONTNEED` (and readahead is off). On Windows the reads go through the file cache. The misses of a layer are read by 4 threads at once, each through its own 8 MiB staging buffer, into the cache, which is a Metal or CUDA buffer. `LLAMA_LONGHAUL_IO_THREADS` sets the number of threads (1 to 64; 1 reads on the compute thread only).
- With CUDA the staging buffers are pinned host memory, so the uploads to the GPU run at full PCIe speed.
- The cache keeps experts by recency and frequency (`src/llama-moe-cache.cpp`). In `inkling2_loop` models the passes of the loop core share the slots of their physical layer, so an expert used in several passes is cached once.
- Layers whose experts have the same types and shapes share one group of slots. When a file gives the experts of some layers other types (common in community quants), each layout is its own group. The budget is then split in two steps: first every group gets the slots for the largest ubatch `U` (`U * n_expert_used` slots, at most all its experts) that fits in half the budget, then the rest is split by the size of each group's experts. The startup log shows the slots of each group and `U`. A file with one layout gets the whole budget, as before. The smallest working cache is the one that gives every group `n_expert_used` slots; a smaller `-lhc` fails with an error that names it.
- The shared experts of Qwen and Ling are small and stay in memory. The experts of a Qwen MTP block or a Ling MTP layer are not loaded, and the block is not run.
- Prompt processing also reads through the cache. If the cache cannot hold the experts that one ubatch selects in a layer, `n_ubatch` is reduced at startup (logged as a warning). A larger cache gives larger ubatches and faster prompt processing.
- A Metal or CUDA GPU is required, with all layers on GPUs (the default `-ngl`). The layers can be split over several GPUs, or over several laptops with RPC (see [Several laptops](#several-laptops)): each device caches the experts of its own layers in its own memory.
- `--fit` reduces the context size if needed, but it does not offload fewer layers or move experts to the CPU. If the model does not fit, use a smaller `-lhc` or context.

The budget covers the expert cache only. The resident weights, the KV cache, the compute buffers and the staging buffers (32 MiB) come on top.

## CUDA

On a discrete GPU the expert cache, the resident weights and the KV cache are in VRAM; the experts on disk use no RAM. A GPU with enough VRAM can hold every expert: set `-lhc` to the size of the experts (about 16 GiB for Mini Q4_K) and after the first reads every expert is a hit. Below that, the hit rate and the SSD speed set the generation speed as on a Mac.

## Several laptops

The layers of one model can run on several Macs connected with Thunderbolt cables. The main laptop runs `llama-cli` or `llama-server` with `--rpc`. Each other laptop runs `ggml-rpc-server` (see [tools/rpc](../tools/rpc/README.md#this-fork-several-macs-over-thunderbolt) for the cable and network setup). All laptops must run a build of the same commit.

- Each device (the GPU of a laptop) holds a contiguous slice of the layers and caches the experts of those layers in its own memory, so the memory of all laptops caches experts.
- `-lhc` is the total for all devices. It is split across the devices like the layers: by `-ts` if given, otherwise by free memory. `-lhcd RPC0=12G` gives a device its own size, and `-lhc` is then the total of the other devices.
- The RPC devices come first and the main laptop's GPU last, so the output layer and sampling run on the main laptop.
- In `inkling2_loop` models `-ngl` and `-ts` count the layers that hold weights (25 for Mini: 24 layers and the output), not the 49 layer passes. Each pass of the loop core runs on the device that holds its layer, so a token crosses the link several times (6 times for Mini on two laptops with an even split).
- Pipeline parallelism stays off: the cache needs the experts that a layer selects before the layer runs.
- A remote MoE layer costs about one round trip on the link even when every expert hits: the main laptop reads the selected expert ids before it fills the cache.

A miss of a remote layer is read in one of two ways:

- **By the remote laptop**, when its `ggml-rpc-server` was started with `--model-dir DIR` and DIR holds the same GGUF file (the same size, the same first 4 MiB and the same 64 blocks of 4 KiB at evenly spaced offsets). Only activations, expert ids and slot maps cross the link. The startup log of the cache ends with `read by the server`.
- **By the main laptop**, which reads the experts from its own file and sends them over the link. This needs no setup on the remote laptop, but the link is likely slower than the SSD. The main laptop logs `RPC0[...] has no copy of <file> (see ggml-rpc-server --model-dir), the experts are sent from this host`.

The file check is not a full hash: two files that differ only outside the sampled blocks would give wrong output. Copy the GGUF file to the other laptops as is, do not re-quantize it there.

Keep a draft model (`-md`) on the main laptop with `-devd MTL0`: a draft on a remote laptop would pay a round trip for every draft token.

```sh
# on the remote laptop
ggml-rpc-server -H 10.0.0.2 -p 50052 --model-dir ~/models

# on the main laptop
llama-cli -m ~/models/Inkling-2-Mini-AD-Q4_K.gguf --rpc 10.0.0.2:50052 -lhc 16G
```

### Inkling 2 Mini on two laptops

With an even split, the remote laptop (first) holds layers 0-11 and the main laptop holds layers 12-23 and the output. A token runs the prefix 0-5 and the core 6-11 on the remote laptop, the core 12-17 on the main laptop, then passes 2 and 3 go remote, main, remote, main, then the suffix 18-23 runs on the main laptop. Each laptop runs 24 of the 48 layer passes.

The experts of Mini Q4_K are about 15.5 GB, so `-lhc 16G` split over two laptops (about 8 GiB each) holds every expert: after the first reads no expert is read from disk, and the round trips on the link set the speed. With a smaller cache each laptop reads about half of the misses. These numbers are computed from the model card, not measured: measure the speed of your own link and SSDs.

## Expert split

With `-es` the laptops do not take turns on the layers. Every MoE layer runs on all of them at the same time, each laptop on its own part of the experts:

```sh
# the remote laptop computes about 60% of the routed experts, the main laptop 40%
llama-cli -m ~/models/Inkling-2-Mini-AD-Q4_K.gguf --rpc 10.0.0.2:50052 -es 60 -lhc 16G
```

- The layers (attention, router, norms, shared experts, dense FFN layers) and the KV cache are on one device, the main laptop's GPU by default. The output layer is on the last device, which is the main laptop's GPU.
- The routed experts of every MoE layer are split over the devices: each device owns a fixed set of experts of each layer, about its share of them, spread over the expert ids. A device caches and reads only the experts it owns, so the caches of all laptops add up with no copies.
- For each MoE layer, each device computes the selected experts that it owns for all tokens of the batch, and the main laptop adds the results. The remote laptop reads its misses from its own SSD (with `--model-dir`) while the main laptop reads its own misses and computes. This applies to generation and to prompt processing.
- At the start each device owns its share of the experts. Routing is not even, so the share of the selected experts that a device computes can differ from its target. While the model runs, the cache counts how often each expert is selected (old counts fade after about 4096 selections per layer) and, between two graphs, moves experts from the device furthest above its target to the device furthest below, until each device is within 2 points of its target. A layer is checked after every 512 selections, with at most 2 moves per check. A moved expert is dropped from the cache of its old device and read by the new device at its next use. A device never owns more experts than its cache has slots, or than it owned at the start if that is more; at that limit, two experts swap devices instead. The exit log shows the target and the measured share (see [Reading the stats](#reading-the-stats)). `LLAMA_EXPERT_SHARE_REBALANCE=0` keeps the experts where they started.
- The share is of the routed experts only. The attention, the router and the shared experts run on the device of the KV cache. For Inkling 2 Mini the routed experts are about two thirds of the weights that a token reads (1.9 GB of experts against about 1 GB resident); for Qwen 3.5 / 3.6 and Ling-3.2-air they are a smaller part, about a third to a half (estimates from the sizes below, not measured).

The forms of `-es`:

| `-es` | |
|---|---|
| `60` | the RPC devices together get 60% (split evenly if there are several), the local GPU 40% |
| `RPC0=60,MTL0=40` | per device; the devices not named split the rest evenly (`RPC0=60` alone is the same as `60` with one server) |
| `60,40` | per device, in the order of the devices (the RPC devices come first, or the order of `-dev`) |

The shares must not add up to more than 100%, and to exactly 100% when every device has one. A share of 0 is allowed: that device then computes no experts. `-es` cannot be given with `-ts` or `-sm`, since all of them set the split. The startup log prints the result, e.g. `expert split: RPC0 60.0%, MTL0 40.0%, layers and KV cache on MTL0`.

Cache sizes: `-lhc` is split over the devices by share, so with `-es 60 -lhc 10G` the remote gets 6 GiB and the main laptop 4 GiB. A device caches only its own experts, so it needs less memory to hold all of them: for Mini at 60/40, about 9.3 GB on the remote and 6.2 GB on the main laptop hold every expert. To give a device more or less memory than its share, name it with `-lhcd`, e.g. `-es 50 -lhcd RPC0=12G,MTL0=4G`.

`-kvd RPC0` puts the layers and the KV cache on the remote laptop instead, for a context that does not fit in the main laptop's memory. This is slower: the main laptop's part of each MoE layer then needs the activations from the remote, which costs about two round trips per MoE layer pass instead of one.

Limits:

- Only for models with experts on disk (LongHaul), with all layers on the GPUs (the default `-ngl`) and no `-ot`.
- Only Metal and RPC devices, and the RPC servers must run on Macs. CUDA does not skip the experts that another device owns, so it is not supported, also not behind an RPC server (which this host cannot detect: the output would be wrong).
- A draft model (`-md`) does not use the expert split. Without `-devd`, it goes on the device of the KV cache.

### Layer split or expert split

Neither always wins, so measure both on your laptops:

- The expert split wins when the expert reads or the expert compute set the speed: both SSDs and both GPUs work at the same time instead of in turns.
- The layer split can win when every expert hits and the link latency sets the speed. For Inkling 2 Mini the expert split costs about one round trip per MoE layer pass (48 per token), the layer split with an even split about 24 (one per remote MoE layer pass) plus 6 hops.
- The share applies on average. One token can select most of its 8 experts on one laptop, and each layer waits for the slower laptop, so find the best share by measuring, not from the GPU speeds.

To compare, run `llama-bench` with the layer split and with a sweep of the expert share. The RPC device comes first, so the first number of `-ts` is the remote's share:

```sh
llama-bench -m M --rpc 10.0.0.2:50052 -lhc 16G -sm layer,expert -ts 40/60,50/50,60/40
```

### The tokenizer

The tokenizer is not split. It runs once per prompt on the CPU of the main laptop at millions of tokens per second (a 100k-token prompt takes tens of milliseconds), and the token ids are needed there. A round trip to the remote laptop takes longer than that. To check on your own prompts: `time llama-tokenize -m MODEL -f prompt.txt --show-count`. The prompt processing (the model work on the prompt) is split, as above.

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

## Sizing for Qwen 3.5 / 3.6

The MoE models have 256 or 512 experts per layer and a shared expert. All layers are MoE layers, including the linear attention ones. The 35B-A3B numbers are from the GGUF headers of the two common Q4_K_M files of Qwen3.6-35B-A3B, the others are estimates for Q4_K experts from the model configs:

| Model | `L` | `k` | `E` | Experts on disk | No-hit read per token |
|---|---|---|---|---|---|
| 35B-A3B (3.5 and 3.6), ggml-org Q4_K_M | 40 | 8 | 1.77 MB | 16.9 GiB | 0.57 GB |
| 35B-A3B (3.5 and 3.6), unsloth UD-Q4_K_M | 40 | 8 | 1.90 MB, 2.04 MB in 3 layers | 18.2 GiB | 0.61 GB |
| 122B-A10B, Q4_K | 48 | 8 | 5.3 MB | 61 GiB | 2.0 GB |
| 397B-A17B, Q4_K | 60 | 10 | 7.1 MB | 202 GiB | 4.3 GB |

The ubatch limit by cache size (the cache holds all experts of a layer from 256 slots, 512 for the 397B):

| Model | 512M | 1G | 2G | 4G |
|---|---|---|---|---|
| 35B-A3B, ggml-org Q4_K_M (300 slots at 512M) | any | any | any | any |
| 35B-A3B, unsloth UD-Q4_K_M | 9 | 18 | any | any |
| 122B-A10B | 12 | 25 | any | any |
| 397B-A17B | 7 | 15 | 30 | any |

The unsloth file gives `ffn_down_exps` Q6_K in 3 of its 40 layers and Q5_K in the others, so the cache builds two groups. Both groups first get the slots of the same ubatch, then the rest is split by the size of the groups, so at 512M the 3-layer group gets about 75 slots and the other about 200, and the 3-layer group sets the ubatch limit. Prefer a file with one expert type in all layers, such as the ggml-org Q4_K_M, or give the cache more memory.

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
llama_moe_cache: ubatch <= 8: hits = ..., misses = ..., hit rate = ...%, read = ... MiB, local ... MiB in ... s (... MiB/s)
```

`read` counts all misses, `local` the ones this host read from its own file (all of them without RPC servers that read for themselves). The time and speed are for the local reads only. If the read time is most of the run time, a larger cache helps. To compare I/O thread counts, run `llama-bench` with `LLAMA_LONGHAUL_IO_THREADS` set to 1, 2, 4 and 8 and compare the MiB/s.

With the expert split it also logs one line per device:

```
llama_moe_cache: RPC0: share target 60.0%, measured ...%, ... of 128 experts per layer, hits = ..., misses = ..., hit rate = ...%, read = ... MiB (by the server), moved in ..., out ...
```

`measured` is the share of the selected experts that the device computed over the whole run, so it includes the time before the moves. The experts per layer are the ones it owns at exit, on average over the layers. `moved in` and `out` count the experts the rebalance moved to and from the device. Many moves on a long run mean that the routing changes a lot; each move costs a read on the new device.

`LLAMA_MOE_CACHE_TRACE=<file>` writes the selected experts of each lookup to a file, to replay the routing offline with other cache sizes.

## Tests

`LLAMA_LONGHAUL_RESIDENT=1` with a cache size of 0 keeps the experts in the mapped file without a RAM bound. It exists for tests that compare against the old behavior. `test-llama-archs` checks LongHaul against the same file with its experts in the mapped file, or against the model in memory, for caches from 2 slots to all experts. For `qwen35moe` it also checks files with separate gate and up experts, with two expert groups, and with an MTP block. For `bailingmoe3` it checks files with several expert layouts, split files and a file with an MTP layer. With `GGML_METAL_DEVICES=2` (two emulated Metal devices) it also checks every LongHaul arch with its layers split over the two devices, and with the expert split at several shares, with the KV cache on either device, with cache sizes per device and with the rebalance moving experts at every graph. `test-expert-share` checks the expert ownership and the rebalance rule, and `test-arg-expert-split` checks how `-es`, `-kvd` and `-lhcd` are read. `test-llama-archs --rpc` (CTest label `rpc`) starts an RPC server on a thread of the test, on `127.0.0.1` only, and checks the experts read by the server and the experts sent from the client against the local GPU.
