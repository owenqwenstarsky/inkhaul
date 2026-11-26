# Plan: expert split over several laptops

Status: phases 1 to 3 committed, phases 4 and 5 open. Builds on THUNDERBOLT_PLAN.md (commit e9e55df).

## 1. Goal

- Set from `llama-cli` how much of the activated expert work each laptop does, as a percentage: `-es 60` makes the remote laptop compute about 60% of the routed experts of every token, the main laptop the other 40%.
- Both laptops work at the same time on every MoE layer, in generation and in prompt processing, and both SSDs read missing experts at the same time.
- Pin the KV cache (and the attention that reads it) to the main laptop (default) or to the remote laptop.
- Set the memory each laptop gives to the expert cache separately from its share of the work.

Answers that shaped this plan: the load split is an expert split within each layer (not a layer split), "tokenization on both laptops" means prompt processing plus the tokenizer itself, the pinned cache is the KV cache, and memory is set apart from compute.

## 2. Current state

- `-ts` splits the layers over the devices (`llama_model_base::load_tensors`, `src/llama-model.cpp:683-745`). Each device holds a contiguous run of layers, their KV cache (`src/llama-kv-cache.cpp:217` uses `model.dev_layer(il)`) and the LongHaul cache banks of their experts.
- The laptops take turns: the scheduler runs the splits in graph order, so the remote runs its layers while the main laptop waits, and the other way around.
- `llama_moe_cache` (`src/llama-moe-cache.cpp`) gives each device a budget (`-lhc` split by `-ts` or free memory, line 552), groups per (device, expert layout), one LRU and one slot map per owner layer. The lookup `get_rows(slot_map, ids)` runs on the device of the layer (`build_lookup`, line 1083). The copy callback reads the selected ids (one sync), plans fills, then either asks the RPC server to read from its copy of the file (`rpc_set_from_file`) or reads locally and uploads.
- Pipeline parallelism is off with LongHaul (`src/llama-context.cpp:428`).
- `ggml_mul_mat_id` already skips an id of -1 and zeroes that output row (`ggml/include/ggml.h:1331`). Verified in the CPU kernel (`ggml-cpu.c:1694`), the Metal matvec kernel (`mul_mv.metal`, zeroes the row) and the Metal matmul path (`ggml-metal-ops.cpp:3053`, a zero kernel for rows no expert owns). `test-backend-ops` covers it. CUDA does not implement it (TAG_MUL_MAT_ID_SKIP).
- RPC graph compute is asynchronous and the RPC device reports async and events (`ggml-rpc.cpp:1516`, `:3093`), and the server runs commands in order on one stream.
- The LongHaul archs (inkling2, inkling2_loop, qwen35moe, bailingmoe3) have no expert output biases and no per-expert scales, only the router bias `exp_probs_b`.

## 3. Design

### 3.1 New split mode

Add `LLAMA_SPLIT_MODE_EXPERT = 4` to `enum llama_split_mode` (appended, the ABI of the other values does not change):

- The home device holds every layer's non-expert weights (attention, norms, router, shared experts, dense FFN layers) and the KV cache. The home device is `main_gpu`.
- The routed experts of every MoE layer are split over all devices by `tensor_split`, read as the share of the activated experts each device computes. A share of 0 is allowed (that device then holds no experts). A device with share > 0 gets cache banks for every MoE layer.
- The output layer stays on the last device (the main laptop's GPU, as today), so sampling stays local.
- The input layer stays on the CPU, as today.

`llama_model_load` keeps all devices in this mode (unlike `NONE`, it does not drop the others). Loading fails with a clear error when:
- the model has no LongHaul experts (resident experts would need sliced expert tensors and could not be rebalanced; out of scope),
- a MoE layer has expert biases or per-expert scales (none of the LongHaul archs do),
- a device with share > 0 is not Metal, CPU or RPC (CUDA does not skip -1 ids),
- `-ngl` does not cover all layers, or tensor overrides (`-ot`) are given.

`tensor_split` and `main_gpu` already exist in `llama_model_params`, so no new model fields are needed.

### 3.2 Expert ownership

Each owner layer (loop passes share the owner's experts) has an owner table `owner[e]` = device, kept in host memory by the cache.

- Initial table: a weighted interleave (Bresenham style over expert ids) so that each device owns `round(share * n_expert)` experts, spread over the id range. Deterministic, no seed.
- A device caches only the experts it owns, so the expert cache of all laptops adds up with no duplicates.
- Ownership by count matches the activated share only if routing is uniform, which it is not. Section 3.6 rebalances the table toward the target.

Rejected alternative: assign each token's selected experts to devices per token (exact share, any device can compute any expert). Both caches would then hold the hot experts, so the total cache holds fewer distinct experts and both SSDs read the same experts. With LongHaul, where disk reads set the speed, static ownership is the better trade.

### 3.3 Graph: one MoE layer

`build_moe_ffn` (`src/llama-graph.cpp:~1796`) in expert mode, for layer `il` with home `H` and the expert devices `D0..Dn`:

1. On `H`: router, top-k ids, weights (unchanged).
2. For each device `d`, in the order "devices other than `H` first, `H` last":
   - `slots_d = get_rows(slot_map_d, ids)`, forced to run on `H` (`ggml_backend_sched_set_tensor_backend`). The slot map of `d` holds the slot of each cached expert owned by `d` and -1 for the experts `d` does not own.
   - `experts_d = build_experts(cur, ids, slots_d)` against the banks of `d`. The matmuls go to `d` because the banks are in its buffer. Rows of experts that `d` does not own come out as zeros.
   - `partial_d = sum over k of experts_d * weights`, forced onto `d` with `ggml_backend_sched_set_tensor_backend`, so only `[n_embd, n_tokens]` crosses the link and not `[n_embd, k, n_tokens]`.
3. On `H`: `moe_out = sum of partial_d`.

What crosses the link per remote device per layer: the expert input and the slots and weights to the remote (`[n_embd, n_tokens]` f32, plus `2 * k * n_tokens` 32-bit values), and the partial back (`[n_embd, n_tokens]` f32). The graph topology does not depend on the ownership: an ownership change only changes slot map data.

`llama_moe_cache::build_lookup` returns one entry per device (`slots`, plus a per-device `get_experts(w, d)`). `build_lora_mm_id` takes the device-specific bank. `check_op` checks each device.

Why this order overlaps work (home = main laptop):
- The remote's lookup split runs on `H`. Its copy callback reads the ids from `H` (a local sync, no link round trip), plans the remote fills, and queues the remote's reads (`rpc_set_from_file`, which the server runs in order before the remote's compute).
- The scheduler then submits the remote experts split (asynchronous over RPC).
- The local lookup's callback then reads the local misses from the local SSD while the remote reads its own SSD and computes.
- The local experts split runs on `H`, then the final add waits for the remote partial (one blocking read over the link).

Per MoE layer pass that is about one round trip on the link (the read of the remote partial), with both GPUs and both SSDs busy at the same time.

### 3.4 Copy callback changes (`llama_moe_cache::impl::copy`)

- The lookup backend is `H`, not the cache device. Replace `GGML_ASSERT(backend == devices[g.id].backend)` with: the backend is the lookup backend, and the cache device is `devices[g.id]`. `rpc_set_from_file` gets the cache device, not `ggml_backend_get_device(backend)`.
- Ids cache: the ids of a layer are read once per graph compute and reused by the other devices' callbacks for that layer. Key: (layer, epoch), where `begin_graph()` increments the epoch before each graph compute. This matters when `H` is the remote laptop (section 3.5): without it each device would pay its own round trip.
- Filter: the plan of device `d` sees only the ids that `d` owns (the others are skipped, not given slots). The copy writes a separate host buffer: `out[e] = owner[e] == d ? slot_map[e] : -1`. The LRU internals do not change.
- Sync: before host-side refills of device `d`'s banks, synchronize `d`'s backend (today the code synchronizes the lookup backend, which was the same device). RPC server reads need no sync, since the server orders them after earlier computes.
- `max_ubatch()` is the minimum over the devices, with each device's need capped at its owned-expert cap (section 3.6).

### 3.5 KV cache pinning

The KV cache sits on the home device, and the attention runs where the KV cache is (moving the KV cache over the link every token would cost far more than the activations). Pinning the KV cache therefore moves the home: `--kv-device RPC0` sets `main_gpu` to the remote, so the remote holds the attention, the router, the shared experts and the KV cache of every layer. The main laptop then computes its expert share and the output layer.

- Default: the main laptop's GPU (the first non-RPC GPU, resolved in `common`).
- Ordering rule (3.3): the devices other than `H` go first, so with `H` = remote the main laptop's experts go first and the remote's part last.
- Cost with `H` = remote: about two round trips per MoE layer pass instead of one (one to read the ids, one to fetch the expert input for the main laptop's part), and less overlap. It is for long contexts that do not fit in the main laptop's memory, not for speed. The docs will say so.
- `--kv-device` needs `-es` (expert mode). In layer mode the KV cache already follows the layers (`-ts`), and the flag errors out with a pointer to `-ts`.

### 3.6 Rebalancing toward the activated share

Rebalancing moves the owner table so that the measured share of activated experts approaches the target.

- Counts: per (owner layer, expert) a u32 activation count, updated in the copy callback from the ids it already reads. Each layer halves its counts after every 4096 selections, so old routing fades out.
- When: in `begin_graph()`, never in the middle of a graph, so all devices plan a layer from the same table. A layer is checked after at least 512 new selections.
- Rule: compute each device's share of the counted activations. If a device is more than 2 percentage points off its target, move experts from the most overloaded device to the most underloaded one: pick the expert whose count fraction is closest to `min(excess, deficit)` without exceeding 1.5 times it, preferring experts not currently cached on the old owner. At most 2 moves per layer per check.
- Move: the old owner drops the expert's slot (a new `moe_cache_lru::drop(owner_layer, expert)` that frees the slot), and the new owner reads it on its next use.
- Cap: a device's owned count never exceeds `max(initial owned, n_slots)` of its group, so the `max_ubatch` computed at startup stays valid.
- Deterministic (no randomness). `LLAMA_EXPERT_SHARE_REBALANCE=0` turns it off (tests, comparisons).
- The rebalance rule is a pure function in `src/llama-expert-share.{h,cpp}`: `(counts, owner, targets, caps, max_moves) -> moves`, unit tested on its own.

The share is of the routed experts only. The attention, the router and the shared experts run on the home device. The docs will say so, with the rough fraction of the active parameters that the routed experts make up for each LongHaul model.

### 3.7 Expert cache sizes per device

New context param `const size_t * longhaul_cache_sizes` (borrowed, `llama_max_devices()` entries, 0 = not named), next to `longhaul_cache_size`.

- Named devices get exactly their size.
- `longhaul_cache_size` (`-lhc`) is the total of the devices not named, split among them by share in expert mode and by `-ts` or free memory in layer mode (as today). With nothing named this is today's behavior.
- Works in both modes.
- The startup log prints each device's budget, owned experts and slots. A warning is printed when a device's cache holds less than its owned experts of one layer (the ubatch limit).

### 3.8 Stats

The exit log adds a line per device:

```
llama_moe_cache: RPC0[10.0.0.2:50052]: share target 60%, measured 58.7%, owned 74/128 per layer (avg), hits ..., misses ..., read ... MiB (by the server), moves ...
llama_moe_cache: MTL0: share target 40%, measured 41.3%, ...
```

To tune the share, compare runs with `llama-bench` (section 4.2). A per-token "main laptop waited for the remote" time would need timing inside the scheduler that breaks its async path, so it is left out.

### 3.9 Concurrency in prompt processing

- The expert split already runs both laptops at the same time in prompt processing: per layer, both compute their expert share of the whole ubatch at once, and both SSDs read their misses at once. The local reads overlap the remote's compute (ordering in 3.3).
- Phase 5 (experimental): two-batch overlap. Split a ubatch into halves A and B in one graph and order the nodes per layer as `attn A`, `experts A (remote queued)`, `attn B`, `experts A (local)`, `combine A`, `experts B (remote queued)`, ... so the home device's attention for B runs while the remote computes the experts of A. Safe because B's attention at layer l only needs A's KV at layer l, which A writes before its experts. This touches `llama_context` (two ubatch slot infos per graph), the KV and recurrent memory (state written by A and read by B through the same buffer, so node order must enforce it) and the scheduler's node order. It only pays if attention is a large part of the prefill time. Gate: do it only if the Phase 2 measurements show attention above about 25% of the prefill time. Otherwise drop it.

### 3.10 The tokenizer itself

Not distributed. Tokenizing runs once per prompt on the CPU at millions of tokens per second (a 100k-token prompt takes tens of milliseconds), while a single round trip on the link is tens to hundreds of microseconds, and the token ids are needed on the main laptop anyway. Running it on the remote laptop, or on both at once, can only add latency. To check this on your own prompts: `time llama-tokenize -m MODEL -f prompt.txt --show-count`.

## 4. User interface

### 4.1 `common` flags (`common/arg.cpp`, `common/common.{h,cpp}`)

| Flag | Meaning |
|---|---|
| `-es, --expert-share SHARE` | Turns on expert mode. `60` gives the RPC devices together 60% (split evenly when there are several) and the local GPU the rest. `RPC0=60,MTL0=40` names devices as `--list-devices` prints them (a name prefix such as `RPC0` is enough). Percentages, which must sum to 100 when every device is named. With some devices named, the rest is split evenly among the others. |
| `-kvd, --kv-device DEV` | Home device (attention, dense weights, KV cache) in expert mode. Default: the local GPU. |
| `-lhcd, --longhaul-cache-device DEV=SIZE,...` | Expert cache size per device, e.g. `RPC0=12G,MTL0=4G`. `-lhc` sizes the devices not named. |

- `-es` sets `split_mode = EXPERT`, fills `tensor_split` in device order and sets `main_gpu` from `-kvd`.
- `-es` with `-ts` or `-sm` is an error (both decide the split). `-es` without `--rpc` and with one GPU is allowed but pointless, and logs a warning.
- `-sm expert` with `-ts`/`-mg` remains available for scripts.
- Env names: `LLAMA_ARG_EXPERT_SHARE`, `LLAMA_ARG_KV_DEVICE`, `LLAMA_ARG_LONGHAUL_CACHE_DEVICE`, following the existing pattern.
- Applies to `llama-cli`, `llama-server` and the other tools that use `common`.

### 4.2 `llama-bench`

- `-sm expert` in the split-mode parser and `split_mode_str`.
- `-ts` already accepts several values (`60/40,50/50,40/60`), so a sweep of the share is `llama-bench -m M --rpc R -sm expert -ts 40/60,50/50,60/40 -lhc 8G`. The RPC device comes first, so the first number is the remote's share.
- `-mg` sets the home. A `-lhcd` flag takes several values like `-lhc`.

### 4.3 Examples

```sh
# remote laptop: same as today
ggml-rpc-server -H 10.0.0.2 -p 50052 --model-dir ~/models

# main laptop: the remote computes about 60% of the routed experts
llama-cli -m ~/models/Inkling-2-Mini-AD-Q4_K.gguf --rpc 10.0.0.2:50052 -es 60 -lhc 16G

# a bigger expert cache on the laptop with more free memory
llama-cli -m M --rpc 10.0.0.2:50052 -es 50 -lhcd RPC0=12G,MTL0=4G

# KV cache (and attention) on the remote, for a long context
llama-cli -m M --rpc 10.0.0.2:50052 -es 40 -kvd RPC0 -c 131072
```

## 5. Files to change

| File | Change |
|---|---|
| `include/llama.h` | `LLAMA_SPLIT_MODE_EXPERT = 4` with a comment. `longhaul_cache_sizes` in `llama_context_params`. Comments on `tensor_split`/`main_gpu` for the expert mode. |
| `src/llama.cpp` | Keep all devices in expert mode. Validate `main_gpu` and the shares. |
| `src/llama-model.cpp` | `load_tensors`: in expert mode `dev_layer(il)` is the home for every layer, `dev_output` is the last device, and no layer split points are computed. Load checks from 3.1. Keep `tensor_split()` for the cache. |
| `src/llama-model.h` | `split_mode()` accessor, if not there yet. A helper `expert_share(dev)`. |
| `src/llama-context.cpp` | Pass `longhaul_cache_sizes`. Call `moe_cache->begin_graph()` before each graph compute. Pipeline parallelism stays off. Default and copy of the new param. |
| `src/llama-cparams.h` | `longhaul_cache_sizes` (an owned vector). |
| `src/llama-moe-cache.h/.cpp` | Devices from the shares in expert mode (every MoE layer on every device with share > 0). Groups per (device, layout) over all layers. Owner tables, per-device slot maps, filtered planning, the -1 copy buffer, the ids cache, the per-device sizes (3.7), rebalance hooks and `lru.drop`, per-device stats. `build_lookup` returns per-device slots. `get_experts(w, d)`. |
| `src/llama-expert-share.h/.cpp` (new) | Initial ownership (weighted interleave) and the rebalance rule, pure functions. |
| `src/llama-graph.cpp/.h` | `build_moe_ffn` expert-mode path (3.3). `build_lora_mm_id` takes the device bank. |
| `src/CMakeLists.txt` | Add `llama-expert-share.cpp`. |
| `common/arg.cpp`, `common/common.h`, `common/common.cpp` | `-es`, `-kvd`, `-lhcd`, the device name parsing, the conflict checks and the param mapping. `-sm expert` in the split-mode help. |
| `tools/llama-bench/llama-bench.cpp` | `-sm expert`, `-lhcd`. |
| `tests/test-llama-archs.cpp` | Expert-mode cases (section 6). |
| `tests/test-expert-share.cpp` (new), `tests/CMakeLists.txt` | Unit test of ownership and rebalance. |
| `docs/longhaul.md` | "Expert split" section: what `-es` means (routed experts only), layer split vs expert split and when each wins, `-kvd` costs, `-lhcd`, the stats lines, tuning with `llama-bench`. |
| `tools/rpc/README.md`, `docs/multi-gpu.md`, `README.md` | Short pointers. |

## 6. Tests (in memory, code level only)

All with the test models that `test-llama-archs` already builds, comparing logits to the single-device reference at NMSE <= 1e-4. The order of the partial sums differs from the single-device sum, so an exact match is not expected.

1. `test-expert-share` (new, label `main`): the initial ownership hits `round(share * n_expert)` for shares 0/30/50/60/100 and 3 devices. The rebalance converges to within 2 points on skewed synthetic counts (Zipf-like), respects the caps and `max_moves`, makes no moves within the tolerance, is deterministic, and does not oscillate under alternating counts.
2. `test-llama-archs` with `GGML_METAL_DEVICES=2` (two emulated Metal devices), for every LongHaul arch, expert mode with shares 50/50, 30/70, 100/0 and 0/100, home on device 0 and device 1, cache "every expert" and "5 slots per device", and per-device sizes via `longhaul_cache_sizes`. Rebalance off, for stable references.
3. Same as 2 with the rebalance on and a forced tiny check interval (an internal knob read only by tests), over several decode steps: logits still match the reference after moves, which checks `lru.drop` and the -1 maps.
4. `test-llama-archs --rpc` (label `rpc`, in-process loopback server as today): expert mode with the RPC device at share 60, for both "read by the server" and "sent from the client", and with the home on the RPC device.
5. Mutation checks done by hand during implementation (not committed): break the -1 filter, the ids cache epoch and `lru.drop`, and confirm a test fails each time.
6. Existing suites must stay green: `ctest -L main`, the LongHaul cases, the `rpc` label.

Not run without your OK: `test-rpc-multi-server` (separate processes), and anything on the two real laptops.

## 7. Phases

| Phase | Content | Done when |
|---|---|---|
| 1 | Library expert mode with a static owner table: split mode, placement, per-device banks and slot maps, graph path, callback changes, KV home (`main_gpu`), per-device cache sizes, stats. | Tests 2, 4 and 6 pass. |
| 2 | `common` flags, `llama-bench`, docs. You measure layer split vs expert split on the two laptops. | Flags work in `llama-cli`/`llama-server`, docs written, your numbers in hand. |
| 3 | Rebalancing (3.6) and its unit test. | Tests 1 and 3 pass, and the measured share at exit is within 2-3 points of the target on a real prompt. |
| 4 | Tuning from your measurements: send the activations as f16 over the link if the link limits prompt processing, and overlap the shared expert with the remote experts (build the shared expert before the combine). | Measured gain, or dropped. |
| 5 | Two-batch overlap in prompt processing (3.9), only if the gate is met. | Gate met and the prefill speed improves, or dropped. |

Each phase is a separate commit, made only with your approval.

## 8. Expected performance (to measure, not measured)

- Expert split wins when disk reads or expert compute dominate: two SSDs and two GPUs work at once instead of in turns.
- Layer split can win when every expert hits the cache and link latency dominates. For Inkling 2 Mini the expert split needs about 48 round trips per token (one per MoE layer pass), while the layer split with an even split needs about 24 for the remote MoE layers plus 6 hops.
- Both modes add up the caches of all laptops.
- The share applies on average. A single token can select, say, 7 of its 8 experts from one laptop, and each layer waits for the slower laptop, so the best share is found by measuring (`llama-bench` sweep), not computed from the GPU speeds.

## 9. Risks

- Per-token imbalance with small `k` (above). Mitigation: measure, and maybe later replicate the hottest experts on both laptops. Not in this plan.
- Scheduler split count: about 3 to 4 extra splits per MoE layer pass (roughly 200 for Mini). Check that `ggml_backend_sched` has no fixed split limit that this hits, and look at the graph-build time in the perf log.
- The ownership check is per graph: a rebalance that ran in the middle of a graph would give inconsistent maps, so it runs only in `begin_graph()`.
- Metal cleanup crash at exit during load (`GGML_ASSERT(rsets count == 0)`): a separate known issue, to look at alongside, not caused by this plan.

## 10. Phase 1 results (2026-11-25)

Tests run on this Mac (Apple M5, Metal), all in one process:

| Run | Result |
|---|---|
| `test-expert-share` (new) | OK |
| `GGML_METAL_DEVICES=2 test-llama-archs -a 'inkling2\|qwen35moe\|bailingmoe3' -s 1` | exit 0, 232 LongHaul checks OK, 0 FAIL (including 7 expert-split cases per file variant) |
| `test-llama-archs -a 'inkling2\|qwen35moe\|bailingmoe3' -s 1` (one device) | exit 0, 92 LongHaul checks OK |
| `test-llama-archs --rpc -s 1` | exit 0, 20 checks OK (4 expert-split cases per arch: server-read and sent experts, layers on the GPU or on the RPC device) |
| `ctest -L "main\|behavior"` | 55 of 60 pass. The 5 failures are missing Python modules (`jinja2`, `numpy`, `torch`) and a missing `models/ggml-vocab-qwen35.gguf`, not this change. |

Mutation checks (reverted):
- Copying the LRU slot map without the -1 mask: 22 expert-split cases fail. Without the -1, an expert that a device does not own has the "empty" slot, past the end of the bank view. The Metal matmul path leaves those rows unwritten, so the output picks up stale memory. In one run this only moved the NMSE from 4.7e-08 to 9.0e-05, so the expert-split cases use a threshold of 1e-5 instead of 1e-4. That is safe, since they run on the same backend as the reference and the largest NMSE seen is 6.7e-07.
- Planning each device with the experts of the other devices: 28 cases fail.

Changes from the plan:
- Per-expert scales (`ffn_*_exps_s`) are allowed: they multiply the output rows, so a skipped expert stays zero. Inkling 2 files have them. Only expert biases are refused, by an assert in `build_moe_ffn` (this fork's `llama_layer` has no expert bias tensors, so no load-time check is needed).
- The ids of a layer are read by the copy of its first part and reused by the later parts, which the graph orders right after it (`layer::next_part` asserts the order). `begin_graph()` is left for phase 3, where the rebalance needs it.
- `test-expert-share` has the label `behavior`, like the other tests of internal functions.
- The router bias of the inkling2 and bailingmoe3 test models is a ramp, so every token selects the same top experts and the measured shares are 50/50 whatever the target. The qwen35moe models route freely: 30/70 by count measured 26/74, which is what phase 3 is for.
- The exit stats keep the device names, because the backends are freed before the cache logs its stats.

## 11. Phase 2 results (2026-11-25)

Done:
- `common`: `-es/--expert-share`, `-kvd/--kv-device`, `-lhcd/--longhaul-cache-device` (env `LLAMA_ARG_EXPERT_SHARE`, `LLAMA_ARG_KV_DEVICE`, `LLAMA_ARG_LONGHAUL_CACHE_DEVICE`) and `-sm expert`. They are resolved after all arguments are read (`common_params_resolve_expert_split` in `common/arg.cpp`), so the order of `--rpc`, `-dev` and `-es` does not matter. Without `-dev`, the resolver builds the device list as `llama_prepare_model_devices` would (RPC first, then GPUs, then iGPUs if there is no GPU) and sets it in `params.devices`, so that the indices of the shares, of `main_gpu` and of the cache sizes match the model. The mmproj default device is set before this, so it does not move to an RPC server.
- A draft model leaves the expert split: layer split, no `tensor_split`, no cache sizes per device, and without `-devd` it goes on the KV device (`common_base_params_to_speculative`).
- `llama-bench`: `-sm expert`, and `-lhcd s0/s1/...` in device order (0 = from `-lhc`).
- Docs: `docs/longhaul.md` (options table, "Expert split" section with the layer vs expert trade-off, the tokenizer and the stats line), `docs/multi-gpu.md`, `tools/rpc/README.md`, `README.md`.

Tests:
- `test-arg-expert-split` (new, label `behavior`, run by CTest with `GGML_METAL_DEVICES=2`): the three forms of `-es`, `-kvd`, `-lhcd`, the env vars, the order of `-dev`, the draft, and 20 invalid inputs. OK. With one device it runs only the checks that need one device.
- `test-expert-share`: OK. The library did not change in this phase.

Changes from the plan:
- `llama-bench -lhcd` takes one value for all tests, not a list: its sizes are per device, so a list of lists would be hard to read. Compare sizes with separate runs.
- A single number (`-es 60`) needs both RPC and local devices; on one Mac with two GPUs the devices must be named (`MTL0=60,MTL1=40`).
- The RPC servers must run on Metal: a CUDA server would not skip the experts of the other devices, and the client cannot see the backend of a server. The docs say so.

Not done: the measurement of layer split vs expert split on the two laptops. That needs your laptops and is not an in-memory test. The commands are in `docs/longhaul.md` ("Layer split or expert split").

## 12. Phase 3 results (2026-11-25)

Done:
- `llama_expert_share_rebalance` (`src/llama-expert-share.cpp`): the rule of 3.6 as a pure function. It returns the moves and applies them to the owner table.
- `moe_cache_lru::drop` frees the slot of a moved expert and puts it at the front of list 0, where `victim()` looks first. The CRF of the expert is kept.
- `llama_moe_cache::begin_graph()`, called by `llama_context::graph_compute` before each graph compute. It checks the layers with at least 512 new selections. The copy of the first part of each layer counts the selections it already reads, and halves the counts of a layer every 4096 selections.
- The exit line per device shows the experts it owns at exit (average over the layers) and the moves in and out.
- Env: `LLAMA_EXPERT_SHARE_REBALANCE=0` turns it off, `LLAMA_EXPERT_SHARE_REBALANCE_INTERVAL=N` sets the selections between checks (for tests).

Tests:
- `test-expert-share`: convergence to within 2 points on Zipf-like counts (2 and 3 devices, 128 to 512 experts), each move makes the sum of the gaps smaller, the caps and `max_moves` hold, no move within the tolerance or without counts, deterministic, no move to a device with a cap of 0, the cached preference, the 1.5x limit, swaps at the cap, and alternating routing with halved counts (6 moves in the first 100 checks, 0 in the next 100). OK.
- `GGML_METAL_DEVICES=2 test-llama-archs -a 'inkling2|qwen35moe|bailingmoe3' -s 1`: exit 0, 261 OK, 0 FAIL. Two new cases per file: 30/70 with a check at every graph, with 4 times the 5-slot cache and with the smallest cache on the device of 30%. For qwen35moe the measured share goes from 26.0/74.0 (no rebalance) to 29.0/71.0 (2 moves), and to 31.7/68.3 with the smallest cache (52 moves, mostly swaps). The other cases now run with the rebalance off and check that nothing moves.
- One device: 93 OK. `--rpc`: 20 OK.

Mutation check (reverted): `drop` that frees the slot but leaves the expert's slot map entry, so that an expert that moves back reads a slot that holds another expert. The NMSE over all logits only went from 2.2e-07 to 2.8e-06, under the 1e-5 threshold. The expert-split cases now also check the NMSE of each token's logits (at most 1.4e-06 on correct code), which catches it with 4.2e-05.

Changes from the plan:
- Swaps: when the device below its target already owns its cap of experts, a step swaps an expert of each device, so that the counts do not change. Without them, a device whose cache has fewer slots than it owns experts could never take experts. A swap counts as two of the 2 moves of a check.
- The preference for experts that are not cached is a penalty (a cached expert counts as a quarter of the gap further away), not a strict order: a strict order would move small uncached experts first and need many more checks.
- No RPC case with moves: the RPC test models route every token to the same two experts, which 60/40 already splits one each, so no move fits. The moves use the same host-side LRU and fill paths as the RPC cases that pass.
- The measured share at exit on a real prompt (the "done when" of phase 3) needs your laptops.

