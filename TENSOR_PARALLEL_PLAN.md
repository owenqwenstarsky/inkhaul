# Plan: two laptops as one machine (tensor parallel over Thunderbolt)

Status: design only, nothing implemented. Builds on THUNDERBOLT_PLAN.md (RPC, LongHaul on several devices) and EXPERT_SPLIT_PLAN.md (`-es`). All numbers below are estimates from model sizes and published link figures, not measurements. Phase 0 measures them before any other work starts.

## 1. Goal

- Both laptops work on every token at the same time: every large matrix of every layer is split between the two GPUs, so the memory bandwidth and the GPU compute of both laptops add up, as if one machine had all the cores.
- The test is speed, not memory: a run on both laptops must be faster than the best run on one laptop that holds the whole model, and faster than today's layer split and expert split.
- Measured with `llama-bench` on the real laptops: `tg128` (generation of one stream), `pp512` and `pp4096` (prompt processing), and `tg128` with 2 and 4 parallel streams (server load).

## 2. Why today's modes do not speed up a token

- **Layer split** (`-sm layer`): the laptops take turns. While one runs its layers, the other waits. It pools memory but a token takes as long as on one laptop that holds the model, plus the link hops.
- **Expert split** (`-es`): better, but four things limit it.
  1. Only the routed experts run on both laptops. Attention, router, shared experts and the output layer run on the home laptop, and the remote laptop waits while they run.
  2. Each MoE layer costs a full round trip (the home laptop sends the activations, then waits for the partial sum), not a one-way trip.
  3. The main process drives the remote laptop command by command, and the LongHaul copy step of every layer stops the GPU and reads the expert ids on the CPU. The main process cannot run ahead of the GPUs.
  4. Each token picks 8 experts. With fixed ownership, a token often gets 5 or 6 of them on one laptop, and each layer waits for the slower one.

## 3. What limits the speed-up

On Apple Silicon the CPU and GPU share one memory bus, so generation speed is set by memory bandwidth, and using the CPU cores as well adds nothing. What two laptops add is a second memory bus and a second GPU. The price is a sync between the laptops inside every layer.

Time per generated token with the work split in half:

```
T2 = T1/2 + N * t
```

- `T1`: time on one laptop that holds the whole model;
- `N`: syncs per token;
- `t`: cost of one sync, which is the one-way link latency plus the time to stop and restart the GPU.

To be faster, `N * t` must be smaller than `T1/2`.

| Model (Q4_K, both laptops M5, ~120 GB/s each) | `T1` | Syncs `N` (heads split / attention copied) | Break-even `t` |
|---|---|---|---|
| Inkling 2 Mini (48 layer passes) | ~23 ms | 96 / 48 | ~120 us / ~160 us |
| Qwen 3.5/3.6 35B-A3B (40 layers) | ~15 ms | 80 / 40 | ~90 us / ~120 us |

The two `N` values are the two ways to split attention, see 5.2. With attention copied, the copied weights stay serial, which is why its break-even is not simply twice as high.

Estimated generation speed by sync cost `t` (tok/s, speed-up over `T1`):

| | t = 50 us | t = 100 us | t = 300 us |
|---|---|---|---|
| Inkling 2 Mini, heads split | 61 (1.4x) | 47 (1.1x) | 25 (0.6x) |
| Inkling 2 Mini, attention copied | 56 (1.3x) | 50 (1.15x) | 34 (0.8x) |
| Qwen 35B-A3B, heads split | 88 (1.3x) | 65 (1.0x) | 32 (0.5x) |
| Qwen 35B-A3B, attention copied | 84 (1.25x) | 72 (1.05x) | 46 (0.7x) |

What this means:

- **Generation depends almost entirely on `t`.**
  - Published figures for TCP over Thunderbolt are about 300 us; RDMA over Thunderbolt 5 is under 50 us. Your laptops are Thunderbolt 4, so TCP only.
  - At 300 us no split of the layers can make a single stream faster. This matches the published llama.cpp results, which got slower with more Macs over TCP.
  - So Phase 0 measures `t` on your cable first. Most of the design (5.3) is about getting each sync down to the bare one-way latency.
- **Prompt processing speeds up even on a slow link.** A batch of 512 tokens has enough work per layer to cover the sync and the transfer. The estimate is 1.4x without overlap, and up to ~1.9x with the overlap of 5.8.
- **Several streams at once** (server load) spread the cost of each sync over all streams in the batch.
- **Speculative decoding** checks several guessed tokens in one pass, so one pass costs the same `N` syncs for several tokens. It makes a split pass relatively cheaper (5.7).

## 4. Current state (checked 2026-11-26)

- **The upstream tensor-parallel core is still in this tree.**
  - `ggml/src/ggml-backend-meta.cpp` (identical to the fork base 5fddd0b, still compiled, `ggml/src/CMakeLists.txt:193`) wraps several devices as one "meta" device.
  - Each tensor gets a split state: split on axis 0-3, MIRRORED (a full copy on each device) or PARTIAL (each device holds a partial sum).
  - The graph is cut into one sub-graph per device at every PARTIAL node, with an all-reduce between them. The MoE all-reduce is delayed until after the expert sum, so it moves `[n_embd, n_tokens]`. Sub-graph splits are cached when the graph uid repeats.
- **RPC has an all-reduce between servers** (`ggml-rpc.cpp`).
  - The client sends `COMM_INIT` (the comm port is the first server's port + 1000, `:3259`).
  - Backends without their own communicator (Metal) use `comm_allreduce_pairwise` (`:2680`), for exactly 2 ranks.
  - The data never passes through the client, and the client's commands are fire-and-forget.
  - The RPC graph cache sends a 12-byte `GRAPH_RECOMPUTE` for a graph it has seen.
- **The pairwise all-reduce is slow in three ways** (`:2680-2788`):
  1. It blocks the server's command loop, which waits for the GPU (`ggml_backend_synchronize`), copies to the host, exchanges, copies back and runs an ADD graph. The GPU is idle for that whole time and for the encoding of the next sub-graph.
  2. It is a ping-pong: rank 0 sends then receives, rank 1 receives then sends. That is a full round trip, where one one-way trip would do.
  3. With bf16 on the wire (payloads of 32768 values or more), each rank adds its own f32 partial to the other's bf16 partial. The "mirrored" result then differs between the ranks, which is a correctness bug for any later MIRRORED tensor.
- **The llama half was removed by the fork's prune.**
  - The split table `llama_meta_device_get_split_state` (5fddd0b `src/llama-model.cpp:432-1100`), the granularity rules and the arch list `llm_arch_supports_sm_tensor` (5fddd0b `src/llama-arch.cpp:1547`) are gone.
  - `-sm tensor` is gone too; `src/llama.cpp:159` rejects the mode.
  - Upstream allowed `qwen35moe` and refused `inkling2`, `inkling2_loop` and `bailingmoe3`. It needed flash attention and could not run with the MoE cache.
- **Metal events are `MTLSharedEvent`s** (`ggml-metal-device.m:1448`). The CPU can set their value, which is what an all-reduce that does not block the GPU needs. ggml has no API for "the host signals an event" yet.
- **Transport:** TCP over the Thunderbolt Bridge, with `TCP_NODELAY` (`transport.cpp:715`). The RPC transport already negotiates RDMA on Thunderbolt 5, so anything built on it gets RDMA for free on such Macs.
- **Speculative decoding** is in `common`: n-gram types (no extra model) and draft models. The Qwen 3.5 MTP block is loaded but not run.

## 5. Design

### 5.1 One logical device

The main model runs on a meta device over two ranks, one per laptop. Each rank is a GPU behind an RPC server:

- Rank 0 is the main laptop's GPU, behind an RPC server on its Thunderbolt address (10.0.0.1). The remote must be able to connect to rank 0 for the comm socket, so it cannot be 127.0.0.1.
- Rank 1 is the remote laptop's GPU (10.0.0.2).
- `llama-cli` / `llama-server` is the client. It builds the graph once, sends small inputs (token ids, positions, masks) to both ranks each pass, streams cached-graph commands, and reads the logits.
- The ranks exchange partial sums directly over the cable.

Why this process model first: it is the upstream multi-node design, and most of it is already in the tree. With every weight resident, the client never waits inside a pass: it sends the whole pass and waits only for the logits.

From Phase 3, `llama-cli` starts rank 0 itself as a server thread (`--tp-host 10.0.0.1`), the way the RPC test already starts one. Then no second process is needed on the main laptop.

The later alternative is one llama process per laptop (section 5.9). It is only needed for models larger than both laptops together.

### 5.2 What is split

| Tensors | Split | Why |
|---|---|---|
| Routed experts `ffn_gate/up(_exps)` | by FFN rows (axis 1) | each rank computes half of every selected expert: no per-token imbalance, unlike `-es` ownership |
| Routed experts `ffn_down_exps` | by FFN columns (axis 0), gives PARTIAL | one all-reduce of `[n_embd, n_tokens]` per MoE layer |
| Shared experts `*_shexp` | like the routed experts (upstream mirrored them for Qwen) | the meta backend merges independent PARTIAL branches into one all-reduce, so the shared expert adds no sync and its reads are halved |
| Router `ffn_gate_inp`, norms, small vectors | mirrored | small; both ranks need the full router output |
| Attention / Gated DeltaNet / KDA weights, KV cache, recurrent state | policy `--tp-attn`, see below | |
| Output `output.weight` | by vocab rows (axis 1) | halves the largest single read per token. The client reads half the logits from each rank; for a 248k vocab that is ~0.5 MB from the remote per token. |
| Token embeddings | mirrored | a lookup reads one row |

Attention policy (`--tp-attn`):

- **`split`**: Q/K/V/gate by heads, KV cache and recurrent state by heads, output projection gives PARTIAL. Two all-reduces per layer. Attention reads and the KV cache are halved, so the possible context doubles.
- **`mirror`**: every rank holds all attention weights and the full KV cache, and computes attention in full. One all-reduce per layer (the MoE one). Attention is not sped up and its memory is doubled.
  - Even so, this is strictly better than today's `-es`: one one-way sync instead of a round trip, both ranks busy, and balanced experts.

The default is chosen in Phase 3 from your measurement. By section 3, `mirror` wins when `t` is above about 100 us.

Upstream's granularity rules carry over:

- attention by whole GQA groups;
- FFN in steps of `lcm(block size, 128)`;
- the vocab in steps of the block size.

Qwen 3.5's full-attention layers have 2 KV heads, so `split` gives one KV head per rank; with more than 2 ranks those layers fall back to `mirror`.

### 5.3 The fast all-reduce

This is the part that decides whether generation speeds up. The goal for one sync is the one-way link latency plus about 20 us.

1. **Exchange at the same time.** Both ranks send, then both receive.
   - Small payloads go inline: 8 KiB fits in the socket buffers, so it cannot deadlock.
   - Large ones (prompt processing) use one sender thread per peer.
   - This replaces the ping-pong and saves one one-way trip per sync.
2. **Do not stop the GPU queue.** `COMM_ALLREDUCE` on the server no longer blocks. It queues:
   - "record `ready[n]`" after the sub-graph that makes the partial;
   - "wait `done[n]`", the add, and right behind them the next `GRAPH_RECOMPUTE`.
   
   So the next sub-graph is encoded and committed while the bytes are on the cable, and the GPU starts it as soon as `done[n]` is set.
   
   A comm thread per rank then:
   - waits for `ready[n]`, spinning on the event value for up to ~200 us before it sleeps;
   - copies the partial out of the shared Metal buffer;
   - exchanges it with the peer and copies the peer's partial into a receive tensor;
   - sets `done[n]` from the CPU.
3. **New ggml API:** `ggml_backend_event_signal_host(ggml_backend_event_t)`, which marks an event as reached from the host. It is a new optional device function in `ggml-backend-impl.h`. On Metal it sets `MTLSharedEvent.signaledValue`. Backends without it keep the blocking path.
4. **The same sum on both ranks.** The mirrored result must be bit-identical on every rank. Each rank adds the same two operands, and IEEE addition of two numbers is order-independent. With bf16 on the wire, each rank first rounds its own partial to bf16 too. This fixes the bug in section 4.
5. **Spin before sleep on receive** (`recv` with `MSG_DONTWAIT` in a short loop), for the decode-sized exchanges only.
6. **Built on the RPC transport.** The peer sockets come from `transport.cpp`, so on Thunderbolt 5 Macs the same code runs over RDMA.
7. **A timeout.** If the peer does not answer within a few seconds (cable pulled, laptop asleep), the comm thread sets `done[n]` with an error flag, and the server fails the pass with a clear message. A GPU must never wait on an event that never comes.

### 5.4 Fewer messages from the client

Today one pass costs each server about `4L + 1` messages (sub-graph and all-reduce commands), with one send each.

- A new RPC command, `GRAPH_STEPS {device, n, (uid, allreduce?) * n}`, sends a whole pass in one message per server. The RPC protocol minor version goes up.
- The mirrored inputs go out as one batched `SET_TENSOR` per server.
- The client then sends about 2 messages per server per pass and reads the logits.

### 5.5 Both ranks must pick the same experts

Each rank runs the router itself on its mirrored input.

- On identical chips with identical kernels, the router output is bit-identical, given 5.3.4.
- On different chips the kernels can sum in another order: Metal picks some kernel paths by GPU family, and the M5 matmul path is not used on older chips. A near tie could then pick different experts on the two ranks, and the output of that layer would be silently wrong.

The guard:

- After each MoE layer, each rank adds an 8-byte checksum of that layer's selected expert ids to the next all-reduce payload. That costs no extra trip; the ids are ready before the partial is.
- A mismatch stops the run with an error that names the layer.
- On two laptops with different chips, a `--tp-router-sync` mode would send rank 0's ids to rank 1. That costs one more one-way trip per MoE layer. It is built only if your laptops differ (open question 1).

### 5.6 Setting the share of each laptop

- `-ts A,B` sets the share of each rank, as in the layer split. Each tensor class rounds the share to its granularity in every layer. The startup log prints the share each class gets, for example `tensor split: experts 50/50 (n_ff 512, step 256), attention 50/50, output 60/40`.
- Every layer syncs, so each layer waits for the slower rank. Averaging uneven layers over the model therefore does not help, and the split must be even within each layer.
- Two laptops with the same chip use 50/50.
- Unequal laptops can only shift the classes with fine steps (the output layer and the Gated DeltaNet heads). A finer expert share would mix FFN slices with a few whole experts owned by the faster laptop. That is left for later, and only if your laptops differ.
- `-es` stays a separate mode. Giving it with `-sm tensor` is an error.

### 5.7 Speculative decoding

- A verify pass of `K` guessed tokens costs the same `N` syncs as a pass of one token, and its weight reads are split across both GPUs. The fixed sync cost is spread over the accepted tokens.
- Nothing new is needed in the library, since a verify pass is just a batch.
- The work is to measure n-gram speculation (no extra memory) and a small draft model, and to write the best default into the docs.
- The draft model runs on the main laptop's GPU directly (`-devd MTL0`), not through RPC.
- The Qwen 3.5 MTP block would be a better drafter, but it is a separate feature (open question 3).

### 5.8 Prompt processing

- The wire carries bf16 above 32768 values, with the consistent rounding of 5.3.4. A 512-token ubatch of Qwen 35B moves 2 MiB per sync, 80 times per ubatch.
- Overlap (a gated phase): split the ubatch into two halves in one graph, so the GPU computes half B while half A's partial is on the cable.
  - The ordering problem (the KV cache and the recurrent state written by A before B reads them) is the same as Phase 5 of EXPERT_SPLIT_PLAN.md.
  - It is done only if the measured transfer time is above ~20% of the prompt processing time.

### 5.9 Models larger than both laptops (gated)

- Ling-3.2-air (75 GB) and Qwen 122B do not fit in two 24 GB laptops, so they still need LongHaul.
- With the client model of 5.1, each MoE layer would need a round trip to read the expert ids, and the client could no longer run ahead.
- The design for this case is one llama process per laptop. Rank 0 takes the prompt and sends each batch to rank 1, and both run `llama_decode` on their own slice.
- Each rank's LongHaul cache reads the expert ids from its own GPU. Its router is mirrored, so the ids are the same, which 5.5 checks. It reads misses from its own SSD.
- Only the batch, the all-reduces and the logits cross the cable.
- This needs:
  - a rank-local meta device;
  - a communicator usable from a library process (`comm_init_rank` on the RPC transport);
  - LongHaul banks for expert slices: half the rows of gate/up are contiguous, but half the columns of `down` are not, so `down` is read whole and half of it is kept.
- It gets its own plan, and only if you need these models (open question 2).

### 5.10 Model support

- **`qwen35moe` first:** the upstream table exists, with full attention and Gated DeltaNet split by heads.
- **`inkling2` / `inkling2_loop` (Phase 4):** heads split for Q/K/V/gate/rel/sink and the K/V short convolutions with their state channels.
  - The `attn_post_conv` and `ffn_post_conv` convolutions are linear per channel. A PARTIAL input can pass through them if each rank keeps its partial history in the conv state, so the all-reduce moves after them. This needs a meta propagation rule: SSM_CONV, add and scale of a PARTIAL stay PARTIAL.
  - `mix()` (the depth attention) needs the full vector and runs after the all-reduce. The loop control messages are mirrored.
  - Still 2 syncs per pass with `split`, 1 with `mirror`.
  - Why upstream refused this arch is not written down. Phase 4 starts by finding out.
- **`bailingmoe3`:** the MLA latent cache cannot be split by heads, so it is mirrored and the heads are split. KDA is split by heads. Last, and only if you need it.

## 6. User interface

```sh
# remote laptop (as today)
ggml-rpc-server -H 10.0.0.2 -p 50052

# main laptop, Phases 1-2: its own rank as a second process
ggml-rpc-server -H 10.0.0.1 -p 50052
llama-cli -m M --rpc 10.0.0.1:50052,10.0.0.2:50052 -dev RPC0,RPC1 -sm tensor -fa on

# from Phase 3: llama-cli runs rank 0 itself
llama-cli -m M --rpc 10.0.0.2:50052 -sm tensor --tp-host 10.0.0.1 --tp-attn mirror -ts 50,50

# compare all modes
llama-bench -m M --rpc 10.0.0.1:50052,10.0.0.2:50052 -dev RPC0,RPC1 -sm layer,expert,tensor -p 512 -n 128
```

New flags:

- `--tp-host ADDR` (env `LLAMA_ARG_TP_HOST`): runs rank 0 in-process.
- `--tp-attn split|mirror` (env `LLAMA_ARG_TP_ATTN`).
- `--tp-router-sync`: only if built, see 5.5.
- `-sm tensor` comes back.

With `-sm tensor` all experts load as normal (sliced) weights. If they do not fit, loading fails with a pointer to `-es` or 5.9.

## 7. Files to change

| File | Phase | Change |
|---|---|---|
| `src/llama-model.cpp`, `src/llama-model.h` | 1, 3, 4 | restore `llama_meta_device_get_split_state` and the granularity code from 5fddd0b; shared experts sliced; `--tp-attn` policy; inkling2 table; resident load of `TENSOR_LONGHAUL` experts with `-sm tensor` |
| `src/llama-arch.cpp` | 1, 4 | `llm_arch_supports_sm_tensor` as an allow list: `qwen35moe`, later `inkling2*` |
| `src/llama.cpp` | 1 | remove the reject at `:159`, restore the meta device setup |
| `src/llama-context.cpp` | 1 | restore the flash-attention requirement and the backend-sampling fallback for tensor mode |
| `src/llama-moe-cache.cpp` | 1 | refuse the cache in tensor mode (clear error) until 5.9 |
| `include/llama.h` | 1, 3 | the comment on `LLAMA_SPLIT_MODE_TENSOR`; `tp_attn` in `llama_model_params` |
| `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend.cpp` | 2 | `ggml_backend_event_signal_host` |
| `ggml/src/ggml-metal/ggml-metal-device.m`, `ggml-metal.cpp` | 2 | host signal on `MTLSharedEvent`; check that graph compute does not wait for completion |
| `ggml/src/ggml-rpc/ggml-rpc.cpp`, `transport.{h,cpp}`, `ggml/include/ggml-rpc.h` | 2, 3 | simultaneous exchange, async all-reduce with comm thread and events, consistent bf16, spin receive, timeout, `GRAPH_STEPS`, ids checksum, protocol minor bump, comm port option |
| `ggml/src/ggml-backend-meta.cpp` | 3, 4 | ids checksum hook; PARTIAL through SSM_CONV, add and scale |
| `tools/rpc/rpc-server.cpp`, `tools/rpc/README.md` | 2, 3 | `--comm-port`; docs |
| `common/arg.cpp`, `common/common.{h,cpp}` | 1, 3 | `-sm tensor`, `--tp-host`, `--tp-attn`, errors for `-es` with `-sm tensor` |
| `tools/llama-bench/llama-bench.cpp` | 1, 3 | `-sm tensor`, `-tpa split,mirror` |
| `tools/tp-probe/` (new) | 0 | link and GPU event latency probe |
| `scripts/tp-budget.py` (new) | 0 | bytes per token by part, from a GGUF header, and the section 3 estimate for a measured `t` |
| `tests/test-llama-archs.cpp`, `tests/test-tp-comm.cpp` (new), `tests/test-arg-expert-split.cpp`, `tests/CMakeLists.txt` | 1-4 | see section 8 |
| `docs/tensor-parallel.md` (new), `docs/longhaul.md`, `docs/multi-gpu.md`, `README.md` | 1-5 | docs |

## 8. Tests (in memory, code level only)

1. **Split correctness in one process:** `GGML_METAL_DEVICES=2 test-llama-archs` with `-sm tensor` against one device. Covers `qwen35moe`, both `--tp-attn` policies and shares 50/50 and uneven. Overall and per-token NMSE at most 1e-5, as in the expert split tests.
2. **Two ranks over RPC:** `test-llama-archs --rpc` starts two RPC server threads on 127.0.0.1 with their own comm ports. The client uses `RPC0,RPC1 -sm tensor`, so the all-reduce runs over loopback between the threads. Logits are compared with one local device.
3. **Mirrored tensors bit-identical:** after a prompt-sized ubatch (bf16 on the wire) and a decode step, read the same mirrored tensor from both ranks and require equal bytes. It must fail on today's pairwise code, which proves that the bug of section 4 is caught.
4. **`test-tp-comm` (new):** two threads, loopback sockets, real Metal events.
   - 10000 decode-sized and 50 payloads of 4 MiB, checking the values, ordering, and no deadlock.
   - The timeout path, with the peer closed in the middle.
   - `ggml_backend_event_signal_host` ordering.
5. **Router guard:** a test hook biases the router of one rank, and the run must stop with the layer named.
6. **Arguments:** `-sm tensor` with `-es` fails; `--tp-attn`; the shares rounded per class.
7. **Mutation checks,** as in the earlier plans:
   - drop the bf16 self-rounding (test 3 fails);
   - skip the wait on `done[n]` (tests 2 and 4 fail);
   - add the shared expert outside the merged all-reduce (test 1 fails).

Nothing runs on the two laptops without your approval. The probe and the benchmarks there are yours to run, or mine after you OK each run.

## 9. Phases

| Phase | Content | Done when / gate |
|---|---|---|
| 0 | `tp-probe`: TCP one-way latency and same-time exchange on the cable at 8 KiB, 64 KiB and 2 MiB (p50/p99, spin vs sleep), and the GPU -> CPU -> GPU event delay on one Mac. `tp-budget.py` for your GGUF files. One-laptop `llama-bench` baseline. | **Gate:** estimated `t` (one-way + event delay). Below ~100 us: all phases. Above ~150 us: generation of one stream cannot speed up on this cable. The plan is cut to prompt processing, server load and speculative decoding, and you decide whether to go on. |
| 1 | Restore `-sm tensor` for `qwen35moe`: upstream split table, resident experts, two RPC ranks, the existing pairwise all-reduce. Tests 1, 2, 6. | Tests pass. You measure tensor vs layer vs expert split. This shows what the slow all-reduce costs. |
| 2 | Fast all-reduce (5.3) and `GRAPH_STEPS` (5.4). Tests 3, 4. | Tests pass. On the laptops, one sync costs about the probe's one-way latency + ~20 us. |
| 3 | Shared experts sliced, `--tp-attn mirror`, `-ts` per class, `--tp-host`, router guard. Test 5. | You measure `split` vs `mirror` and set the default. |
| 4 | `inkling2` / `inkling2_loop` (5.10). Tests 1-3 for them. | Tests pass, measured on the laptops. |
| 5 | Speculative decoding defaults (5.7), docs. | Measured gain written into the docs, or dropped. |
| 6 | Prompt overlap (5.8). | Only if transfers are above ~20% of prompt processing time. |
| 7 | One process per laptop with LongHaul per rank (5.9). | Only if you need models larger than both laptops; its own plan. |

Each phase is one commit.

## 10. Risks

- **The cable may be too slow for generation.** That is why Phase 0 is a gate and not a formality.
- **Upstream calls the tensor mode experimental,** with no guarantee outside CUDA. Its Metal path was tested in upstream CI with emulated devices only.
- **GPU waits on events:** a queue that waits on `done[n]` hangs if the peer dies, so the timeout of 5.3.7 is required. A few seconds of waiting should be within Metal's limits, but this is checked in test 4.
- **Different chips** can give different router results (5.5). The guard turns a silent error into a stop.
- **`mirror` uses more memory:** attention weights and the full KV cache on both laptops.
- **Machine stability:** one published test saw two Macs crash under sustained heavy TCP over Thunderbolt (HPL). The prompt-processing benchmarks push a lot of data over the cable, so every run on the laptops needs your approval first.
- **Security:** `ggml-rpc-server` has no authentication. Bind it to the Thunderbolt addresses only, as today.
- **A dropped link** or a sleeping laptop aborts the run, as today.

## 11. Answers (2026-11-26)

- Phase 0 gate: build all phases now with in-memory tests, you measure later. Phase 6 stays dropped until your numbers show it is needed.
- The other laptop has the same chip (M5): 50/50 and the checksum guard, no `--tp-router-sync`, no mixed expert share.
- Phase 7: not now.
- One commit per phase, nothing pushed.

## 12. Open questions (answered in section 11)

1. What is the other laptop (chip, memory)? The same chip as this M5 makes 50/50 and the router guard enough. Different chips need 5.5's router sync and a finer share.
2. Which models matter most? Inkling 2 Mini and Qwen 35B-A3B fit in two laptops. Ling-3.2-air and Qwen 122B do not, and need Phase 7.
3. What matters most to you: one chat stream, long prompts, or `llama-server` with several users? This decides the order of Phases 4-6, and whether Qwen MTP speculation is worth its own plan.
4. When Phase 0's probe exists, may I run it on both laptops (with your approval each time), or will you run it?

## 13. Phase 0 results (2026-11-26)

Done:
- `llama-tp-probe` (`tools/tp-probe/`), with four modes:
  - `listen` / `connect`: round trip and same-time exchange. The default sizes are 8 KiB, 64 KiB and 2 MiB, each with a blocking and a spinning receive.
  - `gpu`: a tiny graph run plus the wait for it, and the same through an event.
  - `budget`: reads a GGUF header and prints the bytes per token by part, the estimate of section 3 for each mode, and the break-even sync cost.
  - `self-test`: the network code over 127.0.0.1 on two threads, and the budget of a toy loop model with known sizes.
- The budget is written in C++ on the gguf API, not in Python as planned, because the Python `gguf` package needs numpy, which this machine lacks. CTest runs `test-tp-probe` (label `rpc`, since it listens on 127.0.0.1).
- `docs/tensor-parallel.md`: how to measure.

Measured on this Mac (M5, macOS 26.6.2), not on the cable:
- Loopback TCP: 8 KiB round trip p50 35 us, same-time exchange p50 15 us.
- **A tiny graph run plus the wait for it: p50 306 us, p99 679 us** (300 runs; the same through an event record and host wait). That is the cost of stopping the GPU and starting it again, and the blocking all-reduce of section 4 pays it at every sync, before any cable latency. Part of it may be the idle GPU clocking down between the runs of the probe; Phase 2 measures the delay inside a busy queue. Either way, section 5.3 (never drain the queue) matters more than the plan assumed.

Not done: the cable measurement and the budget of your model files. No GGUF model is on this machine, and the cable needs your laptops.

## 14. Phase 1 results (2026-11-26)

Done:
- `-sm tensor` loads `qwen35` and `qwen35moe` again. Other archs fail at load with a clear error (`llm_arch_supports_sm_tensor`).
  - `llama_prepare_model_devices` builds one meta device from `-dev` or from all GPUs (RPC first).
  - It warns when RPC and local devices are mixed, because then the comm between the servers cannot start and every sum goes through the client.
- The split table (`llama_meta_device_get_split_state` in `src/llama-model.cpp`) is upstream's, adapted to this fork:
  - It covers the fused full-attention QKV, the GDN head ratio and the replay caches (`r`, `spk`, `spv`, `spgb`).
  - The KV cache layer is read from the end of the name.
  - The router and the shared experts are mirrored, as upstream. Slicing the shared experts is Phase 3.
- The experts are resident: in tensor mode `create_tensor` drops the LongHaul flag, so the experts load sliced like any other weight. `-lhcd` and `-es` are rejected with `-sm tensor`.
- In tensor mode:
  - Flash attention is forced on, or the load fails if it is set off.
  - Backend sampling is off.
  - `--fit` skips the mode with a warning.
- `llama-bench -sm tensor` and `-sm tensor` in the common arguments.
- The model saver copies the meta tensors to host first. The meta buffer finds tensors by pointer, so the save roundtrip of a tensor-mode model crashed before.
- `ggml-backend-meta.cpp`:
  - The device name was always `Meta()`: the constructor read a vector it had moved from.
  - It logs once when the backend all-reduce refuses a tensor and the slow fallback is used.
- `ggml-rpc.cpp`: rank 0 now listens for its peer on the address its server was started with (`-H`), not on `0.0.0.0`.
- Docs: `docs/tensor-parallel.md` (how to run it on two Macs), `docs/multi-gpu.md`, `tools/rpc/README.md`.

Tests (section 8):
- Test 1:
  - `GGML_METAL_DEVICES=2 test-llama-archs -a 'qwen35|qwen35moe'` adds the rows `Meta` (even) and `Meta 3:1`. All pass.
  - The NMSE against the CPU is 2.1e-07, the same as one Metal device.
  - The save roundtrip and the mixed batch pass.
  - The logs confirm the split: the KV cache of the meta device is half of one device's.
- Test 2:
  - `test-llama-archs --rpc` starts two RPC servers on 127.0.0.1 threads and runs `-sm tensor` over `RPC0,RPC1` at 1:1 and 3:1.
  - It requires that the servers connect to each other and that no sum falls back to the client.
  - The NMSE against one local GPU: `qwen35` 1.0e-15, `qwen35moe` 8.8e-13.
  - Mutation check: with the server's sum skipped, all four cases fail (NMSE 2.6e-03 to 2.7e-03), and 1:1 and 3:1 then give different logits, so the two splits really differ.
- Test 6: `test-arg-expert-split` checks that `-sm tensor` parses and that `-es`, `-lhcd` and `-kvd` fail with it.
- The tests reused before also pass:
  - `test-llama-archs` on one device.
  - `GGML_METAL_DEVICES=2` for the LongHaul archs.
  - The LongHaul RPC cases.
  - `ctest -L "main|behavior|rpc"` passes, except 5 tests that already failed on this machine: the Python tests (no numpy, torch, jinja2) and the missing `ggml-vocab-qwen35.gguf`.

Not done: the speed of tensor vs layer vs expert split on the laptops. That needs your approval and your laptops.
