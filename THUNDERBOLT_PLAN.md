# Multi-laptop LongHaul over Thunderbolt: plan

Goal: run one model across several Macs connected with Thunderbolt cables. One laptop (the main host) runs `llama-cli` / `llama-server`. The others run `ggml-rpc-server`. Each laptop holds some of the layers and a LongHaul expert cache for those layers, so the combined RAM of all laptops caches experts. This builds on the upstream RPC backend (`ggml-rpc`), which this fork removed in the Apple-only prune.

Constraints: no model is downloaded or run. All checks are code-level tests that run in memory (random tiny models, emulated devices). Commits follow AGENTS.md (`Assisted-by:` trailer, each one approved by you, nothing pushed).

Status: implemented, all in-memory tests pass (see section 12). Nothing is committed: each commit in section 11 waits for your approval.

---

## 1. Facts (checked 2026-11-25)

### 1.1 RPC in this fork

- **RPC was deleted in `63a6a89`** (Phase 4 of LONGHAUL_PLAN.md). The fork base `5fddd0b` still has all of it, and it is the restore source:
  - `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/` (`ggml-rpc.cpp` 3131 lines, `transport.{h,cpp}`, `transport-apple.{h,cpp}`, `log.{h,cpp}`, `CMakeLists.txt`);
  - `tools/rpc/` (`rpc-server.cpp`, `CMakeLists.txt`, `README.md`) and the tests `tests/test-rpc-multi-server.{cpp,sh}`, `tests/test-rpc-unreachable.cpp` (registered from `tools/rpc/CMakeLists.txt`);
  - the glue: `GGML_RPC` in `ggml/CMakeLists.txt`, `ggml_add_backend(RPC)` in `ggml/src/CMakeLists.txt`, the registry entries in `ggml/src/ggml-backend-reg.cpp`, `llama_supports_rpc()` and the RPC device ordering in `src/llama.cpp`, `--rpc` in `common/arg.cpp`, `-rpc` in `llama-bench`.
- **ggml did not change since the fork base.** `ggml.h`, `ggml-backend.h` and `ggml-backend-impl.h` are identical to `5fddd0b`, so `GGML_OP_COUNT` is still 101 and the RPC protocol (10.0.2) matches. The restored files should build unchanged.
- **The fork base already has the recent server refactors** (#32651 typed requests, #32896 one request context). Two RPC PRs are still open upstream:
  - #32925: server-side shape checks in `graph_compute` (an out-of-bounds write through `GET_ROWS`). LongHaul uses `GET_ROWS` on the slot maps, so this one matters. Port it once it merges.
  - #32532: `SUPPORTS_OP`. Today the RPC device answers `true` for every op.

### 1.2 Thunderbolt transport

- **Upstream RPC has two transports.** Plain TCP, and RDMA over Thunderbolt on macOS 26.2+ (`transport-apple.cpp`, linked weakly against `librdma`). RDMA is negotiated in the HELLO handshake and falls back to TCP. `GGML_RPC_NO_RDMA=1` forces TCP.
- **RDMA over Thunderbolt needs Thunderbolt 5** on both ends, plus a one-time `rdma_ctl enable` from macOS Recovery (Apple TN3205).
- **This laptop:** Apple M5, 24 GB, macOS 26.6.2. Its ports report "Up to 40 Gb/s", which is Thunderbolt 4, so it **cannot use RDMA**. It talks TCP over the Thunderbolt Bridge network interface. The SDK has `librdma.tbd`, so the RDMA code still compiles.
- **Effect:** between two Thunderbolt 4 laptops, IP over Thunderbolt is slower than the 40 Gb/s link rate and is likely slower than the internal SSD. The network can become the bottleneck for expert misses. See D5.

### 1.3 LongHaul is already shaped for several devices

- **`src/llama-moe-cache.cpp` already has a `device` struct and per-device groups.** Each layer is cached on the device that `model.dev_layer(il)` names. The budget is split across devices by `-ts`, or by free memory by default (lines 542-640).
- **The RPC device reports `GGML_BACKEND_DEVICE_TYPE_GPU`**, so the cache already accepts it.
- **A hard check blocks more than one device** (lines 527-530): "LongHaul supports one GPU". The reason given is the `inkling2_loop` case, where a loop pass placed on another device than the layer that owns its experts would need the banks of that device.
- **`check_op()` always uses the first group's device** (`devices[groups[0].id]`). That is only correct with one device.
- **The test suite can emulate a second GPU:** `GGML_METAL_DEVICES=2` (and `GGML_CUDA_DEVICES=2`). `ci/run.sh` already runs `test-llama-archs` with 1-4 Metal devices. `test_longhaul` uses the second device only for the error case "two GPUs".

### 1.4 Loop models split badly across devices (upstream issue #33040, PR #33058)

- `inkling2_loop` expands 24 physical layers into 48 cache slots. Slots 24-47 are passes 2-3 of the loop core and alias the weights of physical layers 6-17.
- `load_tensors` splits all 48 slots across devices. With an even split, slots 0-24 (every physical layer) go to device 0, and device 1 gets only KV slots and the output layer. Every later pass then runs its matmuls on one device and its attention on another.
- **Upstream PR #33058 (open, 2026-11-25)** fixes this. It adds `layer_phys(il)` and `n_layer_offload()`, splits the layers that hold weights, and puts each alias slot on the device of its physical layer. It is 6 files, about 40 lines. Its tests ran on two `ggml-rpc-server` instances, which is our exact setup.

### 1.5 How a LongHaul miss travels today

`copy()` (`src/llama-moe-cache.cpp` ~867) runs from the scheduler copy callback for each MoE layer:
1. read the selected expert ids from the device (`tensor_get_async` + `synchronize`);
2. plan hits and misses (CRF/LRU);
3. for each miss, `pread` from the GGUF into a staging buffer (4 I/O threads, 8 MiB each), then `ggml_backend_tensor_set` into the bank;
4. upload the slot map.

On an RPC device every step works unchanged: step 1 is one network round trip, step 3 sends the expert bytes over the link, and the RPC dispatcher keeps the commands in order on one socket. Two problems:
- **The RPC hash cache.** If `ggml-rpc-server -c` is on, `ggml_backend_rpc_buffer_set_tensor` hashes any write over 10 MiB to a `WEIGHTS` buffer, and the server saves it as a cache file. The cache banks are `WEIGHTS` buffers (`llama-moe-cache.cpp:758`). A run of experts over 10 MiB would be hashed on every miss and written to disk on the server.
- **All misses come from the main host's SSD and cross the link**, even for layers on another laptop.

---

## 2. Decisions (made by me, override any of them)

| # | Decision | Why |
|---|---|---|
| D1 | **Restore upstream RPC from `5fddd0b`, unchanged** (backend, Apple RDMA transport, `ggml-rpc-server`, tests, `--rpc`, `llama-bench -rpc`, the device ordering in `src/llama.cpp`). `GGML_RPC` is **on by default on Apple** and off elsewhere (your answer to Q3). | You asked to use the existing llama-rpc. Byte-identical files keep future upstream RPC fixes easy to port. Building RPC starts nothing: `ggml-rpc-server` listens only when you run it, and `--rpc` connects only when you pass it. |
| D2 | **Topology:** one main host plus up to 15 `ggml-rpc-server` laptops (`GGML_RPC_MAX_SERVERS = 16`). Layer split only (`-sm layer`, the default). Every laptop runs a build of the same commit. | `-sm tensor` is removed from this fork, and the cache cannot split one layer's experts across devices. RPC rejects other protocol versions, and both ends must have the same ops and kernels. |
| D3 | **One cache per device, one `-lhc` budget.** Each device caches the experts of its own layers in its own memory. The total `-lhc` is split across devices like the layers (by `-ts`, else by free memory). This is the split the cache already does. There is no per-device size list. | No new CLI semantics. The budget follows the layers, so each device caches the same share of its experts. `-ts` sets both if you want a different balance. |
| D4 | **Port #33058 close to verbatim** for loop placement: `virtual int layer_phys(int il)` on `llama_model`, `n_layer_offload()`, and the split over physical layers. `-ngl` and `-ts` then count physical layers for `inkling2_loop` (25 instead of 49). | Without it, a two-laptop split of Inkling 2 Mini puts every weight on one laptop. Porting the upstream code keeps a later merge clean. |
| D5 | **Misses for remote layers, in two steps.** Phase 3: the main host reads them from its SSD and sends them over the link. This needs no setup on the remote. Phase 4 (in, your answer to Q1): when `ggml-rpc-server --model-dir DIR` holds the same GGUF, the remote reads its own misses from its own SSD. A fingerprint check confirms it is the same file. If the file is missing or different, it falls back to Phase 3 with a warning. | Phase 3 works for any setup. Between Thunderbolt 4 laptops the link is probably slower than the SSD, so Phase 4 keeps expert bytes off the network, and only activations, ids and slot maps cross it. |
| D6 | **The RPC hash cache only handles whole-tensor writes** (`offset == 0 && size == ggml_nbytes(tensor)`). | Model loading writes whole tensors, and cache fills always write part of a bank. One condition keeps fills off the hash path. |
| D7 | **Device order stays upstream's:** RPC devices first, local GPU last. The output layer and sampling run on the main host. | One crossing into the remote layers and one back per token, plus the extra crossings of loop passes (see 5.3). |
| D8 | **Pipeline parallelism stays off** with LongHaul (`llama-context.cpp:428`). | The cache needs the expert ids of a layer before that layer runs, which already serializes the layers. |
| D9 | **Draft models are unchanged.** The docs recommend keeping a draft model local (`-devd MTL0`). | A draft on a remote laptop would pay network round trips for every draft token. |
| D10 | **In-memory tests use emulated Metal devices** (`GGML_METAL_DEVICES=2`) for the multi-device cache. The RPC path is tested by an in-process `ggml-rpc-server` thread on `127.0.0.1`, under its own CTest label `rpc` (your answer to Q2). | The emulated devices test the cache logic with no sockets. The RPC path itself needs a server. A thread in the test process needs no outside service. |
| D11 | Commits follow AGENTS.md: `Assisted-by: Claude Opus 5.5`, each approved by you, nothing pushed. | AGENTS.md rule. |

---

## 3. Phase 1 - Restore RPC (no LongHaul change)

Restore from `5fddd0b` with `git checkout 5fddd0b -- <paths>`, then re-apply only the glue the prune removed.

- **Files restored unchanged:** `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/*`, `tools/rpc/*`, `tests/test-rpc-multi-server.cpp`, `tests/test-rpc-multi-server.sh`, `tests/test-rpc-unreachable.cpp`.
- **`ggml/CMakeLists.txt`:** `option(GGML_RPC "ggml: use RPC" ${GGML_RPC_DEFAULT})`, with `GGML_RPC_DEFAULT` ON when `APPLE` and OFF elsewhere (D1), and `include/ggml-rpc.h` in `GGML_PUBLIC_HEADERS`.
- **`ggml/src/CMakeLists.txt`:** `ggml_add_backend(RPC)`.
- **`ggml/src/ggml-backend-reg.cpp`:** the `#ifdef GGML_USE_RPC` include, `register_backend(ggml_backend_rpc_reg())`, and `ggml_backend_load_best("rpc", ...)`.
- **`tools/CMakeLists.txt`:** `if (GGML_RPC) add_subdirectory(rpc) endif()`.
- **`src/llama.cpp`:**
  - `llama_supports_rpc()` looks up the `RPC` registry again (it returns `false` today).
  - `llama_supports_gpu_offload()` includes RPC.
  - `llama_prepare_model_devices` sorts RPC devices into their own list, puts them in front, and keeps the iGPU rule that RPC devices do not count as discrete GPUs.
- **`common/arg.cpp`, `common/arg.h`:** `add_rpc_devices()`, `--rpc SERVERS` (env `LLAMA_ARG_RPC`), the `-mmdev` sort-order note, and the old TODO comment.
- **`tools/llama-bench/llama-bench.cpp`:** `-rpc`, and `RPC` in the backend column.
- **CI:** the Apple build gets RPC from the new default. Add `ctest -L rpc` to the Apple test step, which runs the loopback cases.
- **Not restored:** the VirtGPU RPC front/back files (VirtGPU stays removed), and `LLAMA_RPC` in `llama_option_depr` (a deprecated alias).

Tests:
- Build with the default (RPC on) and with `-DGGML_RPC=OFF`. Both must build with `LLAMA_FATAL_WARNINGS=ON`.
- `test-arg-parser`: `--rpc` parses. It fails cleanly with "RPC not supported in this build" when RPC is off.
- `test-rpc-unreachable`: connects to `127.0.0.1:1`, where nothing listens, and checks for `nullptr` instead of an abort. No server is started.
- `test-rpc-multi-server` starts two `ggml-rpc-server` processes. It is restored, but its CTest label changes from `main` to `rpc`. With RPC on by default, the `main` label would otherwise start server processes in every `ctest -L main`. I will not run it without your OK (your Q2 answer covers in-process threads, not separate processes).

---

## 4. Phase 2 - LongHaul across several devices (no RPC needed)

### 4.1 Loop placement: port #33058 (`src/llama-model.{h,cpp}`, `src/models/models.h`, `src/llama-context.cpp`, `src/llama-ext.h`, `common/fit.cpp`)

- `llama_model`: `virtual int layer_phys(int il) const { return il; }` and `int n_layer_offload() const` (the largest `layer_phys(il) + 1`).
- `models.h`: `llama_model_inkling2_loop::layer_phys` becomes `override`. The fork already has this function and `n_layer_phys()`.
- `load_tensors` (~`llama-model.cpp:713-736`): `i_gpu_start` and `act_gpu_layers` use `n_layer_offload()`, and each slot uses the device of `layer_phys(il)`. Same for the "offloaded N/M layers" log.
- `llama-context.cpp`: the pipeline-parallel and `full_offload` checks compare against `n_layer_offload()`.
- `llama-ext.h`: `llama_model_n_layer_offload()`. `common/fit.cpp:152`: `hp_ngl = llama_model_n_layer_offload(model)`. The fork has no `llama_model_n_layer_nextn` (no MTP graph), so that part of the upstream line is dropped.
- Models without aliasing do not change: `layer_phys` is the identity.

### 4.2 The cache on several devices (`src/llama-moe-cache.cpp`)

- **Remove the "one GPU" check** (lines 527-530). Replace it with a check that every loop pass is on the device of its owner layer. After 4.1 this always holds, so a failure means a bug or a new aliasing model. It throws and names the layer and both devices.
- **`check_op()`:** use the device of the group that holds `w` (find it through `bindings` / `layer_of`), not `groups[0]`.
- **Staging host buffer type** (~line 737): it is taken from the device of `groups[0]`. That is fine for Metal and RPC (neither has a host buffer type, so plain memory is used). Keep it, and add a short comment that it is shared by all devices.
- **Budget split, logs, `max_ubatch()`, the trace format:** already per device. No change. Check that the trace header still holds for several devices (it writes `device` per group).
- **Error text:** "LongHaul requires a GPU backend" names RPC too.

### 4.3 Tests (`tests/test-llama-archs.cpp`, in memory)

Replace the "two GPUs" error case in `test_longhaul` and `test_longhaul_bailingmoe3` with positive cases on `{gpu, gpu2}`. The logits must match the reference within the existing tolerance:
- default split (by free memory), for `inkling2`, `inkling2_loop`, `qwen35moe` and `bailingmoe3`;
- an uneven split (`tensor_split = {1, 3}`), so the two devices get different budgets;
- `inkling2_loop`: check through `llama_model_dev_layer` (or the debug log) that every alias slot is on the device of its physical layer;
- a file with two expert layouts, with each layout on its own device;
- the smallest cache size, so every device evicts.

They run when `GGML_METAL_DEVICES >= 2` (already in `ci/run.sh`) and are skipped with one device. Locally I run `GGML_METAL_DEVICES=2 test-llama-archs` and `GGML_METAL_DEVICES=1 test-llama-archs`.

---

## 5. Phase 3 - LongHaul over RPC (misses sent from the main host)

### 5.1 Changes

- **`ggml/src/ggml-rpc/ggml-rpc.cpp`, `rpc_use_hash_cache()`:** also require `offset == 0 && size == ggml_nbytes(tensor)` (D6). One line plus a short comment.
- **`src/llama-moe-cache.cpp`:** none beyond Phase 2. The RPC buffer `set_tensor` blocks until the bytes are sent, so reusing the staging buffer stays safe. The existing `set_mtx` keeps one upload at a time on the socket.
- **Startup log:** the per-device cache line already prints the buffer type name (`RPC0[host:port]` or similar), so you can see which laptop holds which layers and how many slots it has.

### 5.2 What crosses the link per remote MoE layer

- one `graph_compute` for the part of the layer before the lookup;
- one `get_tensor` of the expert ids, which is a full round trip (the only blocking one when every expert hits);
- for each miss, the expert bytes (Phase 3 only);
- one `set_tensor` of the slot map, then the next `graph_compute`.

So each remote MoE layer costs about one network round trip when every expert hits. With 24 remote layer passes per token, that adds tens of round trips per token. This is an estimate, not a measurement.

### 5.3 Inkling 2 Mini on two laptops (after D4)

With an even split, laptop A (RPC, first) gets physical layers 0-11 and laptop B (main) gets 12-23 plus the output. A token runs prefix 0-5 and core 6-11 on A, core 12-17 on B, then passes 2 and 3 go A, B, A, B, then the suffix 18-23 on B. That is 6 crossings per token, each carrying one hidden state per token. The two laptops each run 24 of the 48 layer passes.

### 5.4 Tests

Loopback test (approved, Q2): `test-llama-archs` gets a case that starts `ggml_backend_rpc_start_server` on a thread at `127.0.0.1:<free port>`. It exposes the local Metal device, registers it with `ggml_backend_rpc_add_server`, and runs the LongHaul comparison on `{RPC0, MTL0}`, once with the server cache (`-c`) on and once off. It checks that no file appears in the server cache directory during the run. The case is registered as its own CTest entry with the label `rpc` (`test-llama-archs --rpc`), so the default `ctest -L main` does not start it.

`ggml_backend_rpc_start_server` never returns, so the test runs it on a detached thread and the process exits at the end. I will check whether a small stop hook is needed for a clean exit, and ask you before adding one.

---

## 6. Phase 4 - Remote laptops read their own misses

### 6.1 Protocol (`ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`)

- `RPC_PROTO_MINOR_VERSION` 0 -> 1. HELLO sets a new flag, `RPC_DISPATCHER_FLAGS_FILE_READ`, when the server was started with `--model-dir`. The client uses the new commands only when the flag is set, so a new client with an old server falls back to Phase 3.
- **`RPC_CMD_FILE_OPEN`:** the request is `{ basename[256], size, fingerprint }`. The response is `{ file_id }`, or -1 if the server has no file with the same size and fingerprint. The basename is only for logs.
- **`RPC_CMD_SET_TENSORS_FROM_FILE`:** the request is `n` records of `{ rpc_tensor, offset, file_id, file_offset, size }`, with no response. The server checks each record (known `file_id`, range inside the file, `rpc_check_region` inside the tensor). It reads the ranges with `pread` (`F_NOCACHE`) on a small pool of threads (`--io-threads`, default 4), copies them in with `ggml_backend_tensor_set`, and drops the connection on any error, like the other commands.
- **Fingerprint:** fnv-1a 64 over the file size, the first 4 MiB, and 64 blocks of 4 KiB at evenly spaced offsets. One function in `ggml-rpc.cpp` computes it for both the client and the server, so the two sides cannot disagree. It costs about 4.3 MiB of reads per file. It catches different quants, different revisions, and a re-quantization that keeps the same size, which basename and size alone would miss.
- **Public API**, also found through `ggml_backend_reg_get_proc_address` so that `src/` does not link `ggml-rpc`:
  - `int32_t ggml_backend_rpc_dev_open_file(ggml_backend_dev_t dev, const char * path)`;
  - `void ggml_backend_rpc_dev_set_tensors_from_file(ggml_backend_dev_t dev, const struct ggml_backend_rpc_file_read * reads, size_t n)`.
- **Security:** the server reads only files it indexed from `--model-dir` at startup. The client never sends a path. Every range is checked against the file size and the tensor region.

### 6.2 Server (`tools/rpc/rpc-server.cpp`)

- `--model-dir DIR`: index the `*.gguf` files in DIR (not recursive). Split GGUFs work because each split is its own file. Log each file with its size and fingerprint.
- `--io-threads N` (default 4).

### 6.3 Client (`src/llama-model.{h,cpp}`, `src/llama-moe-cache.cpp`)

- `llama_model` keeps the paths of the LongHaul files next to `longhaul_files` (`llama-model.cpp:1127`), because `llama_file` does not store its path. Add an accessor `longhaul_src(const ggml_tensor *) -> {file index, offset}`.
- **Cache constructor:** for each RPC device with the flag, open every LongHaul file its layers use. If all open, the device is marked "remote reads". Otherwise it logs one warning that names the device and the first missing file, and that device uses Phase 3.
- **`copy()`:** on a remote-read device, build one `ggml_backend_rpc_file_read` list for the fills (the same coalesced runs as today) and send it in one call. No staging buffer or I/O pool is used. The next `graph_compute` comes after it on the same socket, so the bank is filled before it is read.
- **Stats:** remote fills are counted as "remote" bytes in `log_stats()`. Their read time is on the server, which logs its own totals at `GGML_RPC_DEBUG=1`.

### 6.4 Tests

- **In memory:** a fingerprint test on temp files. The same content gives the same value, while one changed byte in the head, one in a sampled block, or a different size each give a different value. It goes in `tests/test-rpc-fingerprint.cpp`, built when `GGML_RPC=ON`.
- **Loopback test:** the same LongHaul comparison as 5.4, with the server indexing the test's temp dir. Cases: the matching file (remote reads), a file with one changed byte (falls back to Phase 3, checked through the log callback), and no `--model-dir` (Phase 3).

---

## 7. Phase 5 - Docs

- **`docs/longhaul.md`:** replace "LongHaul supports one GPU" with a "Several laptops" section. It covers:
  - the topology, how the budget splits across devices, `-ts`;
  - Phase 3 vs Phase 4 and when each is used;
  - loop-model placement and what a split does to `-ngl` / `-ts` (D4);
  - a sizing example for Inkling 2 Mini on two laptops. With about 15.5 GB of experts, `-lhc 16G` split across two laptops can hold every expert, so after warm-up no expert is read from disk. This is computed from the model card, not measured.
- **`tools/rpc/README.md`:** add a fork section.
  - Thunderbolt Bridge setup: connect the cable, then in System Settings -> Network -> Thunderbolt Bridge give each Mac a static address (for example `10.0.0.1` and `10.0.0.2`).
  - Bind `ggml-rpc-server -H` to that address, never `0.0.0.0`.
  - Thunderbolt 4 (your laptops) uses TCP over the Thunderbolt Bridge. Thunderbolt 5 + RDMA gets a short note only.
  - Optional: a larger MTU on the Thunderbolt Bridge interface, measured by you.
  - `--model-dir`, and the same-commit rule.
- **`docs/build.md`:** RPC is built by default on Apple. `-DGGML_RPC=OFF` turns it off.
- **`docs/multi-gpu.md`, `README.md`:** a short pointer to the docs above.
- **Example commands:**

```sh
# each remote laptop (same commit)
ggml-rpc-server -H 10.0.0.2 -p 50052 --model-dir ~/models

# main laptop
llama-cli -m ~/models/Inkling-2-Mini-AD-Q4_K.gguf --rpc 10.0.0.2:50052 -lhc 16G -devd MTL0
```

---

## 8. Risks

- **Nothing here runs on real laptops.** The emulated devices share one GPU and the loopback test shares one machine. Real throughput, Thunderbolt Bridge speed and round-trip cost are for you to measure.
- **`supports_op` is always true on RPC** (until #32532). An op the remote Metal cannot run fails at compute time on the server, not at load time. Same commit on both ends keeps the kernels identical.
- **`ggml-rpc-server` has no authentication** and the protocol is not hardened (#32925 is open). Use it only on the Thunderbolt cable between your own Macs.
- **A dropped link aborts the main process.** This is upstream RPC behaviour, and a laptop going to sleep triggers it too.
- **`--fit` does not rebalance a LongHaul split.** It reduces the context or fails, as today with one device (`common/fit.cpp:539`).
- **D4 changes `-ngl` / `-ts` for `inkling2_loop`**, which then count 25 layers, not 49. A hand-tuned `-ts` written for the old count gives a different placement.
- **Phase 4 trusts the fingerprint.** Two files that match in size, the first 4 MiB and all 64 sampled blocks but differ elsewhere would give wrong output silently. This is unlikely for real GGUFs, but it is not a full hash.

---

## 9. Answers (2026-11-25)

1. **Phase 4:** included.
2. **Loopback RPC test:** allowed, as an in-process thread on `127.0.0.1` under the CTest label `rpc`. `test-rpc-multi-server` (separate processes) is not run without your OK.
3. **`GGML_RPC` default:** on for Apple builds, off elsewhere.
4. **Other laptops:** all Thunderbolt 4. The docs and examples use TCP over the Thunderbolt Bridge.

## 10. Files touched

| File | Phase | Change |
|---|---|---|
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/*` | 1, 3, 4 | restore; D6; file commands |
| `ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`, `ggml/src/ggml-backend-reg.cpp` | 1 | RPC glue |
| `tools/rpc/*`, `tools/CMakeLists.txt` | 1, 4 | restore; `--model-dir`, `--io-threads` |
| `tests/test-rpc-*.cpp`, `tests/test-rpc-multi-server.sh` | 1, 4 | restore; fingerprint test |
| `src/llama.cpp` | 1 | `llama_supports_rpc`, device order |
| `common/arg.{h,cpp}`, `tools/llama-bench/llama-bench.cpp` | 1 | `--rpc`, `-rpc` |
| `src/llama-model.{h,cpp}`, `src/models/models.h`, `src/llama-ext.h`, `src/llama-context.cpp`, `common/fit.cpp` | 2, 4 | #33058 port; LongHaul file paths |
| `src/llama-moe-cache.cpp` | 2, 4 | several devices; remote reads |
| `tests/test-llama-archs.cpp` | 2, 3, 4 | two-device cases; optional loopback cases |
| `.github/workflows/build-apple.yml` | 1 | `ctest -L rpc` |
| `docs/longhaul.md`, `tools/rpc/README.md`, `docs/build.md`, `docs/multi-gpu.md`, `README.md` | 5 | docs |

## 11. Commits (each needs your approval)

1. `ggml : restore the RPC backend and ggml-rpc-server`
2. `llama : split looped models by the layers that hold weights` (port of #33058)
3. `llama : LongHaul cache on several devices`
4. `ggml-rpc : use the hash cache for whole tensors only`
5. `ggml-rpc : read LongHaul experts from a local copy on the server`
6. `docs : LongHaul on several laptops`

## 12. Results (2026-11-25)

All phases are implemented. Only code-level tests were run, on this Mac (M5, emulated Metal devices, one RPC server thread on `127.0.0.1`). No model was downloaded, and nothing ran on a second laptop.

| Test | Result |
|---|---|
| `test-llama-archs -s 1` (all archs) | 15/15 passed |
| `GGML_METAL_DEVICES=2 test-llama-archs -s 1 -a "inkling2\|inkling2_loop\|qwen35moe\|bailingmoe3"` | 16/16 passed, two-device cases for every LongHaul arch |
| `test-llama-archs --rpc -s 1` (`ctest -L rpc`) | 12/12 cases passed: `inkling2` and `inkling2_loop`, on RPC only (5 slots, 512 MiB) and RPC + GPU 1/3, experts read by the server and sent from the client |
| `test-rpc-loopback` (`ctest -L rpc`) | 5/5 passed: the hash cache skips partial writes, file open and reads |
| `test-rpc-fingerprint` (`ctest -L main`) | 8/8 passed |
| `ctest -L main` | 24/24 passed |
| `test-arg-parser` | passed |
| `-DGGML_RPC=OFF -DLLAMA_FATAL_WARNINGS=ON` build | builds, no RPC targets; `test-llama-archs --rpc` skips |

Checks that the tests catch the bugs they are for:

- With `layer_phys` disabled, the two-device loop cases fail ("failed to create llama context").
- Without D6, `test-rpc-loopback` fails "partial write not cached".
- With the server reads shifted by 64 bytes, the `--rpc` server cases fail. The NMSE alone (4e-6) did not catch it, so the test also requires the logits of the server reads to equal the logits of the experts sent from the client bit for bit.

Changes from the plan:

- The server binary is `ggml-rpc-server` (its name in `5fddd0b`), not `rpc-server`.
- The `--rpc` test files "a" and "b" differ in `general.name` only: the test weights come from a pool keyed on the tensor names, so another seed gives the same weights. The server copy of "b" has its last byte changed.
- The `--rpc` test does not start the server with `-c`: the tiny test tensors are below the 10 MiB hash threshold, so that check would pass without D6. `test-rpc-loopback` checks D6 directly with 12 MiB writes.
- No stop hook was needed: the detached server thread does not block the exit.
- CI runs `ctest -L rpc -E test-llama-archs-rpc`: the CI Mac has the Paravirtual Metal device, on which CI already skips `test-llama-archs`. `test-rpc-multi-server` therefore runs in CI (it was in upstream CI too), but it was not run here.
- `docs/build.md` says that the split over several CUDA GPUs is not tested: the code path is the same, but only emulated Metal devices were tested.

