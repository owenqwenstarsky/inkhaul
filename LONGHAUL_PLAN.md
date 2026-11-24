# LongHaul plan

Goal: run Inkling 2 Mini on an M5 MacBook (24 GB unified memory) by keeping the routed expert weights on disk and holding only a fixed-size expert cache in RAM (default 512 MiB, set with one flag, always on). Then strip the fork down to Inkling 2 on Apple Silicon.

Status: approved. Phases 1-5 are committed (see 12-16).

Target: [AtomicChat/Inkling-2-Mini-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF). The model is not downloaded or run during this work.

---

## 1. Decisions (made by me, override any of them)

| # | Decision | Why |
|---|---|---|
| D1 | The budget covers **routed expert weights only**. Attention, router, norms, conv taps, embeddings and output stay fully resident (mmap'd and wrapped as Metal buffers, as today). KV cache, recurrent state and compute buffers sit outside the budget. | Inkling2 has no shared experts and no dense FFN layers, so experts are nearly all of the file. Non-expert weights run on every token; streaming them would make every token pay for an SSD read of the whole dense part. |
| D2 | **Reuse `llama_moe_cache`** (`src/llama-moe-cache.cpp`). Change only where misses come from: read them from the GGUF with `pread` (macOS `F_NOCACHE`) instead of copying from memory-mapped host tensors. | The slot maps, eviction policy (CRF classes + LRU), graph wiring and scheduler copy callback already exist and are tested. Only the miss source and the "always on" policy are new. |
| D3 | Expert tensors become **never-touched mapped tensors**: a new loader flag puts them in the lazy-tensor path (CPU buffer type, mapped whatever the load mode, `POSIX_MADV_RANDOM`, left out of prefetch, mlock and Metal wrapping). LongHaul never reads them through the mapping. | Tensors keep a valid `data` pointer and host buffer, so the existing cache code still accepts them. A mapping that is never touched costs address space, not RAM. |
| D4 | **All batch sizes go through the cache**, prompt processing included. The 32-token `max_batch` cutoff is removed. When the cache cannot hold one layer's worth of experts for a full ubatch, `n_ubatch` is clamped automatically (and logged). | No path may fall back to the host experts, since that would page the file into RAM. Each layer then reads each expert it needs at most once per ubatch, so a bigger cache gives faster prefill. |
| D5 | **Remove** the CPU-share path (`--moe-cache-cpu-share`, zero slots, CPU maps) and the large-batch `copy_experts` path. | Both read the host experts from memory, which defeats the RAM cap. Once LongHaul is always on they are dead code. |
| D6 | New flag **`--longhaul-cache SIZE`** (`-lhc`, env `LLAMA_ARG_LONGHAUL_CACHE`). It accepts `512M`, `1G`, `2G`, `1536M`, or a bare number in MiB. Default `512M`. No value disables it. It replaces `--moe-cache-mib`, `--moe-cache-cpu-share`, `-cmoe/--cpu-moe`, `-ncmoe/--n-cpu-moe` and their draft variants. | You asked for one argument with sane size units, always on. The removed flags only control expert placement, which LongHaul now owns. |
| D7 | In the public C API the field is renamed `moe_cache_size` -> `longhaul_cache_size`, default 512 MiB. Internal class and file names (`llama_moe_cache`) stay. | A clear public name, with minimal churn inside. |
| D8 | The library rejects `longhaul_cache_size == 0` for MoE models. CLI, server and common never pass 0. The one exception is a test-only path that keeps a fully resident reference model for comparisons (see 6.1). | "Always on" is enforced in the library, not just in the CLI. Tests still need a ground truth. |
| D9 | The cache budget is **per context**. Each `llama_context` on the model gets its own cache (server slots share one context, so one cache). | The cache lives in the context today. A shared cross-context cache would be a new subsystem. |
| D10 | Requires the Metal backend. With no GPU device (for example `-ngl 0` or `--device none`), context creation fails with a clear error. | The cache relies on the scheduler copying a host slot map to the device. With everything on the CPU that copy, and so the callback, never happens. |
| D11 | **Keep `inkling2_loop`**, and dedupe its aliased loop layers in the cache so one physical expert is cached once. | Mini is an `inkling2_loop` model (12 core layers run 3 times). Without the dedupe, the same expert would fill up to 3 slots. |
| D12 | **Order: LongHaul first, then pruning.** | The current test suite (synthetic models, `test-llama-archs`) only works on the intact tree, and many tests build LLaMA models. Pruning first would break the safety net before the feature exists. |
| D13 | **Pruning scope:** keep only `inkling2`, `inkling2_loop`, `dflash` limited to its DSpark variant (the Mini repo ships `dspark-Inkling-2-Mini-Q8_0.gguf` as a draft model), the `clip` stub (needed so `llama-quantize` can open mmproj files) and the `inkling2v` vision projector. Keep backends CPU + Metal + BLAS (Accelerate). Remove the other GPU backends, non-arm CPU arch code, non-macOS build targets, CI for other platforms, and tools/examples that only serve other models or platforms. Generic infrastructure stays (jinja engine, PEG/auto parser, server, bench, quantize, imatrix, perplexity, gguf tools). | This is what "focus the fork on Inkling 2 Mini + LongHaul" means in practice, without breaking shared machinery. Per-arch `#if` blocks inside shared ggml-cpu headers are left alone (editing them carries high risk for little gain). |
| D14 | Commits follow AGENTS.md: `Assisted-by: Claude` trailer, not `Co-authored-by`. Every commit needs your explicit approval. Nothing is pushed. | AGENTS.md rule. |

---

## 2. Current state (what exists today)

- **Model:** `src/models/inkling2.cpp` and `src/models/inkling2-loop.cpp`.
  - Each layer has `ffn_gate_up_exps` `{n_embd, 2*n_ff_exp, n_expert}` and `ffn_down_exps` `{n_ff_exp, n_embd, n_expert}`, plus a router, an optional score-correction bias and SiTU activation.
  - Memory is `llama_memory_hybrid_iswa` (KV with SWA + recurrent conv state).
  - The loop variant aliases physical layers into extra cache slots: `post_arch_tensors` copies `layers[phys]` into `layers[il]`.
  - Conversion lives in `conversion/inkling2.py`. Packed NVFP4/MXFP8 experts are dequantized there and quantized again when the GGUF is written.
- **MoE cache:** `src/llama-moe-cache.{h,cpp}` (1,054 lines).
  - Groups the layers that share an expert layout. Each group has banks (one device tensor per expert tensor, `n_slots` experts each) and one CRF/LRU policy across all its layers.
  - `build_lookup()` adds `get_rows(slot_map, ids)` per layer. The slot map is a host weight, so the scheduler starts a new split there and calls `llama_context::sched_copy_experts`, which calls `llama_moe_cache::copy()`.
  - `copy()` reads the selected ids, runs `plan()` and uploads misses with `ggml_backend_tensor_set_async(bank, w->data + e*nb2, ...)` (line 846). That `w->data` read is the part LongHaul replaces.
  - Constraints today: only layers whose experts sit entirely in host buffers are cached; ubatches over 32 tokens bypass the cache (`get_slot_map`, `max_batch`); a GPU backend is required.
- **Context wiring:** `src/llama-context.cpp`.
  - Cache creation is at ~line 618.
  - `graph_max_nodes` has a cache allowance at ~2915.
  - The copy callback is at ~3143. It also handles the large-batch expert upload and has an `INKLING2_LOOP` bail-out at ~3165.
  - Pipeline parallelism is off when the cache is on.
- **Graph:** `src/llama-graph.cpp` `build_moe_ffn` (~2547, ~2713) and `build_lora_mm_id` (~1925) swap `w` for the cache view when slots are present.
- **Loader:**
  - `src/llama-model-loader.cpp` has a lazy-tensor mechanism (`TENSOR_READ_LAZY`, `lazy_read`). Lazy tensors live in their own context on the CPU buffer type, are mapped whatever the load mode, get `POSIX_MADV_RANDOM`, and are skipped by prefetch and mlock. `src/llama-model.cpp:2204-2216` already keeps such regions out of Metal-wrapped buffers.
  - Tensor file positions are in `llama_tensor_weight{idx, offs}`.
  - `llama_file` (`src/llama-mmap.{h,cpp}`) has only `seek` + `read_raw`, which are not thread-safe. It has no positional read and no macOS `F_NOCACHE`; `DIRECT_IO` on macOS falls back to buffered reads.
- **Metal:** with unified memory the default buffer type is "shared". Its `is_host()` returns **false**, so the cache cannot `pread` straight into `bank->data` through the generic API. Writes go through `ggml_backend_tensor_set` (a memcpy for shared buffers).
- **CLI:** `common/arg.cpp:2729-2770` has `-ot`, `-cmoe`, `-ncmoe`, `--moe-cache-mib` and `--moe-cache-cpu-share`. `common/common.cpp:1779` copies them into the context params. `common/speculative.cpp:3067` forces `moe_cache_size = 0` for draft contexts.
- **Tests:** `tests/test-llama-archs.cpp` already builds random tiny models per arch, round-trips them through a GGUF file, and has device configs with `moe_cache_size` (~lines 898-910, 1337-1340, 1472, 1549). `tests/test-llama-mmap.cpp` and `tests/test-arg-parser.cpp` exist.
- **Docs and scripts:** `docs/moe-cache.md` and `scripts/moe-cache-tune.py` (tunes size + CPU share through the server).

---

## 3. Phase 0 - Facts and baseline (before any code)

1. **Inkling 2 Mini specs** (from the Hugging Face model card and file listing, no weights downloaded):
   - Arch `inkling2_loop`: 24 physical layers, the 12 core layers run 3 times, so a token makes 48 layer passes.
   - 128 routed experts per layer, 8 used per token, expert FFN width 1152. 26.4B parameters, 3.4B active.
   - The experts are about 94% of the weights. `Inkling-2-Mini-AD-Q4_K.gguf` is 16.5 GB, so an expert (gate_up + down) is about 5 MB and the resident part is about 1 GB.
   - The repo also ships a DSpark draft model (`dspark-Inkling-2-Mini-Q8_0.gguf`). DSpark therefore stays during pruning (D13, Phase 3).
   - The GGUFs are already converted, so no converter work is needed for Phase 1.
2. **Pick the GGUF quant** for the experts (for example Q4_K) and confirm Metal supports `MUL_MAT_ID` for it. LongHaul fails at init otherwise (see 4.4).
3. **Work out expectations.** With `E` = bytes per expert (gate_up + down), `L` = MoE layers, `k` = experts per token and `h` = hit rate:
   - Disk read per generated token is about `L * k * (1 - h) * E`.
   - Tokens per second are at most `SSD_bandwidth / that`.
   - Slots at 512 MiB are about `512 MiB / E`.

   With Mini's numbers (`L` = 48 passes, `k` = 8, `E` ~ 5 MB): about 1.9 GB per token with no hits, and about 100 slots at 512 MiB. A 512 MiB cache cannot hold all 128 experts of a layer, so `n_ubatch` is clamped to 12. From 1 GiB up (>= 128 slots) there is no clamp. `docs/longhaul.md` has the table. These are estimates from the card, not measurements.
4. **Optional baseline, run by you** (I won't run it without your OK): `llama-cli -m mini.gguf --cpu-moe --moe-cache-mib 512`. Record tokens/s and how memory grows in Activity Monitor. This gives the "before" number and confirms the page-cache growth.

---

## 4. Phase 1 - LongHaul core

### 4.1 Positional, uncached file reads (`src/llama-mmap.h`, `src/llama-mmap.cpp`)

- Add `void llama_file::read_raw_at(void * ptr, size_t len, size_t offset) const`.
  - POSIX: a `pread` loop that handles short reads and `EINTR`, and throws on error or EOF.
  - Win32: `ReadFile` with an `OVERLAPPED` offset. This keeps the class whole, even though the fork targets macOS.
  - Thread-safe, because it does not touch the file position.
- Add `void llama_file::set_no_cache()`.
  - Apple: `fcntl(fd, F_NOCACHE, 1)`.
  - Linux: `posix_fadvise(..., POSIX_FADV_DONTNEED)` after each read, best effort.
  - Other platforms: no-op.
- Test: extend `tests/test-llama-mmap.cpp` to write a temp file and check `read_raw_at` at several offsets, including a read that spans the end of the file (must throw).

### 4.2 Expert tensors stay on disk (`src/llama-model-loader.{h,cpp}`, `src/llama-model.{h,cpp}`, `src/models/inkling2.cpp`)

- Add `TENSOR_LONGHAUL = 1 << 6` to `llama_model_loader` and mirror it in `llama_model_base` (next to `TENSOR_READ_LAZY`, `llama-model.h:1121`, `llama-model.cpp:4256`).
- In `create_tensor` (`llama-model-loader.cpp` ~1437): `TENSOR_LONGHAUL` calls `lazy.add(..., force = true)`. Force skips the lazy-mode and 4 GiB size checks and also records the tensor in a separate `longhaul` set.
  - Lazy tensors already get the CPU buffer type, a dedicated context, mapping whatever the load mode, `MADV_RANDOM` and no prefetch or mlock. This needs no new loading path.
  - If mmap is unsupported, `lazy.add` would load the tensor in full. For `TENSOR_LONGHAUL` that becomes a hard error.
- Skip `check_tensors` validation for LongHaul tensors (`load_all_data` ~1810), since validation would read the whole tensor.
- Record each LongHaul tensor's source on the model: `struct llama_longhaul_src { uint16_t file; size_t offs; }` in `std::unordered_map<const ggml_tensor *, llama_longhaul_src>`, filled from `llama_tensor_weight`.
- Keep the files open: move `ml.files` into `llama_model::impl` (as `longhaul_files`) when any LongHaul tensor exists, and call `set_no_cache()` on each. Add a const accessor `llama_model::longhaul_read(const ggml_tensor * t, void * dst, size_t offset_in_tensor, size_t len) const`.
- `src/models/inkling2.cpp:155-156`: create `ffn_gate_up_exps` and `ffn_down_exps` with `TENSOR_LONGHAUL`.
  - The loop variant inherits this through `llama_model_inkling2::load_arch_tensors`.
  - The router `ffn_gate_inp` and `ffn_exp_probs_b` stay resident.
- `-ot` overrides that match LongHaul tensors are ignored, with one warning naming the pattern.
- Log at load: `longhaul: N expert tensors, X GiB kept on disk`.

### 4.3 Misses read from disk (`src/llama-moe-cache.{h,cpp}`)

- **Constructor:**
  - Keep the GPU requirement (D10), with an error that names the cause.
  - Cache a layer only if all of its expert tensors are LongHaul tensors (replaces `llama_moe_cache_is_host_weight`).
  - Throw if any MoE layer is left uncached. Today it warns and skips, but in LongHaul an uncached layer would read the mapping.
  - Throw if the budget is below the minimum (`n_expert_used` slots for the largest group). The message names the minimum size.
- **Staging:** one host buffer sized to the largest single coalesced fill. It is capped (for example 64 MiB) and longer runs are split. Reported in the memory breakdown as "LongHaul staging", outside the budget.
- **`copy()` (line ~835):** replace `ggml_backend_tensor_set_async(bank, w->data + ...)` with: for each run of consecutive experts going to consecutive slots, `model.longhaul_read(w, staging, e*nb2, n*nb2)`, then `ggml_backend_tensor_set(bank, staging, slot*nb2, n*nb2)`.
  - Use the synchronous `set` because the staging buffer is reused.
  - Ordering stays safe because `copy()` already synchronizes the backend before it changes slots.
- **`get_slot_map()`:** drop `n_tokens > max_batch`. Keep the capacity check, since 4.4 makes sure it always passes.
- **`build_lookup()`:** in LongHaul, every case that returns `{}` (no slot map, no binding, op not supported on the device) becomes a `GGML_ABORT` with the layer and reason. A silent fallback would read the mapping.
- **Remove:** `cpu_share`, `n_zero`, `slot_map_cpu`, `cpu_map`, `cpu_experts`, `cpu_acc`, the `n_cpu` lambda, the second path in `build_experts()`, `copy_experts()`, and `stats_copy`. `plan()` loses its CPU arguments.
- **Keep:** the `LLAMA_MOE_CACHE_TRACE` routing trace, which is useful for picking a size offline. Drop the CPU-share fields from its header.
- **Stats:** add read time and MiB/s to `log_stats()` so slow runs can be split into I/O time and compute time.

### 4.4 Context policy (`include/llama.h`, `src/llama-cparams.h`, `src/llama-context.cpp`)

- `llama.h`: rename `moe_cache_size` to `longhaul_cache_size` (comment: expert cache in bytes for LongHaul, default 512 MiB) and remove `moe_cache_cpu_share`. `llama_context_default_params()` (~4435) sets `512ull*1024*1024`.
- Constructor (~618):
  - If the model has LongHaul tensors, the cache size must be > 0, otherwise throw (D8).
  - Create the cache after backend setup, as today.
  - **Clamp `n_ubatch`** so every layer's worst case fits: `n_ubatch <= floor(n_slots_group / n_expert_used)` whenever `n_slots_group < n_expert` of that group. Log the clamp and suggest a bigger cache.
  - This needs the slot count before `sched_reserve()`, so add `llama_moe_cache::max_ubatch()`.
- `sched_copy_experts` (~3143): keep only the `moe_cache->copy()` branch for LongHaul models. The large-batch upload code is still needed for non-LongHaul tensors. After pruning (Phase 3) nothing else uses it, so remove it there, along with the `INKLING2_LOOP` bail-out.
- Check that `--fit` / `fit-params` (no_alloc pass) still works. The cache already attaches dummy buffers when `no_alloc` is set.
- `common/speculative.cpp:3067`: remove `moe_cache_size = 0`. A draft context of a LongHaul model gets its own cache (D9).

### 4.5 Loop variant dedupe (D11) (`src/llama-moe-cache.cpp`)

- When building groups, map each layer to a canonical layer: the first layer whose expert tensor pointers match.
- Aliased layers keep their own slot map tensor, because the scheduler copies per split, but `copy()` plans on the canonical layer's LRU keys and slot-map data, then uploads that data into the aliased layer's destination.
- Test with a random `inkling2_loop` model: the hit count of the second pass equals the misses of the first (the same experts are reused, not cached twice).

### 4.6 CLI (`common/arg.cpp`, `common/common.{h,cpp}`)

- Add `--longhaul-cache SIZE` as in D6. Parsing: case-insensitive suffixes `K`/`M`/`G` (optionally followed by `iB` or `B`), base 1024, bare number = MiB. Reject 0, negatives and overflow.
- Put the parser in `common` as `common_parse_size_mib()` if no existing helper fits (I found none).
- `common.h`: `size_t longhaul_cache_size = 512ull*1024*1024;` (replaces the two `moe_cache_*` fields). `common.cpp:1779` copies it.
- Remove `-cmoe`, `-ncmoe`, `--moe-cache-mib`, `--moe-cache-cpu-share` and the draft variants (`arg.cpp` ~2735-2770 and ~4185-4200).
- Update the help text in `tools/cli/README.md`, `tools/completion/README.md` and `tools/server/README.md`.
- Test: `tests/test-arg-parser.cpp` cases for `512M`, `1G`, `1536`, `2GiB`, `0` (rejected) and `abc` (rejected).

### 4.7 Server and bench

- `tools/server`: no code change expected (it uses common params). Check that `/props` or the startup log shows the cache size.
- `tools/llama-bench`: it builds its own context params. Add a `-lhc` column so speed can be measured at different cache sizes. This is the main tuning tool.

---

## 5. Phase 2 - Performance (after Phase 1 is correct; each step measured with llama-bench)

1. **Parallel reads.** Within one `copy()`, issue the run reads on a small persistent thread pool (default 4, env `LLAMA_LONGHAUL_IO_THREADS`). Each thread gets its own staging slice; the cache waits for all of them before the slot map upload. A queue depth above 1 matters on Apple SSDs. Keep it only if it measurably helps.
2. **Direct reads into the Metal shared buffer**, skipping the staging copy. This needs a small ggml-metal accessor, because `is_host()` is false. Keep it only if the memcpy shows up in a profile, which it likely won't next to the SSD.
3. **Tuning the eviction policy** with the existing trace: replay traces from real chats at 512M, 1G and 2G, and check whether the CRF half-life of 128 tokens suits Inkling's routing. Constants change only with data behind them.
4. **Prefill:** report prompt tokens/s at 512M vs 1G vs 2G, and document the recommendation (prefill is the part that benefits most from a bigger cache).

---

## 6. Tests (code-level only, no services, per your standing instructions)

### 6.1 New and changed tests

- **`test-llama-archs` LongHaul configs.** For random `inkling2` and `inkling2_loop` models round-tripped through a temp GGUF:
  - (a) A tiny cache at the minimum slot count, which forces eviction on almost every layer.
  - (b) A cache that holds all experts.
  - (c) A resident reference (the test-only path from D8) computed the old way.

  Logits must match: (a) vs (b) exactly, and (b) vs (c) within the existing tolerance. Cover decode (1 token), small batches, and a prompt larger than the clamped `n_ubatch`. This is the main correctness test for eviction, reads, slot maps and the ubatch clamp.
- **`test-llama-mmap`:** `read_raw_at` (4.1).
- **`test-arg-parser`:** size parsing (4.6).
- **Failure cases:** a cache below the minimum gives a clear error; `longhaul_cache_size = 0` with a MoE model gives an error; no GPU device gives an error.
- **RAM check (Mac only, in the same test binary):** after (a) runs N tokens, the process resident size grows by less than the cache + staging + a margin, and not by the size of the expert data. Uses `task_info` on macOS; skipped elsewhere.

### 6.2 Run before every commit

`cmake --build` for the affected targets, then `ctest -L main` plus the tests above. On the Mac, `test-llama-archs` uses Metal, so it exercises the real path.

---

## 7. Phase 3 - Remove the other architectures

In order, each step building and passing the tests before the next.

1. **Move tests off other archs first:**
   - Switch `test-fp8`, `test-generate-fp8-models`, `test-speculative-rejection-sampling`, `test-llama-context-model` and `test-thread-safety` from LLaMA / stories15M to generated random Inkling2 models.
   - Trim `test-llama-archs` to Inkling2, Inkling2-loop and CLIP.
   - Trim `test-chat` to the Inkling templates plus the generic cases.
   - Trim `test-mtmd-impl` to inkling2v.
   - Drop `tests/snapshots/` entries for other models.
   - Remove the 19 `models/ggml-vocab-*.gguf` files (74 MB) and their tokenizer test registrations.
   - Add an Inkling tokenizer test. It needs a vocab-only GGUF made with `convert_hf_to_gguf.py --vocab-only` from the Mini tokenizer, which you would provide.
2. **`src/models/`:** delete the other model files. Keep `inkling2.cpp`, `inkling2-loop.cpp`, `clip.cpp`, and `dflash.cpp` + `dspark.cpp` trimmed to the DSpark variant. Shrink `models.h` to those structs.
   - Before trimming `dflash`, read the drafter's GGUF header metadata from the Hub API (no weights) to confirm its `general.architecture`, its variant keys and which backbone it uses. Keep that variant and backbone; drop DFlash, DFlash2, LiliCorr and the DeepSeek-V4/V4.1/Mistral Large 4 backbones only if the drafter does not use them.
3. **`src/llama-arch.{h,cpp}`:**
   - Arch enum and names: keep INKLING2, INKLING2_LOOP, DFLASH, CLIP and UNKNOWN.
   - Prune `llm_kv` and `llm_tensor` (with their name and info tables) to the keys and tensors Inkling reads, plus the generic ones `llama_model_base::load_hparams` reads.
   - Simplify the predicate switches.
4. **`src/llama-model.{h,cpp}`:**
   - Prune `llama_model_mapping`, `llm_type` (keep UNKNOWN; `llama-quant.cpp` uses `LLM_TYPE_70B`, so remove those heuristics), `print_info`, the `create_memory` branches and `llama_model_rope_type`.
   - Remove the tensor-split meta-device code (~650 lines, which Inkling rejects) and the Prism Hadamard runtime (LLaMA/Qwen only).
   - Prune `llama_layer` to the Inkling fields plus any that generic helpers touch.
5. **`src/llama-hparams.{h,cpp}`:** remove arch-only fields and branches. Keep the Inkling branch of `n_embd_r()`.
6. **Memory:** delete `llama-kv-cache-dsa`, `-dsa-iswa`, `-dsv4`, `-msa`, `llama-memory-hybrid` and `llama-memory-hybrid-idx`. Keep `llama-memory-hybrid-iswa`, `kv-cache-iswa`, `kv-cache`, `kv-cells`, `memory-recurrent` and `memory`. Remove `rs_replay` and the PLE conv state.
7. **`src/llama-graph.{h,cpp}`:** delete the msa, dsa, dsv4, cross-attention/T5, mem_hybrid(_k), non-rel hybrid_iswa, embd_h and rwkv inputs and builders. Remove the arch checks inside `build_moe_ffn`.
8. **`src/llama-context.cpp`:** keep the `ctx_other` handling DSpark needs; remove the EAGLE3/Gemma4-assistant/MTP parts, T5, DeepSeek/Kimi/GROK special cases, and the large-batch expert upload in `sched_copy_experts`.
9. **Public API (`include/llama.h`, `src/llama-ext.h`):** keep the draft-model API DSpark uses; remove the MTP/nextn functions. Keep LoRA (Inkling uses `build_lora_mm`). Remove the control-vector API (Inkling never calls `build_cvec`).
10. **`src/llama-vocab.cpp`:** keep BPE plus `no_vocab`. Remove SPM, WPM, UGM, RWKV, PLaMo and HybridDNA tokenizers and every pre-tokenizer except the `inkling2` entry. That entry maps to the GPT4O regex, so keep the GPT4O regex.
11. **`src/llama-chat.cpp`:** remove the legacy templates. `llama_chat_apply_template` stays as an API and returns -1 for anything that isn't jinja-based (common uses the jinja path for Inkling).
12. **`src/llama-quant.cpp`:** remove other arch heuristics and keep Inkling's exclusions.
13. **`common/`:**
    - Delete the 21 other specialized parsers (`common/parsers/*`, `sources.cmake`) and their detection checks in `chat.cpp`. Keep the generic autoparser and `inkling.cpp`.
    - Remove the ASR/TTS helpers and `chat-tts-norm.cpp`.
    - `speculative.cpp`: keep the n-gram, lookup and DSpark drafts; remove MTP, EAGLE3 and the other DFlash variants.
14. **`tools/mtmd`:** keep `inkling2v.cpp` and the helpers it uses (`build_vit`, `build_rope_2d`, patch-merge, `build_ffn`, position-embedding resize).
    - Delete the other 71 encoders, the audio and generation/TTS helpers, the ffmpeg video helper and `legacy-models/`.
    - Prune the `projector_type` enum and switches.
    - Remove the internal QWEN3, TINY_VOCA and GRANITE sub-model loads in `mtmd.cpp:700-800`.
15. **Python:**
    - `conversion/`: keep `base.py`, `inkling2.py` and `__init__.py` (trimmed maps).
    - Trim `convert_hf_to_gguf.py` (Mistral, mtp, ple; keep the dspark path for Inkling drafters) and `convert_hf_to_gguf_update.py` (Inkling entry only).
    - `gguf-py`: `constants.py` keeps the INKLING2, INKLING2_LOOP and MMPROJ/inkling2v tensor lists and the keys they use.
    - `tensor_mapping.py`: **prune by checking what Inkling's conversion actually maps, not by comment**. Several generic entries Inkling needs have no `# inkling2` tag. The check: convert a sliced Mini (`skills/slice-model`) before and after, and diff the tensor names and the resulting GGUF.
    - Trim `test_conversion_fp8.py` to Inkling.
16. **Templates and docs:** keep the two Inkling jinja templates plus the generic ones the server tests use. Delete per-model docs. Update `docs/development/HOWTO-add-model.md` or remove it.

---

## 8. Phase 4 - Platform focus (Apple Silicon only)

- **ggml backends:**
  - Delete cuda, hip, musa, vulkan, sycl, opencl, cann, webgpu, zdnn, zendnn, hexagon, openvino, et, virtgpu and rpc.
  - Clean up `ggml/CMakeLists.txt` (options, public headers), `ggml/src/CMakeLists.txt` (`ggml_add_backend` calls; keep CPU variants `apple_m*`) and `ggml-backend-reg.cpp` (includes, registration, DL name list).
  - Remove the RPC string lookups in `src/llama.cpp:115,232`, `common/arg.cpp:1192-1202` and `llama-bench`.
- **ggml-cpu:**
  - Delete the `arch/` subdirectories except `arm/`, plus `spacemit/` and `amx/` (fix the `#include` at `ggml-cpu.cpp:7`).
  - Keep `kleidiai/`: Arm, optional, and worth benchmarking on M5.
  - Keep `llamafile/` and `tiled/` (used on arm64).
  - Leave the shared headers (`simd-mappings.h`, `vec.h`, `arch-fallback.h`) unchanged.
- **Tools and examples:**
  - Delete `tools/rpc`, `connect`, `cvector-generator` and `tts`.
  - Delete `examples/sycl`, `llama.android`, `llama.swiftui`, `batched.swift`, `convert-llama2c-to-ggml`, `diffusion`, `llama-eval`, `model-conversion`, `simple-cmake-pkg`, `test-cmake`, `deprecation-warning` and `pocs/vdot`.
  - Update `app/CMakeLists.txt` if a linked tool is removed (none of the linked ones are).
- **Build and CI:**
  - `CMakePresets.json`: arm64-apple presets only.
  - `build-xcframework.sh`: macOS slice only.
  - Remove the nix options for other backends, `.devops` Dockerfiles, other `cmake/` toolchains and `FindNCCL`.
  - `.github/workflows`: keep code-style, editorconfig, python-lint/type-check, build-apple and the UI workflows if you use the web UI. Delete the rest.
  - `ci/run.sh`: Metal and CPU only.
- **Tests:** remove the backend-name branches in `test-backend-ops`, `test-llama-archs` and `test-backend-sampler`.

---

## 9. Phase 5 - Docs

- Replace `docs/moe-cache.md` with `docs/longhaul.md`. It covers what is cached, the flag, how to size it (the formula from Phase 0 with Mini's real numbers), the prefill vs cache-size trade-off, reading the stats line, and the trace for offline sizing.
- Delete `scripts/moe-cache-tune.py` (it tunes the removed CPU share), or rewrite it later to sweep `-lhc` with llama-bench.
- README: a short fork notice and a quick start for Inkling 2 Mini on a Mac.

---

## 10. Risks and open items

- **Mini's expert types.** The AD-Q4_K file may give some layers' experts another type. LongHaul checks at startup that Metal supports `MUL_MAT_ID` for each type and splits the budget into groups by layout. Not verified on the real file, since it is not downloaded.
- **Speed is bound by the SSD.** At 512 MiB the hit rate may be low. Generation could run at single-digit tokens/s and prefill will be slow. The design keeps RAM fixed; speed comes from cache size and the Phase 2 work.
- **Upstream merges** become hard after Phases 3 and 4. Inkling and Metal fixes from upstream would need manual porting. Phase 1 alone stays easy to rebase, which is another reason it comes first.
- **macOS memory accounting:** `F_NOCACHE` stops our reads from filling the unified buffer cache, but the resident non-expert weights (mmap'd) still count as file-backed memory. The RAM test in 6.1 measures the real footprint.
- **The test-only resident reference (D8)** is the one place where "always on" is relaxed. It is reachable only through a test hook and is never exposed in common or the CLI.
- **Per-context budget (D9):** a server with a draft context on the same model uses two caches. Documented, not shared.

---

## 11. Files touched by Phase 1 (summary)

| File | Change |
|---|---|
| `src/llama-mmap.h/.cpp` | `read_raw_at`, `set_no_cache` |
| `src/llama-model-loader.h/.cpp` | `TENSOR_LONGHAUL`, forced lazy add, skip validation, record sources |
| `src/llama-model.h/.cpp` | flag mirror, keep files open, `longhaul_read`, source map |
| `src/models/inkling2.cpp` | experts created with `TENSOR_LONGHAUL` |
| `src/llama-moe-cache.h/.cpp` | disk misses, staging, no batch cutoff, hard errors, remove CPU share and `copy_experts`, loop dedupe, `max_ubatch()`, I/O stats |
| `include/llama.h` | `longhaul_cache_size`, remove `moe_cache_cpu_share` |
| `src/llama-cparams.h`, `src/llama-context.cpp` | always-on check, ubatch clamp, callback cleanup, defaults |
| `src/llama-graph.cpp` | none expected beyond what the cache API change requires |
| `common/arg.cpp`, `common/common.h/.cpp`, `common/speculative.cpp` | new flag and size parser, remove old flags, draft context |
| `tools/llama-bench/llama-bench.cpp` | `-lhc` |
| `tests/test-llama-archs.cpp`, `tests/test-llama-mmap.cpp`, `tests/test-arg-parser.cpp` | tests from 6.1 |
| `tools/*/README.md`, `docs/moe-cache.md` | flag docs |

---

## 12. Phase 1 status

Implemented as planned (D1-D11), with these changes from the text above:

- The CPU-share path and `copy_experts` are removed from the cache entirely, not only for LongHaul models. `sched_copy_experts` keeps only the `moe_cache->copy()` branch.
- The test-only resident reference (D8) is enabled with the `LLAMA_LONGHAUL_RESIDENT` environment variable plus a cache size of 0. In-memory models (`llama_model_init_from_user`) never have LongHaul tensors, so they serve as the reference too.
- `llama_model_save_with_meta` (`src/llama-ext.h`, test-only) writes a model plus metadata to a GGUF, so tests round-trip random `inkling2`/`inkling2_loop` models through a file.
- `docs/moe-cache.md` and `scripts/moe-cache-tune.py` are removed; `docs/longhaul.md` replaces them.

Verified on Metal (code-level tests only, no real model):

- `test-llama-archs`: LongHaul matches the resident reference (NMSE <= 6e-7) for `inkling2` and `inkling2_loop` with 2, 5, 16 slots and 512 MiB; the ubatch clamp gives 1, 2 and 64 as expected; a 1-slot cache fails at init; loop passes hit the slots of the first pass. The full all-arch run passes.
- `test-llama-mmap` (`read_raw_at`), `test-arg-parser` (size parser), `ctest -L main` and the other non-model suites pass. `test-tokenizers-ggml-vocabs` fails only because `models/ggml-vocabs` is an unfetched git-lfs checkout here.

## 13. Phase 2 status

- Item 1 (parallel reads) is implemented: a persistent pool in `llama-moe-cache.cpp`, default 4 threads, `LLAMA_LONGHAUL_IO_THREADS` (1-64), one 8 MiB staging buffer per thread (32 MiB in total, as before). The uploads to the bank are serialized with a mutex. Tested for equal logits with 1, 4 and 16 threads, and the pool alone under ThreadSanitizer. **Not measured**: the default of 4 comes from the SSD's queue depth, not from a benchmark, since the model is not run here. Measure it with `llama-bench` and `LLAMA_LONGHAUL_IO_THREADS=1/2/4/8`.
- Items 2-4 (direct reads into the Metal buffer, eviction tuning, prefill numbers) need profiles, traces or benchmarks of the real model, so they wait until you can run it.

## 14. Phase 3 status (done)

Committed in fe2a11a (steps 1-9 of section 7, and the speculative part of step 13):

- Tests: `test-llama-archs` keeps Inkling2, Inkling2-loop and the LongHaul cases. `test-quant-type-selection` is build-only, because its snapshot needs `--generate`, which fetches the Mini header. `eval-callback` runs on the generated `inkling2-moe.gguf` on the GPU. The tokenizer tests are build-only. The FP8 KV test is dropped.
- Tensor split mode: `LLAMA_SPLIT_MODE_TENSOR` stays in `llama.h` for ABI reasons but fails at load time. It is removed from `-sm`, fit and llama-bench.
- `llama-arch`, `llama-model`, `llama-hparams`, `llama-graph` and `llama-quant`: pruned to what Inkling2, Inkling2-loop, DSpark and CLIP use.
  - The per-tensor `.scale` and input-scale tensors stay, so files that carry them still load.
  - A `prism.hadamard.version` key is rejected at load time.
  - A `nextn_predict_layers` > 0 is rejected at load time.
- MTP, EAGLE3, DFlash2 and LiliCorr are removed:
  - The spec types, their drafts in `common/speculative.cpp`, `--mtp`, `--eagle3`, imatrix `--nextn`/`-md`, and the mtp/eagle3 download sidecars.
  - Files named `mtp-*` or `eagle3-*` are still kept out of model selection.
  - `draft-dflash` stays as an alias that runs the DSpark draft.
- Public API removed:
  - `LLAMA_CONTEXT_TYPE_MTP` and `llama_model_params::load_mtp`.
  - `llama_model_n_layer_nextn`, `llama_set_nextn_layer_offset` and `llama_model_dflash_selector_top_k`.
  - The DFlash variants other than DSpark.
  - The state embedding API (`llama_batch_ext_set_embd_state[_row]`, `llama_set_ctx_state_src`).
  - The h_nextn device buffer (`llama_set_embeddings_nextn_dev`, `llama_{get,set}_embeddings_nextn_rows`).
- Public API kept for DSpark: `llama_set_embeddings_nextn`, `llama_get_embeddings_nextn[_ith]`, the layer input buffers and `llama_batch_ext_add_embd_row`.
- mtmd: `inkling2v` is the only encoder left, and `mtmd-helper-gen.cpp` is a stub. Unlike step 14, the ffmpeg video helper is kept.

Committed in e1545e1:

- Control vectors: `llama_set_adapter_cvec`, `llama_adapter_cvec`, `common_control_vector_*`, `--control-vector*` and `tools/cvector-generator` are removed.
- Vocab: only the `gpt2` (BPE) and `test` tokenizer models load; the others throw "unknown tokenizer".
  - Pre-tokenizers: empty and `default` map to DEFAULT, `inkling2` maps to GPT4O; anything else throws.
  - A metaspace prepend scheme other than `never` is rejected at load time.
  - The public `llama_vocab_type` enum is unchanged.
- Legacy chat templates: `src/llama-chat.{h,cpp}` are gone.
  - `llama_chat_apply_template` always returns -1, and `llama_chat_builtin_templates` returns 0.
  - In common, `use_jinja = false` throws "try using --jinja", and `common_chat_verify_template(..., false)` returns false. Inkling's templates already returned -1 on the old path, so Inkling behaves the same.
  - `--chat-template chatml` still selects the ChatML jinja template.
  - `examples/simple-chat` is removed, since it only used `llama_chat_apply_template`.
  - `test-chat-template` keeps its jinja cases and checks that the C API rejects every template.
  - Not changed: `llama-completion` and `llama-mtmd-cli` still default to `use_jinja = false`, so Inkling needs `--jinja` there (as before).
- Chat parsers: `common/parsers/` keeps `parsers.{h,cpp}` and `inkling.cpp`; the other 20 specialized parsers and their detection checks are gone. Every other template goes through the generic autoparser. Neither Inkling template matched an earlier check, so Inkling still gets its own parser.
  - Also removed: the gemma4 and minimax-m3 PEG mappers and chat formats, the StepFun trim and DeepSeek V4.1 message rendering, `common_chat_msg::extra_json`, and the GPT-OSS template patch in `common_chat_templates_init`.
  - `test-chat`: the blocks and tests for the removed parsers are gone, along with every reference to their templates. The autoparser and Inkling cases stay. `test_chat_session` and `test_chat_json_depth` now use `Qwen-QwQ-32B` (autoparser) instead of `Qwen3.5-4B`.
- ASR/TTS helpers:
  - `common/chat-tts-norm.{h,cpp}` are removed, along with the "tts" chat template and dictionaries, `common_chat_get_{asr,tts}_prompt` and `common_chat_get_speech_system_prompt`.
  - The public `llama_model_meta_arr_{n,str}` API and `mtmd_gen_audio_g2p` (TTS dictionaries and g2p only) are removed.
  - `tools/tts`, `LLAMA_EXAMPLE_TTS`, `--tts-speaker-{file,text}`, `scripts/chat_tts_norm.py` and `docs/tts-norm-patterns.md` are removed.
  - The server's audio endpoints stay (generic, and already gated off because mtmd has no audio here). Transcription uses the generic "Transcribe audio to text" prompt, and `/tts` sends the input as is.

Committed after e1545e1 (steps 14-16 and the rest of 13, done by three parallel worktree agents and reviewed):

- Python:
  - `conversion/` keeps `base.py`, `inkling2.py`, a new `dspark.py` (the DFlash/DSpark drafter converter, now on `TextModel`) and `__init__.py`.
  - `convert_hf_to_gguf.py` loses the Mistral, MTP, PLE, sentence-transformers and QKV-fusion paths. `convert_hf_to_gguf_update.py` keeps the inkling2 entry only.
  - `convert_llama_ggml_to_gguf.py` and `examples/convert_legacy_llama.py` are deleted.
  - gguf-py:
    - `constants.py` keeps the INKLING2, INKLING2_LOOP, DFLASH and MMPROJ (inkling2v) lists, plus the keys the C++ side or the kept python still uses.
    - `gguf_writer.py` loses 194 unused `add_*` methods.
  - `tensor_mapping.py` was pruned conservatively: only entries for removed tensors went. The plan's before/after conversion on a real checkpoint was not run (no model downloads). Instead, in-memory synthetic DSpark and inkling2v conversions give bit-identical output before and after.
  - `test_conversion_fp8.py` keeps only the Inkling2 tests.
- C++ cleanup:
  - mtmd drops the fields and tensors of other models (TTS, codec, g2p, deepstack, lfm3vl, MoE, sinks, post-norms). The public `mtmd.h` API is unchanged.
  - Removed hparams nothing assigns: PLE, MLA, alibi, softcap, clamp_kqv, group/layer norm, rel buckets, deepstack, swiglu clamps, gating func, dsv4 HC, indexer.
  - `llama-arch` loses 87 KV names and 9 tensors that are now unused.
- Templates and docs:
  - Removed the 25 templates nothing references, `docs/multimodal/` and `test_spec_methods.py` (plus its workflow).
  - Rewrote `docs/multimodal.md`, `docs/speculative.md`, `examples/speculative-simple/README.md`, `docs/multi-gpu.md`, `tools/mtmd/README*.md` and `HOWTO-add-model.md` for the fork.
  - Moved the removed-parser server tests onto kept templates.

Phase 3 follow-ups (not blocking):

- hparams still never assigned: `ssm_*`, `n_embd_head_kda`, and the `rs_replay` code that reads them. Also the `v_mla` plumbing in `build_attn`.
- About 42 LLM_KV keys are used only by the `test-llama-archs` fixture writers.
- Test coverage:
  - The generic FP8 conversion code is now tested only through the Inkling2 packed-expert tests, and nothing tests a synthetic DSpark conversion.
  - The generic `defer_loading`/`tool_reference` path has no test template.
- The server python suite still uses LLaMA presets this fork cannot load (not run here).
- `mtmd-cli` prints llava/MobileVLM hints, `arg.cpp` has the `--vision-gemma-*` presets, and the `/tts` README section lists model-specific params.

---

## 15. Phase 4 status (done)

Done by three parallel worktree agents (backends, ggml-cpu, build/CI), each reviewed, then merged and checked together:

- Backends:
  - Removed cuda, hip, musa, vulkan, sycl, opencl, cann, webgpu, zdnn, zendnn, hexagon, openvino, et, virtgpu and rpc, with their headers, CMake options, registry entries, `FindNCCL` and the CUDA/HIP/SYCL sections of `ggml-common.h`. Kept: CPU, Metal and BLAS (Accelerate).
  - `GGML_CPU_ALL_VARIANTS` builds only `apple_m1`, `apple_m2_m3` and `apple_m4`. The x86, LoongArch, RISC-V and s390x ISA options are gone.
  - RPC: `--rpc` is gone from common and llama-bench, along with `tools/rpc` and its tests. `llama_supports_rpc()` stays in `llama.h` for ABI reasons and returns false.
  - Generic multi-device code (`--device`, `--split-mode`, `GGML_BACKEND_DL`) stays.
  - `test-backend-ops`, `test-backend-sampler` and `test-llama-archs` lose their removed-backend branches. Tolerances on Metal, CPU and BLAS are unchanged.
  - Docs: `docs/build.md` covers only the macOS arm64 build, `docs/backend/` is deleted, and `docs/ops.md` has BLAS, CPU and Metal columns.
- ggml-cpu: only `arch/arm` is left. `amx/`, `spacemit/` and the x86/spacemit CMake helpers are deleted. kleidiai, llamafile, tiled and the shared headers are unchanged. A full `test-backend-ops -b CPU` run passed 25162/25162.
- Tools, examples, build and CI:
  - Removed `tools/connect` and the 11 examples listed in section 8, plus `pocs/`.
  - `CMakePresets.json` keeps the arm64-apple presets only, `build-xcframework.sh` builds a single macOS arm64 slice, and `flake.nix` is aarch64-darwin only.
  - Removed the `.devops` Dockerfiles and specs, `docs/docker.md`, the non-Apple toolchains, and the Android, s390x and RISC-V docs.
  - `.github/workflows` keeps build-apple, code-style, editorconfig, the python checks and the 8 UI workflows. build-apple loses its leak check, which needed a Gemma model. The release scripts, the labeler config and the composite actions that only deleted workflows used are removed.
  - `ci/run.sh`: Metal and CPU only. The upstream model runs (Qwen3, BGE, reranker) are removed because this fork cannot load those archs.
  - README: a fork notice and a quick start for Inkling 2 Mini (done here instead of in Phase 5).
- Checks: fresh configure and build with no compiler warnings. `ctest -L behavior -LE 'model|python'` passed 40/40 and `ctest -L main` passed 22/22. `test-llama-archs`, `test-batch-alloc`, `test-model-resolution`, `test-arg-parser` and `test-backend-ops -o MUL_MAT` (Metal and BLAS) passed. `llama-gen-docs` output is in sync.

Phase 4 follow-ups (not blocking):

- Not run: the KleidiAI build (FetchContent needs the network), nix evaluation, the kept GitHub workflows (only the files they reference were checked; no YAML parser is installed) and a full Metal `test-backend-ops` run.
- `tools/quantize/tests.sh` and `tools/gguf-split/tests.sh` still download Qwen3. They are no longer called from CI.
- The diffusion params in common (`LLAMA_EXAMPLE_DIFFUSION`, `--diffusion-*`) and `llama_model_is_diffusion` are dead now that `examples/diffusion` is gone.
- `ui-publish.yml` has no caller left (`release-publish.yml` is deleted). It is kept with the UI workflows.
- `tools/fit-params/README.md` shows sample output from a CUDA machine.

---

## 16. Phase 5 status (done)

- `docs/longhaul.md` replaces `docs/moe-cache.md`. It covers what is cached, the flag, the sizing formula with Mini's numbers and a table, the prefill vs cache-size trade-off, the stats line, the I/O thread count and the trace for offline sizing. The numbers are estimates from the model card, not measurements.
- `scripts/moe-cache-tune.py` is deleted.
- The README fork notice and quick start were added in Phase 4. The quick start pins `-hf AtomicChat/Inkling-2-Mini-GGUF:Q4_K`, because without a tag `-hf` falls back to the first model file when the repo has no Q4_K_M or Q8_0.
