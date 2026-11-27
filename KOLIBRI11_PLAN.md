# Kolibri 1.1 on LongHaul: plan

Goal: run [Aleph-Alpha/Kolibri-1.1](https://huggingface.co/Aleph-Alpha/Kolibri-1.1) in this fork, with its routed experts on disk behind the LongHaul cache, the same way Inkling 2 Mini, Qwen 3.5 MoE and Ling-3.2-air run today. The official GGUFs ([Aleph-Alpha/Kolibri-1.1-GGUF](https://huggingface.co/Aleph-Alpha/Kolibri-1.1-GGUF)) must load unchanged.

Constraints: the model is not downloaded or run. All checks are code-level tests that run in memory (random tiny models, synthetic conversions, `test-backend-ops`). Commits follow AGENTS.md (`Assisted-by:` trailer, each one approved by you, nothing pushed).

Status: draft, waiting for your answers on section 2.

---

## 1. Facts (checked 2026-11-25)

### 1.1 Upstream status

- **Kolibri 1.1 is not in upstream master yet.** Aleph Alpha's own PR [ggml-org/llama.cpp#32891](https://github.com/ggml-org/llama.cpp/pull/32891) ("model : add Kolibri-1.1 support", by an Aleph Alpha employee) is open since 2026-11-23, with no reviews. The GGUF model card says llama.cpp needs that PR to load the files.
  - One commit, head `5f4d222`, base `6aee508` (upstream 2026-11-20). +979/-105 in 27 files.
  - Fetched locally as `refs/kolibri/pr32891` (no checkout).
  - The bot flagged it as large and multi-backend, so it may change before it merges. The keys in the published GGUFs are already fixed, though (1.3).
- **Kolibri 1.1 is the existing `kolibri1` arch (Kolibri 1) plus three things:**
  1. sparse attention in every full-attention layer,
  2. K/V sharing between full-attention layers,
  3. a dense FFN prefix (layers 0-1).
- **In our history:** `kolibri1` was removed by the prune commit `3453f63`. **Commit `9906149` has `kolibri1` and LongHaul together**, as for Ling. Its `src/models/kolibri1.cpp` equals the PR base except for one `build_moe_ffn` argument list (HEAD uses the same short form as `9906149`). Its `conversion/kolibri1.py` and `conversion/afmoe.py` are byte-identical to the PR base.

### 1.2 The model

- **Size:** 75.1B parameters, 3.44B active. Hidden 2560, vocab 128000, context 262144 (1M by extrapolation).
- **50 layers:**
  - Layers 0-1: dense SwiGLU FFN (3584). Layers 2-49: MoE with 384 experts, 6 used per token, separate `gate`/`up`/`down` (512 wide), plus 1 shared expert (512).
  - Router: sigmoid, a selection bias added before top-6, no weight normalization (`norm_topk_prob = false`).
  - Norms: pre and post norm around attention and around the FFN, per-head RMS QK-norm.
- **Attention, 4:1 pattern:**
  - 40 sliding-window layers: GQA 48/4, head dim 128, window 513 (512 preceding + current), RoPE base 10000.
  - 10 sparse layers (4, 9, ..., 49): NoPE.
    - Only layers 4, 19, 24 and 29 have their own K/V. The others read them: 9, 14, 49 use layer 4; 34, 39, 44 use layer 29. Shared-KV layers have only `attn_q` and `attn_output`.
    - "Indexer" layers (4, 14, 19, 29) choose a pool of keys; the other sparse layers ("reindexers") select from the pool of the closest preceding indexer.
    - The sparse attention has **no learned indexer weights**. A query scores keys with the 12 query heads of the first KV head: the mean of `ReLU(q*k)/sqrt(128)`, with q and k rounded to NVFP4. It attends to the 2048 best keys plus the first 8 (sink) and the last 128 (recent) keys.
    - Contexts longer than 16384 keys first pick the 1024 best blocks of 16 keys (scored by the mean key of the block), and the selection runs inside them.
    - K/V of the sparse layers are rounded to NVFP4 before they go into the cache (`kv_cache_qat`).
- **HF checkpoint:** FP8 e4m3 with 128x128 block scales. The checkpoint keeps K/V/k_norm weights in shared-KV layers that are never used; the converter drops them.
- **Chat:** ChatML (`<|im_start|>`/`<|im_end|>`), `<think>` reasoning, Hermes-style `<tool_call>{json}</tool_call>`. Reasoning effort is a template kwarg (`reasoning_effort` none/low/medium/high, or `enable_thinking`).

### 1.3 The official GGUFs (headers read with HTTP range requests, no weights)

- `Q4_K_M` (45.5 GB) header:
  - `general.architecture = kolibri1`, `block_count = 50`, `leading_dense_block_count = 2`, `feed_forward_length = 3584`, `expert_feed_forward_length = 512`, `expert_count = 384`, `expert_used_count = 6`, `expert_gating_func = 2`, `expert_weights_norm = false`, `sliding_window = 513`, a 50-entry `sliding_window_pattern`.
  - New keys: `attention.indexer.types` (bool array, true at 4, 14, 19, 29), `attention.kv_source_map`, `attention.indexer.top_k = 2048`, `attention.indexer.candidate_block_size = 16`, `attention.indexer.candidate_top_k_blocks = 1024`.
  - `tokenizer.ggml.pre = qwen2`, `gpt2` BPE, 128000 tokens, no BOS, EOS 127906, pad 127901.
  - Shared-KV layers have no `attn_k`/`attn_v`/`attn_k_norm`.
  - **Expert layouts:** 2 groups. 23 layers have Q4_K/Q4_K/Q6_K (gate/up/down), 25 layers all Q4_K.
- `IQ2_XXS-GSQ-RCO-2.00bpw` (18.9 GB) header: same keys. The experts mix IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS and IQ2_S per tensor, which gives **26 expert layouts**. The `IQ1_M`/`IQ2_XS` files are likely the same kind.

### 1.4 What HEAD already has, and what is missing

What is in HEAD already:
- **ggml:**
  - `ggml_lightning_indexer` and `ggml_top_k_mask`, with CPU, Metal and CUDA kernels. The Metal indexer needs head dim 128 and supports any head count.
  - `ggml_top_k`, and `FLASH_ATTN_EXT` with a sparse mask and the `n_kv_max` hint.
  - `ggml_fp32_to_f8_e4m3` (needed by the CPU rounding).
- **llama:**
  - `llama_kv_cache_iswa` with the `layer_reuse_cb` path (K/V sharing). No arch uses it today, so it is untested in this fork.
  - `hparams.kv_source_impl` / `has_kv()` / `kv_source()`, the `KV_SOURCE_MAP` key, `n_layer_dense_lead`, `LEADING_DENSE_BLOCK_COUNT`.
  - The tensors and keys `kolibri1` uses (post norms, `exp_probs_b`, shared expert, QK norms, swa pattern).
  - `build_attn_mha` with the `n_kv_max` argument, `build_attn_inp_kv_iswa`, and the `qwen2` pre-tokenizer.
- **LongHaul:** separate gate/up/down experts and the mixed-layout budget split (Ling D4).

What is missing:

| Area | Missing in HEAD |
|---|---|
| ggml | `GGML_OP_ROUND_NVFP4` / `ggml_round_nvfp4` (new in the PR) on CPU, Metal, CUDA. CUDA lightning indexer for 12 heads (HEAD: 4, 32, 64). CUDA MMA flash attention for the sparse 128/128 GQA-12 shape. |
| `src/llama-arch.*` | `LLM_ARCH_KOLIBRI1`. The keys `ATTENTION_INDEXER_TYPES`, `ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE`, `ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS`. (`INDEXER_TOP_K` and `KV_SOURCE_MAP` exist.) |
| `src/llama-hparams.*` | `indexer_top_k`, `is_indexer_full_impl` + `is_indexer_full()`. |
| `src/llama-model.*` | `LLM_TYPE_78B_A3_5B`, factory case, NEOX rope type, the `create_memory` iswa branch with `reuse` (it passes `nullptr` today). |
| `src/llama-kv-cache.*` | `llama_kv_cache_context::get_cells()` (the class has `llama_kv_cache::get_cells`, the context lacks the wrapper). |
| `src/models/` | `kolibri1.cpp` and its `models.h` struct. |
| Python | the converter (`conversion/kolibri1.py`, which inherits from `afmoe.py` -> `llama.py`; gguf-py in HEAD has no `LLAMA`/`AFMOE` arch, so these cannot be restored as is), `MODEL_ARCH.KOLIBRI1` and its tensor list, tensor mappings (`post_attn_norm`, `post_ffn_norm`, `moe.router.expert_bias`), writer methods for the three new keys, the Kolibri `chkhsh` line. |
| Chat | the Kolibri template file and tests. The autoparser should handle the format (it is the Qwen3 / Hermes 2 Pro layout), to be confirmed by tests. |
| Docs | `docs/longhaul.md` section, README. |

### 1.5 LongHaul numbers (computed from the Q4_K_M header, not measured)

- **Experts:** 48 layers x 384 = 18432 experts.
  - Q4_K/Q4_K/Q6_K layers: 2.43 MiB per expert. All-Q4_K layers: 2.11 MiB per expert.
  - All experts: 40.7 GiB (96% of the file).
- **Resident part:** about 1.6 GiB (attention, dense FFN, shared experts, router, norms, embeddings, output). This is the smallest resident part of the four models.
- **KV cache:** only 4 layers keep full-context K/V: 8 KiB per token at F16, so 2 GiB at 262144 tokens. The 40 sliding layers need about 40 MiB.
- **Disk read per token with no hits:** 48 x 6 x ~2.3 MiB, about 650 MiB. That is less than Ling (~1 GiB) or Inkling 2 Mini (~1.9 GB).
- **Cache split at 512 MiB (Ling D4 rule):** `reserve(U) = U x 6 x (2.43 + 2.11) MiB <= 256 MiB` gives U = 9. The rest is split by expert bytes. That makes about 110 + 115 slots and `max_ubatch` about 18. The minimum cache is `reserve(1)`, about 27 MiB.
- **Hit rate:** 225 slots for 18432 experts is low. I would recommend 2-4 GiB in the docs. The default stays 512M.
- **The IQ files (26 layouts):**
  - `reserve(1)` is about 140 MiB.
  - At 512 MiB, U = 1 and the one-layer groups get about 14 slots, so `max_ubatch` is about 2. Prefill will be very slow.
  - At 2 GiB, `max_ubatch` is about 11.
  - The docs should say this, and recommend Q4_K_M (2 layouts) for LongHaul, or re-quantizing experts to one type.

---

## 2. Decisions (my recommendation first; I need your answer on each)

| # | Question | Recommendation | Why |
|---|---|---|---|
| K1 | Source of the code: port PR #32891 now, or wait for it to merge upstream? | **Port it now, pinned to head `5f4d222`.** Restore `kolibri1` from `9906149`, then apply the PR on top, and keep the code as close to the PR as possible. Before implementation, re-fetch the PR and diff against `5f4d222`. If it moved, I'll tell you what changed before I go on. | The published GGUFs need exactly this code. A near copy keeps later upstream fixes easy to port. Risk: review may still change the design. Since the GGUF keys are already public, a change to them is unlikely. |
| K2 | Also support Kolibri 1 (no sparse attention)? | **Yes.** Same arch: the `indexer_top_k == 0` path is the Kolibri 1 graph. Tests cover both. | It costs no extra code, and the fork had Kolibri 1 before the prune. |
| K3 | Backends for the new `ROUND_NVFP4` op and the CUDA kernel changes | **CPU + Metal + CUDA, as in the PR.** Skip the PR's CUDA `CMakeLists.txt` arch-list change (adds `100a-real`, changes `103-real` -> `103a-real`). | LongHaul needs a GPU, and the fork supports Metal and CUDA. The CMake change only turns on the native FP4 convert path on B200/B300. The fallback path is bit-exact, and the native path still compiles on consumer Blackwell (`120a-real`, already in the list). I can't build or run CUDA here, so you'd run the CUDA tests. |
| K4 | Python converter | **A self-contained `conversion/kolibri1.py`** on `TextModel`, with `Kolibri1Model` and `Kolibri11Model`. It ports the parts of `AfmoeModel`/`LlamaModel` it uses: hparams, merging the experts, renaming `expert_bias`, and the dense prefix. `afmoe.py`/`llama.py` are not restored. Test with a synthetic FP8 checkpoint. | `llama.py` registers many archs that gguf-py no longer has, so restoring it as is would break imports. The official GGUFs exist, so the converter is only for people who want their own quants. Alternative: skip conversion entirely (saves about 200 lines with tests). |
| K5 | Several sequences | **Keep the PR rule:** sparse attention with `n_seq_max > 1` needs `--kv-unified`, otherwise context creation fails with a clear message. | The sparse inputs index the cells of one stream. `llama-server` with `-np auto` already turns on a unified KV cache. |
| K6 | LongHaul with the 26-layout IQ files | **No code change.** Document the numbers from 1.5 and recommend Q4_K_M or one expert type. | The Ling D4 split handles it correctly, just with a small prefill ubatch. A smarter split (for example by bytes across types) would be a new cache design. |
| K7 | Chat and tool calls | **No new parser.** Add the template as `models/templates/Aleph-Alpha-Kolibri-1.1.jinja`, plus `test-chat` cases (reasoning on/off, `reasoning_effort` levels, a tool call, a tool call after reasoning, a JSON schema). If the autoparser fails a case, I'll stop and ask before adding a parser. | The format is the same as the Qwen3 / Hermes 2 Pro layout that the autoparser handles. Kolibri 1 had no special parser upstream either. |

Smaller choices I made (override if you like):
- **`ROUND_NVFP4` goes at the end of `enum ggml_op`** (after `GGML_OP_SIGMOID_BACK`, `GGML_OP_COUNT` 101 -> 102), as the note in HEAD's `ggml.h` asks. The PR puts it after `TOP_K_MASK`, but its base predates that note and `SIGMOID_BACK`.
- **`ggml-backend-meta.cpp` gets a `GGML_OP_ROUND_NVFP4` case** (`handle_generic`, `scalar_only = true`, like `TOP_K_MASK`). The PR misses it. Without it the meta backend (tensor split over several GPUs) aborts with "op not implemented".
- **`n_pool_block` / `n_pool_blocks` are moved into `llama_hparams`** (as `indexer_cand_block_size` / `indexer_cand_top_k_blocks`) if the test round-trip through `llama-model-saver.cpp` needs them. The PR keeps them on `llama_model_kolibri1`, where the saver cannot see them.

---

## 3. Phase 0 - Last facts (no weights)

1. **Re-check the PR** (K1): `git fetch` it again and diff it against `5f4d222`.
2. **Tokenizer hash:**
   - Download only `tokenizer.json` and `tokenizer_config.json` of Kolibri-1.1 into `.cache/kolibri11-tokenizer/` (gitignored).
   - Compute `chkhsh` as `convert_hf_to_gguf_update.py` does. Compare it with Kolibri 1's `6e040dfe...` (maps to `qwen2`).
   - If it differs, add a second line. The GGUF already says `qwen2`, so a match is likely.
   - This needs `transformers`. I'll check that it is installed and ask before installing anything.
3. **Chat template:** save `chat_template` from `tokenizer_config.json` (Apache-2.0) as `models/templates/Aleph-Alpha-Kolibri-1.1.jinja`.
4. **Reference splits:** record about 15 `qwen2` pre-tokenizer splits of German and English strings (umlauts, sharp s, digits, contractions) for `tests/test-unicode.cpp`, as for Ling.

---

## 4. Phase 1 - ggml

### 4.1 `GGML_OP_ROUND_NVFP4`

What it does: round each group of 16 values of a row to NVFP4 and back to f32. The scale is `e4m3(max(amax/6, 2^-9))` and each value becomes `scale*e2m1(x/scale)`. Both roundings are to nearest even and saturate.

- **`ggml/include/ggml.h`, `ggml/src/ggml.c`:** the enum entry (at the end), name, symbol, `GGML_OP_COUNT` asserts, and the constructor `ggml_round_nvfp4` (F32, contiguous rows, `ne0 % 16 == 0`). Taken from the PR.
- **CPU** (`ggml-cpu.c`, `ops.{h,cpp}`): the PR kernel. One thread per row, `n_tasks = n_threads`.
- **Metal** (`ggml-metal-device.{h,cpp,m}`, `ggml-metal-impl.h`, `ggml-metal-ops.{h,cpp}`, `kernels/misc.metal`): the PR kernel, one thread per group, with precise divisions.
- **CUDA** (`round-nvfp4.{cu,cuh}`, `ggml-cuda.cu`): the PR kernel. The native FP4 convert is used when `__CUDA_ARCH__ >= 1000`, otherwise a bit-exact fallback.
- **Meta backend:** the case from section 2.

### 4.2 CUDA kernels used by the sparse layers

- `lightning-indexer.cu`:
  - the PR refactor into `lightning_indexer_few_heads<N_HEAD>`, with instances for 4 and 12 heads;
  - `ggml_cuda_lightning_indexer_supported` accepts 12.
- `fattn-mma-f16.cuh`: the PR's two extra `(128, 128, ncols1 1|8, ncols2 8)` cases in `ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse`.

### 4.3 Tests (`tests/test-backend-ops.cpp`)

- `test_round_nvfp4`, the PR version. It is bit-exact (`max_nmse_err = 0`) and includes the tie and subnormal cases. Shapes: the 4 eval cases.
- Lightning indexer with `nh = 12` in the eval and perf loops.
- The 3 sparse flash-attention eval cases (128/128, GQA 12, `n_kv_max = 2184`) and the perf case.
- Run on Metal here. You run CUDA.

---

## 5. Phase 2 - Runtime: the `kolibri1` model

Restore from `9906149`, then apply the PR's `src/` diff. Each item names what changes for the other models (shared code).

### 5.1 `src/llama-arch.{h,cpp}`

- Add back `LLM_ARCH_KOLIBRI1` / `"kolibri1"`.
- Add the keys `LLM_KV_ATTENTION_INDEXER_TYPES`, `LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE` and `LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS`, with their `9906149` names.
- No new tensors.
- Not hybrid, not in the rs-rollback list.

### 5.2 `src/llama-hparams.{h,cpp}`

- Add `uint32_t indexer_top_k = 0`, `std::array<uint32_t, LLAMA_MAX_LAYERS> is_indexer_full_impl`, and `bool is_indexer_full(uint32_t il) const` (from `9906149`).
- Zero-fill `is_indexer_full_impl` for every arch in `load_hparams`, next to the existing `kv_source_impl` iota (`llama-model.cpp:494`).
- Optionally add the two pool fields (section 2).
- Nothing changes for other archs: they never read these fields.

### 5.3 `src/llama-model.{h,cpp}`

- `LLM_TYPE_78B_A3_5B` and its name. The factory case. `LLAMA_ROPE_TYPE_NEOX`. (`9906149` has no kolibri-specific `print_info` lines.)
- **`create_memory`:** add a `KOLIBRI1` branch before the generic iswa branch. It builds `llama_kv_cache_iswa` with:
  - `reuse = has_kv(il) ? -1 : kv_source(il)`;
  - the PR check that throws when `indexer_top_k > 0 && !kv_unified && n_seq_max > 1`.
- The generic branch (DSpark) stays unchanged.

### 5.4 `src/llama-kv-cache.{h,cpp}`

- Add `const llama_kv_cells & llama_kv_cache_context::get_cells(llama_seq_id) const` (PR, 4 lines).

### 5.5 `src/models/models.h`, `src/models/kolibri1.cpp`

- Restore the struct and the file from `9906149`, then apply the PR diff:
  - load the dense prefix and the sparse hparams, with the PR's validation of indexers and KV sources;
  - for shared-KV layers, create only `attn_q` (and no `attn_k_norm`);
  - dense FFN tensors for the prefix layers;
  - `llm_graph_input_sparse` (`cand`, `always`, `blk_cells`, `blk_mask`, `set_input`, `can_reuse`);
  - `build_attn_sparse`;
  - the graph changes.
- Keep HEAD's `build_moe_ffn` argument form (`..., il, logits, nullptr, selected)`).
- **LongHaul:** `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps` get `TENSOR_LONGHAUL`, as in `bailingmoe3.cpp`. The router, `exp_probs_b`, the shared expert and the dense prefix stay resident.

### 5.6 Other

- `src/llama-model-saver.cpp`: write `INDEXER_TYPES`, `INDEXER_TOP_K` and the two candidate keys when `indexer_top_k > 0`, so the tests can round-trip a Kolibri 1.1 model. (`KV_SOURCE_MAP`, `LEADING_DENSE_BLOCK_COUNT` and the shared-expert keys: I'll check each one is written.)
- `src/llama-context.cpp`: no change expected. I'll check that `graph_max_nodes` covers the block-mean loop (16 adds per indexer layer) at the largest reserve.
- `tests/fusion/MTL.csv` still has the `kolibri1` rows. No change.

---

## 6. Phase 3 - LongHaul for Kolibri

### 6.1 What should work with no code change

- **Separate gate/up/down experts:** covered by the Qwen and Ling tests.
- **Dense prefix layers:** no experts, so the cache skips them.
- **Two expert layouts (Q4_K_M):** two groups, Ling D4 split.
- **Sparse inputs** are host inputs set per ubatch, independent of the slot maps. With `n_ubatch` clamped, the inputs are just smaller.
- **Shared-KV layers** change only attention, not experts.

### 6.2 Tests (`tests/test-llama-archs.cpp`)

**Fixture** (`kolibri1`, restored from `9906149`, then extended):
- head dim 128 (the Metal lightning indexer requires it), `n_head = 8`, `n_head_kv = 2` (4 routing heads, which CUDA supports);
- 6 layers: 0 dense, then MoE; pattern swa, swa, indexer (own KV), swa, reindexer (reads layer 2's KV), swa;
- `exp_probs_b`, shared expert, `expert_weights_norm = false`;
- `indexer_top_k = 16`, `candidate_block_size = 4`, `candidate_top_k_blocks = 8`. The pool path then starts at `n_kv > 32`, so the block path runs in small contexts.
- **A Kolibri 1 variant** with `indexer_top_k = 0` (K2).

**Generic arch tests:**
- `kolibri1` goes into `test_backends` and `test_kv_shift`, and into `moe_mandatory`.
- **CPU vs GPU comparisons:** the indexer scores use ReLU, so random weights give many exact ties at 0. `top_k` could then pick different keys on different backends. So the cross-backend cases use contexts where every candidate is selected (`n_kv` below `top_k + 136`). The selection-heavy cases run in the same-backend LongHaul comparisons below.
- `test_kv_shift` checks a shift with shared-KV layers. This is the first user of the `reuse` path in this fork.
- **New: state save/load** of a sequence with shared-KV layers, if `test-llama-archs` has no generic case for it.

**LongHaul cases** (same as `bailingmoe3`, same backend, compared against `LLAMA_LONGHAUL_RESIDENT`):
- **Variants:** F32, Q8_0 and NVFP4 experts, plus a mixed-layout variant (one layer with another down type, which makes 2 groups).
- **Cache sizes:** the minimum `reserve(1)`, 5 slots, and "all experts".
- **Inputs:**
  - decode;
  - small batches;
  - a prompt longer than the clamped ubatch;
  - **a 400-token prompt and 20 decode steps**, so the sink/recent/top-k selection and the block pool path both run (400 > 32 pool keys, 400 > 136 + 16).
- **Errors:** a cache below `reserve(1)` fails with a message that names the minimum; `n_seq_max = 2` without unified KV fails with the K5 message.

### 6.3 Other tests

- `tests/test-unicode.cpp`: the Phase 0 `qwen2` splits.

---

## 7. Phase 4 - Conversion (K4)

- **`conversion/kolibri1.py`** (new, self-contained):
  - **`Kolibri1Model(TextModel)`:**
    - hparams and GGUF keys as `AfmoeModel` + `LlamaModel.set_gguf_parameters` wrote them for Kolibri 1: shared count 1, expert FF length, leading dense count, expert weights norm, sigmoid gating, sliding window and pattern, shared FF length;
    - the gpt2 vocab;
    - stacks the per-expert gate/up/down into 3D tensors;
    - renames `expert_bias` -> `expert_bias.bias`;
    - the dense prefix (`mlp_only_layers` must be leading, `mlp_only_intermediate_size` becomes `feed_forward_length`).
  - **`Kolibri11Model(Kolibri1Model)`:** the PR's `modify_tensors` (drop the unused K/V/k_norm of shared-KV layers) and `set_gguf_parameters` (the sparse policy checks and the 5 new keys).
- **`conversion/__init__.py`:** register `Kolibri1ForCausalLM` and `Kolibri1_1ForCausalLM`.
- **`conversion/base.py`:** the `chkhsh` line(s) from Phase 0 -> `qwen2`. FP8 128x128 dequant is generic already.
- **`convert_hf_to_gguf_update.py`:** add back the Kolibri entry.
- **gguf-py:**
  - `constants.py`: `MODEL_ARCH.KOLIBRI1`, its name, and its tensor list (`9906149` list plus `FFN_GATE/UP/DOWN` from the PR), `Keys.Attention.Indexer.TYPES/CANDIDATE_BLOCK_SIZE/CANDIDATE_TOP_K_BLOCKS`.
  - `gguf_writer.py`: `add_indexer_types`, `add_indexer_candidate_block_size`, `add_indexer_candidate_top_k_blocks`.
  - `tensor_mapping.py`: `post_attn_norm`, `post_ffn_norm` and `moe.router.expert_bias` (`9906149` entries).
- **Tests** (`tests/test_conversion_fp8.py`, in memory):
  - a synthetic Kolibri 1.1 checkpoint: 6 layers as in the fixture, 4 experts, FP8 128x128 block scales on the projections, unused K/V in a shared-KV layer;
  - checks: tensor names and shapes, all keys against the Q4_K_M header layout (1.3), the dropped tensors, and that an unsupported policy (for example `index_num_sink_tokens = 4`) raises;
  - a Kolibri 1 case without the sparse keys.
- **Lint:** flake8 and ty on the touched files.

---

## 8. Phase 5 - Chat (K7)

- The template file from Phase 0.
- `tests/test-chat.cpp`: the K7 cases on the Kolibri template, with the autoparser.
- Also check that `reasoning_effort` reaches the template through `caps_apply_reasoning_effort` (`common/chat.cpp:905`), and that `enable_thinking=false` yields the empty think block.

---

## 9. Phase 6 - Docs

- **`docs/longhaul.md`:** a "Sizing for Kolibri 1.1" section with:
  - what is cached;
  - the numbers from 1.5;
  - the IQ-file note (K6);
  - the recommended size (2-4 GiB);
  - the `--kv-unified` note (K5).
  - Also update the intro and the "Tests" paragraph.
- **`README.md`:** a quick start next to the others, and the fork notice lists Kolibri.
- **`docs/development/HOWTO-add-model.md`:** one line that `kolibri1` (Kolibri 1 and 1.1) is supported.
- **This file:** a status section at the end, as in `LING32_PLAN.md`.

---

## 10. Tests run before each commit (code-level only)

- **Build:** `cmake --build` for the touched targets, Metal build on this Mac.
- **C++:**
  - `test-backend-ops -o ROUND_NVFP4`, `-o LIGHTNING_INDEXER`, `-o FLASH_ATTN_EXT` (Metal vs CPU);
  - `test-llama-archs` (all existing cases plus the new `kolibri1` ones) and `--kv-shift`;
  - `test-chat`, `test-unicode`, `ctest -L main`, `ctest -L behavior -LE 'model|python'`.
- **Python:** `tests/test_conversion_fp8.py`, flake8, ty.
- **CUDA:** only on your machine (same list).

---

## 11. Risks

- **The PR is unreviewed upstream.** Its design may change in review. A later rename of keys would make our loader and the published GGUFs disagree with upstream. K1's re-check reduces this but cannot remove it.
- **Known deviation in the PR, inherited:** the block score uses the mean of keys already rounded to NVFP4, while training rounds after the mean. Aleph Alpha reports negligible eval differences.
- **No real-model check.** The math matches the PR and the model card, but nothing is compared against vLLM outputs here.
- **First use of K/V sharing (`reuse`) in this fork.** The kv-cache code is there, but no current arch exercises it. K-shift, state save/load and `seq_rm` on shared layers are covered by the new tests only.
- **Shared-code edits:** `create_memory`, hparams fields, the saver, `ggml_op` enum (+1), and the meta backend. Every existing test must keep passing unchanged.
- **Host cost of the sparse inputs:** `set_input` fills `[n_kv, n_tokens]` masks on the CPU each ubatch. At 262k context that is O(n_kv) per token. Correct, but long-context decode on a Mac will pay for it. LongHaul's small ubatch keeps the buffers small (about 9 MiB for `cand` at 262k x 18).
- **Quantized KV cache (`-ctk q8_0`):** the sparse layers write K/V with `cpy_k`/`cpy_v` and attend with `build_attn_mha`, bypassing `build_attn`'s rotation. That is consistent within the layer, but I'll add one LongHaul case with a Q8_0 cache to be sure.
- **Speed:** about 650 MiB read per token with no hits at Q4_K_M, and much slower prefill with the IQ files at small caches.

## 12. Files touched (summary)

| File | Change |
|---|---|
| `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-backend-meta.cpp` | `ROUND_NVFP4` op |
| `ggml/src/ggml-cpu/{ggml-cpu.c,ops.h,ops.cpp}` | CPU kernel |
| `ggml/src/ggml-metal/{ggml-metal-device.h,.cpp,.m,ggml-metal-impl.h,ggml-metal-ops.h,.cpp,kernels/misc.metal}` | Metal kernel |
| `ggml/src/ggml-cuda/{round-nvfp4.cu,.cuh,ggml-cuda.cu,lightning-indexer.cu,fattn-mma-f16.cuh}` | CUDA kernel, 12-head indexer, sparse FA shape |
| `src/llama-arch.{h,cpp}` | arch, 3 keys |
| `src/llama-hparams.{h,cpp}` | indexer fields |
| `src/llama-model.{h,cpp}` | type, factory, rope type, `create_memory` branch, zero-fill |
| `src/llama-kv-cache.{h,cpp}` | `get_cells` on the context |
| `src/models/models.h`, `src/models/kolibri1.cpp` | the model (restored + PR), experts `TENSOR_LONGHAUL` (model sources are globbed, no CMake change) |
| `src/llama-model-saver.cpp` | sparse keys |
| `conversion/kolibri1.py`, `conversion/__init__.py`, `conversion/base.py`, `convert_hf_to_gguf_update.py`, `gguf-py/gguf/{constants,gguf_writer,tensor_mapping}.py` | converter |
| `models/templates/Aleph-Alpha-Kolibri-1.1.jinja` | template |
| `tests/test-backend-ops.cpp`, `tests/test-llama-archs.cpp`, `tests/test-chat.cpp`, `tests/test-unicode.cpp`, `tests/test_conversion_fp8.py` | tests |
| `docs/longhaul.md`, `README.md`, `docs/development/HOWTO-add-model.md` | docs |

## 13. Status (2026-11-25)

| Phase | Status |
|---|---|
| 0 - Facts | Done. The Kolibri-1.1 tokenizer hashes to `6e040dfe...` (`qwen2`). The splits, config and template are in `.cache/kolibri11-tokenizer/`. PR #32891 is still at `5f4d222` (checked again at the end). |
| 1 - ggml | Done. `ROUND_NVFP4` is the last op (CPU, Metal, CUDA); 12-head indexer and the FA 128/128 sparse shape from the PR. |
| 2 - Runtime | Done. `kolibri1` restored from `9906149`, the PR applied on top, pool sizes in hparams, experts `TENSOR_LONGHAUL`, the K5 `--kv-unified` rule. |
| 3 - LongHaul | Done, no LongHaul code change. |
| 4 - Conversion | Done. Self-contained `conversion/kolibri1.py` on `TextModel`. |
| 5 - Chat | Done. The autoparser passes every case, no new parser. |
| 6 - Docs | Done: `docs/longhaul.md` (Kolibri sizing computed from the official Q4_K_M and IQ2_XXS headers), `README.md`, `HOWTO-add-model.md`. |

**Tests run** (Metal on this Mac, no CUDA):
- `test-backend-ops -b MTL0`: `ROUND_NVFP4` (4/4, bit-exact), `LIGHTNING_INDEXER` (240/240, 48 with 12 heads), `FLASH_ATTN_EXT` (5384/5384, with the `n_kv_max=2184` cases).
- `test-llama-archs` (all archs, exit 0): `kolibri1` vs CPU on Metal, BLAS and CPU, roundtrip, mixed batch; 13 `kolibri1` LongHaul cases; 2 state save/load cases. `kolibri1` cases checked with seeds 1-6.
- `test-llama-archs --kv-shift`: `kolibri1` passes (shift, shift x2, logits; the control is sensitive).
- `test-llama-context`: `kolibri1` passes every KV case. The 4 skips are the recurrent-state cases.
- `test-chat`, `test-chat-auto-parser`, `test-unicode`.
- `ctest -L main`: 22/22.
- `ctest -L behavior -LE 'model|python'`: 40/41. `test-tokenizer-0-qwen35` fails because `models/ggml-vocab-qwen35.gguf` was removed in `3453f63ab`. That failure predates this work.
- `tests/test_conversion_fp8.py` (18, 4 new), flake8 and ty on the touched Python files.
- A name check: the tensors the converter writes for dense, MoE and shared-KV layers match the official Q4_K_M header.

**Deviations from the plan:**
- The fixture has 8 layers, not 6: S S I S R R I S, where layer 4 and layer 6 read the K/V of layer 2, and layer 5 is a reindexer with its own K/V. The real model has both kinds: layer 24 is a reindexer with its own K/V, and layer 14 is an indexer that reads layer 4.
- The IQ2_XXS file has **25** expert layouts, not 26. The 26th was the empty layout of the dense layers 0-1. Computed numbers:
  - smallest cache: about 161 MiB (the plan said about 140);
  - ubatch: 2 tokens at 512M, 5 at 1G, 11 at 2G, 22 at 4G.
- Q4_K_M numbers, computed with the split rule of the cache:
  - smallest cache: 32 MiB (the plan said about 27);
  - 512M gives 223 slots and a max ubatch of 18;
  - from 2G, every group holds a whole layer.
- `tensor_mapping.py` needed no change. HEAD already maps `post_attn_norm`, `post_ffn_norm` and `moe.router.expert_bias`.
- The converter does not port two `AfmoeModel`/`LlamaModel` behaviors:
  - Writing `expert_weights_scale` when the config has a `route_scale`. Neither Kolibri config has it, and the loader does not read it.
  - The Q/K row permutation on the NVFP4 checkpoint path. Kolibri ropes with NEOX, and the BF16 path never permuted.
- The chkhsh line is a pre-computed hash with Kolibri-1.1 as the ref, as upstream did for Kolibri-1.
- `test-chat` caps: Kolibri reports the effort levels `none`, `low` and `medium`. `high`, `xhigh` and `max` render the same as an unknown level (the template falls back to high), so the caps probe does not list them. No level is rejected, so the server passes every level through to the template.

**Follow-ups (not done):**
- CUDA was not built or run here. The CUDA files are the PR's, without its `CMakeLists.txt` arch change (K3).
- There is no check against a real model or vLLM outputs.
- The host cost of the sparse inputs at long context (section 11) is unchanged.
