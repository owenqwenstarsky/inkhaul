# GLM-5.3-Flash on LongHaul: plan

Goal: run [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) in this fork, with its routed experts on disk behind the LongHaul cache, the same way Inkling 2 Mini, Ling-3.2-air and Kolibri 1.1 run today.

Constraints: the model is not downloaded or run. All checks are code-level tests that run in memory (random tiny models, synthetic conversions). Only GGUF headers (HTTP range requests), the HF config, the safetensors index and the tokenizer files are read. Commits follow AGENTS.md (`Assisted-by:` trailer, each one approved by you, nothing pushed).

Status: implemented (phases 0-6, decisions as recommended), not committed. See section 14 for what changed from the plan and what is still open.

---

## 1. Facts (checked 2026-11-27)

### 1.1 Upstream status

- Upstream supports the model as arch **`glm5-next`** (`LLM_ARCH_GLM5_NEXT`). It was added in #27773 (merged 2026-09-30), and about 25 follow-up PRs changed it (k-pool selection, fused QSA mask, MTP, replay instead of rollback).
- **In our history:** the fork had `glm5-next` until the prune commit `3453f63`. **Commit `9906149` has `glm5-next` and LongHaul together**, so it is the restore source, as for Ling and Kolibri.
- **No drift:** `9906149` is the upstream base `5fddd0b` plus LongHaul. Upstream changed none of `src/models/glm5-next.cpp`, `src/llama-memory-hybrid-idx.*` or `conversion/glm.py` between `5fddd0b` and tag `b12812` (the newest tag in our clone). Upstream master is 5 commits past `b12812`, and none of them touches GLM, the indexer or the chat code.
- **Still open upstream:**
  - #27917, MTP drafting for GLM-5.3-Flash. Its converter writes `glm5-next.attention.indexer.index_share_mtp`, and the unsloth GGUFs carry that key. Our loader ignores unknown keys.
  - #30268, layer-input embeddings.
  - #31875, a model bug: long reasoning turns into "hmm hmm hmm". It is also reported upstream at zai-org/GLM-5 #146 and #168. It is not ours to fix.
- Closed without merge: #27752 and #27754 (duplicate adds), and #31253 (`-sm tensor` for glm5-next).

### 1.2 The model (HF `config.json`, safetensors index, model card)

- **HF class:** `Glm5NextForConditionalGeneration`, with `model_type` `glm5_next`. It is multimodal: text, image, video and audio tokens.
- **Size:** 320B parameters in total, 18B active. MIT license.
  - hidden 4096, vocab 154880;
  - `max_position_embeddings` 1,048,576;
  - the checkpoint is FP8 e4m3 with 128x128 block scales (328 GB).
- **45 trunk layers:**
  - **34 KDA linear-attention layers.** They have q/k/v short convs (kernel 4), 64 heads of 128, the `f_a`/`f_b` decay gate with `gate_lower_bound` -5, a beta gate, and an RMS-norm output with an `g_a`/`g_b` sigmoid gate.
  - **11 DSA layers (3, 7, ..., 43).** Each is a **nope-only MLA** (`qk_rope_head_dim` 0, `mla_use_nope` true, q_lora 1536, kv_lora 512, 64 heads, 256/256 head dims), so there is no RoPE anywhere in the text model. Each DSA layer has a **k-pool DSA indexer**:
    - 32 heads of 128, top-k 2048;
    - `index_kpool` 4: keys are pooled 4 at a time with a softmax gate and a learned position embedding;
    - `select_tail` true;
    - `indexer_types` is "full" for every layer.
- **mHC (manifold hyper-connections):** 4 residual streams, 20 Sinkhorn iterations, eps 1e-6. Each layer has `hc_attn_*` and `hc_ffn_*` (fn `[4*4096, 24]`, base `[24]`, scale `[3]`).
- **FFN:**
  - layers 0-2: dense (12288);
  - layers 3-44: MoE with 288 experts, 8 used per token, 2048 wide, and 1 shared expert;
  - the router uses sigmoid scoring with `e_score_correction_bias` and `noaux_tc` (1 group), then normalizes the weights and scales them by 2.5;
  - `swiglu_limit` 10 on the routed and shared experts.
- **Layer 45 is an MTP (NextN) layer:** a DSA layer plus MoE, plus `eh_proj`, `enorm`, `hnorm` and `shared_head.norm`. It has no embed or head of its own.
- **HF tensor names** use the `model.language_model.` prefix. The vision tower is `model.visual.*`. The `hc_*` and `index_kpool_compress_*` tensors have no `.weight` suffix.
- **Chat template:**
  - it starts with `[gMASK]<sop>`;
  - roles: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`;
  - the generation prompt ends with `<think>`;
  - `reasoning_effort` low/high/max, default max (written as a system line);
  - `clear_thinking` defaults to false;
  - tool calls look like `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`, the same family as GLM-4.7.
- **Recommended sampling:** temp 1.0, top_p 0.95.

### 1.3 The community GGUFs (unsloth headers, read with HTTP range requests, no weights)

- `general.architecture = glm5-next`, `block_count = 46` (45 trunk + 1 MTP), `nextn_predict_layers = 1`.
- `tokenizer.ggml.pre = glm4`, `tokenizer.ggml.model = gpt2`, eos 154820, bos 154822 (`[gMASK]`), eot 154827 (`<|user|>`), eom 154829 (`<|observation|>`).
- The keys match what the `9906149` converter writes:
  - `head_count_kv` per layer (0 for KDA, 1 for DSA);
  - `key_length` 512 / `value_length` 512, `key_length_mla` 256 / `value_length_mla` 256, `rope.dimension_count` 0;
  - `kda.head_dim` 128, `kda.gate_lower_bound` -5;
  - `indexer.{head_count 32, key_length 128, top_k 2048, kpool 4}`;
  - `hyper_connection.{count 4, sinkhorn_iterations 20, epsilon 1e-6}`;
  - `leading_dense_block_count` 3, `expert_weights_scale` 2.5, `expert_weights_norm` true, `expert_gating_func` 2;
  - `swiglu_clamp_exp/shexp` 10 for all layers.
- **Tensor names match `9906149`.** For example, `indexer_compressor_gate/ape` is the name of `LLM_TENSOR_INDEXER_KPOOL_GATE/APE`.
- unsloth quantized `hc_*_fn` to Q8_0. The `9906149` converter forces it to F32. Both load.

### 1.4 What HEAD already has, and what is missing

What is still in HEAD:
- **ggml:** every op glm5-next uses, on CPU, Metal and CUDA: `LIGHTNING_INDEXER`, `TOP_K_MASK`, `DSV4_HC_PRE/COMB/POST`, `SWIGLU_CLAMP`, `GATED_DELTA_NET` (KDA gate), `SSM_CONV`, and `FLASH_ATTN_EXT` with DK=DV=512 and an `n_kv_max` bound. The only `ggml.h` change since `9906149` is Kolibri's `ROUND_NVFP4`.
- **From the Ling restore:**
  - KDA: `n_embd_head_kda`, `kda_gate_lower_bound`, the KDA branch of `n_embd_r/s`, `SSM_CONV1D_Q/K/V`, `SSM_F_A`, `SSM_G_A`, `SSM_BETA`, `SSM_A_NOSCAN`, `SSM_DT`, `SSM_NORM`;
  - MLA: `is_mla`, the K-only cache, `llm_graph_input_attn_k`, `build_inp_mem_hybrid_k`;
  - NextN skipping (`n_layer_nextn`, `crop_before/after_nextn`);
  - the swiglu clamp arrays, `build_rs`, `build_conv_state_store` and `build_recurrent_attn`.
- **From Kolibri:** the indexer KV keys (`INDEXER_KPOOL`, `INDEXER_KPOOL_SELECT_TAIL`, `INDEXER_TYPES`, ...), `is_indexer_full_impl`, and `llama_kv_cache(..., is_indexer)`.
- **Other:** `build_attn_mha(..., n_kv_max, ...)` (same signature as `9906149`), and the fused-node probe mechanism (`llm_graph_result::add_fused_node`, the probes at `src/llama-context.cpp:44`).

What is gone:

| Area | Missing in HEAD |
|---|---|
| `src/llama-arch.*` | `LLM_ARCH_GLM5_NEXT`. The KV keys `HYPER_CONNECTION_{COUNT,SINKHORN_ITERATIONS,EPSILON}`. The tensors `HC_{ATTN,FFN}_{FN,BASE,SCALE}`, `SSM_F_B`, `SSM_G_B`, `INDEXER_{K_NORM,PROJ,ATTN_K,ATTN_Q_B,KPOOL_GATE,KPOOL_APE}`, with their `LLM_TENSOR_INFOS` entries. `GLM5_NEXT` in `llm_arch_is_hybrid`. |
| `src/llama-hparams.*` | `dsv4_hc_mult`, `dsv4_hc_sinkhorn_iters`, `dsv4_hc_eps`, `indexer_n_head`, `indexer_head_size`, `indexer_kpool`, `indexer_kpool_select_tail`, `indexer_kpool_row`. |
| `src/llama-model.*` | The factory case, `LLM_TYPE_320B_A18B`, the NextN allow list (`src/llama-model.cpp:872`), the `print_info` lines, the rope type NONE (`:2321`), the `create_memory` branch (`:2045`), and the `llama_layer` fields `hc_*`, `ssm_f_b`, `ssm_g_b`, `indexer_*`. |
| Memory | `src/llama-memory-hybrid-idx.{h,cpp}` (deleted, about 1240 lines): recurrent + K-only MLA + an indexer cache with the k-pool layout, the stale tracking on sequence edits, and state save/load of all three caches. |
| `src/llama-graph.*` | The DSV4-style swiglu clamp (`ggml_swiglu_clamp`: the gate is clamped before silu). HEAD only has the bailingmoe3 form (silu clamped after), at `src/llama-graph.cpp:1478` and `:1854`. The `LLM_FUSED_OP_DSV4_HC_PRE/COMB/POST` and `LLM_FUSED_OP_TOP_K_MASK` enums. |
| `src/llama-cparams.h`, `src/llama-context.cpp` | `fused_dsv4_hc_pre/comb/post`, `fused_top_k_mask` and their 4 probes. `GLM5_NEXT` in the FP8-cache refusal (`:4074`). |
| `src/models/` | `glm5-next.cpp` (1077 lines at `9906149`) and its `models.h` struct. |
| `src/llama-vocab.*` | The `glm4`/`glm5` pre-tokenizer (`LLAMA_VOCAB_PRE_TYPE_CHATGLM4`, its regex, `special_bos_id = NULL`, `ignore_merges`, `clean_spaces = false`). HEAD throws "unknown pre-tokenizer" for every GLM GGUF. |
| `src/llama-model-saver.cpp` | The indexer k-pool and `hyper_connection.*` keys (needed for the test round-trip). |
| Python | `Glm5NextModel` (`9906149:conversion/glm.py:409-600`), `_set_vocab_glm`, the gguf-py arch, tensors, keys, writer methods (`add_indexer_kpool*`, `add_hyper_connection_*`) and tensor mappings, the `glm4` tokenizer hashes in `conversion/base.py` and `convert_hf_to_gguf_update.py`. |
| Chat | No GLM-specific parser existed at `9906149`: upstream handles GLM-4.x with the autoparser. `models/templates/GLM-4.6.jinja` and `GLM-4.7-Flash.jinja` are in HEAD with autoparser tests. There is no GLM-5.3 template. |
| mtmd (vision only) | `PROJECTOR_TYPE_GLM5V`, `mtmd_image_preprocessor_glm5v`, its graph, and `Glm5NextVisionModel` (`9906149:conversion/qwen3vl.py:265`). |

### 1.5 LongHaul numbers (computed from the unsloth headers, not measured)

| | UD-Q4_K_XL (186 GiB) | UD-IQ2_XXS (95 GiB) |
|---|---|---|
| Experts | 176.8 GiB (95.1%) | 88.6 GiB (93.4%) |
| Resident (incl. the MTP layer's non-expert part) | 9.2 GiB | 6.2 GiB |
| One expert (gate+up+down) | 14.5 MiB | 7.2 MiB |
| Expert layouts in the trunk | 3 (40 layers / layer 11 / layers 12, 44) | 3 (39 / 11 / 12, 44); layer 45 (MTP) has a 4th |
| Disk read per token with no hits | 42 x 8 x 14.6 MiB = **4.8 GiB** | **2.4 GiB** |
| Smallest cache (8 slots per group) | 381 MiB | 200 MiB |

The cache's split rule (reserve first, then share), applied to these layouts:

| Cache | Q4_K_XL max ubatch / experts cached | IQ2_XXS max ubatch / experts cached |
|---|---|---|
| 512 MiB (default) | 1 / 0.3% | 1 / 0.6% |
| 2 GiB | 2 / 1.1% | 5 / 2.3% |
| 4 GiB | 5 / 2.3% | 10 / 4.6% |
| 8 GiB | 10 / 4.5% | 21 / 9.3% |
| 16 GiB | 22 / 9.0% | no limit / 18% |

- **GLM is the heaviest model so far per token.** It reads about 4x Ling's bytes per token, and its experts are 4.4x Ling's size. Decode speed will be bounded by the SSD and the hit rate. With the 512 MiB default, prefill runs at ubatch 1.
- **Context memory, f16, per token:**
  - MLA latent: 11 layers x 512 x 2 B = 11 KiB;
  - indexer rows (key | gate | pooled): 11 x 384 x 2 B = 8.25 KiB;
  - total 19.3 KiB, so 2.4 GiB at 128K and **18.9 GiB at the 1M default context**.
- **Recurrent state:** about 146 MiB per sequence (34 x 4 MiB KDA state, plus conv).
- The MTP layer is skipped (D2), so its experts are never read, and it does not count as a cache group.

---

## 2. Decisions (my recommendation first; I need your answer on each)

| # | Question | Recommendation | Why |
|---|---|---|---|
| D1 | Scope: text only, or text + vision (mmproj)? | **Text only now.** Vision is a later, separate phase (section 10). | LongHaul is about the LLM experts. The text model has no RoPE, so images need no position work on the LLM side, which makes vision a fairly clean restore later (the `GLM5V` projector and preprocessor from `9906149`). It is still about 600 lines of mtmd plus a converter class. |
| D2 | The MTP layer: skip it on load, or bring back MTP drafting? | **Skip it**, as for Ling and Qwen 3.5. Layer 45 is created with `TENSOR_SKIP \| TENSOR_SKIP_IF_VIRTUAL \| TENSOR_NOT_REQUIRED`, and `n_layer()` excludes it. The converter keeps the layer by default, so files match the community GGUFs, and `--no-mtp` drops it. | Community GGUFs include the layer, so the loader must accept it. MTP drafting needs the MTP context type, graph and draft code that the prune removed on purpose. It also needs `index_share_mtp` (#27917, still open upstream) and LongHaul for the MTP experts. |
| D3 | `llama-memory-hybrid-idx`: restore verbatim, or trim the qwen4exp-only paths? | **Restore it, then trim** the qwen4exp-only paths: `indexer_kpool_by_order` (pools by cache order), `sinfos_kpool`, and `new_pool_pos` (M-RoPE pool positions). Keep everything glm5-next runs, including the stale tracking and state I/O. | qwen4exp is not in the fork. These branches would be dead code (about 150 lines), and AGENTS.md asks for simple code. The trimmed paths are never taken for glm5-next (`by_order` is always false), and the test plan (section 9) covers the paths that are kept. Alternative: verbatim, which is easier to diff against upstream but carries dead code. |
| D4 | Fused mHC and selection ops | **Restore the 4 fused-op probes** (`DSV4_HC_PRE/COMB/POST`, `TOP_K_MASK`) with their unfused fallbacks, as at `9906149`. | The unfused Sinkhorn alone is about 160 graph nodes per call, and there are 90 calls per token (2 per layer). The fused ops exist on CPU, Metal and CUDA, and the probe mechanism is already in HEAD, so this is 4 table entries and 4 flags. The fallback keeps backends that lack an op working. |
| D5 | Recurrent rollback (`rs_replay`) and draft-model speculative decoding | **Leave it off for glm5-next**, as for bailingmoe3 (Ling D6). Drop the `cparams.rs_replay` branch of `build_kda_conv1d`. | Only speculative decoding needs it, and D2 removes MTP. A draft model (`-md`) on a hybrid recurrent model needs rollback, so `-md` is refused or clamped for GLM, as it is for Ling today. Turning it on later is a small change: two arch lists and the restored branch. |
| D6 | LongHaul cache size for GLM | **Keep the 512M default.** The docs say to use at least 4 GiB (IQ2_XXS) or 8 GiB (Q4_K_XL), and print the table in section 1.5. | The default is shared by every model. At 512M GLM still loads, but at ubatch 1 with almost no hits. A per-arch default would be a new mechanism. Alternative: a one-line warning at context creation when the cache holds fewer than 2 ubatch tokens of every group. |
| D7 | Multi-device features (`-sm tensor`, `--kv-remote`, Thunderbolt/RPC) | **Out of scope.** `llm_arch_supports_sm_tensor` stays false, so `-sm tensor` fails with the existing clear error. The LongHaul expert split (`-es`) and two-GPU LongHaul are generic, so they get tests, not code. | `-sm tensor` needs per-arch split rules for MLA, the indexer, KDA and the mHC streams. Upstream never merged it (#31253 was closed). |
| D8 | Chat and tool calls | **Autoparser first.** Add the GLM-5.3-Flash template and test cases. Write a dedicated parser only if a case fails. | GLM-4.6 and 4.7 already pass the autoparser with the same `<arg_key>`/`<arg_value>` format. The GLM-5.3 differences are the reasoning effort line and `<think>` in the generation prompt, which the autoparser handles for Kolibri. |
| D9 | Context length default | **No code change.** The docs tell GLM users to set `-c` (for example 131072). `--fit` already lowers the context to fit memory. | The 1M default costs about 19 GiB of f16 cache plus the indexer inputs. Changing defaults is a policy for all models. |

---

## 3. Phase 0 - Last facts (no weights)

1. **Tokenizer hash:** download `tokenizer.json`, `tokenizer_config.json` and `config.json` to `.cache/glm53-tokenizer/` (hidden, in the workspace, as before). In a throwaway venv with `transformers>=5`, compute `chkhsh` the way `convert_hf_to_gguf_update.py` does. I expect it to be one of the existing `glm4` hashes, or a new one that resolves to `glm4`.
2. **Template:** save the HF `chat_template.jinja` and check it is byte-identical to the template in the unsloth GGUF. A copy from the header is already in my scratchpad.
3. **Backend check:** run `test-backend-ops -b MTL0` (and CPU) on the ops at GLM shapes:
   - `LIGHTNING_INDEXER` with 32 heads x 128;
   - `TOP_K_MASK` with pool indices and a tail;
   - `DSV4_HC_*` with hc=4;
   - `FLASH_ATTN_EXT` at DK=DV=512 with `n_kv_max = 4*512 + 3 = 2051`;
   - `GATED_DELTA_NET` with a KDA gate, and `SWIGLU_CLAMP`.

   If a shape has no case, add it (section 4). If Metal falls back for any op, the plan grows a ggml phase. I don't expect it: ggml is unchanged since `9906149`, where these ran.
4. **Upstream re-check** just before coding: `gh api compare b12812...master` for `glm5-next`, `hybrid-idx`, `conversion/glm.py`.
5. **Old quant rules:** read `9906149:src/llama-quant.cpp` for any glm5-next, `hc_` or `indexer_compressor` rules, and restore what is there.

---

## 4. Phase 1 - ggml (tests only, expected)

- No op or kernel change is planned.
- `tests/test-backend-ops.cpp`: add the GLM shapes from Phase 0 step 3 only where no case covers them. Likely candidates are `LIGHTNING_INDEXER` with 32 heads and `FLASH_ATTN_EXT` 512/512 with an `n_kv_max` near 2051.

---

## 5. Phase 2 - Runtime: the `glm5-next` model

Restore source: `9906149`. Keep upstream names (`dsv4_hc_*`, `LLM_TENSOR_INDEXER_KPOOL_*`), so a later upstream sync stays a plain diff.

### 5.1 `src/llama-arch.{h,cpp}`
- `LLM_ARCH_GLM5_NEXT` named `"glm5-next"`.
- KV keys `LLM_KV_HYPER_CONNECTION_COUNT`, `_SINKHORN_ITERATIONS`, `_EPSILON` (`%s.hyper_connection.*`).
- Tensors, with names and `LLM_TENSOR_INFOS` entries from `9906149`:
  - `HC_ATTN_FN/BASE/SCALE`, `HC_FFN_FN/BASE/SCALE`;
  - `SSM_F_B`, `SSM_G_B`;
  - `INDEXER_K_NORM`, `INDEXER_PROJ`, `INDEXER_ATTN_K`, `INDEXER_ATTN_Q_B`, `INDEXER_KPOOL_GATE` (`indexer_compressor_gate`), `INDEXER_KPOOL_APE` (`indexer_compressor_ape`).
- `GLM5_NEXT` in `llm_arch_is_hybrid` (`src/llama-arch.cpp:532`). It is not added to `llm_arch_supports_rs_rollback` (D5) or `llm_arch_supports_sm_tensor` (D7).

### 5.2 `src/llama-hparams.{h,cpp}`
- `dsv4_hc_mult`, `dsv4_hc_sinkhorn_iters`, `dsv4_hc_eps`.
- `indexer_n_head`, `indexer_head_size`, `indexer_kpool`, `indexer_kpool_select_tail`, and `indexer_kpool_row = 3` (key | gate | pooled).
- No `indexer_kpool_by_order` (D3).

### 5.3 `src/llama-model.{h,cpp}`
- The factory case; `LLM_TYPE_320B_A18B` ("320B.A18B").
- `GLM5_NEXT` in the NextN allow list (`:872`).
- `print_info`: add the MoE lines (shared with bailingmoe3), plus the indexer, k-pool and hc lines.
- `LLAMA_ROPE_TYPE_NONE` in `llama_model_rope_type` (`:2321`).
- `create_memory` (`:2045`): a `llama_memory_hybrid_idx` branch with the three filters from `9906149`:
  - attn = trunk, not recurrent;
  - idx = trunk, not recurrent, full indexer;
  - recr = trunk, recurrent.

  The MTP context branch is dropped (D2).
- `llama_layer`: `hc_attn_fn/base/scale`, `hc_ffn_fn/base/scale`, `ssm_f_b`, `ssm_g_b`, and `indexer_k_norm(_b)`, `indexer_proj`, `indexer_attn_k`, `indexer_attn_q_b`, `indexer_kpool_gate`, `indexer_kpool_ape`, where they are missing.

### 5.4 Memory: `src/llama-memory-hybrid-idx.{h,cpp}` (new, restored)
- Restore both files from `9906149`, add them to `src/CMakeLists.txt`, and trim per D3.
- Adapt the `llama_kv_cache` call to HEAD's constructor, which gained `is_indexer` for Kolibri. The indexer cache passes `rope_type = NONE`, so `is_indexer` stays false.
- `llama-memory-hybrid.h` in HEAD has the same constructor as at `9906149`, so the base class needs no change. Also check every `llama_kv_cache` method the file calls: `state_read_sinfo`, `state_clear`, `get_cells`, `seq_pos_min`, `prepare`, `get_stream`. I found all of them in HEAD.

### 5.5 `src/llama-graph.{h,cpp}`, `src/llama-cparams.h`, `src/llama-context.cpp`
- **Swiglu clamp:**
  - in `build_ffn` (`:1478`) and `build_moe_ffn` (`:1854`), `GLM5_NEXT` takes `ggml_swiglu_clamp(gate, up, limit)`, which clamps the gate from above before silu and clamps up on both sides;
  - bailingmoe3 keeps its own form;
  - the arch check is the one at `9906149`.

  The two forms differ for gate values above the limit. The LongHaul path goes through the same `build_moe_ffn`, so it is covered.
- **Fused ops (D4):**
  - `LLM_FUSED_OP_DSV4_HC_PRE/COMB/POST` and `LLM_FUSED_OP_TOP_K_MASK`;
  - the 4 probe entries next to the FLASH_ATTN ones (`src/llama-context.cpp:44`);
  - the 4 `cparams` flags (default true, resolved by the probes).
- **FP8 cache:** add `GLM5_NEXT` to the FP8 cache refusal (`:4074`). The graph reads the cache directly.

### 5.6 `src/models/models.h`, `src/models/glm5-next.cpp`
- Restore from `9906149`, minus:
  - `graph_mtp`, the `LLM_GRAPH_TYPE_DECODER_MTP` dispatch and `nextn_flags()` (D2);
  - the `rs_replay` branch of `build_kda_conv1d` (D5);
  - `build_cvec` (the fork removed control vectors).
- MTP layer flags: `TENSOR_SKIP | TENSOR_SKIP_IF_VIRTUAL | TENSOR_NOT_REQUIRED`, as in `bailingmoe3.cpp:119`.
- Routed experts: `TENSOR_LONGHAUL` (section 6).
- The graph is otherwise unchanged:
  - mHC pre, then norm, then KDA or DSA, then mHC post;
  - mHC pre, then norm, then dense FFN, or MoE plus shared expert, then mHC post;
  - then the stream mean, `output_norm` and the head.

  DSA layers attend only to the indexer selection, through `build_attn_mha(..., n_kv_max = n_sel)`.

### 5.7 `src/llama-vocab.{h,cpp}`
- `LLAMA_VOCAB_PRE_TYPE_CHATGLM4` for `tokenizer_pre` `"glm4"` or `"glm5"`, with the regex, `special_bos_id = LLAMA_TOKEN_NULL`, `ignore_merges = true` and `clean_spaces = false` from `9906149`.
- The template writes `[gMASK]<sop>` itself, so no BOS is added.

### 5.8 Other
- `src/llama-model-saver.cpp`: write `INDEXER_HEAD_COUNT/KEY_LENGTH/TOP_K/KPOOL/KPOOL_SELECT_TAIL/TYPES`, `HYPER_CONNECTION_*`, and the KDA and MLA keys when set, so the test models round-trip.
- `src/llama-quant.cpp`: whatever Phase 0 step 5 finds. Today 1-D tensors and `ssm_conv1d` already stay unquantized.

---

## 6. Phase 3 - LongHaul for GLM

### 6.1 What should work with no LongHaul code change
- The routed experts are `{4096, 2048, 288}` gate/up/down per MoE layer (3-44). The cache groups them by layout: 3 groups in both unsloth files (section 1.5).
- The Ling D4 rule (reserve first, then share) handles the one-layer and two-layer groups. The smallest cache is the exact sum: 381 MiB at Q4_K_XL, 200 MiB at IQ2_XXS.
- The MTP layer's experts are skipped, so they are not on disk in the cache.
- The router path (sigmoid, `exp_probs_b`, norm, scale 2.5, 1 group) is the existing `build_moe_ffn` with slots.
- The shared expert and the dense layers stay resident.
- mHC does not touch the MoE inputs: the MoE sees one `[n_embd, n_tokens]` input, as in every other model.
- `max_ubatch()` limits the ubatch as in section 1.5.

### 6.2 Tests (`tests/test-llama-archs.cpp`)
- Restore the `glm5-next` fixture from `9906149:tests/test-llama-archs.cpp` (`:313`, `:385-440`, `:537`, `:1143`), adapted to HEAD's helpers. The fixture is a small model with:
  - KDA and DSA layers, both full and shared indexer;
  - 1 dense lead layer and MoE layers;
  - hc=4 and kpool=4 with a small top-k, so that selection really drops cells.
- Cases:
  - CPU vs Metal vs BLAS;
  - roundtrip (saver);
  - mixed batch;
  - multi-sequence;
  - state save/load of all three caches, including a `seq_rm` mid-pool (the stale-pool path) and a `seq_cp`.
- Add `{ LLM_ARCH_GLM5_NEXT, true }` to the LongHaul table (`:2473`). This runs base, gate/up split, F16 down, two GPUs and the expert split.
- Add a mixed-layout variant (one layer's experts in another type) so two groups exist, as the real files have 3.
- Add the MTP-layer case (`add_nextn_layer`), checking that the file loads and the layer is skipped.
- `test-llama-archs --kv-shift`: the model has no RoPE and is recurrent, so shifting must be refused cleanly, or be a no-op where the hybrid memory allows it. Check whichever HEAD's hybrid does for bailingmoe3.
- `test-llama-context`: the KV cases. The recurrent-state cases are skipped, as for Kolibri and Ling.

---

## 7. Phase 4 - Conversion

- `conversion/glm.py`: add `Glm5NextModel` from `9906149:conversion/glm.py:409-600`, with `_set_vocab_glm`, register it in `conversion/__init__.py`, and check the `model.language_model.` prefix strip in `base.py`. Behavior:
  - FP8 dequantization through the existing `conversion/base.py` path;
  - experts merged per layer;
  - `kv_b_proj` split into `k_b` (transposed) and `v_b`;
  - conv1d reshaped to `{1, d_inner, 1, d_conv}`;
  - `A_log` written as `-exp(A_log)`;
  - `dt_bias` renamed to `dt_proj.bias`;
  - the `hc_*` and `index_kpool_compress_*` names get `.weight`;
  - small mHC and gating tensors forced to F32;
  - MTP kept by default, `--no-mtp` drops it, `--mtp-only` per the existing flag;
  - `model.visual.*` skipped by the text model.
- `gguf-py`:
  - `MODEL_ARCH.GLM5_NEXT` and its tensor list;
  - the keys `KPOOL`, `KPOOL_SELECT_TAIL` (if missing) and `HYPER_CONNECTION_*`;
  - writer methods `add_indexer_head_count/key_length/top_k/kpool/kpool_select_tail/types` and `add_hyper_connection_*`, where missing;
  - tensor mappings for `indexer.{wk, wq_b, weights_proj, k_norm, index_kpool_compress_gate/ape}`, `hc_{attn,ffn}_{fn,base,scale}`, `f_b_proj`, `g_b_proj`, from `9906149:gguf-py/gguf/tensor_mapping.py:1776-1830`.
- Tokenizer:
  - the `glm4` hash entries in `conversion/base.py` and `convert_hf_to_gguf_update.py`;
  - a GLM-5.3-Flash entry if Phase 0 finds a new hash (pre-computed line, as upstream does).
- Tests in `tests/test_conversion_fp8.py`:
  - a synthetic GLM5-Next checkpoint with 1 KDA layer (dense), 1 DSA layer (MoE, FP8 block scales), the MTP layer and a few vision tensors;
  - check the written keys, names, shapes and dtypes, the `--no-mtp` output, and that `visual.*` is skipped;
  - a name check of the converter's names against the unsloth header tensor list (section 1.3).

---

## 8. Phase 5 - Chat (D8)

- Add `models/templates/GLM-5.3-Flash.jinja` (the HF file).
- Add it to `tests/test-chat-analysis.cpp` and the autoparser tables in `tests/test-chat-auto-parser.cpp`. Cases:
  - a plain reply;
  - reasoning (`<think>` opened by the generation prompt);
  - a tool call with 2 args (string and JSON);
  - parallel tool calls;
  - a tool response turn;
  - `reasoning_effort` low/high/max, and the caps probe;
  - `clear_thinking` true/false history.
- Run `test-chat`, `test-chat-auto-parser` and `test-chat-analysis`. Write a parser (`common/parsers/glm5.cpp`) only if a case fails, and ask you first.

---

## 9. Phase 6 - Docs

- `docs/longhaul.md`:
  - GLM sizing (section 1.5 tables);
  - the cache size advice (D6);
  - the `-c` advice (D9);
  - text only (D1), MTP skipped (D2), no `-md` (D5), no `-sm tensor` (D7);
  - the known model bug #31875.
- `README.md`: add GLM-5.3-Flash to the model list. `docs/development/HOWTO-add-model.md`: note the hybrid-idx memory.

---

## 10. Later phases (not in this plan's scope; listed so nothing is lost)

- **Vision (D1):**
  - `Glm5NextVisionModel` (`9906149:conversion/qwen3vl.py:265`);
  - `PROJECTOR_TYPE_GLM5V` with its preprocessor and graph in `tools/mtmd`;
  - the image tokens.

  No LLM-side position work is needed (no RoPE).
- **MTP drafting (D2):** after #27917 settles upstream.
- **`-sm tensor`, `--kv-remote` (D7).**
- **Rollback for `-md` (D5).**

---

## 11. Tests run before each commit (code-level only)

- Build: Metal (this Mac).
- `test-backend-ops -b MTL0` for the touched ops.
- `test-llama-archs`: all archs, plus `--kv-shift`.
- `test-llama-context`.
- `ctest -L main`.
- `test-chat`, `test-chat-auto-parser`, `test-chat-analysis`, `test-unicode`, with the GLM regex cases added the way Ling added `bailingmoe2`.
- `tests/test_conversion_fp8.py`, plus flake8 and ty on the touched Python files.
- CUDA is not built here, as before.

---

## 12. Risks

- **No reference check:** under the constraints there is no check against real weights or vLLM. Mitigations:
  - the restored code is byte-for-byte upstream's at `b12812`, apart from the D2/D3/D5 removals, and upstream users run it;
  - the tests compare backends and paths.
- **Speed:** 4.8 GiB of expert reads per token at Q4_K_XL with no hits. Decode will be SSD-bound until the hit rate is high. Real numbers need a run you would do yourself.
- **Host cost:** the k-pool inputs (`set_input_kpool`) are built on the CPU every ubatch, in O(n_kv). Their cost at long context is not measured (the same open item as Kolibri).
- **The D3 trim** touches code that also has stale-pool and state logic. The section 6.2 state tests are the guard.
- **The swiglu clamp form** is easy to get wrong, because it silently changes outputs slightly. A unit test compares the GLM path against `ggml_swiglu_clamp` on values above the limit.
- **Model bug #31875** (reasoning loops) will look like a fork bug to users. The docs point to the upstream reports.
- **Upstream may change `glm5-next`** before this lands. Phase 0 step 4 re-checks.

---

## 13. Files touched (summary)

| File | Change |
|---|---|
| `src/llama-arch.{h,cpp}` | arch, 3 KV keys, 14 tensors, hybrid list |
| `src/llama-hparams.h` | 8 fields |
| `src/llama-model.{h,cpp}` | factory, type, NextN allow, print_info, rope type, create_memory, layer fields |
| `src/llama-memory-hybrid-idx.{h,cpp}` | new (restored, trimmed) |
| `src/CMakeLists.txt` | add the memory file and `models/glm5-next.cpp` |
| `src/llama-graph.{h,cpp}` | swiglu clamp form, 4 fused-op enums |
| `src/llama-cparams.h`, `src/llama-context.cpp` | 4 fused flags and probes, FP8 refusal |
| `src/models/models.h`, `src/models/glm5-next.cpp` | new (restored, MTP/replay/cvec removed, experts LongHaul) |
| `src/llama-vocab.{h,cpp}` | `glm4`/`glm5` pre-tokenizer |
| `src/llama-model-saver.cpp` | indexer and hc keys |
| `src/llama-quant.cpp` | only if Phase 0 finds rules |
| `tests/test-backend-ops.cpp` | GLM shapes, only if missing |
| `tests/test-llama-archs.cpp` | fixture, LongHaul table, mixed layout, MTP, state |
| `tests/test-chat*.cpp`, `models/templates/GLM-5.3-Flash.jinja` | template and cases |
| `tests/test-unicode.cpp` | GLM regex cases |
| `conversion/glm.py`, `conversion/__init__.py`, `conversion/base.py`, `convert_hf_to_gguf_update.py` | converter, registration, hashes |
| `gguf-py/gguf/{constants,gguf_writer,tensor_mapping}.py` | arch, keys, writers, mappings |
| `tests/test_conversion_fp8.py` | synthetic GLM checkpoint |
| `docs/longhaul.md`, `README.md`, `docs/development/HOWTO-add-model.md` | docs |

---

## 14. Implementation notes (2026-11-27)

All phases 0-6 are done with the section 2 recommendations. Nothing is committed. The notes below are what differs from the plan or what was learned.

- **Phase 0:**
  - the tokenizer hash is `cdf5f353...`, the same as GLM-4.7-Flash; it is added as a pre-computed `glm4` entry;
  - the released `tokenizer_config.json` names `TokenizersBackend`, so converting needs transformers >= 5 (the repo pins 4.57.6). The workspace venv `.cache/venv-glm53` has transformers 5.24.1, torch 2.11.0, flake8 and ty 0.0.100.
- **Phase 2:**
  - HEAD had lost `f_norm_eps` and `LLM_NORM` in the prune; both are back, for the indexer key LayerNorm (eps 1e-6);
  - the NextN tensors are created but not stored, since `llama_layer` has no `nextn` member in HEAD.
- **Phase 3:**
  - `test_longhaul_bailingmoe3` became `test_longhaul_nextn(arch)` and runs for `bailingmoe3` and `glm5-next`;
  - `test_state_glm5_next` checks save/load, `seq_rm` all, `seq_cp` (split and unified), and that a `seq_rm` inside a pool is refused;
  - `--kv-shift` reports SKIP for `glm5-next` (no RoPE).
- **Stale pool tracking:** a mutation that turned off the stale marking in `llama-memory-hybrid-idx.cpp` failed no test. The stale paths cannot be reached today: the recurrent cache refuses a `seq_rm` inside a sequence, and a full clear re-pools everything. They become reachable with rollback (section 10, `-md` or MTP). That work must add a test that fails when the marking is off.
- **Phase 4:**
  - the converter follows the `bailingmoe3.py` conventions of HEAD;
  - checked against the real safetensors index (names only) and both unsloth headers: all 1412 tensor names match, and all KV values match. Differences, all expected:
    - the converter also writes `indexer.kpool_select_tail` and `indexer.types` (the loader defaults are the same);
    - it does not write `indexer.index_share_mtp`, which only MTP drafting reads (#27917);
  - `expert_shared_feed_forward_length` is written, as in the unsloth files;
  - the vocab step on the released tokenizer gives the same `pre` and special token ids as the unsloth files;
  - FP8 128x128 blocks go through the generic dequant. The preserve-FP8 path only takes per-tensor scales, so it never applies to this checkpoint.
- **Phase 5 (D8):**
  - the autoparser handles the template, so no GLM parser was written;
  - the prompts that llama.cpp renders (effort levels, a tool turn, `clear_thinking`) match transformers' jinja2 output byte for byte;
  - the template has no `enable_thinking` switch: the generation prompt always opens `<think>`. The caps probe reports effort levels `low` and `high` (max is the fallback), with none rejected.
- **Phase 6:** `docs/longhaul.md` has a GLM sizing section with the section 1.5 tables and the D6, D9, D1, D2, D5 and D7 notes. `README.md` and `docs/development/HOWTO-add-model.md` list the arch.
- **Still open:** section 10, plus a real run (speed, hit rate, output quality), which only you can do.

