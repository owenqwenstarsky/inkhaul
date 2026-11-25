# Ling-3.2-air on LongHaul: plan

Goal: run [inclusionAI/Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air) in this fork, with its routed experts on disk behind the LongHaul cache, the same way Inkling 2 Mini runs today.

Constraints: the model is not downloaded or run. All checks are code-level tests that run in memory (random tiny models, synthetic conversions). Commits follow AGENTS.md (`Assisted-by:` trailer, each one approved by you, nothing pushed).

Status: approved and implemented (uncommitted), all decisions as recommended. See section 13 for what was done, the deviations and the follow-ups.

---

## 1. Facts (checked 2026-11-24)

### 1.1 Upstream status

- Upstream llama.cpp has **no Ling-3.2-specific PR or issue**, and none is needed. Ling-3.2-air uses the HF class `BailingMoeV3VLForConditionalGeneration`, which upstream already converts and runs as arch `bailingmoe3`:
  - added for Ling 3.0 in #26608 (merged 2026-08-17);
  - VL support in #29151 (merged 2026-09-26);
  - DSpark drafter support in #27508 (merged 2026-08-22).
- Upstream master (79c69f4) has not changed `bailingmoe3` since our fork point. Our fork base matches upstream 5fddd0b.
- Community GGUFs already exist. bartowski's Q4_K_M header, read with HTTP range requests and no weights, shows:
  - `general.architecture = bailingmoe3`, `block_count = 43` (42 trunk layers + 1 MTP layer), `nextn_predict_layers = 1`;
  - `tokenizer.ggml.pre = bailingmoe2`;
  - `rope.dimension_sections = [8,12,12,0]`;
  - `expert_group_count = 8`, `expert_group_used_count = 4`, `expert_gating_func = 2` (sigmoid);
  - per-layer `swiglu_clamp_exp` / `swiglu_clamp_shexp`;
  - split into 3 files.
- **In our history:** the fork had `bailingmoe3` until the prune commit `3453f63`. **Commit `9906149` has `bailingmoe3` and LongHaul together**, so it is the restore source for everything below. Its `conversion/bailingmoe3.py` and `tools/mtmd/models/ling3vl.cpp` are byte-identical to upstream master.

### 1.2 The model

- **Size:** 124B parameters in total, about 5.5B active per token. 2560 hidden, vocab 157184, context 262144.
- **42 trunk layers**, in groups of 6:
  - **35 KDA linear-attention layers**: recurrent, with q/k/v short convs, a safe gate and an RMS-norm output gate.
  - **7 MLA full-attention layers** (5, 11, ..., 41): no q LoRA, kv_lora_rank 512, rope dim 64, a head-wise sigmoid output gate, and a K-only cache.
- **FFN:**
  - Layers 0-1: dense FFN (6144).
  - Layers 2-41: MoE with 512 experts, 8 used per token, separate `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` (768 wide), and one shared expert.
  - The router uses sigmoid scoring with an expert bias, picks 4 of 8 groups, then normalizes the weights and scales them by 2.5.
- **SwiGLU clamps** on layers 34-41 (shared expert) and 35-41 (routed experts).
- **Layer 42 is an MTP (NextN) layer:** MLA attention plus MoE, plus `eh_proj`, `enorm` and `hnorm`.
- **The HF checkpoint is FP8 e4m3 with 128x128 block scales.** The converter in our tree still dequantizes these (`conversion/base.py:426-490`).
- **What changed from Ling 3.0** (from a diff of the reference modeling code):
  - The SwiGLU limits are applied now. llama.cpp already applies them.
  - The rotary class is M-RoPE in the "video_rope" layout. For text this is the same as 1-D RoPE.
  - The vision output width is 4608 instead of 4096.
  - Nothing else in the text path changed.

### 1.3 A bug in every current GGUF of this model (upstream too)

- **The HF model rotates the MLA rope dims in adjacent pairs** `(2i, 2i+1)`: `rope_interleave=True`, see `apply_rotary_pos_emb_interleave`.
- **llama.cpp pairs them differently.** For a VL file (one with `rope.dimension_sections`), `llama_model_rope_type` returns MROPE (`9906149:src/llama-model.cpp:4008-4010`). M-RoPE always uses NEOX pairing `(i, i+32)`: `ggml.h:1724` says it "cannot be disabled".
- **The converter does not permute the weights to make up for it.**
- **Effect:** the 7 MLA layers and the MTP layer get the wrong rotation. Output stays coherent, so the bug goes unnoticed, but it differs from the reference.
- **The fix here, for text:** run `bailingmoe3` with NORM rope (GPT-J pairing, which is the HF pairing) and ignore the sections.
  - With text-only positions all three M-RoPE axes are equal, so this matches HF exactly.
  - It fixes existing GGUFs with no reconversion.
- **For images**, an "interleaved-pair M-RoPE" mode would be needed in ggml (CPU, Metal, CUDA), plus HF's centred image positions in mtmd. See section 9.

### 1.4 What the prune removed that `bailingmoe3` needs

What is still in HEAD:
- **ggml:** every op and kernel `bailingmoe3` uses, on CPU, Metal and CUDA. That covers GATED_DELTA_NET with a KDA gate, SSM_CONV, ARGSORT/TOP_K, FILL, SET_ROWS, FLASH_ATTN_EXT with DK=576/DV=512, and MROPE. The ggml diffs against `9906149` are empty.
- **Graph plumbing:**
  - the recurrent state helpers `build_rs` and `build_conv_state_store`;
  - the `rs_replay` code;
  - the NextN embedding API;
  - in `build_moe_ffn`: expert-group routing, sigmoid gating and `exp_probs_b`.

What is gone:

| Area | Missing in HEAD |
|---|---|
| `src/llama-arch.*` | `LLM_ARCH_BAILINGMOE3`. The KV keys `KEY/VALUE_LENGTH_MLA`, `SWIGLU_CLAMP_EXP/SHEXP`, `EXPERT_WEIGHTS_NORM`. The tensors `SSM_CONV1D_Q/K/V`, `SSM_F_A`, `SSM_BETA`, `SSM_A_NOSCAN`, `SSM_DT`, `SSM_G_A`, `SSM_NORM`, `ATTN_Q_A/Q_B/KV_A_MQA/K_B/V_B/Q_A_NORM/KV_A_NORM`, `FFN_*_SHEXP`, `NEXTN_*` and `LAYER_OUT_NORM`, with their `LLM_TENSOR_INFOS` entries. `BAILINGMOE3` in `llm_arch_is_hybrid`. |
| `src/llama-hparams.*` | `n_layer_nextn` (and `n_layer()` excluding the NextN layers), the MLA dims (`is_mla()`, `n_embd_head_k/v_mla()`), `n_layer_dense_lead`, `n_lora_q/kv`, `n_ff_shexp`, `n_expert_shared`, `expert_weights_norm`, `expert_gating_func`, `kda_safe_gate`, `kda_gate_lower_bound`, the swiglu clamp arrays. **The KDA branch of `n_embd_r()`**: HEAD returns 0 for this model, which would break the conv state. |
| `src/llama-model.*` | The factory case, `llm_type` 7.9B-A1.3B / 124B-A5.1B, `print_info` lines, the rope type, the MLA `n_offs` in `get_rope_params` (for K-shift), the `create_memory` branch, `nextn_flags()`, and the `llama_layer` fields (`wq_a`, `wkv_a_mqa`, `attn_q_a_norm`, `attn_kv_a_norm`, `ffn_*_shexp`, `ssm_*`, `nextn`). HEAD throws on `nextn_predict_layers > 0`. |
| Memory | `src/llama-memory-hybrid.{h,cpp}` (deleted). The K-only KV cache (`has_v = !is_mla()`); HEAD always allocates V. |
| `src/llama-graph.*` | `llm_graph_input_attn_k` + `build_attn_inp_k`, the K-only `build_attn` overload, `llm_graph_input_mem_hybrid_k` + `build_inp_mem_hybrid_k`, and the swiglu clamps in `build_moe_ffn` and `build_ffn`. |
| `src/models/` | `bailingmoe3.cpp`, `delta-net-base.cpp`, and their `models.h` structs. |
| `src/llama-vocab.*` | The `bailingmoe2` pre-tokenizer (`LLAMA_VOCAB_PRE_TYPE_BAILINGMOE`, regex, `clean_spaces = false`). HEAD throws "unknown pre-tokenizer". |
| `src/llama-quant.cpp` | The `ssm_conv1d` "do not quantize" rule. |
| `src/llama-model-saver.cpp` | The keys the tests need to round-trip a `bailingmoe3` model. |
| Python | `conversion/bailingmoe3.py`, `Qwen3VLVisionModel` (for the mmproj), the `--mtp`/`--no-mtp` flags, the `bailingmoe2` tokenizer hash and update entry, and the gguf-py arch, tensors, keys, writer methods and tensor mappings. |
| Chat | `common/parsers/ling3.cpp` and its detection, the test-chat cases, and the Ling template. HEAD's autoparser "Bailing V3" patch looks for `Bailing V3 chat template`, but the 3.2 template starts with `Bailing V3 unified chat template`, so the patch does not fire. The removed ling3 detection (`<role>ASSISTANT</role>` + `<arg_key>`) does match the 3.2 template. |
| mtmd (only for vision) | `ling3vl.cpp`, `PROJECTOR_TYPE_LING3VL`, the dyn-size preprocessor, `build_inp_temporal_merge`, the vision position input, and the image start/end tokens. |

### 1.5 LongHaul numbers (estimates from the config, not measured)

- **Experts:** 40 MoE layers x 512 experts x 3 matrices of 2560x768 = 120.8B parameters, about 97% of the model.
- **Resident part:** about 3.6B parameters (KDA and MLA projections, shared experts, dense FFN, router, embeddings and output). That is about 2.5-3 GB at Q4_K_M. It fits a 24 GB Mac with room to spare.
- **Expert size:** about 3.3 MB at Q4_K (gate+up+down).
- **Disk read per token with no hits:** 40 x 8 x 3.3 MB, about 1.1 GB.
- **At 512 MiB the cache holds about 150 slots out of 20480 experts**, so expect a low hit rate. Ling is much heavier per token than Inkling 2 Mini. I would recommend at least 2-4 GiB for Ling in the docs. The default stays 512M.
- **Expert types vary by layer in community quants.** bartowski Q4_K_M has 6 layouts:
  - layers 2-10: Q4_K/Q6_K/Q6_K;
  - 27 layers: all Q4_K;
  - layers 31-32: Q5_K down;
  - layer 35: Q6_K down;
  - layer 41: Q8_0;
  - MTP layer 42: Q4_0.
- **That breaks the cache's budget split.** Each layout becomes one cache group, and the budget is split by bytes. A one-layer group gets about 1/41 of the budget: at 512 MiB that is about 3 slots, fewer than the 8 one token needs. So:
  - **context creation fails below about 1.1 GiB;**
  - the ubatch clamp drops to 1.
- See decision D4.

---

## 2. Decisions (my recommendation first; I need your answer on each)

| # | Question | Recommendation | Why |
|---|---|---|---|
| D1 | Scope: text only, or text + vision (mmproj)? | **Text only now.** Vision is a later, separate phase (section 9). | LongHaul is about the LLM experts. Correct vision needs a new ggml rope mode on 3 backends plus HF's centred image positions (section 1.3). Restoring `ling3vl` as is would reproduce the upstream mismatch. |
| D2 | The MTP layer: skip it on load, or bring back MTP speculative decoding? | **Skip it.** `nextn_predict_layers` is accepted for `bailingmoe3`, the NextN layer's tensors are created with `TENSOR_SKIP`, and `n_layer()` excludes it. The converter keeps the MTP layer by default (so files match upstream GGUFs) and gets `--no-mtp` back. | Community GGUFs include the layer, so the loader must accept it. Restoring MTP drafting means bringing back a whole subsystem (`LLAMA_CONTEXT_TYPE_MTP`, the graph type, the draft in `common/speculative.cpp`, about 600 lines) that the fork removed on purpose, and its experts would need LongHaul too. |
| D3 | MLA rope: how to fix the pairing bug (section 1.3)? | **Return `LLAMA_ROPE_TYPE_NORM` for `bailingmoe3` always.** The converter stays identical to upstream (it still writes the sections). | This fixes the existing community GGUFs with no reconversion, and it is a one-line change. The alternative, permuting rope rows in the converter, would fix only files we convert, and it would break those files under NORM. Reporting the bug upstream is your call; I won't file anything. |
| D4 | LongHaul budget split with several expert layouts | **Reserve first, then share.** (1) Find the largest ubatch `U >= 1` such that giving every group `min(U*n_expert_used, its expert count)` slots costs at most half the budget. (2) Split the rest by expert bytes, as today. The minimum size becomes the exact sum for `U = 1`, not `max(.../share)`. A single group behaves exactly as today. | The ubatch need is the same for every group whatever its layer count, because slots are reused layer by layer within a ubatch. The hit rate grows with bytes. Today one-layer groups starve, so context creation fails and the ubatch clamp drops to 1. This changes LongHaul for any model with mixed expert types, including Inkling files with mixed quants. Alternative: a floor of `n_expert_used` slots per group only (smaller change, but prefill stays at ubatch 1 for mixed files). |
| D5 | Chat and tool calls | **Restore the `ling3` parser and its detection, and add the Ling-3.2 template and tests.** Also fix the autoparser marker to match "Bailing V3" templates in general. | Without a parser, tool calls and thinking are not parsed for Ling-3.2. The removed parser's detection matches the 3.2 template (checked against the GGUF's template bytes). |
| D6 | Recurrent rollback (`rs_replay`, `n_rs_seq`) | **Leave it off for `bailingmoe3`.** Restore `delta-net-base` without the replay path. | Only speculative decoding needs it, and D2 removes that for Ling. That is about 150 fewer lines. |

---

## 3. Phase 0 - Last facts (no weights)

1. **Tokenizer hash.**
   - Download only `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` of Ling-3.2-air (about 8 MB) into a hidden cache dir `.cache/ling32-tokenizer/` in the workspace (gitignored).
   - Compute its `chkhsh` with the same code as `convert_hf_to_gguf_update.py`, and confirm it equals the `bailingmoe2` hash `9b1be57e...`. If it differs, add a second hash line that maps to `bailingmoe2`. The community GGUFs already say `bailingmoe2`, so a match is likely.
   - This needs `transformers` locally; I'll check that it is installed and ask before installing anything.
2. **Reference splits for the pre-tokenizer test.** With the `tokenizers` library and the same files, record the pre-tokenizer splits for about 20 strings (contractions, digits, CJK, whitespace, newlines). These are data for the C++ test in 4.7.
3. **Chat template.** Save the 3.2 `chat_template.jinja` (MIT) as `models/templates/inclusionai-Ling-3.2-air.jinja`, and restore `inclusionai-ling-3.0-flash.jinja` from `9906149`.

---

## 4. Phase 1 - Runtime: the `bailingmoe3` text model

Restore from `9906149`, trimmed to what the text model uses. Each item names what changes for Inkling (shared code).

### 4.1 `src/llama-arch.{h,cpp}`

- Add back `LLM_ARCH_BAILINGMOE3` / `"bailingmoe3"`.
- Add back these KV keys: `ATTENTION_KEY_LENGTH_MLA`, `ATTENTION_VALUE_LENGTH_MLA`, `SWIGLU_CLAMP_EXP`, `SWIGLU_CLAMP_SHEXP`, `EXPERT_WEIGHTS_NORM`.
- Add back these tensors, each with its `LLM_TENSOR_NAMES` and `LLM_TENSOR_INFOS` entry (`llm_tensor_info_for` uses `.at()`): `SSM_CONV1D_Q/K/V`, `SSM_F_A`, `SSM_BETA`, `SSM_A_NOSCAN`, `SSM_DT`, `SSM_G_A`, `SSM_NORM`, `ATTN_KV_A_MQA`, `ATTN_K_B`, `ATTN_V_B`, `ATTN_KV_A_NORM`, `ATTN_Q_A`, `ATTN_Q_B`, `ATTN_Q_A_NORM`, `FFN_GATE_SHEXP`, `FFN_UP_SHEXP`, `FFN_DOWN_SHEXP`, `NEXTN_EH_PROJ`, `NEXTN_ENORM`, `NEXTN_HNORM`, `LAYER_OUT_NORM`.
- Add `BAILINGMOE3` to `llm_arch_is_hybrid`. Do not add it to `llm_arch_supports_rs_rollback` (D6).

### 4.2 `src/llama-hparams.{h,cpp}`

- Add back the fields listed in 1.4, plus `use_mrope()` if the vision phase comes later. It is not needed for D3.
- `n_layer_nextn`: `n_layer()` returns `n_layer_all - n_layer_nextn`. Inkling has 0, so nothing changes for it.
- `n_embd_r()`: add back the KDA branch `3*(ssm_d_conv-1)*n_head*n_embd_head_kda`, placed so that Inkling's `n_sconv_channels` branch is checked first.
- Add back `is_n_embd_k_gqa_variable` only if the K-only cache needs it. I'll check during implementation.

### 4.3 `src/llama-model.{h,cpp}`

- **Factory:** `case LLM_ARCH_BAILINGMOE3`.
- **`llm_type`:** add back `LLM_TYPE_7_9B_A1_3B` and `LLM_TYPE_124B_A5_1B` with their names (Ling-3.0-tiny and flash/air).
- **`load_hparams`:**
  - Keep reading `n_layer_nextn`, but throw only for archs other than `BAILINGMOE3`.
  - Zero-fill the swiglu clamp arrays for every arch, because `build_ffn` reads them.
- **`print_info`:** add back the bailing block (dense lead, shared FF, weights scale/norm, gating, nextn).
- **`llama_model_rope_type`:** return `LLAMA_ROPE_TYPE_NORM` for `BAILINGMOE3` (D3), with a one-line comment about the pairing.
- **`get_rope_params`:** add back `n_offs = n_embd_head_k - n_rot` when `n_lora_kv > 0`, so a K-shift rotates only the k_pe tail of the MLA cache. `llama-kv-cache.cpp` still honours `n_offs`.
- **`create_memory`:** add a `BAILINGMOE3` branch before the Inkling hybrid-iswa branch. It builds `llama_memory_hybrid` with null filters (attention = not recurrent) and `rs_replay = false`.
- **`llama_layer`:** add back `wq_a`, `wkv_a_mqa`, `attn_q_a_norm`, `attn_kv_a_norm`, `ffn_gate/up/down_shexp`, `ssm_q/k/v_conv`, `ssm_f_a`, `ssm_beta`, `ssm_a`, `ssm_dt_b`, `ssm_g_a`, `ssm_o_norm`, and the `llama_layer_nextn nextn` struct. All of these are `ggml_tensor *`, because the saver walks the struct as an array of pointers.
- **`nextn_flags()`:** simplified, since there is no `load_mtp`. The trunk is always required, and the NextN layer is always `TENSOR_SKIP`, marked `TENSOR_NOT_REQUIRED` if the file has no NextN layer.

### 4.4 Memory

- **Restore `src/llama-memory-hybrid.{h,cpp}`** from `9906149` and add it to `src/CMakeLists.txt`. Its constructor calls still match HEAD's `llama_kv_cache` and `llama_memory_recurrent` signatures.
- **K-only cache:** in `src/llama-kv-cache.cpp`, set `has_v = !hparams.is_mla()` again (HEAD hard-codes `true`). Inkling is not MLA, so nothing changes for it.

### 4.5 `src/llama-graph.{h,cpp}`

- **Restore** `llm_graph_input_attn_k` (with `set_input`, `can_reuse`), the K-only `build_attn(llm_graph_input_attn_k *, ..., v_mla, ...)`, `llm_graph_input_mem_hybrid_k`, and `build_inp_mem_hybrid_k()`.
- **Do not restore** `build_attn_inp_k`, which only the MTP graph used.
- **`build_moe_ffn`:** add back the routed-expert swiglu clamp for `swiglu_clamp_exp[il] > 0`:
  - `up = clamp(up, -l, l)`, `gate = clamp(silu(gate), -inf, l)`, `cur = gate*up`;
  - placed before the LongHaul slot lookup consumers, with no change to the cache path;
  - only reached when the array value is non-zero, so Inkling (SITU) is unaffected.
- **`build_ffn`:** add back the shared-expert clamp for `LLM_FFN_SILU + LLM_FFN_PAR` with `swiglu_clamp_shexp[il] > 0`. DSpark and Inkling are unaffected because the arrays are zero-filled.

### 4.6 `src/models/`

- **`delta-net-base.cpp` and `models.h`:**
  - Restore `llm_build_delta_net_base` with `build_delta_net` and `build_recurrent_attn` (non-replay path only, asserting `n_rs_seq == 0`), plus `build_gdn_l2_norm`.
  - Drop `build_conv_state`, `build_conv_state_replay` and `build_recurrent_attn_replay` (D6).
- **`bailingmoe3.cpp`:** restore it and drop `graph_mtp`, the `LLM_GRAPH_TYPE_DECODER_MTP` dispatch, the `rs_replay` branch in `build_causal_conv1d`, and the `build_cvec` call (control vectors are gone).
  - Change the experts to `TENSOR_LONGHAUL`: `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps` of trunk MoE layers get `trunk_flags | TENSOR_LONGHAUL`, as in `inkling2.cpp:155-156`. The router, `exp_probs_b` and the shared expert stay resident.
  - The NextN layer: every tensor is created with `TENSOR_SKIP`, so the file loads and its experts are never mapped or cached.

### 4.7 `src/llama-vocab.{h,cpp}`

- Add back `LLAMA_VOCAB_PRE_TYPE_BAILINGMOE` and its regex (`9906149:src/llama-vocab.cpp:493-499`).
- Map `bailingmoe`, `bailingmoe2` and `llada-moe` to it with `clean_spaces = false`.
- `tokenizer.ggml.normalizer.nfc` stays ignored. Upstream never applied it to BPE.

### 4.8 Other

- **`src/llama-quant.cpp`:** add back `ssm_conv1d` to the "do not quantize" rule.
- **`src/llama-model-saver.cpp`:** add back the keys `bailingmoe3` needs, so the tests can round-trip it: `LEADING_DENSE_BLOCK_COUNT`, `EXPERT_SHARED_*`, `SWIGLU_CLAMP_*`, `EXPERT_WEIGHTS_NORM`, `EXPERT_GATING_FUNC`, `NEXTN_PREDICT_LAYERS`, `Q/KV_LORA_RANK`, `KEY/VALUE_LENGTH_MLA`, `KDA_SAFE_GATE`, `KDA_GATE_LOWER_BOUND`.
- **`src/llama-context.cpp`:** no change expected. `rs_replay` stays false (D6), and `n_rs_seq` is clamped to 0 because the arch is not in the rollback list.

---

## 5. Phase 2 - LongHaul for Ling

### 5.1 Separate gate/up/down, dense lead layers, the NextN layer

- **Separate tensors:** `llama_moe_cache_layer_experts` already collects separate gate/up/down. A group's banks then hold 3 tensors per slot. No code change, but it is untested today, so the tests in 5.4 cover it.
- **Dense lead layers** have no experts and are skipped by the cache, as today.
- **The NextN layer** has null expert tensors (`TENSOR_SKIP`), so it is skipped too.

### 5.2 Budget split (D4) in `src/llama-moe-cache.cpp`

- **New split**, replacing the per-group `share` loop (around line 580):
  1. `reserve(U) = sum_g alloc_size(g, min(U*n_expert_used, n_owners*n_expert))`.
  2. Take the largest `U >= 1` with `reserve(U) <= size/2`. If `reserve(1) > size`, throw. The message gives `reserve(1)` as the minimum.
  3. Split `size - reserve(U)` among the groups by `host_bytes`, and add it to each group's reserved slots. The existing per-device split stays in front of this.
- **One group:** the result is the same as today, because the whole budget goes to it either way. The tests check this.
- **Startup log:** each group's slots and the resulting `max_ubatch()` are already logged. Add the reserved `U`.
- **Trace header:** `LLAMA_MOE_CACHE_TRACE` writes the split. Update the header fields if the simulator needs them, and bump `MCT1` to `MCT2` only if the format changes.

### 5.3 Hybrid memory with the cache

- **Batch splitting:** recurrent layers need `split_equal` ubatches. The `n_ubatch` clamp (`llama-context.cpp:456-466`) applies before the scheduler reserve, as for Inkling. Nothing changes, but the tests run a prompt longer than the clamped ubatch on the hybrid model.
- **`graph_max_nodes`:** the cache allowance is per MoE layer. `bailingmoe3` overrides `graph_nodes_per_tensor` (32). Check the reserve with 40 cached layers in the test.

### 5.4 Tests (`tests/test-llama-archs.cpp`)

**Fixture.** Restore the `bailingmoe3` fixture keys from `9906149`:
- hybrid per-layer `HEAD_COUNT_KV`;
- the MLA dims (576/512/64/192/128);
- KDA head dim 128 and conv 4;
- the clamp arrays;
- sigmoid gating, a shared expert and a dense lead layer.

Add `EXPERT_GROUP_COUNT = 4` / `USED = 2` so group routing runs through the cache, and `NEXTN_PREDICT_LAYERS = 1` with an MTP layer in the file, so the skip path is covered.

**Generic arch tests.** `bailingmoe3` goes back into `test_backends` and `test_kv_shift`. The `kda.safe_gate` key-removal case is restored. Add it to `moe_mandatory`.

**LongHaul cases**, as for Inkling:
- **Variants:** F32, Q8_0 and NVFP4 experts.
- **Cache sizes:** the minimum, 5 slots, and "all experts".
- **Resident reference:** the cache must match `LLAMA_LONGHAUL_RESIDENT` within the existing NMSE bound.
- **Inputs:** decode, small batches, and a prompt longer than the clamped ubatch.
- The `bank_size` helper reads the 3 expert tensors of the first MoE layer (blk.1, because blk.0 is dense).

**New LongHaul cases:**
- **Mixed layouts:** quantized variants give one layer a different expert type, which makes a one-layer group. The test checks that:
  - context creation succeeds at `reserve(1)`;
  - `max_ubatch()` follows D4;
  - logits match the reference.
- **Single group:** slot counts for a single-group Inkling model are unchanged by D4 (regression check).
- **Split GGUF:**
  - write a random `bailingmoe3` model as 2 split files with the gguf API: KVs copied, `split.no`/`split.count`/`split.tensors.count`, tensors divided between the files, experts in both;
  - load it with LongHaul and compare against the single-file logits.
  - This covers `llama_tensor_weight.idx` in `longhaul_read`.
- **Error cases:** a cache below `reserve(1)` fails with a message that names the minimum.

### 5.5 Other tests

- **`tests/test-tokenizer-bailing.cpp`** (new, small): `unicode_regex_split` with the `bailingmoe2` regex on the Phase 0 strings, compared with the HF splits. It is in memory and needs no vocab file.
- **Rope type:** a unit check that a `bailingmoe3` file with `rope.dimension_sections` gets `LLAMA_ROPE_TYPE_NORM` and `n_pos_per_embd() == 1`.

---

## 6. Phase 3 - Conversion (Python)

- **`conversion/bailingmoe3.py`:** restore it from `9906149` (identical to upstream), with `BailingMoeV3Model` and `BailingMoeV3VLModel`. Skip `BailingMoeV3VLVisionModel` under D1.
- **`conversion/__init__.py`:** register `BailingMoeV3ForCausalLM` and `BailingMoeV3VLForConditionalGeneration` in the text map. Add the mmproj map entry only with vision.
- **`conversion/base.py`:**
  - add back the `supports_mtp_export` / `mtp_only` / `no_mtp` class attributes;
  - add the `bailingmoe2` chkhsh line to `get_vocab_base_pre` (Phase 0 result);
  - add the `FFN_*_SHEXP`, `SSM_BETA` and `NEXTN_EH_PROJ` entries back to the FP8 `scale_types`.
- **`convert_hf_to_gguf.py`:** add back `--mtp` (MTP layer only) and `--no-mtp`, with their mutual exclusion and the class-flag setting. Drop the parts for `--dspark` and `--ple`.
- **`convert_hf_to_gguf_update.py`:** add back the `bailingmoe2` entry (`inclusionAI/Ling-mini-base-2.0`).
- **gguf-py:**
  - `constants.py`: `MODEL_ARCH.BAILINGMOE3` and its tensor list, the tensor enums and names from 1.4, `Keys.KDA.SAFE_GATE` / `GATE_LOWER_BOUND`, `Keys.Attention.Q/KV_LORA_RANK`, `LLM.LEADING_DENSE_BLOCK_COUNT`, `EXPERT_SHARED_FEED_FORWARD_LENGTH`, `EXPERT_SHARED_COUNT`, `EXPERT_WEIGHTS_NORM`.
  - `gguf_writer.py`: `add_kda_safe_gate`, `add_kda_gate_lower_bound`, `add_q_lora_rank`, `add_kv_lora_rank`, `add_leading_dense_block_count`, `add_expert_shared_feed_forward_length`, `add_expert_shared_count`, `add_expert_weights_norm`.
  - `tensor_mapping.py`: the shared-expert, `final_layernorm`, `dt_proj`, `A_log`, `o_norm`, q/k/v conv1d, `b_proj`, MLA and `eh_proj`/`enorm`/`hnorm` entries.
- **Tests** (`tests/test_conversion_fp8.py`, in memory, no weights):
  - Restore the generic `convert` / `_convert` harness (it writes a tiny `config.json` + `model.safetensors` to a temp dir) without the imports of removed modules.
  - Add a synthetic Ling-3.2-air case: a VL wrapper config with `text_config`, 2 trunk layers (one KDA, one MLA, so `layer_group_size = 2`), 1 dense lead layer, 4 experts in 2 groups, an MTP layer, FP8 128x128 block scales, and vision tensors that must be skipped.
  - The case checks the output tensor names and shapes, the KVs (sections, clamps, groups, `nextn`), the k_b/v_b split, `A_log -> exp`, the conv reshape, and `--no-mtp` dropping layer 2.
- **Lint:** flake8 and ty on the touched files, as CI does.

---

## 7. Phase 4 - Chat (D5)

- **Parser:**
  - restore `common/parsers/ling3.cpp`, its `parsers.h` declaration and its `sources.cmake` entry;
  - restore the detection in `common/chat.cpp` (`<role>ASSISTANT</role>` + `<arg_key>`) before the autoparser.
- **Autoparser patch:** `common/chat-diff-analyzer.cpp:196-202` matches on `Bailing V3` plus `chat template`, so it covers "unified" too.
- **Tests:**
  - `tests/test-chat.cpp`: restore the Ling 3.0 cases (about 20 `peg_tester` cases) and run the same cases on the 3.2 template;
  - add a `json_schema` case, which the old block lacked;
  - extend `tests/test-chat-auto-parser.cpp` for the "unified" header.

---

## 8. Phase 5 - Docs

- **`docs/longhaul.md`:** a Ling-3.2-air section covering:
  - what is cached;
  - the sizing table from 1.5;
  - the mixed-layout note and D4, with the advice to quantize experts to one type (`llama-quantize --tensor-type 'ffn_.*_exps=q4_k'`) for the best prefill;
  - the recommended cache size.
- **`README.md`:** a quick start for Ling-3.2-air next to Inkling. The fork notice mentions both models.
- **`docs/development/HOWTO-add-model.md`:** one line that `bailingmoe3` is supported.
- **`docs/multimodal.md`:** says that Ling vision is not supported (under D1).
- **This file:** a status section per phase, as in `LONGHAUL_PLAN.md`.

---

## 9. Later (not in this plan unless you choose it in D1/D2)

- **Vision:**
  - restore `Qwen3VLVisionModel` (alone; `conversion/qwen3vl.py` imports removed modules), `BailingMoeV3VLVisionModel` and `ling3vl.cpp` (deriving from `mtmd_graph` directly);
  - in mtmd: the dyn-size preprocessor, `build_inp_temporal_merge`, the 4-section vision positions and the image start/end tokens;
  - **also**:
    - (a) a new ggml rope mode, M-RoPE with GPT-J pair rotation and the "video_rope" axis layout, on CPU, Metal and CUDA, with `test-backend-ops` cases;
    - (b) HF's centred image positions and "+1 after image" text positions in mtmd;
    - (c) exact erf GELU in the projector.
- **MTP drafting:** restore the MTP context and draft. The NextN layer's experts would go through LongHaul as their own group.
- **DSpark drafter for Ling** (upstream #27508): not checked for Ling-3.2.

---

## 10. Tests run before each commit (code-level only)

- **Build:** `cmake --build` for the touched targets, Metal build on this Mac.
- **C++ tests:** `test-llama-archs` (all Inkling cases plus the new `bailingmoe3` cases), `test-llama-mmap`, `test-arg-parser`, `test-chat`, `test-chat-auto-parser`, the new tokenizer test, `ctest -L main`, and `ctest -L behavior -LE 'model|python'`.
- **Python:** `tests/test_conversion_fp8.py`, flake8 and ty.
- **CUDA:** built and run only if you run it on your CUDA machine. I can't run CUDA here. The kernels are unchanged since `9906149`, but the CUDA `ssm_conv` needs `d_inner % 128 == 0`, so the fixture uses `d_inner = 128`.

---

## 11. Risks

- **The math is checked against the reference code, not against real outputs.** No model is run. A future run should compare a few tokens against the HF model.
- **D3 diverges from upstream on purpose.** Our files and outputs will differ from upstream llama.cpp until upstream fixes the pairing.
- **D4 changes LongHaul behaviour** for every file with mixed expert types (Inkling included). Single-layout files keep exactly today's split, and the tests check that.
- **Shared-code edits touch Inkling paths:** `n_layer()` with NextN, the `n_embd_r()` branch order, `has_v`, the swiglu clamp in `build_ffn`, and `create_memory`. Every Inkling test must keep passing unchanged.
- **Speed:** about 1.1 GB read per token with no hits. Ling will be slow at small cache sizes, more so than Inkling.
- **Fork maintenance:** `bailingmoe3` adds about 2.5k lines back. Upstream fixes to `bailingmoe3`, delta-net or MLA need manual porting.

## 12. Files touched (summary)

| File | Change |
|---|---|
| `src/llama-arch.{h,cpp}` | arch, keys, tensors, infos, hybrid predicate |
| `src/llama-hparams.{h,cpp}` | MLA, KDA, bailing fields, `n_layer_nextn`, `n_embd_r` KDA branch |
| `src/llama-model.{h,cpp}` | factory, types, nextn handling, clamps, print_info, rope type NORM, `n_offs`, `create_memory`, layer fields, `nextn_flags` |
| `src/llama-memory-hybrid.{h,cpp}`, `src/CMakeLists.txt` | restored |
| `src/llama-kv-cache.cpp` | `has_v = !is_mla()` |
| `src/llama-graph.{h,cpp}` | K-only attention input and `build_attn`, hybrid K input, swiglu clamps |
| `src/models/models.h`, `bailingmoe3.cpp`, `delta-net-base.cpp` | restored (no MTP, no replay), experts `TENSOR_LONGHAUL`, NextN `TENSOR_SKIP` |
| `src/llama-vocab.{h,cpp}` | `bailingmoe2` pre-tokenizer |
| `src/llama-quant.cpp`, `src/llama-model-saver.cpp` | `ssm_conv1d` rule, saver keys |
| `src/llama-moe-cache.cpp` | D4 budget split |
| `conversion/`, `convert_hf_to_gguf*.py`, `gguf-py/` | Ling converter, MTP flags, tokenizer hash, gguf-py entries |
| `common/parsers/ling3.cpp`, `parsers.h`, `sources.cmake`, `common/chat.cpp`, `common/chat-diff-analyzer.cpp` | Ling chat parser |
| `models/templates/` | Ling 3.0 and 3.2 templates |
| `tests/test-llama-archs.cpp`, `tests/test-chat*.cpp`, `tests/test_conversion_fp8.py`, `tests/test-tokenizer-bailing.cpp`, `tests/CMakeLists.txt` | tests from 5.4, 5.5, 6, 7 |
| `docs/longhaul.md`, `README.md`, `docs/development/HOWTO-add-model.md`, `docs/multimodal.md` | docs |

---

## 13. Status (2026-11-24)

| Phase | Status |
|---|---|
| 0 - Facts | Done. The Ling-3.2-air tokenizer hashes to the `bailingmoe2` chkhsh `9b1be57e...`. The splits and the template are in `.cache/ling32-tokenizer/`. |
| 1 - Runtime | Done. |
| 2 - LongHaul | Done, D4 included. |
| 3 - Conversion | Done. |
| 4 - Chat | Done. The 3.0 parser and template are byte-identical to `9906149`; the 3.2 template is the HF file. |
| 5 - Docs | Done: `docs/longhaul.md` (D4, Ling sizing computed from the bartowski Q4_K_M header), `README.md`, `HOWTO-add-model.md`, `multimodal.md`. |

**Tests run** (all pass, Metal on this Mac, no CUDA): `test-llama-archs` (backends, 62 LongHaul cases, rope type) and `--kv-shift`, `ctest -L main` (22 tests), `test-chat`, `test-chat-auto-parser`, `test-unicode`, `tests/test_conversion_fp8.py` (14), flake8 and ty on the touched Python files. A name check found every tensor the converter writes for KDA, MLA and MTP layers in the real GGUF header.

**Deviations from the plan:**
- D4 reserves `min(U*n_expert_used, n_expert)` slots per group, not `n_owners*n_expert`: the need of a ubatch stops at the experts of one layer, as in `max_ubatch()`.
- MLA caches are not rotated (`attn_rot`): V is a view of the K row and goes through `wv_b`.
- `llama-model-saver.cpp` wrote per-layer arrays for `n_layer` blocks only; it now writes `n_layer_all`, as the loader reads them (needed for files with a NextN layer).
- LongHaul always copies the selected experts before the slot lookup (`llama-moe-cache.cpp`). The copy was skipped for a contiguous top-k, which happens when the last layer is MoE and only 1 token is output, so the graph topology changed with the ubatch and `test-llama-context` (rf case, `GGML_SCHED_DEBUG_REALLOC=1`) aborted. Inkling's test model did not reach it.
- The `bailingmoe2` regex test is in `tests/test-unicode.cpp` (15 HF cases), not a new `test-tokenizer-bailing.cpp`. The rope type check is in `test-llama-archs`.
- gguf-py has no `ATTN_KV_B`: the converter always splits `kv_b_proj` into `attn_k_b` / `attn_v_b`, and this fork's C++ has no `attn_kv_b`.
- The test-model file for the MTP case is the saved fixture plus a copied MoE layer and the NextN tensors (`add_nextn_layer`), and the split case writes 2 files with the gguf API (`write_splits`).

**Follow-ups (not done):**
- **NFC normalizer:** Ling's GGUF sets `tokenizer.ggml.normalizer.nfc = true`, and the prune (`cc3378b`) removed the code that applied it. Input that is not NFC (decomposed accents, Hangul jamo) tokenizes differently from HF. Restoring it is about 30 lines (`9906149:src/llama-vocab.cpp`).
- The Bailing V3 autoparser patch in `common/chat-diff-analyzer.cpp` is now unreachable for Ling templates (the `ling3` parser matches first). It can be removed.
- CUDA was not built or run here.
- Section 9 (vision, MTP drafting) is unchanged.

