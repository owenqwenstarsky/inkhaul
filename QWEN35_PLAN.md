# Qwen 3.5 / 3.6 + LongHaul plan

Goal: bring back Qwen 3.5 and Qwen 3.6 (dense and MoE) to the Inkling fork, and run the MoE models with LongHaul (routed experts on disk, fixed-size expert cache in RAM), the same way Inkling 2 Mini runs today.

Status: implemented (Phases 0-6), not committed. No real model was run. See section 9a for what changed from this plan.

Reference trees:
- Upstream: `ca45ee8` (last upstream commit merged into master). It has full `qwen35` / `qwen35moe` support.
- Fork: `HEAD` (`ce1c159`). Qwen was removed in `3453f63`, `cc3378b` and `979a756`.

No model is downloaded or run during this work. Only code-level tests that run in memory (and the existing Metal test binaries).

---

## 1. Decisions (made by me, override any of them)

| # | Decision | Why |
|---|---|---|
| Q1 | **Two archs: `qwen35` (dense) and `qwen35moe` (MoE).** Qwen 3.6 needs no new arch. | Upstream maps Qwen 3.6 onto the same HF classes (`Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`) and the same archs. No `qwen36` exists anywhere upstream. |
| Q2 | **Port the upstream model files close to verbatim**, minus what the fork removed: no MTP graph, no Hadamard (`hdmd`), no control vectors (`build_cvec`), no `-sm tensor` split code. | Every ggml op they use (`ggml_gated_delta_net_ext`, `ggml_ssm_conv`, IMROPE, softplus) is still in HEAD with CPU and Metal kernels. A near copy keeps future upstream fixes easy to port. |
| Q3 | **Restore `llama-memory-hybrid.{h,cpp}`** and `llm_graph_input_mem_hybrid` from upstream. Do not bend `llama_memory_hybrid_iswa` to fit. | Qwen 3.5 needs a non-SWA KV cache plus recurrent state with `rs_replay`. `hybrid_iswa` requires SWA, hard-codes `rs_replay = false` and exposes Inkling's `rel` inputs. The memory, kv-cache and batch APIs the upstream file uses are unchanged in HEAD. |
| Q4 | **MTP layers: load and skip.** Real Qwen 3.5/3.6 GGUFs carry `nextn_predict_layers = 1` and one extra `blk.{n_layer}` block. Restore `hparams.n_layer_nextn` (`n_layer() = n_layer_all - n_layer_nextn`) and create the MTP tensors with `TENSOR_SKIP`. No MTP speculative decoding. | Without this, the common GGUFs do not load (HEAD throws "NextN/MTP layers are not supported"). Bringing back MTP drafting means about 530 lines in `common/speculative.cpp` plus a context type and batch plumbing the fork removed on purpose. It can come later. |
| Q5 | **LongHaul applies to `qwen35moe` routed experts only.** The router (`ffn_gate_inp`), the shared expert (`ffn_*_shexp`, `ffn_gate_inp_shexp`), attention, gated delta net, norms, embeddings and output stay resident. Dense `qwen35` has no LongHaul (nothing to stream). | Same rule as Inkling (D1 of `LONGHAUL_PLAN.md`): weights that run on every token stay in memory. The shared expert runs on every token. |
| Q6 | **Reuse `llama_moe_cache` unchanged** where possible. It already handles separate `gate`/`up`/`down` experts, merged `gate_up`, and groups layers by expert layout (type, shape, stride), so mixed-quant GGUFs work. | Fewer moving parts. The Inkling-only pieces are the `TENSOR_LONGHAUL` flags in `inkling2.cpp` and the loop dedupe, which does nothing for layers that do not alias. |
| Q7 | **Fix `create_tensor_gate_up_exps`** so the merged `ffn_gate_up_exps` tensor gets `TENSOR_LONGHAUL` from the caller's flags. Today it creates it with `TENSOR_NOT_REQUIRED` only and drops the flags. | Without the fix, a GGUF with merged gate_up experts would map them resident. The cache would then throw "layer has experts both on disk and in memory". |
| Q8 | **Tool calls: restore the specialized `common/parsers/qwen3-coder.cpp`** (191 lines) and its detection in `common/chat.cpp`. | The autoparser does not handle Qwen 3.5's real output well: it expects tool arguments in definition order (Qwen does not always follow it), it does not accept `<tool_call>` closing a think block with no `</think>`, and it keeps extra newlines in the reasoning. The upstream parser handles all three and has tests. This partly undoes the Phase 3 parser removal, but only for this one format. |
| Q9 | **Vocab: restore the `qwen35` and `qwen2` pre-tokenizers** (two enum values, two regex cases, two name branches). | Qwen 3.5 GGUFs use `tokenizer.ggml.pre = "qwen35"`, older ones `"qwen2"`. HEAD throws on any pre-tokenizer other than `inkling2`. The fast regex splitters for both are still in `src/unicode.cpp`. |
| Q10 | **Vision: restore the `qwen3vl_merger` projector (images only, no video)** as a separate, last phase. | Qwen 3.5/3.6 are natively multimodal and their mmproj files use this projector. It's about 450 lines of C++ in `tools/mtmd`, independent of LongHaul, so it should not block the text work. |
| Q11 | **Python conversion: port a trimmed converter** (text + vision): `Qwen2MoeModel` -> `Qwen3NextModel` -> V-reorder base -> `Qwen3_5TextModel` / `Qwen3_5MoeTextModel`, and `Qwen3VLVisionModel`. Always drop `mtp.*`. Drop Hadamard, Mistral-format and the NVFP4 V-reorder repack. FP8 checkpoints keep working through the generic base code. | Lets users make their own GGUFs. It also fixes the DSpark converter, which looks up the target's model class and cannot convert a Qwen 3.6 drafter today. NVFP4 repack is about 95 lines for a rare input format and can come later. |
| Q12 | **No `LLM_TYPE_*` restore.** Qwen models report `UNKNOWN` size type in the log. | Purely cosmetic. HEAD keeps only `LLM_TYPE_UNKNOWN`. |
| Q13 | **Order: core model -> LongHaul -> chat -> conversion -> vision -> docs.** Every phase builds and passes the tests before the next starts. | Phase 1 gives a resident reference that Phase 2's LongHaul tests compare against. Chat, conversion and vision are independent of each other. |
| Q14 | Commits follow AGENTS.md: `Assisted-by: Claude` trailer, not `Co-authored-by`. Each commit needs your explicit approval. Nothing is pushed. | AGENTS.md rule, same as `LONGHAUL_PLAN.md` D14. |

---

## 2. Current state (HEAD)

- **Gone:** the arch enum/names, tensor tables, `llama_layer` fields, model files, memory class, graph input, vocab pre-types, converter, gguf-py lists, mtmd projector and parser for Qwen 3.5.
- **Still there (no work):**
  - ggml: `ggml_gated_delta_net_ext` (`ggml/include/ggml.h:2442`), `ggml_ssm_conv` (`:2307`), the Metal `gated_delta_net.metal` kernel, IMROPE. `test-backend-ops` still covers both.
  - Recurrent memory with replay rows (`src/llama-memory-recurrent.cpp:105-135`), `n_embd_r()` / `n_embd_s()` for the delta-net case, the `ssm_*` hparams.
  - Graph helpers: `build_rs`, `build_conv_state_store`, `build_qkv`, `build_ffn`, `build_moe_ffn`, `build_attn(llm_graph_input_attn_kv *)`, `crop_*_nextn`, `t_layer_inp`.
  - Model base: `create_tensor_qkv`, `create_tensor_gate_up_exps` (both with `TENSOR_SKIP` handling), `TENSOR_SKIP_IF_VIRTUAL`.
  - All the KV keys Qwen 3.5 reads.
  - The template `models/templates/Qwen3.5-4B.jinja`. There is no separate Qwen 3.6 template upstream.
  - `unicode_regex_split_custom_qwen2` / `_qwen35` (`src/unicode.cpp:648,784`).
  - Generic M-RoPE plumbing in mtmd (`MTMD_POS_TYPE_MROPE`, chosen from `llama_model_rope_type`).
  - `tests/fusion/MTL.csv` still has `qwen35` / `qwen35moe` rows.
- **LongHaul today:** generic cache in `src/llama-moe-cache.cpp`. `TENSOR_LONGHAUL` is set only in `src/models/inkling2.cpp:155-156`. `test_longhaul` (`tests/test-llama-archs.cpp:997-1129`) hard-codes the Inkling archs and the tensor names `blk.0.ffn_gate_up_exps` / `ffn_down_exps`.

---

## 3. Phase 0 - Facts (before any code)

1. Read `config.json` of the target models from the Hugging Face API (no weights): Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, Qwen3.6-35B-A3B, and the dense 27B for each. Record `hidden_size`, layers, experts, experts per token, `moe_intermediate_size`, shared expert size, `full_attention_interval`, and the linear-attention head counts.
2. Read the GGUF header of one common quant of Qwen3.6-35B-A3B (header only, by HTTP range request, as `test-quant-type-selection --generate` already does). Confirm:
   - The pre-tokenizer string.
   - `nextn_predict_layers` and the MTP tensor names.
   - Whether the experts are merged `ffn_gate_up_exps` or separate `ffn_gate_exps` + `ffn_up_exps`.
   - The quant type per expert tensor and per layer. This sets how many cache groups LongHaul builds.
3. Work out the LongHaul numbers with the formula from `docs/longhaul.md`: `E` = bytes per expert (gate + up + down), `L` = MoE layers, `k` = experts per token. Disk read per token with no hits is about `L*k*E`, slots at 512 MiB are about `512 MiB / E`, and the `n_ubatch` clamp is `slots / k`. Rough numbers from memory, which step 1 must confirm:

   | Model | L | k | E at Q4_K | No-hit read/token | Slots at 512 MiB | n_ubatch clamp |
   |---|---|---|---|---|---|---|
   | 35B-A3B (3.5 and 3.6) | 40 | 8 | ~1.8 MB | ~0.57 GB | ~290 | 36 |
   | 122B-A10B | 48 | 8 | ~5.3 MB | ~2.0 GB | ~100 | 12 |
   | 397B-A17B | 60 | 10 | ~7.1 MB | ~4.3 GB | ~72 | 7 |

   The 35B-A3B is the realistic target on a 24 GB Mac. The resident part of the 397B is several GB and needs checking against 24 GB.

---

## 4. Phase 1 - Core model, resident (`src/`)

Experts are plain resident tensors in this phase. This gives the reference for Phase 2.

### 4.1 Arch and tensor tables (`src/llama-arch.{h,cpp}`)

- `LLM_ARCH_QWEN35` / `LLM_ARCH_QWEN35MOE` with names `"qwen35"` / `"qwen35moe"`. Add both to `llm_arch_is_hybrid` and `llm_arch_supports_rs_rollback`.
- `llm_tensor` enum, `LLM_TENSOR_NAMES` and `LLM_TENSOR_INFOS` for:
  - `ATTN_QKV_GATE`
  - `SSM_CONV1D`, `SSM_DT`, `SSM_A_NOSCAN`, `SSM_BETA_ALPHA_CAT`, `SSM_BETA`, `SSM_ALPHA`, `SSM_NORM`, `SSM_OUT`
  - `FFN_GATE_INP_SHEXP`, `FFN_GATE_SHEXP`, `FFN_UP_SHEXP`, `FFN_DOWN_SHEXP`
  - The six `NEXTN_*` tensors. The loader looks up tensor info before it honors `TENSOR_SKIP`, so the MTP tensors need entries even though they are skipped.
- Not `CLS_OUT` (sequence-classification variants only).

### 4.2 Layer and hparams (`src/llama-model.h`, `src/llama-hparams.{h,cpp}`)

- `llama_layer`: `ssm_conv1d`, `ssm_dt`, `ssm_a`, `ssm_beta`, `ssm_alpha`, `ssm_norm`, `ssm_out`, `wqkv_z`, `ssm_beta_alpha_cat` and the four `ffn_*_shexp`.
- `hparams.n_ff_shexp`.
- `hparams.n_layer_nextn`, with `n_layer()` returning `n_layer_all - n_layer_nextn` (HEAD returns `n_layer_all`; for Inkling both are the same, since `n_layer_nextn` is 0).

### 4.3 Base loader (`src/llama-model.cpp`)

- `load_hparams`: store `nextn_predict_layers` in `hparams.n_layer_nextn` instead of throwing. Inkling and DSpark still reject a value > 0 in their own `load_arch_hparams`.
- Factory (`llama_model_mapping`, line 38): the two new classes.
- `create_memory`: a `qwen35`/`qwen35moe` branch before the Inkling hybrid branch. It uses `llama_memory_hybrid` with filters `il < n_layer() && !is_recr(il)` (attention) and `il < n_layer() && is_recr(il)` (recurrent), and passes `n_rs_seq` and `rs_replay`.
- `llama_model_rope_type`: both archs -> `LLAMA_ROPE_TYPE_IMROPE`. Without it RoPE silently falls to `NONE`.
- The `.scale` / `.input_scale` pass for `ssm_out`, `ssm_beta`, `ssm_alpha` and `*_shexp`, so pre-quantized FP8/NVFP4 GGUFs do not fail with "wrong number of tensors" (about 30 lines, from `ca45ee8:src/llama-model.cpp:~1994-2030`).
- `print_info`: the SSM block (about 10 lines).

### 4.4 Memory and graph input

- `src/llama-memory-hybrid.{h,cpp}` from `ca45ee8` (141 + 286 lines), added to `src/CMakeLists.txt`.
- `llm_graph_input_mem_hybrid` (class, `set_input`, `can_reuse`) and `build_inp_mem_hybrid()` in `src/llama-graph.{h,cpp}` (about 80 lines).
- `src/llama-context.cpp:85`: `cparams.rs_replay = n_rs_seq > 0 && arch is qwen35 or qwen35moe`.

### 4.5 Model files (`src/models/`)

- `delta-net-base.cpp` (274 lines) plus `llm_build_delta_net_base` and `build_gdn_l2_norm` in `models.h`. Drop the `rep_cols_p` use (HEAD's `llm_graph_input_rs` has only `rep_cols_r`, since the PLE state is gone).
- `qwen35.cpp` and `qwen35moe.cpp` (about 510 + 530 lines after trimming). The rule for the MTP layer: `mtp_flags = TENSOR_SKIP`, so `create_tensor_qkv` and `create_tensor_gate_up_exps` skip their tensors. `build_arch_graph` returns only `graph`.
- Class declarations in `models.h`.

### 4.6 Vocab (`src/llama-vocab.{h,cpp}`)

- `LLAMA_VOCAB_PRE_TYPE_QWEN2` and `LLAMA_VOCAB_PRE_TYPE_QWEN35`, their regexes (`ca45ee8:src/llama-vocab.cpp:392-406`) and the `"qwen2"` / `"qwen35"` name branches with `clean_spaces = false` (about 25 lines).

### 4.7 Quantize (`src/llama-quant.cpp`)

- Re-add `quantize &= name.find("ssm_conv1d") == npos;`. Without it, `llama-quantize` would quantize the 2-D conv kernel.

### 4.8 Tests

- `tests/test-llama-archs.cpp`:
  - `SSM_INNER_SIZE = 256` for both archs (the fixture default `2*n_embd` doesn't fit them).
  - `QWEN35MOE` in `moe_mandatory` / `moe_implemented`.
  - Both archs run in `test_backends` (CPU vs Metal), `test_kv_shift` and `save_models` automatically, since those loop over `llama_arch_all()`.
  - A case with `nextn_predict_layers = 1` and an extra MTP block in the file, which must load and give the same logits as the file without it.
- `tests/test-tokenizer-0` (build-only in HEAD) with `models/ggml-vocab-qwen35.gguf` (+ `.inp` / `.out`) restored from git history (`3453f63^`, a real 5.9 MB GGUF, not an LFS pointer). Registered as a test in `tests/CMakeLists.txt` as upstream did. It becomes the fork's only running tokenizer test.
- `test-backend-ops` already covers `GATED_DELTA_NET`, `SSM_CONV` and IMROPE. Run those ops on Metal.

---

## 5. Phase 2 - LongHaul for `qwen35moe`

### 5.1 Experts on disk

- `src/models/qwen35moe.cpp`: create `ffn_gate_exps`, `ffn_up_exps` / `ffn_gate_up_exps` and `ffn_down_exps` of the main layers with `TENSOR_LONGHAUL`. MTP layer experts stay `TENSOR_SKIP` (never created, never cached).
- `src/llama-model.cpp` `create_tensor_gate_up_exps`: create the merged tensor with `TENSOR_NOT_REQUIRED | (flags & TENSOR_LONGHAUL)` (Q7).
- Nothing changes in `src/llama-moe-cache.cpp` unless Phase 0 or the tests show it must:
  - Layers whose expert pointers are null (the skipped MTP layer) already get no cache entry.
  - The shared expert goes through `build_ffn`, not `build_moe_ffn`, so the cache never sees it.
  - Each layer is its own owner (no loop aliasing), so the dedupe is a no-op.

### 5.2 Checks to do, not expected to need code

- **Mixed quant types.** Dynamic quants can give `ffn_down_exps` a different type in some layers. The cache then builds one group per layout, each needing at least `k` slots, and splits the budget by bytes. Check that 512 MiB stays above the minimum for the layouts Phase 0 finds. If it doesn't, the existing error names the minimum size.
- **Recurrent state + ubatch clamp.** A clamped `n_ubatch` (36 for the 35B) with the hybrid memory's equal split. The test covers a prompt longer than the clamp.
- **`rs_replay` rollback** re-runs the last ubatch, which simply means another set of lookups.
- **Node budget.** `graph_max_nodes` already adds `32*n_layer_all` for LongHaul. Check the Qwen MoE graph fits with a tiny tensor count (the test does this).
- **Server.** Context checkpoints for hybrid models are unchanged. One context, so one cache.

### 5.3 Tests (`tests/test-llama-archs.cpp` `test_longhaul`)

- Make the arch list and the tensor names data-driven: `{INKLING2, INKLING2_LOOP, QWEN35MOE}`.
- For `qwen35moe` run two expert layouts: merged `gate_up` and separate `gate` + `up`. Add one file with a different type for `ffn_down_exps` in one layer (two cache groups).
- The same cache configs as Inkling: minimum slots (eviction on nearly every lookup), a cache that holds everything, and 512 MiB. Logits must match the resident reference (`LLAMA_LONGHAUL_RESIDENT=1`, cache 0) within the existing tolerance. Cover decode, small batches and a prompt larger than the clamped `n_ubatch`.
- Failure cases: a cache below the minimum, cache size 0 without the env var.
- A file with an MTP block under LongHaul: loads, and the MTP experts are not in the cache.

### 5.4 Docs

- `docs/longhaul.md`: the supported archs, and the Qwen rows of the sizing table from Phase 0.

---

## 6. Phase 3 - Chat and tool calls (`common/`)

- Restore `common/parsers/qwen3-coder.cpp` and its declaration in `parsers.h`, and add it to `common/sources.cmake`.
- `common/chat.cpp`: the upstream detection (`ca45ee8:common/chat.cpp:1466-1474`). The template has `<tool_call>`, `<function=` and `<parameter=` and not the no-newline `'<tool_call><function=' ~ tool_call.name ~ '>'` form. It runs after the Inkling check.
- `tests/test-chat.cpp`: port the Qwen 3.5 block (`ca45ee8:tests/test-chat.cpp:2369-2842`, about 470 lines) and the generation-prompt checks. Leave `test_chat_session` on `Qwen-QwQ-32B`.
- Check that Inkling's detection still wins for both Inkling templates (the existing Inkling tests).

---

## 7. Phase 4 - Python conversion and gguf-py

- `gguf-py/gguf/constants.py`:
  - `MODEL_ARCH.QWEN35` / `QWEN35MOE`, their names and `MODEL_TENSORS` lists.
  - `MODEL_TENSOR` members and names: `ATTN_QKV`, `ATTN_QKV_GATE`, the eight `SSM_*`, the four `*_SHEXP`, `V_ENC_ATTN_QKV` (also in the `MMPROJ` list).
  - `VisionProjectorType.QWEN3VL = "qwen3vl_merger"`.
  - `LLM.EXPERT_SHARED_FEED_FORWARD_LENGTH`, `ClipVision.IS_DEEPSTACK_LAYERS`.
- `gguf-py/gguf/tensor_mapping.py`: the `linear_attn.*`, `A_log`, `dt_proj`, `conv1d`, `shared_expert*` and `visual.blocks.{bid}.attn.qkv` entries (with the `layers.{bid}.*` variants).
- `gguf-py/gguf/gguf_writer.py`: `add_expert_shared_feed_forward_length`, `add_vision_is_deepstack_layers`.
- `conversion/qwen.py` (new, about 400 lines): the class chain from Q11, minus MTP, Hadamard, Mistral and NVFP4 V-reorder repack. `conversion/qwen3vl.py` (new, about 130 lines). Register both in `conversion/__init__.py`.
- `conversion/base.py`: the `qwen35` chkhsh in `get_vocab_base_pre`, and `_try_set_pooling_type` if the ported classes still call it. `convert_hf_to_gguf_update.py`: the `qwen35` entry.
- Tests: in-memory synthetic conversions (random tiny HF state dicts, no download), as Phase 3 did for DSpark and inkling2v. Check tensor names, shapes and keys, and that the result loads in `test-llama-archs`-style code. Run flake8 and ty (CI runs them).

---

## 8. Phase 5 - Vision (`tools/mtmd`, images only)

- `PROJECTOR_TYPE_QWEN3VL` enum and name, `TN_PATCH_EMBD_1`, the deepstack names (`mtmd-impl.h`). Deepstack loading stays optional, because Qwen 3.5 does not use it.
- hparams, tensor loading and output token counts in `mtmd-model.{h,cpp}`.
- The 4 x n_patches M-RoPE position input (`mtmd-subcontext.cpp`), `build_inp_temporal_merge` and the dispatch (`mtmd-graph.cpp`).
- `models/qwen3vl.cpp` (186 lines), the `qwen3vl` image preprocessor and `dyn_size` (`mtmd-image.{h,cpp}`), and the `<|vision_start|>` / `<|vision_end|>` markers (`mtmd.cpp`).
- Tests: extend `test-mtmd-impl` with a random tiny `qwen3vl_merger` mmproj, plus an image token count check. A real image run is left to you.

---

## 9. Phase 6 - Docs

- `README.md`: the fork notice now says Inkling 2 and Qwen 3.5/3.6, plus a Qwen 3.6 35B-A3B quick start. The GGUF repo and tag come from Phase 0.
- `docs/multimodal.md`, `docs/function-calling.md`, `docs/speculative.md` (DSpark is still Inkling-tested only; MTP is not supported).
- `tools/*/README.md` if help text changes. Regenerate with `llama-gen-docs`.
- Mark `LONGHAUL_PLAN.md` D13 as superseded for the arch list.
- Remove the stale `-a qwen35moe` usage text in `test-llama-archs.cpp`, which becomes valid again.

---

## 9a. Implementation notes (changes from the plan)

- **Weight scales of the new tensors.** `llama_layer` has `_s` and `_in_s` fields for `ssm_out`, `ssm_alpha`, `ssm_beta` and the three shared expert weights, as upstream. Without fields the scales were not saved by `llama_model_saver`, which walks the layer fields.
- **Model saver.** It writes `expert_shared_feed_forward_length` and `nextn_predict_layers`, and writes per-layer arrays for all blocks (`n_layer_all`), so a file with an MTP block reloads.
- **Tokenizer test file.** The restored `models/ggml-vocab-qwen35.gguf` had arch `qwen2`, which the fork does not know. Its header was rewritten to arch `qwen35` (keys renamed, tokenizer data unchanged). It passes against the restored `.inp` / `.out`.
- **LongHaul test.** The in-memory model always builds the merged `ffn_gate_up_exps`, so the test rewrites the saved GGUF for the other cases: separate gate and up, `ffn_down_exps` of layer 1 in F16 (two cache groups), and an MTP block (one more full attention layer plus the `nextn.*` tensors). The reference is the same file with its experts in the mapped file. The expected `n_ubatch` and cache sizes come from the file, per group.
- **Draft rollback.** `check_draft_rollback` was defined but not called. It now runs in the backend test for the archs with recurrent state rollback (Inkling and Qwen), which covers `rs_replay` for Qwen.
- **Mixed expert types limit the ubatch.** The cache splits its budget by the size of the groups, and a group with fewer slots than `n_expert` limits `n_ubatch` to its slots / `k`. The unsloth UD-Q4_K_M file of Qwen3.6-35B-A3B has a 3-layer group (Q6_K `ffn_down_exps`), which gets about 21 slots at 512 MiB, so `n_ubatch` is 2 (5 at 1G, 10 at 2G). The ggml-org Q4_K_M file has one group and no limit at 512 MiB. `src/llama-moe-cache.cpp` is unchanged; `docs/longhaul.md` documents this. A budget split that gives each group at least the slots of one ubatch would fix it, but it changes the hit rates of every mixed file, Inkling included. Update: the Ling-3.2-air work (LING32_PLAN.md, D4) added that split, so the unsloth file now gets `n_ubatch` 9 at 512 MiB (18 at 1G, no limit from 2G).
- **Sizing from Phase 0.** The plan's `n_ubatch` column ignored that the cache has no limit from `n_expert` slots. Corrected numbers are in `docs/longhaul.md`.

---

## 10. Out of scope

- MTP speculative decoding (Q4).
- DSpark drafting for Qwen 3.6 targets. It may work once the Qwen graph sets `t_layer_inp`, and the converter fix makes the drafter convertible, but it is not tested or claimed.
- Video input, NVFP4 V-reorder repack, `LLM_TYPE` names, `-sm tensor`, Hadamard files.
- Restoring the `qwen3.6-27b` quant-selection snapshot (the test needs the network to generate it).

---

## 11. Risks and open items

- **Real-file layout is unverified** until Phase 0 reads a GGUF header: merged vs separate experts, per-layer quant types, the MTP tensor set.
- **Speed.** Qwen 3.5 MoE routes per token over 256 experts (512 for the 397B), so at 512 MiB the hit rate may be low. The 35B-A3B reads up to about 0.57 GB per token with no hits. Bigger caches help prefill most.
- **Metal `MUL_MAT_ID` support for every expert type** in the chosen GGUF. LongHaul already aborts at init with the type name if it is missing.
- **The parser restore re-adds a removed component** (Q8). If you would rather stay on the autoparser, Phase 3 changes to fixing the three autoparser gaps instead, which touches shared code all templates use.
- **Upstream merges.** Close-to-verbatim ports (Q2, Q3) keep Qwen fixes easy to port.

---

## 12. Files touched (summary)

| Phase | Files |
|---|---|
| 1 | `src/llama-arch.{h,cpp}`, `src/llama-model.{h,cpp}`, `src/llama-hparams.{h,cpp}`, `src/llama-memory-hybrid.{h,cpp}` (new), `src/llama-graph.{h,cpp}`, `src/llama-context.cpp`, `src/models/{models.h,delta-net-base.cpp,qwen35.cpp,qwen35moe.cpp}`, `src/llama-vocab.{h,cpp}`, `src/llama-quant.cpp`, `src/CMakeLists.txt`, `tests/test-llama-archs.cpp`, `tests/CMakeLists.txt`, `models/ggml-vocab-qwen35.gguf{,.inp,.out}` |
| 2 | `src/models/qwen35moe.cpp`, `src/llama-model.cpp` (`create_tensor_gate_up_exps`), `tests/test-llama-archs.cpp`, `docs/longhaul.md` |
| 3 | `common/parsers/{parsers.h,qwen3-coder.cpp}`, `common/sources.cmake`, `common/chat.cpp`, `tests/test-chat.cpp` |
| 4 | `gguf-py/gguf/{constants.py,tensor_mapping.py,gguf_writer.py}`, `conversion/{__init__.py,base.py,qwen.py,qwen3vl.py}`, `convert_hf_to_gguf_update.py`, python tests |
| 5 | `tools/mtmd/{mtmd-impl.h,mtmd-model.h,mtmd-model.cpp,mtmd-subcontext.cpp,mtmd-graph.cpp,mtmd-image.h,mtmd-image.cpp,mtmd.cpp,models/qwen3vl.cpp,models/models.h}`, `tests/test-mtmd-impl.cpp` |
| 6 | `README.md`, `docs/*.md`, `tools/*/README.md` |

Rough size: about 2,000 lines of C++ for Phases 1-2 (most of it near-verbatim upstream), about 650 for Phase 3 (mostly tests), about 600 Python for Phase 4, about 450 C++ for Phase 5.
