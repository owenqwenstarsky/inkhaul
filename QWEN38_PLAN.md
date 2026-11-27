# Qwen3.8-Flash-Next on LongHaul: plan

Goal: run [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in this fork, with its routed experts on disk behind the LongHaul cache and its 51B-parameter n-gram table read from disk on demand, the same way Inkling 2 Mini, Qwen 3.5/3.6 and Ling-3.2-air run today.

Constraints: the model is not downloaded or run. All checks are code-level tests that run in memory (random tiny models, synthetic conversions) plus the existing Metal test binaries. GGUF headers and HF config files may be read over HTTP (no weights). Commits follow AGENTS.md (`Assisted-by:` trailer, each one approved by you, nothing pushed).

Status: plan only (2026-11-25). Nothing implemented.

Reference trees:
- Upstream: `ca45ee8` (last upstream commit merged into master). It has full `qwen4exp` support: `src/models/qwen4exp.cpp` (1513 lines), `src/llama-memory-hybrid-idx.{h,cpp}` (237 + 1003), `conversion/qwen4exp.py` (259).
- Fork: `HEAD` (`df3517e`). `qwen4exp` was removed in `3453f63` / `979a756`, but most of its infrastructure survived (section 2).

---

## 1. Facts (checked 2026-11-25)

### 1.1 The model

From `config.json`, the model card and the safetensors index:

- **Size:** 125B params in the trunk (6B active), plus a 51B n-gram embedding table, plus a 4B MTP block. 180B total, 360 GB in BF16.
- **Text:** 48 layers, hidden 2560, vocab 248320, context 262144 (YaRN to 1M per the card).
- **Layout:** 12 x (3 x (Gated DeltaNet -> MoE) + 1 x (QSA attention -> MoE)). Full attention layers are 3, 7, ..., 47.
- **Gated DeltaNet:** 16 QK heads, 48 V heads, head dim 128, conv kernel 4. Same as Qwen 3.5, except that the output gate is a sigmoid, not a silu.
- **QSA (Qwen Sparse Attention):**
  - The main attention: 24 Q heads, 2 KV heads, head dim 256, rope dim 64 (IMROPE, sections [11, 11, 10]), a sigmoid output gate packed into `q_proj`, and q/k norms.
  - The indexer: 4 query heads and 1 shared key head, head dim 128, budget 2048 tokens. The keys are pooled over blocks of 4 cells (`compress_ratio`), so the budget is 512 blocks. The tail block is always kept.
- **MoE:** 512 routed experts, 10 per token, softmax gating with normalized top-k, expert FFN 640. One shared expert (640) with a sigmoid gate.
- **Hyper-connections (HC)** replace every layer norm:
  - The residual is 4 streams wide (`[n_embd, 4, T]`).
  - Each layer has 2 mixers (before the mixer, before the MoE). A mixer is a grouped RMS norm, then a low-rank (320) sigmoid gate, then the mean of the streams.
  - The block output goes back into each stream with its own `2*sigmoid` weight.
  - There is no `output_norm`: a final mixer (`output_hc_*`) takes its place.
- **PLE n-gram embedding** on layer 1 (0-based):
  - The table is `[160, 320001536]`. Each token gathers 16 rows: 2 n-gram sizes (2 and 3) x 8 heads. The row index is a hash of the token and its 2 predecessors, `(t*m0) ^ (t1*m1) ^ (t2*m2) mod vocab_h + offset_h`, and EOS resets the window.
  - The 16 rows (2560 values) feed a key/value gate on the HC streams and a dilated depthwise causal conv (kernel 4, dilation 3) with its own recurrent history.
- **MTP:** 1 QSA block with its own HC head mixer.
- **Vision:** an unmodified Qwen3-VL ViT (27 blocks, 1152 wide, patch 16, merge 2, no deepstack, output 2560).
- **Tokenizer:** Qwen2Tokenizer with the Qwen 3.5 regex. Both GGUFs below have `tokenizer.ggml.pre = qwen35`.
- **Chat template:** the Qwen 3.5 template plus three things:
  - `reasoning_effort` (`xhigh` default, `medium`, `low`; anything else raises);
  - `preserve_thinking` (default true: past turns keep their thinking);
  - `{% generation %}` blocks.
  - The tool-call format is the same `<tool_call><function=...><parameter=...>` XML.

### 1.2 Upstream and community files

- Upstream llama.cpp supports the model as arch **`qwen4exp`** (HF class `Qwen4ExpForConditionalGeneration`). Every community GGUF uses that arch name, so we must too.
- **unsloth/Qwen3.8-Flash-Next-GGUF** UD-Q4_K_XL (headers read by HTTP range):
  - 4 files, 111.3 GB in total. The PLE table is alone in the last file (28.8 GB, IQ4_NL).
  - No MTP block (`block_count = 48`). The MTP files are shipped separately (`mtp-*.gguf`).
  - **Separate** `attn_qkv` / `attn_gate`, `ssm_beta` / `ssm_alpha` and `hc_*_down` / `hc_*_inject`. The indexer projections are BF16, the HC norms F32.
  - Experts are separate `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`, in **3 layouts**:
    - 43 layers: Q4_K / Q4_K / Q5_1, 2.93 MiB per expert;
    - 4 layers: Q4_K / Q4_K / Q8_0, 3.42 MiB;
    - layer 2: Q5_K / Q5_K / Q8_0, 3.81 MiB.
  - `ple.layers = [1]`, `ple.image_token_id` present.
- **ggml-org/Qwen3.8-Flash-Next-GGUF** Q8_0:
  - 162.6 GB. **Merged** `attn_qkv_gate`, `ssm_beta_alpha` and `hc_*_down_inject`. Separate gate/up experts, all Q8_0 (one layout, 4.98 MiB per expert).
  - The PLE table is Q4_0. No MTP block.
  - It also ships `mmproj-*.gguf` (`clip.projector_type = qwen3vl_merger`, `is_deepstack_layers` all false), `mtp-*.gguf` and `dflash-*.gguf` drafters.
- **Delgon/Qwen3.8-Flash-Next-PLE-GGUF** ships the PLE table alone (BF16, Q8_0, IQ4_NL), to swap in for the one in the main files.

### 1.3 Sizes for LongHaul

| File | Experts on disk | PLE table (lazy) | Resident weights | Bytes per expert | No-hit read per token | Minimum cache |
|---|---|---|---|---|---|---|
| unsloth UD-Q4_K_XL | 77.0 GB | 28.8 GB | ~5.5 GB | 2.93 / 3.42 / 3.81 MiB (3 groups) | ~1.40 GiB | ~102 MiB |
| ggml-org Q8_0 | 128.3 GB | 28.8 GB | ~5.5 GB | 4.98 MiB (1 group) | ~2.33 GiB | ~50 MiB |

Other memory:
- **KV cache:** 12 attention layers x 24 KiB per token (F16 K and V), so 768 MiB at 32k and 6 GiB at 262k. The indexer cache is about 6 KiB more per token.
- **Recurrent state:** about 108 MiB per sequence, plus the PLE conv row (360 KiB).
- **Slots and ubatch:** at 512 MiB, the Q8_0 file gets about 100 slots, so `n_ubatch` is clamped to 10. The 3-group unsloth file gets about 170 slots split over 3 groups (D4 split from LING32_PLAN.md), so `n_ubatch` is about 5 at 512 MiB, about 11 at 1 GiB and about 22 at 2 GiB. Phase 2 computes the exact numbers for the docs.

The resident part fits a 24 GB Mac with room for the cache and a 32k-64k context. With 10 of 512 experts per token, hit rates will be low at small caches. Expect SSD-bound decode (a few tokens/s) unless the cache is large.

---

## 2. Current state (HEAD)

**Still there (no work):**
- **ggml:** all the ops `qwen4exp` uses, with CPU, Metal and CUDA kernels:
  - `ggml_dsv4_hc_pre_gated`, `ggml_dsv4_hc_post`, `ggml_lightning_indexer`, `ggml_top_k_mask`;
  - `ggml_top_k`, `ggml_cumsum`, `ggml_fill`, `ggml_sgn`, `ggml_abs`, `ggml_softplus`;
  - `ggml_gated_delta_net`, `ggml_ssm_conv`, IMROPE, and the `GGML_TYPE_F8_E4M3` type.
  - `test-backend-ops` already has the Qwen4-Exp shapes (HC pre/post, top-k mask, F8 get_rows of the n-gram table).
- **Lazy row reads:** `TENSOR_READ_LAZY`, `llama_lazy_mode` (`--lazy`, default AUTO: lazy for marked tensors > 4 GiB), `llama_prefetch_rows`, and the loader rule that accepts a separate PLE-only shard (`is_ple_only`, `src/llama-model-loader.cpp:644`).
- **KV cache:**
  - the indexer-cache mirror (`is_indexer`, `get_rope_params_indexer`);
  - `ext.tok` in the cells, and `get_prev_tokens`.
- **Graph:**
  - `t_h_nextn`, `crop_before/after_nextn`, `build_attn_mha` with an external mask and `n_kv_max`;
  - `self_k_rot`, `build_rope`;
  - the fused-op probe framework (`llm_fused_op_probe`, `resolve`).
- **Qwen 3.5 port:** `llama-memory-hybrid.{h,cpp}` (identical to upstream), `delta-net-base.cpp`, `create_tensor_qkv`, `create_tensor_gate_up_exps` (with the LongHaul flag fix), `n_layer_nextn`.
- **Arch tables:** the `qwen35` pre-tokenizer, the `ATTN_QKV_GATE` / `SSM_*` / `*_SHEXP` tensors, the indexer KV keys (`LLM_KV_ATTENTION_INDEXER_*`).
- **Chat, conversion, vision:**
  - the `qwen3-coder` chat parser, Jinja `generation` tag handling (ignored), `reasoning_effort` caps;
  - `conversion/qwen.py` (`_LinearAttentionVReorderBase`, `_Qwen35MRopeMixin`, `_SHARED_INPUT_MERGES`), `conversion/qwen3vl.py`;
  - the `qwen3vl_merger` projector in mtmd;
  - `gguf_writer.add_hyper_connection_count`, `add_indexer_top_k`, `add_ple_layers`, `add_ple_ngram_size`, `add_embedding_length_per_layer_input`.
- `tests/fusion/MTL.csv` still has `qwen4exp` rows.

**Gone:**

| Area | Missing in HEAD |
|---|---|
| `src/llama-arch.*` | `LLM_ARCH_QWEN4EXP`. KV keys `HYPER_CONNECTION_COUNT/LOW_RANK`, `ATTENTION_COMPRESS_RATIOS`, `PLE_*` (layers, ngram size, heads per ngram, conv kernel, eos/image token, multipliers, head offsets, head vocab sizes). Tensors `HC_{ATTN,FFN}_{NORM,UP,DOWN,INJECT,DOWN_INJECT}`, `HC_HEAD_{NORM,DOWN,UP}`, `INDEXER_{Q,K}_{PROJ,NORM}`, `PLE_{KEY,VALUE,NORM_KEY,NORM_QUERY,NORM_CONV,CONV1D}`, `PER_LAYER_TOKEN_EMBD`, `ATTN_GATE`, `NEXTN_HC_HEAD_*`, with `LLM_TENSOR_INFOS`. |
| `src/llama-hparams.*` | `dsv4_hc_mult`, `hc_low_rank`, `indexer_{n_head,head_size,top_k,kpool,kpool_row,kpool_by_order,kpool_select_tail}`, `dsv4_compress_ratios`, the `ple_*` fields, `is_ple_impl`, `n_embd_per_layer`, `ple_conv_state()`, `is_ple()`, `LLAMA_MAX_PLE_NGRAM/HEADS`. |
| `src/llama-model.*` | Factory case, layer fields (`hc_*`, `index_*`, `ple_*`, `wqkv_gate`, the `_s` scale fields), model fields (`hc_head_*`, `per_layer_tok_embd{,_s}`, `per_layer_tok_embd_scale`, `can_prefetch`), the `create_memory` branch, rope type, print_info. |
| Memory | `src/llama-memory-hybrid-idx.{h,cpp}`. The PLE conv row `p_l` in `llama-memory-recurrent` (alloc, state io, clear, `get_p_l`). The PLE parts of `llama-kv-cache` (`has_cell_ext` and the embd-batch `ext.tok`). |
| `src/llama-graph.*`, `src/llama-cparams.h`, `src/llama-context.cpp` | `rep_cols_p` (PLE replay window), the fused ops `DSV4_HC_PRE/POST` and `TOP_K_MASK` with their cparams and probes, `QWEN4EXP` in the `rs_replay` list and the FP8-cache guard. |
| `src/models/` | `qwen4exp.cpp` and its `models.h` struct. The 1-line `rep_cols_p` pick in `delta-net-base.cpp`. |
| `src/llama-quant.cpp` | The "do not quantize `indexer.{q,k}_proj`" rule. |
| `gguf-py/`, `conversion/` | `MODEL_ARCH.QWEN4EXP`, its tensor list, names and mappings, writer helpers for the missing keys, `conversion/qwen4exp.py`. |

---

## 3. Decisions (made by me, override any of them)

| # | Decision | Why |
|---|---|---|
| D1 | **One arch, `qwen4exp`, ported close to verbatim from `ca45ee8`.** Drop only what the fork removed: the MTP graph (`graph_mtp`, `LLM_GRAPH_TYPE_DECODER_MTP`), `-sm tensor` split code, the Hadamard indexer-key path (`lid_k_hdmd`), and `LLM_TYPE_A3B` (Q12 of QWEN35_PLAN.md). | The community GGUFs use this name and tensor set. A near copy keeps later upstream fixes easy to port. Every ggml op is already in HEAD with kernels. |
| D2 | **Restore `llama-memory-hybrid-idx.{h,cpp}` verbatim.** It is a third cache (one indexer key per token) beside the hybrid attention and recurrent caches. Keep its generic k-pool code (`kpool_row`, `kpool_by_order`) even where only glm5-next used the other setting. Remove only code that needs symbols the fork lacks. | QSA needs the persistent pooled keys, the block layout across ubatches and the stale tracking on sequence edits. That is about 1,000 lines of subtle state that upstream tested. Trimming it gains little and makes ports harder. |
| D3 | **The PLE table is read lazily from the mapped file (`TENSOR_READ_LAZY`), not through the LongHaul cache.** Restore `can_prefetch` and the prefetch call in the PLE input. The default `--lazy auto` makes every real table lazy (28.8-102 GB > 4 GiB). | Each token reads 16 rows of 90-320 bytes, which is tiny random reads. The LongHaul cache is built for whole expert slices of MUL_MAT_ID and would add nothing here. The lazy path, the host gather and the separate-PLE-shard loader rule already exist and are what upstream uses. |
| D4 | **LongHaul applies to the routed experts of the 48 trunk layers only** (`ffn_gate_exps`, `ffn_up_exps`, or merged `ffn_gate_up_exps`, and `ffn_down_exps`). Router, shared expert and its gate, HC, attention, indexer, GDN, the PLE layer weights, embeddings and output stay resident. | Same rule as D1 of LONGHAUL_PLAN.md and Q5 of QWEN35_PLAN.md: weights used by every token stay in memory. |
| D5 | **MTP: load and skip.** The main GGUFs have no MTP block, but a file that has one (`nextn_predict_layers = 1`) must load. The MTP block's tensors (including `nextn.hc_head_*` and its experts) are created with `TENSOR_SKIP`. `mtp-*.gguf` drafters are not supported. | Matches Q4 of QWEN35_PLAN.md and the Ling work. MTP drafting was removed from the fork on purpose. |
| D6 | **Fused ops:** port the HC pre, HC post and top-k mask probes, their cparams (`fused_dsv4_hc_pre`, `fused_dsv4_hc_post`, `fused_top_k_mask`, `auto_fhc`, `auto_ftkm`) and their `LLM_FUSED_OP_*` values. Skip HC comb (DeepSeek-V4 only). | The kernels exist on Metal and CUDA. Without the fused path, each of the 97 mixers expands into many small ops. The probe framework is already in HEAD, so this is about 40 lines. |
| D7 | **The converter always drops `mtp.*`** (`supports_mtp_export = False`, no `_MTP_EXTRA` remap). It keeps the upstream merges (`attn_qkv_gate`, `ssm_beta_alpha`, `hc_*_down_inject`), the FP8 PLE scale path and the chunked PLE table writer. | The runtime cannot use MTP (D5), so exporting it only makes files bigger. This matches `conversion/qwen.py` for Qwen 3.5 (Q11 of QWEN35_PLAN.md). Merged weights mean fewer matmuls, and the runtime takes both forms. |
| D8 | **No new chat parser.** The template is detected by the existing `qwen3-coder` rule. Add the HF template as `models/templates/Qwen3.8-Flash-Next.jinja` and tests for detection, `reasoning_effort` levels (xhigh/medium/low accepted, `high` rejected), `preserve_thinking` and `{% generation %}`. | The tool-call format is unchanged from Qwen 3.5. The fork already handles `reasoning_effort` caps and ignores `generation` blocks. |
| D9 | **Vision needs no C++ work.** The converter registers `Qwen4ExpVisionModel` (a `Qwen3VLVisionModel` subclass), and the existing `qwen3vl_merger` projector loads the ggml-org mmproj. An image batch gets PLE rows from `ple.image_token_id`, as upstream does. | The ViT is unmodified Qwen3-VL. Images arrive as embd batches, so the n-gram hash needs a stand-in token. |
| D10 | **Quantize rule:** port "do not quantize `indexer.q_proj` / `indexer.k_proj`". Nothing else changes in `llama-quant.cpp`: `per_layer_token_embd` already follows `--token-embedding-type`, and `ssm_conv1d` is already excluded. | Matches upstream and the unsloth file (BF16 indexer). The indexer picks the attended blocks, so its error compounds. |
| D11 | **Reject an FP8 KV cache for `qwen4exp`** (the upstream guard in `llama_init_from_model`). | QSA reads the cache directly, and upstream marks it unsupported. |
| D12 | **Keep the `t_layer_inp` hook** (`hc_stream_mean`) in the graph, but do not claim DFlash support for `dflash-*.gguf`. | It is 15 lines and costs nothing when unused. DFlash with a hybrid HC target is not tested here (same position as DSpark in QWEN35_PLAN.md). |
| D13 | **Order: Phase 0 facts -> 1 resident runtime -> 2 LongHaul -> 3 conversion -> 4 chat -> 5 vision check -> 6 docs.** Each phase builds and passes its tests before the next. | Phase 1 gives the resident reference for the Phase 2 LongHaul tests. |
| D14 | Commits follow AGENTS.md: `Assisted-by: Claude` trailer, not `Co-authored-by`. Each commit needs your approval. Nothing is pushed. | AGENTS.md rule, same as the earlier plans. |

---

## 4. Phase 0 - Remaining facts (before code)

1. **Upstream after `ca45ee8`:** read upstream master's `src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp` and `conversion/qwen4exp.py` on GitHub (read-only) and list the fixes made since our merge base. Port bug fixes only.
2. **Tokenizer hash:** compute the `get_vocab_base_pre` chkhsh of the HF `tokenizer.json` and confirm it matches the existing `qwen35` entry. If it doesn't, add a `qwen35` alias entry, not a new pre-type.
3. **HF keys the converter reads:** confirm `hc_count`, `hc_lowrank`, `indexer_*`, `ple_layer_ids` (1-based), `ngram_size`, `heads_per_ngram`, `ple_conv_kernel_size`, `split_ngram_parts` and `image_token_id` against `config.json` (already fetched: all present).
4. **Record the full tensor name sets** of both real GGUFs (unsloth with separate tensors, ggml-org with merged ones) in a test fixture file. Phase 3 checks the converter output against it, and Phase 1 checks that the loader accepts both.

---

## 5. Phase 1 - Resident runtime (`src/`)

Experts are plain resident tensors in this phase.

### 5.1 Arch tables (`src/llama-arch.{h,cpp}`)

- `LLM_ARCH_QWEN4EXP` / `"qwen4exp"`. Add it to `llm_arch_is_hybrid` and `llm_arch_supports_rs_rollback`.
- The KV keys and tensors from the "Gone" table, with names copied from `ca45ee8`. They must match the GGUFs exactly, e.g. `blk.%d.hc_attn_down_inject`, `output_hc_norm`, `blk.%d.indexer.q_proj`, `per_layer_token_embd`.
- `LLM_TENSOR_INFOS` entries for all of them, `NEXTN_HC_HEAD_*` included. The loader looks up infos before it honors `TENSOR_SKIP`.

### 5.2 Hparams (`src/llama-hparams.{h,cpp}`)

- The fields and helpers from the "Gone" table: `ple_conv_state() = (kernel-1) * ngram * hc * n_embd`, and `is_ple(il)`.
- `n_embd_out_impl` already exists. Set it to `hc * n_embd` in the arch loader, as upstream does.

### 5.3 Model (`src/llama-model.{h,cpp}`, `src/llama-model-loader.cpp`)

- **Factory and layer fields:** the factory case; the `llama_layer` fields (`hc_attn_*`, `hc_ffn_*`, `index_{q,k}_{proj,norm}`, `ple_*`, `wqkv_gate`, `wqkv_gate_s`, and the other `_s` fields upstream has); the model fields `hc_head_{norm,down,up}`, `per_layer_tok_embd`, `per_layer_tok_embd_s`, `per_layer_tok_embd_scale`, `can_prefetch`.
- **After load:** read `per_layer_tok_embd_s` into `per_layer_tok_embd_scale`. Insert every lazy tensor into `can_prefetch` (`ca45ee8:src/llama-model.cpp:2121,2325`).
- **`create_memory`:** a `qwen4exp` branch that builds `llama_memory_hybrid_idx`:
  - filters: attention `il < n_layer() && !is_recr(il)`, recurrent `il < n_layer() && is_recr(il)`, indexer = attention filter when `indexer_head_size > 0`;
  - it passes `n_rs_seq` and `rs_replay`;
  - the MTP-context branch is dropped.
- **Rope and info:** `llama_model_rope_type`: `qwen4exp` -> `LLAMA_ROPE_TYPE_IMROPE`. `print_info`: the HC, indexer and PLE lines.
- **Loader:** `get_arr` instantiations for `std::array<uint64_t, LLAMA_MAX_PLE_NGRAM>` and `<..., LLAMA_MAX_PLE_HEADS>`.
- **Saver:** `src/llama-model-saver.cpp` writes the new keys (uint64 arrays included), so the test fixtures round-trip.

### 5.4 Memory

- **New files:** `src/llama-memory-hybrid-idx.{h,cpp}` from `ca45ee8` (D2), added to `src/CMakeLists.txt`.
- **`src/llama-memory-recurrent.{h,cpp}`:** the `p_l` row from `ca45ee8`:
  - allocation (6 tensors per layer when `ple_conv_state() > 0`), with the `rs_replay` width;
  - `size_p_bytes` and the log line;
  - `state_write` / `state_read`, and the clear in the replay reset;
  - `get_p_l` on the context. About 70 lines.
- **`src/llama-kv-cache.cpp`:** `has_cell_ext()` also when `ple_n_heads > 0`. An embd batch stores `ext.tok = ple_embd_token`.

### 5.5 Graph and context

- **`src/llama-graph.{h,cpp}`:**
  - `rep_cols_p` in `llm_graph_input_rs`: `n_conv = max(delta-net window, PLE window)`, and `rep_cols_r` / `rep_cols_p` as views (`ca45ee8:src/llama-graph.cpp:4193-4214`);
  - the matching `set_input`;
  - `LLM_FUSED_OP_DSV4_HC_PRE`, `LLM_FUSED_OP_DSV4_HC_POST`, `LLM_FUSED_OP_TOP_K_MASK`.
- **`src/models/delta-net-base.cpp:132`:** pick `rep_cols_r` or `rep_cols_p` by window width (the upstream line).
- **`src/llama-cparams.h`, `src/llama-context.cpp`:**
  - the fused cparams and the 3 probes (D6);
  - `QWEN4EXP` in the `rs_replay` list;
  - the FP8-cache guard (D11).

### 5.6 Model file

- `src/models/qwen4exp.cpp` and its `models.h` struct, minus `graph_mtp` (D1). `graph_nodes_per_tensor() = 32` as upstream.
- MTP blocks load with `TENSOR_SKIP` (D5), using inline flags as `qwen35moe.cpp` does. HEAD has no `nextn_flags()` helper, so the plan does not restore one.

### 5.7 Quantize

- `src/llama-quant.cpp`: the indexer rule (D10).

### 5.8 Tests

- **`tests/test-llama-archs.cpp`:**
  - Fixture keys for `qwen4exp` from `ca45ee8:tests/test-llama-archs.cpp:535-583`: HC count 4, low rank 8, compress ratio 4 on all layers, and a PLE layer 0 with ngram 3, 2 heads per ngram and conv 4. Also `SSM_INNER_SIZE = 256`, and `QWEN4EXP` in the MoE lists.
  - CPU vs Metal (`test_backends`), `--kv-shift` and `save_models` then run on it, since they loop over `llama_arch_all()`. The fixture's indexer top-k is huge (131072), so CPU and Metal agree.
  - New cases:
    - separate vs merged tensors (both `create_merged` paths) give the same logits;
    - `--lazy on` vs `off` for the PLE table gives the same logits;
    - a file whose PLE table sits in its own split loads (`is_ple_only`);
    - a small indexer top-k (8) so the sparse selection really runs, checked against itself across ubatch sizes (chunked prefill = single shot);
    - draft rollback (`check_draft_rollback`, `n_rs_seq > 0`) covers `rs_replay` with the PLE window;
    - a file with an MTP block loads and gives the same logits.
- **`tests/test-kv-cache.cpp`:** add `QWEN4EXP` to the arch list, as upstream does.
- **`test-backend-ops`** on Metal for `DSV4_HC_PRE`, `DSV4_HC_POST`, `LIGHTNING_INDEXER`, `TOP_K`, `TOP_K_MASK`, `GATED_DELTA_NET`, `SSM_CONV`, `GET_ROWS` (F8_E4M3, IQ4_NL, Q4_0 at width 160). These are existing cases; just run them.

---

## 6. Phase 2 - LongHaul for `qwen4exp`

### 6.1 Experts on disk

- `src/models/qwen4exp.cpp`: trunk blocks create `ffn_down_exps` and `create_tensor_gate_up_exps(...)` with `flags | TENSOR_LONGHAUL`. MTP blocks stay `TENSOR_SKIP`.
- `src/llama-moe-cache.cpp`: no change expected.
  - The shared expert goes through `build_ffn`, so the cache never sees it.
  - The 3 unsloth layouts become 3 groups under the D4 split.
  - Every layer is its own owner.

### 6.2 Checks (expected to need no code)

- **LongHaul and lazy PLE together.** Both go through `lazy_read`. LongHaul tensors are "always mapped", and the PLE tensor is lazy by mode. Check that `llama-model.cpp:999-1010` maps the lazy context once and that the memory accounting excludes both.
- **Node budget:** 32 nodes per tensor + 32 per layer for LongHaul. The test runs a tiny model with few tensors, which is the tight case.
- **`n_ubatch` clamp with QSA:** the k-pool inputs are padded to a stable bound (`n_new >= 1`), so a clamped ubatch only changes the pooling cadence. The test covers a prompt longer than the clamp.
- **The `rs_replay` rollback** re-runs the last ubatch, which means another set of lookups.

### 6.3 Tests (`tests/test-llama-archs.cpp` `test_longhaul`)

- `archs.push_back({ LLM_ARCH_QWEN4EXP, ..., { base, merged, split_down, mtp } })`:
  - base: separate gate/up, as in both real files;
  - merged: `ffn_gate_up_exps`;
  - split_down: `ffn_down_exps` of one layer in F16, giving 2 groups;
  - mtp: an MTP block whose experts are not cached.
- Two more cases: LongHaul with the PLE table lazy (`--lazy on`), and LongHaul with indexer top-k 8.
- The same cache configs as the other archs: minimum slots, everything fits, and 512 MiB. Logits must match `LLAMA_LONGHAUL_RESIDENT=1` within the existing tolerance, for decode, small batches and a prompt longer than the clamped `n_ubatch`.
- Failure cases: a cache below the minimum, and cache size 0 without the env var.

### 6.4 Docs

- `docs/longhaul.md`: add `qwen4exp` to the arch list, the rows of 1.3 with exact clamp numbers, and a PLE note:
  - the table is read lazily from the file, keep `--lazy auto|on`;
  - its reads go through the OS file cache;
  - the PLE shard can be swapped for the Delgon file.

---

## 7. Phase 3 - Conversion and gguf-py

- **`gguf-py/gguf/constants.py`:** `MODEL_ARCH.QWEN4EXP`, its name, its `MODEL_TENSORS` list, the missing `MODEL_TENSOR` members and names, and the missing `Keys.*` (PLE, HC low rank, compress ratios, indexer head count/key length).
- **`gguf-py/gguf/tensor_mapping.py`:** the `qwen4exp` entries from `ca45ee8` (HC, indexer, PLE, `linear_attn.in_proj_z` as `ATTN_GATE`).
- **`gguf-py/gguf/gguf_writer.py`:** the missing writers: `add_hyper_connection_low_rank`, `add_indexer_head_count`, `add_indexer_key_length`, `add_attention_compress_ratios`, `add_ple_heads_per_ngram`, `add_ple_conv_kernel`, `add_ple_eos_token_id`, `add_ple_image_token_id`, `add_ple_layer_multipliers`, `add_ple_head_offsets`, `add_ple_head_vocab_sizes`.
- **`conversion/qwen4exp.py`** from `ca45ee8`, minus MTP (D7): `filter_tensors` drops every `mtp.*`. Register it in `conversion/__init__.py`. `convert_hf_to_gguf_update.py` only if Phase 0 step 2 needs it.
- **Tests (`tests/test_conversion_fp8.py`):**
  - port `TestQwen4ExpPleScale`;
  - add a synthetic tiny checkpoint (random state dict, 4 layers, 2 PLE shards, hash constants above 2^32) and check:
    - the PLE rows are concatenated in shard order;
    - the int64 hash constants are exact;
    - the `+1` norm folds;
    - the indexer qk split;
    - the merged tensors;
    - `ple.layers` is 0-based;
    - every written name is in the real-file name set from Phase 0 step 4;
    - no `mtp.*` / `nextn` tensors are written.
  - flake8 and ty on the touched files.

---

## 8. Phase 4 - Chat

- `models/templates/Qwen3.8-Flash-Next.jinja`: the HF `chat_template.jinja`, byte for byte.
- `tests/test-chat.cpp`, a Qwen3.8 block:
  - the `qwen3-coder` parser is selected;
  - tool call round-trip;
  - reasoning with and without `</think>`;
  - `reasoning_effort` caps (`xhigh`, `medium`, `low`; `high` rejected);
  - `preserve_thinking` default (history thinking kept);
  - the `generation` tag does not break rendering;
  - the Inkling and Qwen 3.5 detections are unchanged.

---

## 9. Phase 5 - Vision check

- Converter: `Qwen4ExpVisionModel` registered (Phase 3), and an mmproj written from a synthetic tiny ViT has `qwen3vl_merger`, deepstack flags all false and projection dim = `n_embd`.
- `tests/test-mtmd-impl.cpp`: load that synthetic mmproj and check the image token count. If the Qwen 3.5 test already covers an identical file, extend it with `projection_dim = 2560` only.
- `test-llama-archs`: an embd batch through a `qwen4exp` model with PLE uses `ple.image_token_id` for its rows, so no crash and deterministic output.

---

## 10. Phase 6 - Docs

- `README.md`: the fork notice lists Qwen3.8-Flash-Next, with a quick start for the unsloth UD-Q4_K_XL file (`-lhc`, `--lazy auto`, context advice for 24 GB).
- `docs/multimodal.md`: the mmproj line. `docs/function-calling.md`: the template.
- `docs/development/HOWTO-add-model.md`, if it lists the fork's archs.
- This file: status, deviations, follow-ups.

---

## 11. Out of scope

- MTP drafting and the `mtp-*.gguf` files (D5).
- DFlash with `dflash-*.gguf` (D12).
- Video input.
- YaRN to 1M context: the generic `--rope-scaling yarn` flags may work, but they are not tested.
- `-sm tensor`, multi-GPU, Hadamard indexer keys, the FP8 KV cache.
- CUDA runs. The code paths are shared and the kernels exist, but this Mac only runs Metal (as in the Ling work).

---

## 12. Risks and open items

- **Size of the port.** About 2,900 lines of C++ return (1,400 model + 1,240 hybrid-idx memory + about 300 glue), plus about 450 Python and about 600 test lines. Most of it is upstream code, but it is the largest arch so far. Please confirm you want it as one feature branch with a commit per phase.
- **QSA and PLE numerics are not checked against real outputs.** No model is run. The tests check self-consistency (CPU vs Metal, LongHaul vs resident, chunked vs single prefill, lazy vs loaded). A later real run should compare a few tokens with the HF reference.
- **PLE with shared tokens.** Upstream asserts (aborts) when a token belongs to several sequences (`n_seq_id > 1`), because its n-gram predecessors are ambiguous. The plan keeps that and documents it. llama-server's normal path does not produce such tokens, but `llama-batched`-style shared prompts would.
- **The PLE page cache.** Lazy reads go through the OS file cache, not `F_NOCACHE`. The pages are clean and evictable, so this is pressure, not a leak, but RSS will look high on long runs.
- **Speed.** About 1.4 GiB read per token with no hits (unsloth Q4) and about 2.3 GiB (Q8_0). Ten of 512 experts per token gives low hit rates at 512 MiB. A 2-4 GiB cache is the realistic setting.
- **The 3-group unsloth file limits `n_ubatch`** (about 5 at 512 MiB), so prefill is slow at small caches. This is the D4 split working as designed.
- **Fork maintenance.** `qwen4exp` adds about 3k lines back. Later upstream fixes need manual porting (Phase 0 step 1 does the first round).

---

## 13. Files touched (summary)

| Phase | Files |
|---|---|
| 1 | `src/llama-arch.{h,cpp}`, `src/llama-hparams.{h,cpp}`, `src/llama-model.{h,cpp}`, `src/llama-model-loader.cpp`, `src/llama-model-saver.cpp`, `src/llama-memory-hybrid-idx.{h,cpp}` (new), `src/llama-memory-recurrent.{h,cpp}`, `src/llama-kv-cache.cpp`, `src/llama-graph.{h,cpp}`, `src/llama-cparams.h`, `src/llama-context.cpp`, `src/models/{models.h,qwen4exp.cpp (new),delta-net-base.cpp}`, `src/llama-quant.cpp`, `src/CMakeLists.txt`, `tests/test-llama-archs.cpp`, `tests/test-kv-cache.cpp` |
| 2 | `src/models/qwen4exp.cpp`, `tests/test-llama-archs.cpp`, `docs/longhaul.md` |
| 3 | `gguf-py/gguf/{constants.py,tensor_mapping.py,gguf_writer.py}`, `conversion/{__init__.py,qwen4exp.py (new)}`, maybe `convert_hf_to_gguf_update.py`, `tests/test_conversion_fp8.py`, a name-set fixture |
| 4 | `models/templates/Qwen3.8-Flash-Next.jinja` (new), `tests/test-chat.cpp` |
| 5 | `tests/test-mtmd-impl.cpp`, `tests/test-llama-archs.cpp` |
| 6 | `README.md`, `docs/*.md`, this file |
