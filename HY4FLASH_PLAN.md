# Hy4-flash (AtomicChat GGUF) on LongHaul: plan

Goal: run [AtomicChat/Hy4-flash-GGUF](https://huggingface.co/AtomicChat/Hy4-flash-GGUF) (Tencent Hy4-flash, arch `hy_v4_flash`) in this fork, text and vision, with its routed experts on disk behind the LongHaul cache, as Inkling 2, Qwen 3.5 MoE, Ling-3.2-air and Kolibri 1.1 run today. The published files must load as they are.

Constraints: the model is not downloaded or run. All checks are code-level tests that run in memory (random tiny models, `test-backend-ops`, chat template tests). Commits follow AGENTS.md (`Assisted-by:` trailer, each one approved by you, nothing pushed).

Status: draft, waiting for your answers on section 2.

---

## 1. Facts (checked 2026-11-27)

### 1.1 Upstream status

- **The model card points to a closed PR.** The card says the files need [ggml-org/llama.cpp#32517](https://github.com/ggml-org/llama.cpp/pull/32517), tested at `0a054d0b`. That PR was **closed on 2026-11-26** with "changes requested". The maintainer's closing note: "support is already added in #32960". `0a054d0b` is no longer on GitHub (the PR branch was force-pushed). The final head `6234f7e` is fetched locally as `refs/hy4/pr32517`.
- **[#32960](https://github.com/ggml-org/llama.cpp/pull/32960) "model : add hy_v4_flash on the DSA iSWA cache" was merged on 2026-11-25** (squash commit `c943331`).
  - It is a simpler version of #32517. It adds no new memory type and runs on the existing `llama_kv_cache_dsa_iswa` (as `dots3note` does). The indexer pools (4 tokens each) are rebuilt from the cache cells at each ubatch. So sequence edits, checkpoints and state save/load need nothing special.
  - 12 commits, head `80bfd75`, fetched locally as `refs/hy4/pr32960`. +2343/-20 in 29 files. No MTP.
  - The PR says its results "match #32517 on the same GGUF", and its converter is the one from #32517.
  - Known cost: decode at long context is slower than #32517 (70.6 / 36.2 / 17.1 t/s at 8k / 32k / 107k on CUDA, against 90.3 / 85.5 / 73.6). Two open upstream PRs (#33187 CUDA, #33189 Metal: "one thread per row for soft_max and sum_rows on short rows") recover most of it.
  - No later upstream commit touches the model (upstream master `15fcfc7`, 2026-11-27, fetched as `refs/hy4/upstream-master`). No open issue about it.
- **The base matches this fork.** This repo is shallow. Its root commit `f8c75f8ef` has the same tree as upstream `5fddd0b99`, which is also the base of #32960's head. So #32960 applies to the same upstream code the fork started from.
- **Our prune removed most of what #32960 builds on.** `3453f63` ("remove the other architectures ...") deleted `hy_v4`, `dots3note`, `deepseek4*`, the DSA caches, the hyper-connection and Engram code, `hunyuanvl` in mtmd, the `hy_v4` pre-tokenizer, and their keys, tensors and fused-op probes. **`990614933` (the parent of the prune) has all of it**, on the same upstream base.

### 1.2 The model

- **Size:** 219B parameters in the GGUF (155.5B network + 63.5B Engram tables), about 11.3B active. Hidden 3072, vocab 120832, context 1048576.
- **42 layers.** Layer 0 has a dense SwiGLU FFN (13824). Layers 1-41 are MoE: 512 routed experts, 16 used per token, 1 shared expert (1536 wide).
  - **Latent experts:** the routed experts run at width 1536. `ffn_routed_down` [3072 -> 1536] goes before them and `ffn_routed_up` [1536 -> 3072] after them. The router and the shared expert use the full 3072.
  - Router: sigmoid, selection bias, weights normalized and then scaled by 3.998. Routed SwiGLU clamped at 10 (DeepSeek V4 style `ggml_swiglu_clamp`, not the bailingmoe3 clamp in HEAD).
- **Hyper-connections (iHC, as hy_v4):** 4 residual streams. Each layer mixes them before attention and before the FFN (`hc_*_fn/base/scale`), and an output head reduces them (`output_hc_*`). Sandwich norms (pre and post norm around attention and FFN).
- **Attention:** MLA run as MQA, with a sigmoid output gate (`attn_gate`) and learned sinks.
  - q_lora 1024, kv_lora 512, qk 192 nope + 64 rope, v 256.
  - 12 sparse full-attention layers (0, 1, 5, 9, ..., 41): 32 heads, RoPE base 1e7. Each owns an indexer: 16 heads x 128. Keys are pooled in groups of 4 consecutive tokens (`kpool = 4`), the pooled key is a softmax-weighted sum of projected keys plus a position bias. Each token attends to the cells of the best 2048/4 - 1 pools plus its own pool (`kpool_select_tail`).
  - 30 sliding-window layers: 64 heads, window 512, RoPE base 1e4.
- **Engram** (hashed n-gram embeddings) on layers 1, 11, 21, 31: 3-gram hashes, 8 heads per n-gram, so 16 lookups per module and token (64 in total), 96 wide each. All modules share one table of 660M rows (`per_layer_token_embd`). Its output is added to the hc streams before attention.
- **RoPE:** the reference uses interleaved M-RoPE with sections [12, 10, 10]. For text tokens the 3 position parts are equal, so this is the same as plain rope. #32960 runs NORM rope when the file has no sections, and IMROPE (rows reordered to NeoX in the graph) when it has them.
- **Chat:** `<｜hy_start:opensource｜>role<｜hy_middle:opensource｜>...<｜hy_end:opensource｜>`. Reasoning in `<think:opensource>...</think:opensource>`. Reasoning effort is a template variable: `no_think`, `high` (default) or `max`. Tool calls are XML style, as in GLM-4.6: `<tool_calls:opensource><tool_call:opensource>name<arg_key:opensource>k</arg_key:opensource><arg_value:opensource>v</arg_value:opensource></tool_call:opensource></tool_calls:opensource>`.
- **Vision:** 27-layer ViT (1152 wide, patch 16), RMS norm, 2x2 conv merge, an MLP to 3072, plus one newline embedding per row. Projector type `hy_v4_flash`.

### 1.3 The AtomicChat files (headers read with HTTP range requests, no weights)

Header dumps are in the scratchpad (`h1.json`, `h2.json`, `hm.json`). The chat template was extracted from the header.

- `Hy4-flash-AD-IQ2_S-IQ2_XXS-00001-of-00002.gguf` (45.7 GiB): 1379 tensors, all weights except the Engram table.
  - `general.architecture = hy_v4_flash`, `block_count = 42`, `leading_dense_block_count = 1`, `expert_count = 512`, `expert_used_count = 16`, `expert_latent_length = 1536`, `expert_gating_func = 2`, `expert_weights_scale = 3.998`, `expert_weights_norm = true`, `swiglu_clamp_exp` = 42 x 10.0.
  - `hyper_connection.count = 4`, `.epsilon = 1e-6`, `.magnitude = 2.0`.
  - `attention.head_count` is a per-layer array (32 or 64), `head_count_kv = 1`, `key_length_mla = value_length_mla = 256`, `q_lora_rank = 1024`, `kv_lora_rank = 512`, `sliding_window = 512`, 42-entry `sliding_window_pattern`.
  - `attention.indexer.head_count = 16`, `.key_length = 128`, `.top_k = 2048`, `.kpool = 4`, `.kpool_select_tail = true`, `.types` (true at 0, 1, 5, ..., 41).
  - `rope.freq_base = 1e7`, `rope.freq_base_swa = 1e4`, `rope.dimension_count = 64`. **No `rope.dimension_sections`.**
  - `ple.layers = [1, 11, 21, 31]`, `ple.ngram_size = 3`, `ple.heads_per_ngram = 8`, `embedding_length_per_layer_input = 96`, 12 multipliers, 64 head offsets and 64 bucket counts, `engram.pad_token_id = 120002`, `engram.token_map` (120832 entries).
  - No MTP block (`nextn_predict_layers` absent).
  - `tokenizer.ggml.model = gpt2`, **`tokenizer.ggml.pre = hy_v4`**, 120832 tokens, 119744 merges, BOS 120000, EOS 120025, pad 120002, no BOS/EOS added. Chat template included (17 KB).
  - Sampling defaults in the header: temp 0.9, top_p 1.0, top_k -1.
- `...-00002-of-00002.gguf` (33.2 GiB): one tensor, `per_layer_token_embd.weight` [96, 660006416], Q4_0.
- `mmproj-Hy4-flash-F16.gguf` (0.97 GB): `clip.projector_type = hy_v4_flash`, 441 tensors (`v.blk.*`, `v.position_embd` F32 [1152, 16384], `v.image_newline`, `mm.0`, `mm.2`, `mm.pre_norm`).
- **Check against #32960's loader:**
  - Every tensor name in the files is one the loader creates. The only loader names missing from the files are the optional MTP tensors.
  - Every required key is present. The 3 missing keys are optional: `key_length_mla_swa`, `value_length_mla_swa`, `rope.dimension_sections`.
  - All shapes match what the loader expects for these hparams, including the Engram table row count (660006416).
- **Types:** routed gate/up IQ2_XXS, routed down IQ2_S (one layout for all 41 MoE layers). Attention, shared expert, latent projections, indexer, Engram projections, embeddings and output Q8_0. Router, hc, norms, sinks and biases F32. Engram table Q4_0.
- **How they were converted:** AtomicChat used an early #32517 converter plus `conversion/prep.py` and a small patch (both in their repo), which only rename things in the public checkpoint. Their `validate.py` checked the token map and multipliers against Tencent's GGUF. The rope rows are kept in the published (interleaved) order, which #32960 expects.

### 1.4 The mmproj problem with #32960

#32960's mtmd refuses the `hy_v4_flash` projector unless the text model uses M-RoPE: "the hy_v4_flash projector needs a text model with M-RoPE rope sections". The text model only uses M-RoPE when the GGUF has `rope.dimension_sections`. **The AtomicChat files do not have it**, so with #32960 as merged they run text only, and `--mmproj` fails. Tencent's own files (`AngelSlim/Hy4-flash-GGUF`) carry the sections. #32960's arch test even says "the published files carry the VL sections". #32517 at the time AtomicChat tested it defaulted missing sections to [12, 10, 10, 0] (commit `c167c512`: "Published files carry no rope sections, so default them to the checkpoint's [12, 10, 10, 0]"). See H2.

### 1.5 What HEAD already has, and what is missing

What is in HEAD already:
- **ggml:** every op the model needs, with CPU, Metal and CUDA kernels: `ggml_dsv4_hc_pre`, `ggml_dsv4_hc_post`, `ggml_top_k_mask`, `ggml_lightning_indexer` (Metal: head dim 128, any head count; CUDA: 4, 12, 32 and 64 heads), `ggml_swiglu_clamp`, `ggml_rope_multi` with IMROPE, `FLASH_ATTN_EXT` with sinks.
- **llama:**
  - `TENSOR_READ_LAZY` and `llama_prefetch_rows` (for the Engram table);
  - `get_prev_tokens` on the KV cache;
  - `llama_kv_cache_iswa`, the K-only attention input (`llm_graph_input_attn_k`), `build_attn_mha`;
  - MLA hparams (`n_lora_q`, `n_lora_kv`, `n_embd_head_k/v_mla_impl`), `indexer_top_k`, `is_indexer_full_impl`, `swiglu_clamp_exp`, per-layer `n_head_arr`, `rope_sections`, `n_layer_all`, `crop_before/after_nextn`, `TENSOR_SKIP`, `LLAMA_LOAD_LOCALS`;
  - the fused-op probe system (used for flash attention today).
- **LongHaul:** separate gate/up/down experts, one expert layout, `TENSOR_LONGHAUL`.

What is missing:

| Area | Missing in HEAD (source) |
|---|---|
| ggml | CUDA lightning indexer for 16 heads (#32960). |
| `src/llama-arch.*` | `LLM_ARCH_HY_V4_FLASH`. 14 keys: hyper-connection (3), PLE/Engram (8: `PLE_LAYERS`, `PLE_NGRAM_SIZE`, `PLE_HEADS_PER_NGRAM`, `PLE_LAYER_MULTIPLIERS`, `PLE_HEAD_OFFSETS`, `PLE_HEAD_VOCAB_SIZES`, `ENGRAM_PAD_TOKEN_ID`, `ENGRAM_TOKEN_MAP`), `EMBEDDING_LENGTH_PER_LAYER`, `KEY/VALUE_LENGTH_MLA_SWA`. Keys present in HEAD: check `INDEXER_HEAD_COUNT/KEY_LENGTH/KPOOL/KPOOL_SELECT_TAIL` and `EXPERT_LATENT_LENGTH` one by one. 26 tensors: `HC_*` (9), `ENGRAM_*` (6), `INDEXER_*` (8), `FFN_ROUTED_UP/DOWN`, `PER_LAYER_TOKEN_EMBD`. (`990614933` + #32960) |
| `src/llama-hparams.*` | `dsv4_hc_mult`, `dsv4_hc_eps`, `hc_magnitude`, `indexer_n_head`, `indexer_head_size`, `indexer_kpool`, `indexer_kpool_select_tail`, `n_expert_latent`, the `ple_*` group, `is_ple_impl` + `is_ple()`, `engram_pad_id`, `LLAMA_MAX_PLE_HEADS` (64, exactly what this model uses), `use_mrope()`. (`990614933`) |
| `src/llama-model.*` | layer fields (`hc_*`, `engram_*`, `indexer_*`, `ffn_routed_*`); model fields (`hc_head_*`, `per_layer_tok_embd`, `can_prefetch`, `engram_token_map`, `engram_multipliers`); factory, rope type, `create_memory` DSA-iSWA branch, `print_info`, the `can_prefetch` insert at load. (`990614933` + #32960) |
| KV cache | `src/llama-kv-cache-dsa.{h,cpp}` and `src/llama-kv-cache-dsa-iswa.{h,cpp}` (`990614933`, about 880 lines), #32960's 7-line pooled-indexer change in the DSA cache, and `set_input_kpool` in `llama-kv-cache.{h,cpp}` (#32960, about 135 lines). |
| `src/llama-graph.*` | `llm_graph_input_attn_k_dsa` and `..._dsa_iswa`, `build_attn_inp_k_dsa(_iswa)`, `build_top_k_mask` and `build_fused_top_k_mask`; the `ggml_swiglu_clamp` branch for this arch in `build_moe_ffn`. (`990614933`) |
| `src/llama-context.*`, `llama-cparams.h` | fused-op probes and flags for `DSV4_HC_PRE`, `DSV4_HC_POST`, `TOP_K_MASK`; the "force a unified KV cache" rule (#32960). |
| `src/llama-vocab.cpp` | `LLAMA_VOCAB_PRE_TYPE_HY_V4` and its regex (`990614933`). |
| `src/llama-model-saver.cpp`, `src/llama-quant.cpp` | the keys this arch needs for a test round-trip; the "do not quantize" rules (#32960 and `990614933`). |
| `src/models/` | `hy-v4-flash.cpp` and its `models.h` struct (#32960). |
| mtmd | the `hy_v4_flash` projector (#32960) and the `hunyuanvl` parts it reuses (position embedding resampled on the host, `TN_MM_PRE_NORM`, `TN_IMAGE_NEWLINE`, ...) (`990614933`). |
| Python | gguf-py constants (`MODEL_ARCH.HY_V4_FLASH`, keys, tensors); converter (see H5). |
| Chat | the template file and tests. |
| Docs | `docs/longhaul.md` section, README. |

### 1.6 Sizes and LongHaul numbers (computed from the header, not measured)

- **File:** 78.9 GiB in total. Shard 1 is 45.7 GiB, the Engram shard 33.2 GiB.
- **Routed experts:** 41 layers x 512 = 20992 experts, 1.881 MiB each (gate 0.580 + up 0.580 + down 0.721), 38.6 GiB in total (84% of shard 1). One layout, so one cache group.
- **Resident weights:** about 7.1 GiB. Attention is 4.9 GiB of it (`attn_gate` 1.8, `attn_output` 1.8, `attn_q_b` 0.6). Then shared experts 0.57, latent projections 0.38, router 0.24 (F32), output 0.37, token embedding 0.37 (host), rest under 0.3.
- **Engram table:** 33.2 GiB Q4_0. It stays in the file, memory-mapped and read lazily on the host (upstream behavior). Each token reads 64 random rows of 54 bytes. With a cold page cache that is up to 64 page-ins (16 KiB each on Apple Silicon) per token. The pages it touches stay in the OS page cache.
- **KV cache (F16):**
  - 12 sparse layers: MLA latent 576 x 2 B plus indexer 256 x 2 B per layer, so 19.5 KiB per token in total: 624 MiB at 32k, 2.4 GiB at 128k, 19.5 GiB at 1M.
  - 30 sliding layers: about 34 MiB at `-ub 512` (window 512 + ubatch).
- **Disk read per token with no hits:** 41 x 16 x 1.881 MiB, about 1.2 GiB. That is between Ling (~1 GiB) and Inkling 2 Mini (~1.9 GB).
- **Cache sizes (one group):**
  - smallest working cache, `reserve(1)`: 16 slots, about 30 MiB;
  - 512M (default): 272 slots, so `n_ubatch` is clamped to 17;
  - 963 MiB holds a whole layer (512 slots), and from there on `n_ubatch` is not clamped.
  - **Recommendation for the docs: `-lhc 1G` at least, 2-4 GiB for the hit rate.**
- **Images:** a whole image is decoded in one ubatch with one M-RoPE position, so the sliding window cannot free its cells. Upstream says `-b` and `-ub` must each be at least image tokens + 512 (up to 4096 + 512 for the largest images). With LongHaul that needs the cache to hold a whole layer (>= 963 MiB). Otherwise the clamped ubatch fails the image.
- **Memory on a Mac, rough:** 7 GiB resident + cache (1-4 GiB) + KV (0.6 GiB at 32k) + compute buffers. The Engram page cache comes on top. The card's non-LongHaul figure is 51.4 GB.

---

## 2. Decisions (my recommendation first; I need your answer on each)

| # | Question | Recommendation | Why |
|---|---|---|---|
| H1 | Which upstream code to port: #32960 (merged) or #32517 (closed, the one the card names)? | **#32960, pinned to head `80bfd75`.** First restore what it builds on from `990614933`, then apply its diff, keeping both as close to upstream as possible. | #32960 is the merged code and it sits on our exact base. Its loader reads every tensor and key of the AtomicChat files (1.3). #32517 adds two new memory classes and a k-pool cache type. It was closed in favor of #32960, so we would maintain it alone. Cost: slower decode at long context (1.1); the follow-up kernels can be ported later. |
| H2 | The files have no `rope.dimension_sections` (1.4). What happens when the key is missing? | **Default to [12, 10, 10, 0]** in `load_arch_hparams` when the key is absent (as #32517 `c167c512`). | It gives the AtomicChat files the same behavior as Tencent's files on upstream: IMROPE, vision works. Text output does not change (equal positions, same q/k reorder). Cost: `n_pos_per_embd() = 4`, so K-shift (context shift) is off for every `hy_v4_flash` user, as for Tencent's files upstream and every Qwen-VL model. Alternative: keep upstream as is: text only with these files, `--mmproj` refused, K-shift available. |
| H3 | Vision (mmproj) in this round? | **Yes, as its own phase (Phase 5), done last so it can be cut.** | The repo ships the mmproj and the card's run command uses it. The text model does not depend on it. It needs the `hunyuanvl` parts of mtmd from `990614933`. |
| H4 | Fused ops for the hyper-connections and the top-k mask | **Restore the 3 probes (`DSV4_HC_PRE`, `DSV4_HC_POST`, `TOP_K_MASK`) and their cparams flags from `990614933`.** Skip `DSV4_HC_COMB`, which this model does not use. | The kernels are in HEAD already. The unfused path is about 15 small nodes per hc call, 85 calls per token. The probe system is in HEAD (flash attention uses it). |
| H5 | Python converter | **No converter. Add only the gguf-py constants** (arch, keys, tensors), so gguf-py stays in sync with `llama-arch` as the header asks. | The converter (650 lines) needs `hy_v4.py`, `deepseek41.py` and their bases, and a 440 GB checkpoint. AtomicChat and Tencent already publish GGUFs. It can come later. |
| H6 | MTP and the DFly2 drafter | **Out of scope.** The loader keeps creating the MTP tensors with `TENSOR_SKIP | TENSOR_NOT_REQUIRED` (as `bailingmoe3` does in HEAD), so a file with MTP still loads. | The AtomicChat files have no MTP. The fork removed MTP. llama.cpp has no DFly2 graph. |
| H7 | What LongHaul keeps on disk | **Only the routed experts** (`ffn_gate/up/down_exps` get `TENSOR_LONGHAUL`). The router, `exp_probs_b`, the latent projections, the shared expert and the dense layer 0 stay resident. The Engram table stays lazily mapped on the host as upstream does. | Same rule as the other four models. Reading the Engram rows with `pread` (no page cache) would be a new subsystem; noted as a follow-up. |
| H8 | `-sm tensor` | **Not in this round.** `hy_v4_flash` stays out of `llm_arch_supports_sm_tensor`, so `-sm tensor` fails at load with the existing message. Layer split over RPC (`--rpc`) and the expert split (`-es`) have no arch-specific code and should work, see 6.3. | Upstream lists the arch for `-sm tensor`, but in this fork that path needs per-arch work in the meta backend (`SWIGLU_CLAMP` has no meta case, the DSA cache and pool inputs are untested there). Kolibri is not in the list either. |
| H9 | Several sequences | **Keep the #32960 rule:** with `n_seq_max > 1` and no `--kv-unified`, the context switches to a unified KV cache with a warning. | The pools gather cells from one KV stream. `llama-server -np auto` already uses a unified cache. |
| H10 | Chat and tool calls | **No new parser.** Save the template from the GGUF as `models/templates/tencent-Hy4-flash.jinja`, add `test-chat` cases. If the autoparser fails a case, I stop and ask before writing a parser. No mapping of `none` to `no_think`: the docs say to use `--chat-template-kwargs '{"reasoning_effort":"no_think"}'`. | The tool format is the GLM-4.6 XML layout, which the autoparser handles. Upstream keeps the same rule for Hy3. The template license is Tencent's (Apache 2.0 with extra commercial terms), the same as `tencent-Hy3.jinja` in upstream. |
| H11 | CUDA | **Port #32960's 16-head kernel instances** into our `lightning-indexer.cu`, which already has Kolibri's 12-head refactor. I cannot build or run CUDA here, so you'd run the CUDA tests. | Without it, the indexer of this model falls back to the CPU on CUDA (the PR says so). |

Smaller choices I made (override if you like):
- **SwiGLU clamp:** `build_moe_ffn` in HEAD clamps the bailingmoe3 way. `hy_v4_flash` must use `ggml_swiglu_clamp` (the upstream branch for hy_v4 / DeepSeek V4). I add only `LLM_ARCH_HY_V4_FLASH` to that branch, so bailingmoe3 does not change.
- **`LLM_GRAPH_TYPE_DECODER_MTP`** does not exist in HEAD. The check that throws for it is dropped.
- **`layer.nextn`** does not exist in HEAD. The MTP tensors are created with `TENSOR_SKIP` without storing them.
- **K-shift:** #32960's "do not shift the pooled indexer cache" (the indexer rows hold no position) comes with the DSA cache change. It only matters for files without sections under the H2 alternative.
- **The `hy_v4` pre-tokenizer** is restored with its `990614933` regex. No `chkhsh` line is needed without a converter.

---

## 3. Phase 0 - Last facts (no weights)

1. **Re-check upstream (H1).** Fetch upstream master again. List commits after `c943331` that touch `hy-v4-flash.cpp`, `llama-kv-cache-dsa*`, `set_input_kpool`, `tools/mtmd/models/hy-v4.cpp`, or the 16-head indexer. Also check whether #33187 / #33189 merged. If anything changed, I'll tell you before going on.
2. **Restore manifest.** List every symbol #32960 uses that HEAD lacks, with its definition in `990614933` (the table in 1.5, made exact). This also shows which `990614933` code to leave out (`deepseek41`, `glm5-next`, `qwen4exp` users of the same fields).
3. **Chat template:** save the GGUF's `tokenizer.chat_template` (already extracted, 17250 bytes) as `models/templates/tencent-Hy4-flash.jinja`. Compare it with `tencent/Hy4-flash`'s `chat_template.jinja` on HF (small text file) and note any difference.
4. **Reference splits:** about 15 `hy_v4` pre-tokenizer splits (Chinese, Japanese, digits, punctuation runs, CRLF, spaces) for `tests/test-unicode.cpp`. They come from `tencent/Hy4-flash`'s `tokenizer.json` (small text file, saved under `.cache/hy4-flash-tokenizer/`, gitignored) through the `tokenizers` pre-tokenizer. I'll check that `tokenizers` is installed and ask before installing anything.
5. **Optional:** read the header of one Tencent file (`AngelSlim/Hy4-flash-GGUF`, Q4_K_M_MTP) to confirm that its MTP tensors are skipped cleanly and its sections match [12, 10, 10, 0]. The plan does not need it.

---

## 4. Phase 1 - ggml

- **`ggml/src/ggml-cuda/lightning-indexer.cu`:** the `n_head == 16` branch from #32960 (MMA kernel on Turing+/AMD MFMA/WMMA, vector kernel otherwise), placed in our refactored dispatch. `ggml_cuda_lightning_indexer_supported` accepts 16.
- **`tests/test-backend-ops.cpp`:** `nh = 16` in the lightning indexer eval and perf loops (#32960).
- Metal: no change (any head count). Run `test-backend-ops -o LIGHTNING_INDEXER` on Metal here; you run CUDA.

---

## 5. Phase 2 - Shared runtime code (restored from `990614933`)

Each item names its effect on the other models. Nothing restored here may change their behavior; all existing tests must pass unchanged.

### 5.1 `src/llama-arch.{h,cpp}`
- The arch `LLM_ARCH_HY_V4_FLASH` / `"hy_v4_flash"`.
- The keys and tensors of 1.5, with their `990614933` names and `LLM_TENSOR_INFOS` entries, plus #32960's `INDEXER_KPOOL_VALUE` and the 4 new `ENGRAM_*` tensors.
- Not hybrid, not in the rs-rollback list, not in `llm_arch_supports_sm_tensor` (H8).

### 5.2 `src/llama-hparams.{h,cpp}`
- The fields of 1.5, zero-filled for every arch in `load_hparams`, next to the existing `kv_source_impl` / `is_indexer_full_impl` fills.
- `use_mrope()`: `rope_sections[0] > 0 && rope_sections[1] > 0` (`990614933`).
- Other archs never read them.

### 5.3 `src/llama-model.{h,cpp}`
- The layer and model fields of 1.5.
- **`can_prefetch`:** the set and its insert of `per_layer_tok_embd` after load (`990614933` line ~2129), so `llama_prefetch_rows` batches the Engram reads.
- **Placement of the Engram table:** it is an input-layer tensor, so it goes to the CPU buffer list (`dev_input`), memory-mapped. A test checks this with LongHaul on (6.2). The card's `-ot 'per_layer_token_embd\.weight=CPU'` then is not needed.

### 5.4 KV cache
- Restore `src/llama-kv-cache-dsa.{h,cpp}` and `src/llama-kv-cache-dsa-iswa.{h,cpp}` as in `990614933`, then apply #32960's change (a pooled indexer caches value + gate: head size x 2, no rope pattern).
- `llama_kv_cache::set_input_kpool` and the context wrapper (#32960).
- New files are picked up by `src/CMakeLists.txt` (I'll check whether it globs or lists them).
- `--kv-remote` with this arch: I'll check whether the DSA cache creates its layers through the same per-layer device callback as `llama_kv_cache`. If it does, `--kv-remote` works. If not, it fails at context creation with a clear message, and a test covers the message.

### 5.5 `src/llama-graph.{h,cpp}`
- The DSA input classes, `build_attn_inp_k_dsa(_iswa)`, `build_top_k_mask`, `build_fused_top_k_mask`.
- The `ggml_swiglu_clamp` branch for `LLM_ARCH_HY_V4_FLASH` in `build_moe_ffn` (section 2). It must also apply on the LongHaul path, which runs the same activation code.

### 5.6 `src/llama-context.cpp`, `src/llama-cparams.h`
- The 3 fused-op probes and flags (H4).
- The unified-KV rule (H9).
- Check that `graph_max_nodes` covers the graph: `graph_nodes_per_tensor() = 32` for this arch (#32960). Unfused hc paths are counted in case a backend fails a probe.

### 5.7 Other
- `src/llama-vocab.cpp`: `hy_v4` pre-tokenizer.
- `src/llama-quant.cpp`: do not quantize `engram_gate`, `engram_kv_b`, `engram_hc_scale`, `indexer_compressor_ape`.
- `src/llama-model-saver.cpp`: write the hc, indexer, PLE/Engram, latent and SWA keys this arch reads, so the tests can round-trip a model. Write `rope.dimension_sections` only when set.

---

## 6. Phase 3 - The `hy_v4_flash` model

### 6.1 Code
- `src/models/models.h`, `src/models/hy-v4-flash.cpp`: #32960's files, with these changes:
  - **H2:** when `rope.dimension_sections` is absent, set [12, 10, 10, 0] before the `use_mrope()` check, and log it once.
  - **LongHaul (H7):** `TENSOR_LONGHAUL` on `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`.
  - **MTP (H6):** no `layer.nextn` stores, `TENSOR_SKIP | TENSOR_NOT_REQUIRED` creation; drop the `LLM_GRAPH_TYPE_DECODER_MTP` check.
  - `build_moe_ffn`: keep HEAD's argument form (`..., il, logits)`), which matches #32960.
- `src/llama-model.cpp`: factory case; rope type (`IMROPE` with sections, `NORM` without); `create_memory` DSA-iSWA branch (no `dots3note` case); `print_info` MLA lines; `LLM_TYPE_UNKNOWN` as upstream.
- `src/llama-context.cpp`: the unified-KV rule (H9).

### 6.2 Tests (`tests/test-llama-archs.cpp`)

**Fixture** (#32960's, adapted to our test file):
- n_embd 128, 4 layers: sparse, sliding, sparse, sliding; the sliding layers have twice the heads; `head_count_kv = 1`;
- layer 0 dense, the rest MoE with latent width, shared expert, `exp_probs_b`;
- hc 4, indexer on the sparse layers (2 heads x 96), one Engram module on layer 1;
- rope sections [12, 10, 10, 0].
- I'll set `indexer.top_k` small (for example 8 with `kpool = 4`), so a 400-token prompt runs the real pool selection and the tail.

**Variants:**
- **No sections:** the text path with NORM rope. Under H2 this also checks that the default turns on M-RoPE.
- **No Engram.**

**Generic arch tests:**
- `hy_v4_flash` goes into `test_backends` and `moe_mandatory`.
- `test_kv_shift`: with sections the arch is skipped (`n_pos_per_embd = 4`). Under H2 that is every file. With the H2 alternative, the no-sections variant runs the K-shift test, which covers "do not shift the pooled indexer cache".
- **CPU vs GPU:** the indexer scores and `top_k` can tie on random weights, as in Kolibri. The cross-backend cases use contexts where every pool is selected. The selection-heavy cases run in the same-backend LongHaul comparisons below.
- **State save/load** of a sequence, and `seq_rm` in the middle of a pool (pools are rebuilt per ubatch, so this must just work).
- **Two sequences without `--kv-unified`:** the warning path gives the same logits as with `--kv-unified`.

**LongHaul cases** (as `bailingmoe3` and `kolibri1`, same backend, compared against `LLAMA_LONGHAUL_RESIDENT`):
- **Variants:** F32 and Q8_0 experts, and one mixed layout (one layer with another down type, so 2 groups).
- **Cache sizes:** the minimum `reserve(1)`, 5 slots, and "all experts".
- **Inputs:** decode; small batches; a prompt longer than the clamped ubatch; a 400-token prompt with 20 decode steps.
- **Errors:** a cache below `reserve(1)` fails with a message that names the minimum.
- **Placement:** the Engram table is in a host buffer and the experts are LongHaul tensors.

**Other tests:**
- `tests/fusion/MTL.csv`: the 3 `hy_v4_flash` rows from #32960, regenerated if the fixture differs.
- `tests/test-unicode.cpp`: the Phase 0 `hy_v4` splits.
- `tests/test-llama-context.cpp`: the KV cases run for the new arch (as for Kolibri).

### 6.3 What should work with no code change (checked by tests)
- **Expert split (`-es`) and RPC layer split:** no arch-specific code. If `test-expert-share` / `test-arg-expert-split` list archs, `hy_v4_flash` is added. Otherwise one LongHaul case with two CPU-side devices covers it, if the existing harness supports that.
- **One expert layout:** one group, the whole budget.
- **The pool inputs and Engram inputs** are host inputs set per ubatch, independent of the slot maps. With a clamped `n_ubatch` they are just smaller.

---

## 7. Phase 4 - gguf-py (H5)

- `gguf-py/gguf/constants.py`: `MODEL_ARCH.HY_V4_FLASH`, its name and tensor list (#32960), the missing `Keys` (hc, PLE/Engram, indexer kpool, latent length, MLA SWA lengths).
- `gguf-py/gguf/tensor_mapping.py`: only the entries the constants need for consistency. No HF names without a converter.
- Lint: flake8 and ty on the touched files.

---

## 8. Phase 5 - Vision (H3)

- **mtmd (`990614933` + #32960):**
  - `PROJECTOR_TYPE_HY_V4_FLASH` / `"hy_v4_flash"`;
  - the loader case (bicubic resize, round half even, tanh GELU in the ViT, merge 2, 4 to 4096 image tokens, min/max pixels);
  - the tensors (`mm.0`, `mm.2`, `mm.pre_norm`, `v.image_newline`, F32 2D `v.position_embd`);
  - `tools/mtmd/models/hy-v4.cpp`;
  - the host position-embedding resample shared with `hunyuanvl` (restore only that part, not the `hunyuanvl` projector);
  - the output token counts;
  - the image markers `<｜hy_image▁start:opensource｜>` / `<｜hy_image▁end:opensource｜>`;
  - the "needs M-RoPE" check.
- **LongHaul and images:** if an image needs an ubatch larger than the clamped `n_ubatch`, fail with a message that names the image token count and the `-lhc` that would fit. #32517 had such a message and reverted it; upstream returns a 500. This is a small fork-only addition. Tell me if you'd rather leave it out.
- **Tests:** `tests/test-mtmd-impl.cpp`: token counts and the marker layout for a few image sizes, and the loader on a tiny synthetic mmproj, if the harness supports it (I'll check). The M-RoPE positions of an image chunk with the text model fixture from 6.2.

---

## 9. Phase 6 - Chat (H10)

- The template file from Phase 0.
- **`tests/test-chat.cpp` cases with the autoparser:**
  - plain answer;
  - reasoning on (default `high`), `max`, and `no_think` through template kwargs (empty think block in the prompt);
  - one tool call and two tool calls;
  - a tool call after reasoning;
  - string and non-string argument values (the template JSON-encodes non-strings);
  - a JSON schema.
- **Caps:** `assert_effort_levels` = {high, max}, `assert_effort_rejected` = {none, minimal, low, medium, xhigh} (to be confirmed by the probe). I'll check what the server does with `--reasoning off` for this template and document it.
- `tests/test-chat-template.cpp`: one render check against the reference prompt string for a system + user + tool turn.

---

## 10. Phase 7 - Docs

- **`docs/longhaul.md`:** the intro lists Hy4-flash, plus a "Sizing for Hy4-flash" section:
  - what is cached and what is resident;
  - the numbers from 1.6;
  - the recommended `-lhc` (1G minimum for full-size prompt ubatches and images, 2-4G for hit rate);
  - the Engram table and the page cache;
  - the image ubatch rule;
  - `no_think`;
  - the run command (no `-ot` needed).
  - Also update the "Tests" paragraph.
- **`README.md`:** a quick start next to the others, and the fork notice lists Hy4-flash.
- **`docs/development/HOWTO-add-model.md`:** one line that `hy_v4_flash` is supported.
- **This file:** a status section at the end, as in `KOLIBRI11_PLAN.md`.

---

## 11. Tests run before each commit (code-level only)

- **Build:** `cmake --build` for the touched targets, Metal build on this Mac.
- **C++:**
  - `test-backend-ops -o LIGHTNING_INDEXER`, `-o TOP_K_MASK`, `-o DSV4_HC_PRE`, `-o DSV4_HC_POST` (Metal vs CPU);
  - `test-llama-archs` (all existing cases plus the new ones) and `--kv-shift`;
  - `test-llama-context`, `test-chat`, `test-chat-auto-parser`, `test-chat-template`, `test-unicode`, `test-mtmd-impl`;
  - `ctest -L main`, `ctest -L behavior -LE 'model|python'`.
- **Python:** flake8 and ty on gguf-py.
- **CUDA:** only on your machine (same list).
- **Optional, yours to run (not done by me):** load the AtomicChat files with `llama-cli` and a short prompt, with and without `--mmproj`, at `-lhc 2G`.

---

## 12. Risks

- **No real-model check.** The loader matches every name, key and shape of the AtomicChat files (1.3). But AtomicChat converted with an early #32517 converter, and upstream compared #32960 to #32517 on a different file. A difference in tensor contents (for example the order inside `engram_kv`) would not show in any test here. Only your optional run (section 11) can show it.
- **H2 deviates from upstream** for files without sections, and turns off K-shift for them.
- **Size of the change.** About 880 lines of restored KV cache, a few hundred lines of restored graph and hparams code, and #32960's ~1100-line model. It touches shared code (`create_memory`, hparams, graph inputs, fused-op probes, `build_moe_ffn`, the saver). AGENTS.md asks to confirm large changes first, which is what this plan is for.
- **Decode speed at long context:** #32960 rebuilds the pools every ubatch on the host (`pool_mask` is [n_kv/4, n_tokens] F16). At 128k that is about 32 MiB per 512-token ubatch. Decode slows down at long context (1.1). On LongHaul the disk reads dominate at short context.
- **Engram reads go through the page cache** (H7). Random 54-byte reads fault in 16 KiB pages. The page cache grows with use and competes with the LongHaul cache for unified memory under pressure.
- **Images with LongHaul** need a cache of at least one layer (963 MiB), or the image fails (1.6).
- **Quantized KV cache (`-ctk q8_0`):** the indexer rows (value + gate) are cached at the KV type. One LongHaul case with a Q8_0 cache checks it.
- **CUDA is untested here** (H11).

## 13. Files touched (summary)

| File | Change |
|---|---|
| `ggml/src/ggml-cuda/lightning-indexer.cu` | 16-head indexer |
| `src/llama-arch.{h,cpp}` | arch, keys, tensors |
| `src/llama-hparams.{h,cpp}` | hc, indexer, PLE/Engram, latent fields, `use_mrope()` |
| `src/llama-model.{h,cpp}` | layer/model fields, factory, rope type, `create_memory`, `print_info`, `can_prefetch` |
| `src/llama-kv-cache-dsa.{h,cpp}`, `src/llama-kv-cache-dsa-iswa.{h,cpp}` (restored), `src/llama-kv-cache.{h,cpp}` | DSA caches, `set_input_kpool` |
| `src/llama-graph.{h,cpp}` | DSA inputs, top-k mask builders, swiglu clamp branch |
| `src/llama-context.cpp`, `src/llama-cparams.h` | fused-op probes, unified-KV rule |
| `src/llama-vocab.cpp` | `hy_v4` pre-tokenizer |
| `src/llama-quant.cpp`, `src/llama-model-saver.cpp` | quant rules, keys |
| `src/models/models.h`, `src/models/hy-v4-flash.cpp` | the model (#32960 + H2, H6, H7) |
| `tools/mtmd/{mtmd.cpp,mtmd-impl.h,mtmd-model.cpp,mtmd-graph.cpp,mtmd-subcontext.cpp,CMakeLists.txt,models/models.h,models/hy-v4.cpp}` | vision |
| `gguf-py/gguf/{constants,tensor_mapping}.py` | constants |
| `models/templates/tencent-Hy4-flash.jinja` | template |
| `tests/test-backend-ops.cpp`, `tests/test-llama-archs.cpp`, `tests/test-llama-context.cpp`, `tests/test-chat.cpp`, `tests/test-chat-template.cpp`, `tests/test-unicode.cpp`, `tests/test-mtmd-impl.cpp`, `tests/fusion/MTL.csv` | tests |
| `docs/longhaul.md`, `README.md`, `docs/development/HOWTO-add-model.md` | docs |

## 14. Commits (each one shown to you before it is made)

1. `ggml : CUDA lightning indexer for 16 heads` (Phase 1)
2. `llama : restore the DSA caches, hyper-connection and Engram support` (Phase 2, no new arch yet; all existing tests pass)
3. `llama : add hy_v4_flash with LongHaul` (Phase 3)
4. `gguf-py : add hy_v4_flash` (Phase 4)
5. `mtmd : add the hy_v4_flash projector` (Phase 5)
6. `chat : add the Hy4-flash template and tests` (Phase 6)
7. `docs : Hy4-flash` (Phase 7)

## 15. Status (2026-11-27)

| Phase | Status |
|---|---|
| 0 - Facts | Done. The Hy4-flash tokenizer is in `.cache/hy4-flash-tokenizer/`. The GGUF chat template differs from `tencent/Hy4-flash`'s `chat_template.jinja` in one place: the GGUF also accepts `reasoning_effort = max` (HF: `no_think` and `high` only). The GGUF version is saved. |
| 1 - ggml | Done. CUDA 16-head lightning indexer (H11), not compiled here. Metal and CPU ops from `990614933`. |
| 2 - Runtime | Done. DSA / DSA-iSWA caches, hyper-connection, Engram, pool inputs, `set_input_kpool`, the H9 unified-KV rule. |
| 3 - Model | Done. #32960 at `80bfd75`, with H2 (default sections), H6 (MTP tensors skipped), H7 (routed experts `TENSOR_LONGHAUL`). |
| 4 - gguf-py | Done. Constants only (H5). |
| 5 - Vision | Done. `hy_v4_flash` projector, host position-embedding resample, markers, M-RoPE check. |
| 6 - Chat | Done. The autoparser passes every case, no new parser. |
| 7 - Docs | Done: `docs/longhaul.md` (Hy4-flash sizing), `README.md`, `HOWTO-add-model.md`. |

**Tests run** (Metal on this Mac, no CUDA):
- `test-backend-ops -b MTL0`: `LIGHTNING_INDEXER` 288/288 (48 with 16 heads), `TOP_K_MASK` 38/38, `DSV4_HC_PRE` 18/18, `DSV4_HC_POST` 11/11.
- `test-llama-archs` (all archs, exit 0, also with `GGML_METAL_DEVICES=2`): `hy_v4_flash` vs CPU on Metal, BLAS and CPU, roundtrip, mixed batch. `hy_v4_flash` cases checked with seeds 1, 3, 5, 7 and 42.
  - LongHaul: F32 and Q8_0 + F32 experts (2 groups), caches `reserve(1)`, 5 slots and 512 MiB, a 400-token prompt and 20 steps, the error below `reserve(1)`, two GPUs (layer split) and the expert split, the Engram table in a CPU buffer, a Q8_0 KV cache, an image-sized run.
  - Without LongHaul: no rope sections (IMROPE, same logits), no Engram, steps vs one batch, state save/load in the middle of a pool, `seq_rm` in the middle of a pool, 2 sequences without `--kv-unified`.
- `test-llama-archs --kv-shift`: `hy_v4_flash` is skipped (M-RoPE, H2).
- `test-llama-context`: `hy_v4_flash` passes every KV case. The 4 skips (rollback, rollback_bound, replay, shared) are the same as for `kolibri1`.
- `test-fusion --check tests/fusion/MTL.csv` on MTL0: the 3 `hy_v4_flash` rows match.
- `test-chat` (7 parser cases with reconstruction, effort caps, 2 exact renders), `test-chat-auto-parser`, `test-chat-template`, `test-unicode` (16 `hy_v4` splits), `test-mtmd-impl` (tiny synthetic mmproj: token counts, newline tokens, encode on CPU and Metal).
- `ctest -L main`: 24/24.
- `ctest -L behavior -LE 'model|python'`: 42/43. `test-tokenizer-0-qwen35` fails because `models/ggml-vocab-qwen35.gguf` was removed in `3453f63ab`. That failure predates this work.
- gguf-py: `py_compile` and a consistency check of the new constants (every `hy_v4_flash` tensor has a name, no duplicate values). flake8 and ty are not installed here, so they did not run.

**Deviations from the plan:**
- **Images with LongHaul (1.6, 12):** the premise was wrong. The memory is created before LongHaul reduces `n_ubatch`, so the sliding-window cache is sized with the requested `-ub`. An image fits when `-b` and `-ub` are large enough, whatever the expert cache size. A test checks it: a 300-row run at one position decodes with a 1-token ubatch, and fails with decode error 1 at `-ub 64`. So the fork-only message names `-b` and `-ub`, not `-lhc`. It is in `mtmd_helper_decode_image_chunk`, for any M-RoPE model with a sliding window, when decode returns 1.
- **GPU selection:** the GPU cases select every pool. With random weights many pools score exactly 0, and the Metal kernels for other ubatch sizes break these ties in another order. The CPU cases use top-k 8, so the selection drops pools.
- **Engram placement test:** it loads with `lazy_mode = on`. The real table (33 GiB) is lazy in the default mode (above 4 GiB), the tiny test table is not.
- **Render check:** it is in `test-chat.cpp`, from the template file. `test-chat-template.cpp` only takes inline templates.
- **Effort caps:** levels `high` and `max`; rejected `minimal`, `low`, `medium`, `xhigh`. The probe never lists `none`. The server answers `"reasoning_effort": "none"` and `--reasoning off` with `enable_thinking = false` and no effort kwarg, which this template ignores, so thinking stays on. The docs name `no_think`.
- **Added:** the two-GPU and expert-split cases (6.3), and the Q8_0 KV cache case (12).

**Follow-ups (not done):**
- CUDA was not built or run here (H11).
- There is no check against the real files. A difference in tensor contents (12) would not show in any test here.
- The M-RoPE positions of an image chunk and the marker layout are not tested through `mtmd_tokenize`. `mtmd_image_tokens` is private to `mtmd.cpp`, and `test-mtmd-impl` has no text model. The positions use the generic M-RoPE 2D path, and the newline column comes from `n_output_tokens_x`, which is tested.
- The host cost of the pool inputs at long context (12) is unchanged.
