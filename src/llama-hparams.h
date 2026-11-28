#pragma once

#include "llama.h"

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>

// bump if necessary
#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_EXPERTS 1024 // Kimi K3
#define LLAMA_MAX_PLE_HEADS 64 // hy_v4_flash (all Engram modules)

enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE           = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX        = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID        = 2,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3, // applied to the router weights instead of the logits
    LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS  = 4,
};

enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE      = 0,
    LLAMA_SWA_TYPE_STANDARD  = 1,
    LLAMA_SWA_TYPE_CHUNKED   = 2,
    LLAMA_SWA_TYPE_SYMMETRIC = 3,
};

// M-RoPE: which input position slot feeds each RoPE section
enum llama_rope_section_order {
    LLAMA_ROPE_SECTION_ORDER_UNSPECIFIED = -1,
    LLAMA_ROPE_SECTION_ORDER_TYXZ        = 0, // default, slot i feeds section i
    LLAMA_ROPE_SECTION_ORDER_ZYXT        = 1, // MiniCPM-V 4.7: time last
};

// forward declaration; full definition in llama-graph.h
enum llm_ffn_op_type : int;

struct llama_hparams {
    // note: use the `_impl` suffix to avoid name conflict between members and getters
    //       for example: n_embd_out() vs n_embd_out_impl

    bool vocab_only;
    bool rope_finetuned;

    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_layer_all;
    uint32_t n_layer_nextn = 0; // MTP blocks at the end of the file, loaded but not run
    uint32_t n_expert = 0;

    // layer il reads the KV cache of layer kv_source_impl[il] (own cache if equal to il)
    std::array<uint32_t, LLAMA_MAX_LAYERS> kv_source_impl;

    // sparse attention (Kolibri-1.1): top-k keys per query, 0 = dense attention
    uint32_t indexer_top_k = 0;
    // a long context query selects keys only from its indexer_cand_top_k_blocks best blocks of indexer_cand_block_size positions
    uint32_t indexer_cand_block_size   = 0;
    uint32_t indexer_cand_top_k_blocks = 0;
    // indexer layers (1) choose the key pool, the other sparse layers (0) select from the pool of the preceding indexer
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_indexer_full_impl;

    // DSA (hy_v4_flash): learned indexer over pools of indexer_kpool consecutive tokens
    uint32_t indexer_n_head    = 0;
    uint32_t indexer_head_size = 0;
    uint32_t indexer_kpool     = 0; // k-pool size
    bool     indexer_kpool_select_tail = true;

    // hyper-connections (hy_v4_flash)
    uint32_t dsv4_hc_mult = 0;
    float    dsv4_hc_eps  = 0.0f;
    // scale of the hyper-connection post gate
    float    hc_magnitude = 0.0f;

    // Engram (hy_v4_flash): n-gram hash embeddings, one module per Engram layer
    uint32_t ple_ngram_size      = 0;
    uint32_t ple_heads_per_ngram = 0;
    uint32_t ple_n_heads         = 0;   // (ngram_size - 1) * heads_per_ngram
    uint32_t ple_head_dim        = 0;
    uint32_t ple_n_modules       = 1;
    // the token that pads the n-gram history, an index into llama_model::engram_token_map
    uint32_t engram_pad_id       = 0;
    // the file lists PLE layer indices, so this is never a per-layer gguf array and can hold one bit per layer
    std::bitset<LLAMA_MAX_LAYERS> is_ple_impl;
    // head offsets and vocab sizes are token-space indices; the gather truncates them to int32 anyway
    std::array<uint32_t, LLAMA_MAX_PLE_HEADS> ple_head_offsets;
    std::array<uint32_t, LLAMA_MAX_PLE_HEADS> ple_head_vocab_sizes;

    // different head size for full_attention and SWA layers
    uint32_t n_embd_head_k_full; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v_full; // dimension of values (d_v) aka n_embd_head
    uint32_t n_embd_head_k_swa;
    uint32_t n_embd_head_v_swa;

    // MLA: key/value head size before the absorption of wk_b/wv_b
    uint32_t n_embd_head_k_mla_impl = 0;
    uint32_t n_embd_head_v_mla_impl = 0;

    // different RoPE dimensions for full_attention and SWA layers
    uint32_t n_rot_full;
    uint32_t n_rot_swa;

    uint32_t n_shortconv_l_cache  = 0;
    uint32_t n_sconv_channels     = 0;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    // per-layer expert feed-forward size
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_exp_arr;
    // per-layer top-k expert routing count
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_expert_used_arr;

    uint32_t n_expert_groups    = 0;
    uint32_t n_group_used       = 0;
    uint32_t n_ff_shexp         = 0;

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q           = 0;
    uint32_t n_lora_kv          = 0;
    uint32_t n_expert_shared    = 0;

    float    expert_weights_scale = 0.0f;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    float    expert_weights_norm_eps = 0.0f; // > 0: weights / (sum + eps), else the sum is clamped
    bool     expert_router_f32    = false;   // compute the router logits in F32

    float f_norm_rms_eps;

    float f_final_logit_softcapping  = 30.0f;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_base_train_swa  = 10000.0f;
    float    rope_freq_scale_train;
    float    rope_freq_scale_train_swa = 1.0f;

    uint32_t n_ctx_orig_yarn;

    float    yarn_ext_factor  = -1.0f;
    float    yarn_attn_factor =  1.0f;
    float    yarn_beta_fast   = 32.0f;
    float    yarn_beta_slow   =  1.0f;

    std::array<int, 4> rope_sections;

    enum llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ;

    // Per-layer RoPE enable flags (1 = use RoPE, 0 = NoPE)
    // by default, all layers use RoPE (controlled by rope_finetuned)
    std::array<uint32_t, LLAMA_MAX_LAYERS> rope_pattern;

    // Sliding Window Attention (SWA)
    llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
    // the size of the sliding window (0 - no SWA)
    uint32_t n_swa = 0;

    // if is_swa_impl[il] == 1, then layer il is SWA
    // if is_swa_impl[il] == 0, then layer il is dense (i.e. non-SWA)
    // by default, all layers are dense
    // note: using uint32_t type for compatibility reason
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_swa_impl;

    // for hybrid state space models
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_recr_impl;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    uint32_t ssm_n_group = 0;

    // for Kimi Linear KDA
    uint32_t n_embd_head_kda      = 0;
    bool     kda_safe_gate        = false;
    float    kda_gate_lower_bound = -INFINITY;

    // SwiGLU clamp limits per layer, 0 = no clamp
    std::array<float, LLAMA_MAX_LAYERS> swiglu_clamp_exp;   // routed experts
    std::array<float, LLAMA_MAX_LAYERS> swiglu_clamp_shexp; // shared expert

    // kimi-k3
    uint32_t n_expert_latent      = 0;      // routed_expert_hidden_size (0 = experts run at n_embd)
    uint32_t attn_res_block_size  = 0;      // 0 = no cross-layer attention residuals
    float    situ_beta            = 1.0f;
    float    situ_linear_beta     = 0.0f;   // 0 = no linear-beta transform on the up branch

    // inkling2
    uint32_t n_relative        = 0;
    uint32_t n_relative_extent = 0;
    uint32_t n_vocab_unpadded  = 0;
    uint32_t n_vocab_padded    = 0;
    uint32_t n_log_scale_floor = 0;
    float    f_log_scale_alpha = 0.0f;
    float    f_qk_clip         = 0.0f;
    float    f_relative_clip   = 0.0f;

    float f_logit_scale    = 0.0f;

    // Additional scale factors (Granite/Granite MoE)
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;

    float    f_attn_value_scale = 0.0f;

    bool causal_attn   = true;

    // for Classifiers
    uint32_t n_cls_out = 1;

    // encoder input embedding dimension (0 = use n_embd_inp())
    // e.g. the dflash encoder fuses target_layers * target_hidden features
    uint32_t n_embd_inp_enc_impl = 0;

    // output embedding dimension (0 = use n_embd)
    uint32_t n_embd_out_impl = 0;

    uint32_t dflash_block_size = 0;

    // each sequence is paired with an unconditional one
    bool     classifier_free_guidance = false;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_pooling_type      pooling_type_cls        = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling before the classifier head (RANK)
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    // Resolved FFN gated activation flavor for archs that read
    // `<arch>.hidden_activation` from the GGUF (e.g. ModernBert derivatives).
    // Defaults to LLM_FFN_NONE (sentinel = 0); the mapping from the GGUF
    // string to a real op is done at hparam-load time via
    // llm_ffn_op_type_from_string() in llama-model.cpp, mirroring how
    // rope_scaling_type_train is handled.
    enum llm_ffn_op_type llm_ffn_op;
    enum ggml_unary_op   act_cls = GGML_UNARY_OP_TANH; // activation of the classifier head (RANK)

    // this value n_pattern means that every nth layer is dense (i.e. non-SWA)
    // dense_first means whether the pattern is start with a dense layer
    // note that if n_pattern == 0, all layers are SWA
    //           if n_pattern == 1, all layers are dense
    // example 1: n_pattern = 3, dense_first = false
    //   il == 0: swa
    //   il == 1: swa
    //   il == 2: dense
    //   il == 3: swa
    //   il == 4: swa
    //   il == 5: dense
    //   il == 6: swa
    //   etc ...
    // example 2: n_pattern = 2, dense_first = true
    //   il == 0: dense
    //   il == 1: swa
    //   il == 2: dense
    //   il == 3: swa
    //   etc ...
    void set_swa_pattern(uint32_t n_pattern, bool dense_first = false);

    // return true if one of the layers is SWA
    bool is_swa_any() const;

    bool is_swa(uint32_t il) const;

    // whether or not the given layer is recurrent (for hybrid models)
    bool is_recr(uint32_t il) const;

    uint32_t n_head(uint32_t il = 0) const;

    uint32_t n_head_kv(uint32_t il = 0) const;

    uint32_t n_ff(uint32_t il = 0) const;

    uint32_t n_ff_exp(uint32_t il = 0) const;

    uint32_t n_expert_used(uint32_t il = 0) const;

    // return the maximum n_expert_used across all layers
    uint32_t n_expert_used_max() const;

    uint32_t n_gqa(uint32_t il = 0) const;

    uint32_t n_rot(uint32_t il = 0) const;

    // dimension of main + auxiliary input embeddings
    uint32_t n_embd_inp() const;

    // dimension of the encoder input embeddings
    uint32_t n_embd_inp_enc() const;

    // dimension of output embeddings
    uint32_t n_embd_out() const;

    // dimension of key/value embeddings for each head (per layer)
    uint32_t n_embd_head_k(uint32_t il = 0) const;
    uint32_t n_embd_head_v(uint32_t il = 0) const;

    // dimension of key embeddings across all k-v heads
    uint32_t n_embd_k_gqa(uint32_t il = 0) const;

    // dimension of value embeddings across all k-v heads
    uint32_t n_embd_v_gqa(uint32_t il = 0) const;

    bool is_mla() const;

    uint32_t n_embd_head_k_mla() const;
    uint32_t n_embd_head_v_mla() const;

    // true if any layer has a different n_embd_v_gqa
    bool is_n_embd_v_gqa_variable() const;

    // return the maximum n_embd_k_gqa/n_embd_v_gqa across all layers
    uint32_t n_embd_k_gqa_max() const;
    uint32_t n_embd_v_gqa_max() const;

    // dimension of the rolling state embeddings
    // corresponds to Mamba's conv_states size or RWKV's token_shift states size
    uint32_t n_embd_r() const;

    // dimension of the recurrent state embeddings
    uint32_t n_embd_s() const;

    uint32_t n_pos_per_embd() const;

    // the file carries M-RoPE sections
    bool use_mrope() const;

    bool has_kv(uint32_t il) const;

    uint32_t kv_source(uint32_t il) const;

    bool is_indexer_full(uint32_t il) const;

    bool is_ple(uint32_t il) const;

    bool has_rope(uint32_t il) const;

    // number of layers, without the NextN (MTP) layers
    uint32_t n_layer() const;

    // note that this function uses different SWA parameters from those in the hparams
    // note: inlined on purpose for performance reasons
    // TODO: think of a better place for this function
    // TODO: pack the SWA params in a struct?
    static bool is_masked_swa(uint32_t n_swa, llama_swa_type swa_type, llama_pos p0, llama_pos p1) {
        assert(p0 >= 0 && p1 >= 0);

        switch (swa_type) {
            case LLAMA_SWA_TYPE_NONE:
                {
                } break;
            case LLAMA_SWA_TYPE_STANDARD:
                {
                    if (p1 - p0 >= (int32_t) n_swa) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_CHUNKED:
                {
                    const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                    if (p0 < pos_chunk_start) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_SYMMETRIC:
                {
                    const int32_t half_n_swa = (int32_t) n_swa / 2;
                    const int32_t pos_diff = p1 - p0;

                    // Mask if outside the symmetric window
                    if (pos_diff < -half_n_swa || pos_diff > half_n_swa) {
                        return true;
                    }
                } break;
        }

        return false;
    }

};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");
