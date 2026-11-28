#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-memory.h"
#include "llama-vocab.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_cparams;
struct llama_ubatch;
struct llama_model_loader;
struct llama_model;

// available models
enum llm_type {
    LLM_TYPE_UNKNOWN,
    LLM_TYPE_7_9B_A1_3B,
    LLM_TYPE_124B_A5_1B,
    LLM_TYPE_78B_A3_5B,  // Kolibri1
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);
std::string llama_rope_section_order_name(llama_rope_section_order rope_section_order);

// Map a GGUF activation-name string to llm_ffn_op_type. Returns `fallback` if
// the string is empty or not recognized.
llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback);

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_post_norm  = nullptr;

    // attention
    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * k_cache_s = nullptr;
    struct ggml_tensor * v_cache_s = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * wqkv      = nullptr;
    struct ggml_tensor * wq_b      = nullptr;
    struct ggml_tensor * wk_b      = nullptr;
    struct ggml_tensor * wv_b      = nullptr;
    struct ggml_tensor * wqkv_b    = nullptr;
    struct ggml_tensor * wqkv_gate = nullptr;

    // normalization
    struct ggml_tensor * ffn_norm         = nullptr;
    struct ggml_tensor * ffn_post_norm    = nullptr;

    // ff
    struct ggml_tensor * ffn_gate     = nullptr; // w1
    struct ggml_tensor * ffn_down     = nullptr; // w2
    struct ggml_tensor * ffn_up       = nullptr; // w3

    // MLA
    struct ggml_tensor * wq_a           = nullptr;
    struct ggml_tensor * wkv_a_mqa      = nullptr;
    struct ggml_tensor * attn_q_a_norm  = nullptr;
    struct ggml_tensor * attn_kv_a_norm = nullptr;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp      = nullptr;
    struct ggml_tensor * ffn_gate_exps     = nullptr;
    struct ggml_tensor * ffn_down_exps     = nullptr;
    struct ggml_tensor * ffn_up_exps       = nullptr;
    struct ggml_tensor * ffn_gate_up_exps  = nullptr;

    // ff shared expert
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp     = nullptr;
    struct ggml_tensor * ffn_down_shexp     = nullptr;
    struct ggml_tensor * ffn_up_shexp       = nullptr;

    // ff bias
    struct ggml_tensor * ffn_exp_probs_b = nullptr;

    // KDA (ssm_a and ssm_beta are with the gated delta net tensors below)
    struct ggml_tensor * ssm_q_conv = nullptr;
    struct ggml_tensor * ssm_k_conv = nullptr;
    struct ggml_tensor * ssm_v_conv = nullptr;
    struct ggml_tensor * ssm_f_a    = nullptr;
    struct ggml_tensor * ssm_dt_b   = nullptr;
    struct ggml_tensor * ssm_g_a    = nullptr;
    struct ggml_tensor * ssm_o_norm = nullptr;

    // rope frequency factors
    struct ggml_tensor * rope_freqs = nullptr;

    // weight scales, applied through llama_model::get_w_scale
    struct ggml_tensor * wq_s       = nullptr;
    struct ggml_tensor * wk_s       = nullptr;
    struct ggml_tensor * wv_s       = nullptr;
    struct ggml_tensor * wo_s       = nullptr;
    struct ggml_tensor * wqkv_s     = nullptr;
    struct ggml_tensor * ffn_gate_s = nullptr;
    struct ggml_tensor * ffn_up_s   = nullptr;
    struct ggml_tensor * ffn_down_s = nullptr;
    struct ggml_tensor * wqkv_gate_s     = nullptr;
    struct ggml_tensor * ffn_gate_exps_s = nullptr; // per expert
    struct ggml_tensor * ffn_down_exps_s = nullptr;
    struct ggml_tensor * ffn_up_exps_s   = nullptr; // also of ffn_gate_up_exps
    struct ggml_tensor * ffn_gate_shexp_s = nullptr;
    struct ggml_tensor * ffn_up_shexp_s   = nullptr;
    struct ggml_tensor * ffn_down_shexp_s = nullptr;
    struct ggml_tensor * ssm_out_s        = nullptr;
    struct ggml_tensor * ssm_alpha_s      = nullptr;
    struct ggml_tensor * ssm_beta_s       = nullptr;

    // input scales, loaded so that files that carry them load, not applied
    struct ggml_tensor * wq_in_s            = nullptr;
    struct ggml_tensor * wk_in_s            = nullptr;
    struct ggml_tensor * wv_in_s            = nullptr;
    struct ggml_tensor * wo_in_s            = nullptr;
    struct ggml_tensor * wqkv_in_s          = nullptr;
    struct ggml_tensor * wqkv_gate_in_s     = nullptr;
    struct ggml_tensor * ffn_gate_in_s      = nullptr;
    struct ggml_tensor * ffn_down_in_s      = nullptr;
    struct ggml_tensor * ffn_up_in_s        = nullptr;
    struct ggml_tensor * ffn_gate_exps_in_s = nullptr;
    struct ggml_tensor * ffn_down_exps_in_s = nullptr;
    struct ggml_tensor * ffn_up_exps_in_s   = nullptr;
    struct ggml_tensor * ffn_gate_shexp_in_s = nullptr;
    struct ggml_tensor * ffn_up_shexp_in_s   = nullptr;
    struct ggml_tensor * ffn_down_shexp_in_s = nullptr;
    struct ggml_tensor * ssm_out_in_s        = nullptr;
    struct ggml_tensor * ssm_alpha_in_s      = nullptr;
    struct ggml_tensor * ssm_beta_in_s       = nullptr;

    // gated delta net (qwen35)
    struct ggml_tensor * ssm_conv1d         = nullptr;
    struct ggml_tensor * ssm_dt             = nullptr;
    struct ggml_tensor * ssm_a              = nullptr;
    struct ggml_tensor * ssm_beta           = nullptr;
    struct ggml_tensor * ssm_alpha          = nullptr;
    struct ggml_tensor * ssm_norm           = nullptr;
    struct ggml_tensor * ssm_out            = nullptr;
    struct ggml_tensor * wqkv_z             = nullptr; // [wqkv | wqkv_gate] in one weight
    struct ggml_tensor * ssm_beta_alpha_cat = nullptr; // [ssm_beta | ssm_alpha] in one weight

    // dflash
    struct ggml_tensor * attn_sinks = nullptr;
    struct ggml_tensor * out_scale  = nullptr;

    // inkling2
    struct ggml_tensor * attn_rel         = nullptr;
    struct ggml_tensor * attn_rel_bias    = nullptr;
    struct ggml_tensor * attn_q_clip      = nullptr;
    struct ggml_tensor * attn_rel_clip    = nullptr;
    struct ggml_tensor * attn_sink_proj   = nullptr;
    struct ggml_tensor * attn_q_inp_norm  = nullptr;
    struct ggml_tensor * attn_k_inp_norm  = nullptr;
    struct ggml_tensor * attn_v_inp_norm  = nullptr;
    struct ggml_tensor * attn_k_conv      = nullptr;
    struct ggml_tensor * attn_v_conv      = nullptr;
    struct ggml_tensor * attn_k_conv_proj = nullptr;
    struct ggml_tensor * attn_v_conv_proj = nullptr;
    struct ggml_tensor * attn_post_conv   = nullptr;
    struct ggml_tensor * attn_res_query   = nullptr;
    struct ggml_tensor * ffn_post_conv    = nullptr;
    struct ggml_tensor * ffn_res_query    = nullptr;

    // hy_v4_flash: hyper-connections
    struct ggml_tensor * hc_attn_fn    = nullptr;
    struct ggml_tensor * hc_attn_base  = nullptr;
    struct ggml_tensor * hc_attn_scale = nullptr;
    struct ggml_tensor * hc_ffn_fn     = nullptr;
    struct ggml_tensor * hc_ffn_base   = nullptr;
    struct ggml_tensor * hc_ffn_scale  = nullptr;

    // hy_v4_flash: latent routed experts
    struct ggml_tensor * ffn_routed_down = nullptr; // n_embd -> n_expert_latent
    struct ggml_tensor * ffn_routed_up   = nullptr; // n_expert_latent -> n_embd

    // hy_v4_flash: pooled lightning indexer
    struct ggml_tensor * indexer_k_norm      = nullptr;
    struct ggml_tensor * indexer_proj        = nullptr;
    struct ggml_tensor * indexer_attn_k      = nullptr;
    struct ggml_tensor * indexer_attn_q_b    = nullptr; // note: for lora a/b, not bias
    struct ggml_tensor * indexer_comp_norm   = nullptr;
    struct ggml_tensor * indexer_kpool_gate  = nullptr;
    struct ggml_tensor * indexer_kpool_ape   = nullptr;
    struct ggml_tensor * indexer_kpool_value = nullptr;

    // hy_v4_flash: Engram
    struct ggml_tensor * engram_kv          = nullptr;
    struct ggml_tensor * engram_gate        = nullptr; // q_weight*k_weight folded
    struct ggml_tensor * engram_kv_b        = nullptr;
    struct ggml_tensor * engram_lookup_norm = nullptr;
    struct ggml_tensor * engram_pre_gamma   = nullptr;
    struct ggml_tensor * engram_hc_scale    = nullptr;
};

struct llama_device {
    bool is_meta;

    ggml_backend_dev_t dev;
};

// tensor parallelism (LLAMA_SPLIT_MODE_TENSOR): how each tensor is split over the devices of the meta device
struct llama_meta_device_get_split_state_userdata {
    size_t                     n_devices;
    const struct llama_model * model;
};

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata);

struct llama_prec_policy {
    // the key is the weight tensor `res->src[0]`, stores the recommended accumulation type of the op
    std::unordered_map<const ggml_tensor *, ggml_prec> prec_acc;

    // the key is the weight tensor `res->src[0]`, stores the recommended activation precision type
    std::unordered_map<const ggml_tensor *, ggml_prec> prec_src1;

    bool apply(ggml_tensor * res, const ggml_tensor * w = nullptr) const;

    void set_acc(const ggml_tensor * w, ggml_prec prec) {
        if (w) {
            prec_acc[w] = prec;
        }
    }

    void set_src1(const ggml_tensor * w, ggml_prec prec) {
        if (w) {
            prec_src1[w] = prec;
        }
    }

    ggml_prec get_src1(const ggml_tensor * w) const {
        const auto it = prec_src1.find(w);
        return it != prec_src1.end() ? it->second : GGML_PREC_UNDEFINED;
    }

    void load(llama_model_loader & ml, const llama_model & model);
};

struct llama_rope_params {
    int32_t n_rot      = 0; // 0 if the layer has no RoPE
    int32_t n_offs     = 0; // first rotated dim in a cached K head
    int32_t type       = LLAMA_ROPE_TYPE_NONE;
    int32_t n_ctx_orig = 0;

    float freq_base   = 0.0f;
    float freq_scale  = 0.0f;
    float ext_factor  = 0.0f;
    float attn_factor = 0.0f;
    float beta_fast   = 0.0f;
    float beta_slow   = 0.0f;

    ggml_tensor * factors = nullptr;
    ggml_tensor * k_norm  = nullptr; // K norm weight applied after RoPE, a K-shift has to undo it
};

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    // per-tensor activation precision policy
    llama_prec_policy prec_policy;

    // weight -> "<name>.scale" tensor (e.g. NVFP4 scale2), applied by build_lora_mm and build_lora_mm_id
    std::unordered_map<const ggml_tensor *, ggml_tensor *> w_scales;

    ggml_tensor * get_w_scale(const ggml_tensor * w) const;

    // see llama_model_set_attn_rot
    bool attn_rot = true;

    // for classifier models
    std::vector<std::string> classifier_labels;
    std::vector<std::string> decision_labels;

    struct ggml_tensor * tok_embd   = nullptr;
    struct ggml_tensor * tok_norm   = nullptr;

    // hy_v4_flash: hyper-connection head, Engram table
    struct ggml_tensor * hc_head_fn         = nullptr;
    struct ggml_tensor * hc_head_base       = nullptr;
    struct ggml_tensor * hc_head_scale      = nullptr;
    struct ggml_tensor * per_layer_tok_embd = nullptr;

    // hy_v4_flash Engram: token id -> id of its normalized form, hashed into the n-grams
    std::vector<int32_t> engram_token_map;
    // hy_v4_flash Engram: n_modules*ngram_size hash multipliers
    std::vector<uint64_t> engram_multipliers;

    // lazily read tensors whose rows llama_prefetch_rows may read ahead
    std::unordered_set<const ggml_tensor *> can_prefetch;

    struct ggml_tensor * output_norm     = nullptr;
    struct ggml_tensor * output_res_query = nullptr; // inkling2: query of the final attention-residual mix
    struct ggml_tensor * output          = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    // per-tensor scale and input scale of the LM head
    struct ggml_tensor * output_s    = nullptr;
    struct ggml_tensor * output_in_s = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;

    // dflash feature fusion layer
    struct ggml_tensor * fc      = nullptr;
    struct ggml_tensor * fc_s    = nullptr;
    struct ggml_tensor * fc_in_s = nullptr;
    struct ggml_tensor * d2t = nullptr;  // draft to target vocabulary mapping

    // dspark
    struct ggml_tensor * dspark_markov_w1   = nullptr;
    struct ggml_tensor * dspark_markov_w2   = nullptr;
    struct ggml_tensor * dspark_markov_w2_s = nullptr;
    struct ggml_tensor * dspark_conf_proj   = nullptr;
    struct ggml_tensor * dspark_conf_proj_b = nullptr;
    struct ggml_tensor * dspark_joint_proj   = nullptr; // rnn markov head
    struct ggml_tensor * dspark_joint_proj_b = nullptr;

    // target-model layer ids whose hidden states feed the dflash drafter
    std::vector<int32_t> target_layer_ids;

    std::vector<llama_layer> layers;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // list of devices used in this model
    std::vector<llama_device> devices;

    // kv_gpu: the device of the KV cache and the attention of the full-attention layers, it holds no layers; nullptr if not used
    ggml_backend_dev_t kv_dev = nullptr;

    // for the meta device of LLAMA_SPLIT_MODE_TENSOR
    struct llama_meta_device_get_split_state_userdata get_split_state_ud = {};

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    // for keeping track of associated LoRA adapters
    std::unordered_set<llama_adapter_lora *> loras;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit llama_model(const llama_model_params & params);
    virtual ~llama_model();

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    llama_ftype ftype() const;

    size_t size() const; // file size
    size_t n_tensors() const;
    size_t n_devices() const;
    const float * tensor_split() const;

    uint32_t n_gpu_layers() const;
    llama_split_mode split_mode() const;
    llama_tp_attn    tp_attn() const;
    bool no_alloc() const;

    // LongHaul: expert tensors kept on disk (TENSOR_LONGHAUL), read by the expert cache
    bool has_longhaul() const;
    bool is_longhaul(const ggml_tensor * t) const;
    // read size bytes of t at byte offs of its data, from the model file
    void longhaul_read(const ggml_tensor * t, void * dst, size_t offs, size_t size) const;
    // the model file and the file offset of the data of t, false if not known
    bool longhaul_src(const ggml_tensor * t, size_t & file, size_t & offs) const;
    // the path of a model file, empty if it was opened from a FILE pointer
    const std::string & longhaul_path(size_t file) const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // total number of parameters in the model
    uint64_t n_elements() const;

    void print_info() const;

    // with LLAMA_SPLIT_MODE_TENSOR: the share of each device per tensor class, after rounding
    void log_tensor_split() const;

    ggml_backend_dev_t dev_layer(int il) const;
    ggml_backend_dev_t dev_output() const;

    // true if the KV cache and the attention of layer il are on kv_dev
    bool kv_remote(int il) const;
    // the device of the KV cache of layer il
    ggml_backend_dev_t dev_kv(int il) const;

    bool has_tensor_overrides() const;

    const struct ggml_tensor * get_tensor(const char * name) const;

    // the RoPE of a layer, used by both the model graph and the K-shift
    virtual llama_rope_params get_rope_params(const llama_cparams & cparams, int il) const;

    // same, for the keys of the lightning indexer
    virtual llama_rope_params get_rope_params_indexer(const llama_cparams & cparams, int il) const;

    float get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float get_rope_freq_scale(const llama_cparams & cparams, int il) const;

    ggml_tensor * get_rope_factors(const llama_cparams & cparams, int il) const;

    llama_memory_i * create_memory(const llama_memory_params & params, const llama_cparams & cparams) const;

    // graph size budget: max graph nodes per model tensor
    virtual uint32_t graph_nodes_per_tensor() const { return 8; }

    // the layer whose weights cache slot il uses
    // looped architectures run some layers more than once and give each later pass its own cache slot
    virtual int layer_phys(int il) const { return il; }

    // number of layers that hold weights, the unit of n_gpu_layers and tensor_split
    // cache slots that alias another layer follow the device of that layer
    int n_layer_offload() const;

    ggml_cgraph * build_graph(const llm_graph_params & params) const;

    virtual void load_stats  (llama_model_loader & ml) = 0;
    virtual void load_hparams(llama_model_loader & ml) = 0;
    virtual void load_vocab  (llama_model_loader & ml) = 0;
    virtual bool load_tensors(llama_model_loader & ml) = 0; // returns false if cancelled by progress_callback

    // model must define these
    virtual void load_arch_hparams(llama_model_loader & ml) = 0;
    virtual void load_arch_tensors(llama_model_loader & ml) = 0;
    virtual std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const = 0;

    // optional, runs after the scale pass
    virtual void post_arch_tensors(llama_model_loader &) {}

protected:
    llama_model_params params;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params);
llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params);

// model must inherit from this
struct llama_model_base : public llama_model {
    friend struct llama_model;

    llama_model * model;
    llama_model_loader * ml = nullptr;
    const LLM_TN tn;

    // llama_model_loader is not yet defined at this point, so we will set it after construction
    const int TENSOR_DUPLICATED;
    const int TENSOR_NOT_REQUIRED;
    const int TENSOR_SKIP;
    const int TENSOR_SKIP_IF_VIRTUAL;
    const int TENSOR_ALLOW_RESHAPE;
    const int TENSOR_READ_LAZY;
    const int TENSOR_LONGHAUL;

    explicit llama_model_base(const llama_model_params & params);
    virtual ~llama_model_base() = default;

    ggml_tensor * create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    // convenience overload of create_tensor that doesn't require llama_model_loader
    ggml_tensor * create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    // helper: try merged gate_up_exps first, fall back to separate gate and up
    void create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_,
                int64_t n_ff_, int64_t n_expert_, int flags);

    // helper: try to load merged qkv first, fall back to separate q, k, v
    void create_tensor_qkv(llama_layer & layer, int bid,
                int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
                int flags);

    // helper: read the SWA pattern as one flag per layer, or as a period expanded by set_swa_pattern
    void load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first = false);

    void load_stats  (llama_model_loader & ml) override;
    void load_hparams(llama_model_loader & ml) override;
    void load_vocab  (llama_model_loader & ml) override;
    bool load_tensors(llama_model_loader & ml) override;
    void load_prec_policy(llama_model_loader & ml);

    // model must define these
    void load_arch_hparams(llama_model_loader & ml) override = 0;
    void load_arch_tensors(llama_model_loader & ml) override = 0;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override = 0;
};

const char * llm_type_name(llm_type type);

// convenience macro for loading local variables for load_tensors() in llama_model_base
// note: cast to int64_t since we will use these for the tensor dimensions
#define LLAMA_LOAD_LOCALS \
    const int     n_layer        = hparams.n_layer();        GGML_UNUSED(n_layer); \
    const int     n_layer_all    = hparams.n_layer_all;      GGML_UNUSED(n_layer_all); \
    const int64_t n_head         = hparams.n_head();         GGML_UNUSED(n_head); \
    const int64_t n_head_kv      = hparams.n_head_kv();      GGML_UNUSED(n_head_kv); \
    const int64_t n_embd         = hparams.n_embd;           GGML_UNUSED(n_embd); \
    const int64_t n_embd_k_gqa   = hparams.n_embd_k_gqa();   GGML_UNUSED(n_embd_k_gqa); \
    const int64_t n_embd_v_gqa   = hparams.n_embd_v_gqa();   GGML_UNUSED(n_embd_v_gqa); \
    const int64_t n_embd_head_k  = hparams.n_embd_head_k();  GGML_UNUSED(n_embd_head_k); \
    const int64_t n_embd_head_v  = hparams.n_embd_head_v();  GGML_UNUSED(n_embd_head_v); \
    const int64_t n_ff           = hparams.n_ff();           GGML_UNUSED(n_ff); \
    const int64_t n_embd_gqa     = n_embd_v_gqa;             GGML_UNUSED(n_embd_gqa); \
    const int64_t n_vocab        = vocab.n_tokens();         GGML_UNUSED(n_vocab); \
    const int64_t n_token_types  = vocab.n_token_types();    GGML_UNUSED(n_token_types); \
    const int64_t n_rot          = hparams.n_rot();          GGML_UNUSED(n_rot); \
    const int64_t n_expert       = hparams.n_expert;         GGML_UNUSED(n_expert); \
    const int64_t n_expert_used  = hparams.n_expert_used();  GGML_UNUSED(n_expert_used); \
    const int64_t n_ctx_train    = hparams.n_ctx_train;      GGML_UNUSED(n_ctx_train);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);
