#pragma once

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-adapter.h"

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <memory>
#include <set>
#include <functional>
#include <map>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_cparams;
struct llama_layer;
struct llama_model;
struct llama_prec_policy;

class llama_moe_cache;

struct llama_memory_context_i;

class llama_kv_cache_context;
class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_iswa_context;

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
};

enum llm_fused_op {
    LLM_FUSED_OP_FLASH_ATTN,
    LLM_FUSED_OP_FLASH_ATTN_KV_ROWS,
    LLM_FUSED_OP_DSV4_HC_PRE,
    LLM_FUSED_OP_DSV4_HC_COMB,
    LLM_FUSED_OP_DSV4_HC_POST,
    LLM_FUSED_OP_TOP_K_MASK,
};

enum llm_ffn_op_type : int {
    LLM_FFN_NONE = 0,           // sentinel: unset; archs must assign before use
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_GELU_ERF,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_GEGLU_ERF,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
    LLM_FFN_SITU,           // kimi-k3
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};


struct llm_graph_params;

//
// llm_graph_input
//

class llm_graph_input_i {
public:
    llm_graph_input_i() {
        const char * LLAMA_GRAPH_INPUT_DEBUG = getenv("LLAMA_GRAPH_INPUT_DEBUG");
        debug = LLAMA_GRAPH_INPUT_DEBUG ? atoi(LLAMA_GRAPH_INPUT_DEBUG) : 0;
    }

    virtual ~llm_graph_input_i() = default;

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const llm_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }
protected:
    // env: LLAMA_GRAPH_INPUT_DEBUG
    int debug = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens       = nullptr; // I32 [n_batch]
    ggml_tensor * embd         = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * mixed_tokens = nullptr; // I32 [n_tok_rows], mixed path: ids of the token rows
    ggml_tensor * mixed_slots  = nullptr; // I64 [n_tok_rows], mixed path: batch index of the token rows
    ggml_tensor * mixed_embd   = nullptr; // F32 [n_embd, n_batch], mixed path: embd rows, token rows are overwritten
    ggml_tensor * scale_rows   = nullptr; // F32 [1, n_batch], per-row scale: scale_tok for token rows, 1 for embd rows
    ggml_tensor * embd_rows    = nullptr; // I32 [n_batch], embd row path: rows of ctx_other's layer input device buffer

    float scale_tok = 1.0f;

    const int64_t n_embd = 0;
};


// original batch index of each ubatch token, used as set_rows destination rows
class llm_graph_input_batch_idxs : public llm_graph_input_i {
public:
    llm_graph_input_batch_idxs() = default;
    virtual ~llm_graph_input_batch_idxs() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * idxs = nullptr; // I64 [n_batch]
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
    llm_graph_input_pos(uint32_t n_pos_per_embd, llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ)
        : n_pos_per_embd(n_pos_per_embd), rope_section_order(rope_section_order) {}
    virtual ~llm_graph_input_pos() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    const uint32_t n_pos_per_embd = 1;
    const llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ;
};




class llm_graph_input_out_ids : public llm_graph_input_i {
public:
    llm_graph_input_out_ids(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const uint32_t n_outputs;
};

// random expert ids for the MoE GEMMs, for benchmarks only (env: LLAMA_MOE_RANDOM_ROUTING)
class llm_graph_input_moe_ids : public llm_graph_input_i {
public:
    llm_graph_input_moe_ids(int64_t n_expert) : n_expert(n_expert) {}
    virtual ~llm_graph_input_moe_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * ids = nullptr; // I32 [n_expert_used_max, n_tokens]

    const int64_t n_expert;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
    llm_graph_input_mean(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_mean() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const llama_cparams cparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
    llm_graph_input_cls(const llama_cparams & cparams, const llm_arch arch) : cparams(cparams), arch(arch) {}
    virtual ~llm_graph_input_cls() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const llama_cparams cparams;
    const llm_arch arch;
};

class llm_graph_input_rs : public llm_graph_input_i {
public:
    llm_graph_input_rs(const llama_memory_recurrent_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_rs() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main; // I32 [n_seqs]
    ggml_tensor * s_copy_tail; // I32 [n_rs - 1]

    // rs_replay: the ubatch first replays the kept pending tokens of the last ubatch (none is a no-op)
    bool          has_rep  = false;
    int64_t       n_conv   = 0;       // conv window columns, the widest conv of the model
    int64_t       n_pend   = 0;       // pending slots per seq
    ggml_tensor * rep_n    = nullptr; // I32 [n_seqs], kept pending tokens
    ggml_tensor * rep_cols = nullptr; // I32 [n_conv, n_seqs], conv history columns of the window after the kept tokens

    // views of rep_cols for the delta-net conv and the PLE conv windows, computed once per graph and shared across layers
    ggml_tensor * rep_cols_r = nullptr;
    ggml_tensor * rep_cols_p = nullptr;

    void set_input_rep(const llama_memory_recurrent_context * mctx, const llama_ubatch * ubatch);

    const llama_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;
};


// KQ mask and the optional per-sequence cell rows of a unified cache (see ggml_flash_attn_ext_rows)
// the mask has two shapes depending on whether `rows` is used:
//   rows != nullptr: [n_kv_rows, n_batch/n_slices, 1, n_slices]
//   rows == nullptr: [n_kv,      n_batch/n_stream, 1, n_stream]
struct llm_attn_mask {
    ggml_tensor * raw  = nullptr;
    ggml_tensor * rows = nullptr; // I32 [n_kv_rows, n_slices]
    uint32_t      n_kv = 0;

    // fill every token of a sequence with the mask row of its first token (dspark draft blocks)
    bool block_uniform = false;
};

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    llm_attn_mask self_attn_mask;

    // note: assumes v_rot^2 == I
    std::vector<ggml_tensor *> self_k_rot;
    std::vector<ggml_tensor *> self_v_rot;

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};





class llm_graph_input_attn_kv_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_v_idxs()     const { return self_v_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }
    ggml_tensor * get_v_idxs_swa() const { return self_v_idxs_swa; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs     = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    llm_attn_mask self_attn_mask;
    llm_attn_mask self_attn_mask_swa;

    std::vector<ggml_tensor *> self_k_rot;
    std::vector<ggml_tensor *> self_v_rot;

    std::vector<ggml_tensor *> self_k_rot_swa;
    std::vector<ggml_tensor *> self_v_rot_swa;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};








// hybrid iswa memory with a per-head attention bias that depends on the distance from the query to the cell (Inkling)
// the KQ masks are always dense (rows == nullptr) and are filled from the same cells as the distances, so that the bias
// gathered by distance lines up with the mask
class llm_graph_input_mem_hybrid_iswa_rel : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa_rel(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx,
            uint32_t extent,
            uint32_t extent_swa) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        extent(extent),
        extent_swa(extent_swa),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa_rel() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    // distance from the token to the cell, extent for the empty, future and more distant cells
    ggml_tensor * self_rel_pos     = nullptr; // I32 [n_kv,     n_batch]
    ggml_tensor * self_rel_pos_swa = nullptr; // I32 [n_kv_swa, n_batch]

    const uint32_t extent;
    const uint32_t extent_swa;

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

class llm_graph_input_sampling : public llm_graph_input_i {
public:
    llm_graph_input_sampling(std::map<llama_seq_id, llama_sampler *> samplers) :
        samplers(std::move(samplers)) { }
    virtual ~llm_graph_input_sampling() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::map<llama_seq_id, llama_sampler *> samplers;
};

//
// llm_graph_result
//

// these objects deliver the result from the graph build process back to the llama_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the set_inputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings, etc.
//   these are used by the llama_context to extact the relevant data, based on the compute parameters

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

class llm_graph_result;

struct llm_graph_params {
    const llama_model * model = nullptr;

    llama_cparams cparams;

    llama_ubatch ubatch; // note: intentionally make a copy

    llm_graph_type gtype;

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;

    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_moe_cache        * moe_cache;

    ggml_tensor * layer_inp_dev = nullptr; // own layer input device buffer (write)
    ggml_tensor * layer_inp_src = nullptr; // ctx_other's layer input device buffer (read)

    std::map<llama_seq_id, llama_sampler *> samplers;

    static bool samplers_equal(
          const std::map<llama_seq_id, llama_sampler *> & lhs,
          const std::map<llama_seq_id, llama_sampler *> & rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (const auto & [seq_id, sampler] : lhs) {
            auto it = rhs.find(seq_id);
            if (it == rhs.end() || it->second != sampler) {
                return false;
            }
        }
        return true;
    }

    uint32_t n_outputs;

    llm_graph_cb cb;

    llm_graph_result * res;

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const llm_graph_params & other) const {
        // first check the ubatch
        bool can_reuse_ubatch =
            ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
            ubatch.n_tokens     == other.ubatch.n_tokens &&
            ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
            ubatch.n_seqs       == other.ubatch.n_seqs &&
            ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq &&
            ubatch.is_mixed()   == other.ubatch.is_mixed() &&
            (
                (!ubatch.token && !other.ubatch.token) ||
                (!ubatch.embd  && !other.ubatch.embd)  ||
                (ubatch.token && other.ubatch.token && ubatch.embd && other.ubatch.embd)
            ) &&
            (!ubatch.embd_row == !other.ubatch.embd_row);

        // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
        //   the reason is because the set of attention streams would be different for different sequences
        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {
                // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
                //   therefore we cannot perform the sequence id check. normally should never happen
                can_reuse_ubatch = false;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
                }
            }
        }

        if (!can_reuse_ubatch) {
            return false;
        }

        if (n_outputs != other.n_outputs) {
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            return false;
        }

        if (samplers.size() > 0) {
            if (!ubatch.data || !other.ubatch.data) {
                return false;
            }

            // check that the outputs are the same for all samplers
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.output[i]    != other.ubatch.output[i] ||
                    ubatch.seq_id[i][0] != other.ubatch.seq_id[i][0]) {
                    return false;
                }
            }
        }

        return
            cparams.embeddings              == other.cparams.embeddings              &&
            cparams.embeddings_nextn        == other.cparams.embeddings_nextn        &&
            cparams.embeddings_nextn_masked == other.cparams.embeddings_nextn_masked &&
            cparams.causal_attn             == other.cparams.causal_attn             &&
            layer_inp_dev == other.layer_inp_dev &&
            layer_inp_src == other.layer_inp_src &&
            model == other.model &&
            gtype == other.gtype &&
            loras == other.loras;
    }
};

struct llm_graph_fused_node {
    llm_fused_op op;
    ggml_tensor * tensor;
    int il;
};

class llm_graph_result {
public:
    llm_graph_result(int64_t max_nodes);

    virtual ~llm_graph_result() = default;

    ggml_tensor * get_inp_tokens()  const { return t_inp_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }
    ggml_tensor * get_h_nextn()     const { return t_h_nextn; }

    ggml_tensor * get_layer_inp(int il) const { return t_layer_inp[il]; }

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;

    void set_max_nodes(int64_t max_nodes);

    void reset();

    void set_inputs(const llama_ubatch * ubatch);
    void set_outputs(const llm_graph_params & params);

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const llm_graph_params & params);

    llm_graph_input_i * add_input(llm_graph_input_ptr input);

    void add_fused_node(llm_graph_fused_node result);


    const std::vector<llm_graph_fused_node> & get_fused_nodes() const { return fused_nodes; }

    void set_params(const llm_graph_params & params);

    // important graph nodes
    ggml_tensor * t_inp_tokens  = nullptr;
    ggml_tensor * t_inp_embd    = nullptr; // [n_embd_inp, n_tokens]
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;
    ggml_tensor * t_h_nextn     = nullptr; // [n_embd, n_outputs] hidden state before final output norm

    std::vector<ggml_tensor *> t_layer_inp;

    std::vector<ggml_tensor *> t_sampled;
    std::vector<ggml_tensor *> t_sampled_probs;
    std::vector<ggml_tensor *> t_sampled_logits;
    std::vector<ggml_tensor *> t_candidates;

    std::vector<ggml_tensor *> t_sampling_all;

    // [0] is the LLAMA_MOE_RANDOM_ROUTING input, the rest are views of it
    std::vector<ggml_tensor *> t_moe_ids;

    std::vector<llm_graph_input_ptr> inputs;
    std::vector<llm_graph_fused_node> fused_nodes;


    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::unique_ptr<uint8_t[]> buf_compute_meta;
    size_t buf_compute_meta_size = 0;

    ggml_cgraph * gf;

private:
    int64_t max_nodes;

    // keep a copy of the previous graph parameters
    // we will use this to determine whether the graph can be reused by comparing them with the new parameters
    // note: these are updated after constructing the new graph
    llm_graph_params params;

    // env: LLAMA_GRAPH_RESULT_DEBUG
    int debug = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

//
// llm_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using llm_graph_get_rows_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor * states, ggml_tensor * ids)>;

struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};

struct llm_graph_amax_src1 {
    ggml_tensor * res;     // cur rescaled
    ggml_tensor * scale_s; // scale to undo on the matmul result, or nullptr
};

struct llm_graph_context {
    const llama_model & model;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    const llama_prec_policy & prec_policy;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int64_t n_tokens;
    const int64_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_moe_cache        * moe_cache;


    ggml_tensor * layer_inp_dev;
    ggml_tensor * layer_inp_src;

    std::map<llama_seq_id, llama_sampler *> samplers;

    const llm_graph_cb & cb_func;

    llm_graph_result * res;

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    llm_graph_context(const llm_graph_params & params);
    virtual ~llm_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;

    // true when the last layer must be narrowed to the output rows before the nextn hidden state is captured
    bool crop_before_nextn(const ggml_tensor * inp_out_ids) const {
        return inp_out_ids != nullptr && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    }

    // true when the nextn hidden state must be narrowed to the output rows after it is captured
    bool crop_after_nextn(const ggml_tensor * inp_out_ids) const {
        return inp_out_ids != nullptr && cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;
    }

    //
    // common
    //



    // do mat_mul, while optionally apply lora and per-tensor scale
    // w_s defaults to the weight scale of w
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr) const;

    // do mat_mul_id, while optionally apply lora and per-expert scale
    // if slots is set, the experts are selected by slots instead of ids, from the MoE cache if cached (see llama_moe_cache::build_lookup)
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids,
              ggml_tensor * slots = nullptr) const;

    // rescale `cur` (the src1 of a matmul) by a power of two, computed from its maximum absolute
    // keep both individual values and Q8 activation block sums within F16 range
    llm_graph_amax_src1 build_amax_src1_f16(
              ggml_tensor * cur) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const;

    // Set reshape to false to return contiguous projections before clamp/reshape.
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape = true) const;

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
             ggml_tensor * gate,
             ggml_tensor * gate_b,
             ggml_tensor * gate_s,
             ggml_tensor * down,
             ggml_tensor * down_b,
             ggml_tensor * down_s,
             ggml_tensor * act_scales,
         llm_ffn_op_type   type_op,
       llm_ffn_gate_type   type_gate,
                     int   il) const;

    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * post_exp_norm = nullptr,
                   float   post_exp_norm_eps = 1e-5f) const;

    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * up_exps,
             ggml_tensor * up_exps_b,
             ggml_tensor * gate_exps,
             ggml_tensor * gate_exps_b,
             ggml_tensor * down_exps,
             ggml_tensor * down_exps_b,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * gate_up_exps_b = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * post_exp_norm = nullptr,
                   float   post_exp_norm_eps = 1e-5f) const;

    //
    // inputs
    //

    // tok_scale: applied to token rows only
    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd, float tok_scale = 1.0f) const;


    ggml_tensor * build_inp_pos() const;

    // applies the RoPE of layer il, cur is returned as is if the layer has none
    ggml_tensor * build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) const;

    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;
    ggml_tensor * build_inp_moe_ids(int64_t n_used, int64_t n_rows) const;


    //
    // attention
    //

    ggml_tensor * build_attn_mha(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans = false)
            ggml_tensor * kq_b,
            ggml_tensor * kq_mask,
            ggml_tensor * kv_rows, // [n_kv, n_stream], k and v are shared by the streams
            ggml_tensor * sinks,   // [n_head_q]
            ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                int64_t   n_kv_max,
                  float   kq_scale,
                    int   il) const;


    llm_graph_input_attn_kv * build_attn_inp_kv() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
                  float   kq_scale,
                    int   il) const;







    // ggml_top_k_mask with top_k viewed in the row layout of kq_mask, see ggml.h
    ggml_tensor * build_fused_top_k_mask(
            ggml_tensor * kq_mask, // [n_kv, n_tokens/n_stream, 1, n_stream]
            ggml_tensor * top_k,   // [n_top, n_tokens]
                    int   il) const;


    llm_graph_input_attn_kv_iswa * build_attn_inp_kv_iswa() const;


    // note: if k_cur or v_cur are not provided, they will not be stored in the memory
    ggml_tensor * build_attn(
            llm_graph_input_attn_kv_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;





    //
    // recurrent
    //

    // TODO: move this implementation to llama_memory_recurrent.
    //       this is analogous to llama_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `llama_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy,
            ggml_tensor * state_copy_main,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const llm_graph_get_rows_fn & get_state_rows = nullptr) const;


    ggml_tensor * build_rs(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const llm_graph_get_rows_fn & get_state_rows = nullptr) const;

    // store the conv history of the ubatch cells, the last n_cols columns of conv_x {n_cols + n_seq_tokens, channels, n_seqs},
    // at byte offset offs of each row of s, and with n_rs_seq > 0 once per rollback slot of this ubatch
    void build_conv_state_store(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
            ggml_tensor * conv_x,
                int64_t   n_cols,
                 size_t   offs) const;


    //
    // hybrid
    //



    // extent, extent_swa: the distance that the bias tables of the full and the SWA layers stop at
    llm_graph_input_mem_hybrid_iswa_rel * build_inp_mem_hybrid_iswa_rel(
            uint32_t extent,
            uint32_t extent_swa) const;

    // kq_b is added to the KQ mask, after the KQ scale
    ggml_tensor * build_attn(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,  // [n_kv, n_tokens/n_stream, n_head_q, n_stream]
            ggml_tensor * sinks, // [n_head_q]
                  float   kq_scale,
                    int   il) const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b,
            ggml_tensor * cls_norm) const;

    //
    // logits
    //

    void build_logits() const;

    //
    // sampling (backend sampling)
    //

    void build_sampling() const;

    // write enabled layer inputs into layer_inp_dev at their batch rows
    void build_layer_inp_dev() const;

};

// TODO: better name
int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional);
