#pragma once

#include "llama-model.h"
#include "llama-graph.h"
#include "llama-model-loader.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>
#include <map>

// ref: https://github.com/ggml-org/llama.cpp/pull/28068
static inline ggml_tensor * build_gdn_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

//
// base classes
//

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s);

    // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
    // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
    ggml_tensor * build_conv_state(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            ggml_tensor *        qkv_mixed,
            int64_t              conv_kernel_size,
            int64_t              conv_channels,
            int                  il);

    // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
    // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
    ggml_tensor * build_recurrent_attn(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);

    // rs_replay variants: replay the kept pending tokens, then keep or commit the new ones [TAG_RECURRENT_REPLAY]
    // x: [channels, n_seq_tokens, n_seqs] new conv inputs; returns [n_conv + n_seq_tokens, channels, n_seqs]
    // a row of conv_states_all can hold several histories [channels, n_conv + n_pend] one after the other:
    // blk picks one, and hist_rows are the rows of the ubatch cells (build_rs) shared by the blocks, nullptr gathers them
    ggml_tensor * build_conv_state_replay(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            ggml_tensor *        x,
            int64_t              n_conv,
            int64_t              channels,
            int                  il,
            ggml_tensor *        hist_rows = nullptr,
            int64_t              blk       = 0);

    ggml_tensor * build_recurrent_attn_replay(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);
};

// Quant-only stub for mmproj GGUFs
// none of these are ever called, they only exist to satisfy the llama_model_base interface
struct llama_model_clip : public llama_model_base {
    llama_model_clip(const struct llama_model_params & params) : llama_model_base(params) {}

    [[noreturn]]
    void load_arch_hparams(llama_model_loader & ml) override;

    [[noreturn]]
    void load_arch_tensors(llama_model_loader & ml) override;

    [[noreturn]]
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_inkling2 : public llama_model_base {
    llama_model_inkling2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    uint32_t graph_nodes_per_tensor() const override { return 32; }

    // the layers stored in the file
    virtual int n_layer_phys() const { return hparams.n_layer(); }

    // message controllers of the q, k, v and ffn inputs of a looped layer (inkling2_loop)
    enum { LOOP_CTRL_Q, LOOP_CTRL_K, LOOP_CTRL_V, LOOP_CTRL_FFN, LOOP_CTRL_COUNT };

    struct loop_layer {
        ggml_tensor * msg_query = nullptr;
        ggml_tensor * ctrl_down[LOOP_CTRL_COUNT]   = {};
        ggml_tensor * ctrl_gain[LOOP_CTRL_COUNT]   = {};
        ggml_tensor * ctrl_forget[LOOP_CTRL_COUNT] = {};
    };

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
        ggml_tensor * mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query);
        ggml_tensor * clip(ggml_tensor * cur, ggml_tensor * scale, float threshold);
        ggml_tensor * convolve(
                ggml_tensor * cur,
                ggml_tensor * weight,
                ggml_tensor * dynamic,
                ggml_tensor * state,
                ggml_tensor * history,
                llm_graph_input_rs * inp,
                int64_t offset);
        ggml_tensor * control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl);

    protected:
        // the helpers alone, a derived graph builds its own body
        struct no_build {};
        graph(const llm_graph_params & params, no_build) : llm_graph_context(params) {}

        // the inputs and the token embeddings
        ggml_tensor * build_inputs();
        // cache slot il of physical layer ip; messages [LOOP_CTRL_COUNT, n_embd, n_tokens] are mixed into its inputs
        ggml_tensor * build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
                ggml_tensor * messages = nullptr, const loop_layer * loop = nullptr);
        void build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks);

        llm_graph_input_mem_hybrid_iswa_rel * inp  = nullptr;
        llm_graph_input_attn_kv_iswa        * attn = nullptr;
        llm_graph_input_rs                  * recr = nullptr;

        ggml_tensor * inp_out_ids = nullptr;
        ggml_tensor * tau         = nullptr; // log-scaled positions
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// the cache slots after the physical layers repeat the core [core_start, core_end) for each later pass
struct llama_model_inkling2_loop : public llama_model_inkling2 {
    llama_model_inkling2_loop(const struct llama_model_params & params) : llama_model_inkling2(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    void post_arch_tensors(llama_model_loader & ml) override;

    int n_layer_phys() const override { return n_phys; }
    int layer_phys(int il) const;

    int      n_phys     = 0;
    uint32_t core_start = 0;
    uint32_t core_end   = 0;
    uint32_t n_passes   = 0;
    uint32_t gate_rank  = 0;
    bool     varying    = false; // trained for a varying number of passes

    // indexed by physical layer - core_start
    std::vector<loop_layer> loop_layers;

    struct graph : public llama_model_inkling2::graph {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// DSpark draft model for Inkling 2 (DFlash + a Markov head, dense backbone)
struct llama_model_dflash : public llama_model_base {
    llama_model_dflash(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    void load_tensors_dspark(int64_t n_vocab_draft);
    void load_tensors_layers_base();

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    // inputs and stages shared by the decoder graphs of the heads
    struct graph_base : public llm_graph_context {
        int64_t n_embd_head;
        bool    use_iswa;
        int     sections[4];

        ggml_tensor * inp_pos;

        llm_graph_input_attn_kv      * inp_attn      = nullptr;
        llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;

        graph_base(const llm_graph_params & params);

        ggml_tensor * build_inject();
        ggml_tensor * build_inp_tokens();
        ggml_tensor * build_layers(ggml_tensor * inp_tokens);
        ggml_tensor * build_logits(ggml_tensor * cur);
        ggml_tensor * build_rope(ggml_tensor * cur, ggml_tensor * pos, int il);
    };

    struct graph_dspark : public graph_base {
        graph_dspark(const llm_graph_params & params);
    };

    // graph helpers
    static ggml_tensor * build_inject_inp(llm_graph_context & g, const llama_model & model);
    static ggml_tensor * get_tok_embd(const llm_graph_context & g, const llama_model & model);
    static std::pair<ggml_tensor *, ggml_tensor *> get_lm_head(const llm_graph_context & g, const llama_model & model);

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// Qwen 3.5 / 3.6: gated delta net layers with a full-attention layer every few layers
// the MTP block at the end of the file is skipped
struct llama_model_qwen35 : public llama_model_base {
    llama_model_qwen35(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    uint32_t graph_nodes_per_tensor() const override { return 32; }

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// the routed experts stay on disk, see TENSOR_LONGHAUL
struct llama_model_qwen35moe : public llama_model_base {
    llama_model_qwen35moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    uint32_t graph_nodes_per_tensor() const override { return 32; }

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_kolibri1 : public llama_model_base {
    llama_model_kolibri1(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    class llm_graph_input_sparse;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

        // sparse attention of a full-attention layer of Kolibri-1.1
        //   pool: in: key pool of the closest preceding indexer, out: the key pool for the next reindexers
        ggml_tensor * build_attn_sparse(
                llm_graph_input_attn_kv_iswa * inp_attn,
                llm_graph_input_sparse * inp_sparse,
                ggml_tensor * q_cur,  // [n_embd_head, n_head,    n_tokens]
                ggml_tensor * k_cur,  // [n_embd_head, n_head_kv, n_tokens], nullptr for layers reading another layer's K/V
                ggml_tensor * v_cur,  // [n_embd_head, n_head_kv, n_tokens], nullptr for layers reading another layer's K/V
                ggml_tensor * & pool, // F16 [n_kv, n_tokens]
                        float   kq_scale,
                          int   il) const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_bailingmoe3 : public llama_model_base {
    llama_model_bailingmoe3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    uint32_t graph_nodes_per_tensor() const override { return 32; }

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);

        // causal conv1d over one of Q/K/V, qkv selects which third of the conv state row is used
        ggml_tensor * build_causal_conv1d(llm_graph_input_rs * inp_rs, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                          int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                          int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
