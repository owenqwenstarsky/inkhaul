#pragma once

#include "llama-model.h"
#include "llama-graph.h"
#include "llama-model-loader.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>
#include <map>

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
