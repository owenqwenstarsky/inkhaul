#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llm_graph_context;

// the experts of a MoE layer, as llm_graph_context::build_moe_ffn passes them to the MoE cache
struct llama_moe_cache_layer {
    int32_t       il;
    ggml_tensor * up_exps;
    ggml_tensor * gate_exps;
    ggml_tensor * down_exps;
    ggml_tensor * gate_up_exps;
};

// the part of a lookup on one device
struct llama_moe_cache_part {
    int32_t       dev   = -1;      // the device of the cache, see llama_moe_cache::backend
    ggml_tensor * slots = nullptr; // [n_expert_used, n_tokens] slots of the selected experts, -1 for the experts of other devices (expert split)
};

// the lookup of the selected experts of a layer in the MoE cache, see llama_moe_cache::build_lookup
struct llama_moe_cache_lookup {
    int32_t il = -1;
    std::vector<llama_moe_cache_part> parts; // empty if the layer does not read the cache, one per device with the expert split (the home device last)
};

// LongHaul: keeps recently and repeatedly used experts in a device buffer, the experts stay on disk (see TENSOR_LONGHAUL)
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback reads the missing experts from the model file
class llama_moe_cache {
public:
    // backends are all the backends of the context, each layer is cached on the GPU of the layer
    // with LLAMA_SPLIT_MODE_EXPERT the experts of every layer are split over the GPUs by the expert shares (see llama_model_params::tensor_split)
    // size is the total budget of the devices without a size in sizes ([n_devices] of the model, 0 = none, may be empty)
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts,
            size_t size, const std::vector<size_t> & sizes);
    ~llama_moe_cache();

    // the experts of w in the cache of device dev, nullptr if not cached there
    ggml_tensor * get_experts(const ggml_tensor * w, int32_t dev) const;

    // the backend of device dev of the cache
    ggml_backend_t backend(int32_t dev) const;

    // expert split: the backend of the layers, which adds the outputs of the devices; nullptr without the expert split
    ggml_backend_t home() const;

    // the largest ubatch whose selected experts always fit in the cache, INT64_MAX if any ubatch fits
    int64_t max_ubatch() const;

    // the lookup of the selected experts in the cache, built after the expert input and before the expert matmuls
    // the expert matmuls then read the cache with the slots in place of the expert ids
    // returns no slots if the layer is not cached
    llama_moe_cache_lookup build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const;

    // aborts if device dev cannot compute op, an expert matmul that reads the cached experts of w
    void check_op(const ggml_tensor * w, const ggml_tensor * op, int32_t dev) const;

    // call before each graph compute: with the expert split, moves experts between the devices toward the target shares
    void begin_graph();

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
