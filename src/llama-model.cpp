#include "llama-model.h"

#include "llama-arch.h"
#include "llama-ext.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include "llama.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static llama_model * llama_model_mapping(llm_arch arch, const llama_model_params & params) {
    switch (arch) {
        case LLM_ARCH_BAILINGMOE3:
            return new llama_model_bailingmoe3(params);
        case LLM_ARCH_CLIP:
            return new llama_model_clip(params);
        case LLM_ARCH_INKLING2:
            return new llama_model_inkling2(params);
        case LLM_ARCH_INKLING2_LOOP:
            return new llama_model_inkling2_loop(params);
        case LLM_ARCH_DFLASH:
            return new llama_model_dflash(params);
        case LLM_ARCH_QWEN35:
            return new llama_model_qwen35(params);
        case LLM_ARCH_QWEN35MOE:
            return new llama_model_qwen35moe(params);
        default:
            throw std::runtime_error(std::string("unsupported model architecture: '") + llm_arch_name(arch) + "'");
    }

}

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params) {
    llama_model * model = llama_model_mapping(arch, params);

    if (model != nullptr) {
        model->arch = arch;
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR && !llm_arch_supports_sm_tensor(arch)) {
            delete model;
            throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
        }
    }

    return model;
}

llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params) {
    llm_arch arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }

    return llama_model_create(arch, params);
}

// the split of each tensor over the devices of the meta device (LLAMA_SPLIT_MODE_TENSOR), adapted from upstream:
// - a weight that produces a split activation is split on axis 1 (rows), one that consumes it on axis 0, which gives a partial sum
// - the KV cache and the recurrent state are split by heads, like the attention that reads them
// - the rest is mirrored (a full copy on every device)
struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata) {
    const llama_meta_device_get_split_state_userdata * ud = (const llama_meta_device_get_split_state_userdata *) userdata;
    const llama_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;
    const bool is_qwen35 = ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE;

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    static const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    static const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    static const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_([a-z]+_)?(k|v)_l\\d*");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    static const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");

    static const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    static const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    static const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    static const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    static const std::regex pattern_r_cache         ("cache_r_l\\d*");
    static const std::regex pattern_s_cache         ("cache_s_l\\d*");
    static const std::regex pattern_spk_cache       ("cache_spk_l\\d*");
    static const std::regex pattern_spv_cache       ("cache_spv_l\\d*");
    static const std::regex pattern_spgb_cache      ("cache_spgb_l\\d*");
    static const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    static const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");

    static const std::regex pattern_ffn_up_weight     ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_gate_weight   ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_ffn_up_shexp_weight   ("blk\\.\\d*\\.ffn_up_shexp.weight");
    static const std::regex pattern_ffn_gate_shexp_weight ("blk\\.\\d*\\.ffn_gate_shexp.weight");
    static const std::regex pattern_ffn_down_shexp_weight ("blk\\.\\d*\\.ffn_down_shexp.weight");

    static const std::regex pattern_output_weight("output\\.weight");

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // count only the previous layers of the same type, to avoid aliasing with the layer pattern
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recr = hparams.is_recr(il);
            const bool il_is_swa  = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recr(il_prev) == il_is_recr && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            const bool is_attn = tensor_name.compare(prefix.size(), 5, "attn_") == 0;
            // attention heads must be on the same device as the KV of the source layer
            rotation = get_il_eff(is_attn ? hparams.kv_source(il) : il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.rfind("_l");
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer() % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    // with --tp-attn mirror every device holds and computes all of the attention, so the layer has one sync
    const bool attn_mirror = ud->model->tp_attn() == LLAMA_TP_ATTN_MIRROR;
    auto get_attn_config = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        return attn_mirror ? get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED) : get_tensor_config_impl(axis, suffix, suffix_fallback);
    };

    auto get_tensor_config = [&]() -> tensor_config {
        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_attn_config(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // linear attention (gated delta net)
        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache) ||
                std::regex_match(tensor_name, pattern_spk_cache) || std::regex_match(tensor_name, pattern_spv_cache) ||
                std::regex_match(tensor_name, pattern_spgb_cache)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_attn_config(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN, the routed experts split along the FFN dimension so that each device computes a slice of every selected expert
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        // shared experts split like the routed ones: the meta backend sums both partials in one all-reduce
        if (std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) || std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight");
        }

        // output, split by vocab rows; mirrored if the graph cuts the padded rows off the logits,
        // since the meta backend splits a view of a split tensor in the same ratio, which a cut does not keep
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            if (hparams.n_vocab_unpadded > 0 && hparams.n_vocab_unpadded != tensor->ne[1]) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }

        // everything else, including the router and the norms
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    // fused tensors are split per segment, see the comments on ggml_backend_meta_split_state
    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        if (is_qwen35) {
            // fused full attention layers with Q gate tensors that need n_embd doubled:
            if (!hparams.is_recr(il) && (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias))) {
                const int64_t n_embd      = hparams.n_head(il) * hparams.n_embd_head_k(il) * 2;
                const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
                GGML_ASSERT(hparams.n_embd_k_gqa(il) == n_embd_gqa);
                GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
                return {{n_embd, 1}, {n_embd_gqa, 2}};
            }

            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            // Qwen 3.5 broadcasts the K heads over the V heads as [k0_v0, k1_v1, k0_v2, k1_v3], so V is segmented on the scale of K
            if (n_k_heads > 0) {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, uint32_t(2 + head_ratio)}};
                }
                if ((hparams.is_recr(il) && std::regex_match(tensor_name, pattern_attn_gate_weight)) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return {{key_dim, uint32_t(head_ratio)}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return {{n_k_heads, uint32_t(head_ratio)}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    const int64_t n_conv_ch = 2*key_dim + value_dim;
                    if (tensor->ne[axis] != n_conv_ch * (hparams.ssm_d_conv - 1)) {
                        // replay: [channels, window + pending] per cell, every column is a q, k, v block
                        GGML_ASSERT(tensor->ne[axis] % n_conv_ch == 0);
                        return {{key_dim, uint32_t((2 + head_ratio) * (tensor->ne[axis] / n_conv_ch))}};
                    }
                    return {{key_dim * (hparams.ssm_d_conv - 1), uint32_t(2 + head_ratio)}};
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return {{n_k_heads * head_v_dim * head_v_dim, uint32_t(head_ratio)}};
                }
                // replay pending inputs, per token: k [S_k, H_k], v [S_v, H_v], then the gates and betas [H_v]
                if (std::regex_match(tensor_name, pattern_spk_cache) || std::regex_match(tensor_name, pattern_spv_cache)) {
                    GGML_ASSERT(tensor->ne[axis] % key_dim == 0);
                    return {{key_dim, uint32_t(tensor->ne[axis] / key_dim)}};
                }
                if (std::regex_match(tensor_name, pattern_spgb_cache)) {
                    GGML_ASSERT(tensor->ne[axis] % n_k_heads == 0);
                    return {{n_k_heads, uint32_t(tensor->ne[axis] / n_k_heads)}};
                }
            }
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd_q = hparams.n_head(il) * hparams.n_embd_head_k(il);
            const int64_t n_embd_k = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v = hparams.n_embd_v_gqa(il);
            GGML_ASSERT(tensor->ne[axis] == n_embd_q + n_embd_k + n_embd_v);
            if (n_embd_k == n_embd_v) {
                return {{n_embd_q, 1}, {n_embd_k, 2}};
            }
            return {{n_embd_q, 1}, {n_embd_k, 1}, {n_embd_v, 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff = tensor->ne[axis] / 2;
            GGML_ASSERT(tensor->ne[axis] == 2*n_ff);
            return {{n_ff, 2}};
        }
        return {{tensor->ne[axis], 1}};
    };

    // the smallest unit of a split, per segment: whole heads for attention, whole quant blocks (and 128 for speed) for the FFN
    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        if (hparams.is_recr(il)) {
            // linear attention
            const int64_t head_dim        = hparams.ssm_d_state;
            const int64_t blck_size_perf  = std::lcm(blck_size, (int64_t) 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                // replay: [channels, window + pending], each segment is a single conv column
                const int64_t n_conv_ch = 2*hparams.ssm_n_group*hparams.ssm_d_state + hparams.ssm_d_inner;
                const bool    replay    = tensor->ne[0] != n_conv_ch * (hparams.ssm_d_conv - 1);
                return std::vector<int64_t>(segments.size(), granularity_qkv * (replay ? 1 : hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
            if (std::regex_match(tensor_name, pattern_spk_cache) || std::regex_match(tensor_name, pattern_spv_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_spgb_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
        } else {
            // regular attention, each device holds whole GQA groups
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);

            // to handle head sizes like 80, only increase granularity while it does not cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf*(int64_t) ud->n_devices < n_embd_q) {
                blck_size_perf *= 2;
            }

            const int64_t granularity_q    = std::lcm((int64_t) n_embd_q, blck_size_perf);
            const int64_t granularity_head = granularity_q / hparams.n_embd_head_k(il); // for tensors with one value per head
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                GGML_ASSERT(segments.size() == 1);
                // Qwen 3.5 has a Q gate in the Q tensor, which doubles the granularity
                if (is_qwen35) {
                    return {std::lcm((int64_t) 2*n_embd_q, blck_size_perf)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_head * hparams.n_embd_head_v(il)};
            }
            if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
                GGML_ASSERT(segments.size() == 1);
                if (tensor->ne[1] == hparams.n_head(il)) {
                    return {granularity_head};
                }
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            // V heads can be smaller than K heads, align V to whole V heads at the same head index as Q and K
            const int64_t granularity_v  = (granularity_kv / hparams.n_embd_head_k(il)) * hparams.n_embd_head_v(il);
            if (std::regex_match(tensor_name, pattern_kv_weight) || std::regex_match(tensor_name, pattern_kv_bias) ||
                    std::regex_match(tensor_name, pattern_kv_cache)) {
                GGML_ASSERT(segments.size() == 1);
                const bool is_v = tensor_name.find("attn_v") != std::string::npos || tensor_name.find("v_l") != std::string::npos;
                return {is_v ? granularity_v : granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                if (is_qwen35) {
                    return {std::lcm((int64_t) 2*n_embd_q, blck_size_perf), granularity_kv};
                }
                if (segments.size() == 3) {
                    return {granularity_q, granularity_kv, granularity_v};
                }
                GGML_ASSERT(segments.size() == 2);
                return {granularity_q, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) || std::regex_match(tensor_name, pattern_ffn_down_weight) ||
                std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) || std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight) ||
                std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
            const int64_t blck_size_perf = std::lcm(blck_size, (int64_t) 128);
            return std::vector<int64_t>(segments.size(), blck_size_perf);
        }

        // everything else
        GGML_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        std::vector<float> tensor_split_scan;
        tensor_split_scan.reserve(ud->n_devices);
        for (size_t j = 0; j < ud->n_devices; j++) {
            tensor_split_scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + tc.rotation) % ud->n_devices]);
            if (j > 0) {
                tensor_split_scan[j] += tensor_split_scan[j - 1];
            }
        }
        const std::vector<std::pair<int64_t, uint32_t>> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        GGML_ASSERT(granularity.size() == segments.size());
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t  ne_s = segments[is].first;
            const uint32_t nr_s = segments[is].second;
            const int64_t  g_s  = granularity[is];
            int64_t low = 0;
            size_t j = 0;
            for (; j < ud->n_devices - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ?
                    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
                if (high % g_s != 0) {
                    high -= high % g_s;
                }
                split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = high - low;
                low = high;
            }
            split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            GGML_ASSERT(tc.tensor_axis_0 != tensor);
            const ggml_backend_meta_split_state source_split_state = llama_meta_device_get_split_state(tc.tensor_axis_0, userdata);
            GGML_ASSERT(source_split_state.axis >= 0 && source_split_state.axis < GGML_MAX_DIMS);
            for (size_t j = 0; j < ud->n_devices; j++) {
                for (size_t is = 0; is < source_split_state.n_segments; is++) {
                    split_state.ne[j] += source_split_state.ne[is*ud->n_devices + j] * source_split_state.nr[is];
                }
            }
        }
    }
    return split_state;
}

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_7_9B_A1_3B: return "7.9B.A1.3B";
        case LLM_TYPE_124B_A5_1B: return "124B.A5.1B";
        default:                  return "?B"; // the size is not detected for the Inkling models
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS: return "sqrtsoftplus";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

static const std::map<llama_rope_section_order, const char *> LLAMA_ROPE_SECTION_ORDERS = {
    { LLAMA_ROPE_SECTION_ORDER_TYXZ, "tyxz" },
    { LLAMA_ROPE_SECTION_ORDER_ZYXT, "zyxt" },
};

std::string llama_rope_section_order_name(llama_rope_section_order rope_section_order) {
    return LLAMA_ROPE_SECTION_ORDERS.at(rope_section_order);
}

static llama_rope_section_order llama_rope_section_order_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SECTION_ORDERS) {
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLAMA_ROPE_SECTION_ORDER_UNSPECIFIED;
}

// Maps GGUF activation names to the FFN op type used by the graph builders.
static const std::map<std::string, llm_ffn_op_type> LLM_FFN_OP_TYPES_FROM_STRING = {
    { "gelu",              LLM_FFN_GEGLU_ERF },
    { "gelu_python",       LLM_FFN_GEGLU_ERF },
    { "gelu_pytorch_tanh", LLM_FFN_GEGLU     },
    { "gelu_new",          LLM_FFN_GEGLU     },
    { "gelu_fast",         LLM_FFN_GEGLU     },
    { "gelu_accurate",     LLM_FFN_GEGLU     },
    { "gelu_python_tanh",  LLM_FFN_GEGLU     },
    { "geglu",             LLM_FFN_GEGLU     },
    { "silu",              LLM_FFN_SWIGLU    },
    { "swish",             LLM_FFN_SWIGLU    },
    { "swiglu",            LLM_FFN_SWIGLU    },
    { "relu",              LLM_FFN_RELU      },
    { "reglu",             LLM_FFN_REGLU     },
};

// transformers names, "gelu" is the exact (erf) variant
static const std::map<std::string, ggml_unary_op> LLM_CLS_ACT_TYPES_FROM_STRING = {
    { "gelu", GGML_UNARY_OP_GELU_ERF },
    { "silu", GGML_UNARY_OP_SILU     },
    { "tanh", GGML_UNARY_OP_TANH     },
};

llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback) {
    const auto it = LLM_FFN_OP_TYPES_FROM_STRING.find(name);
    if (it != LLM_FFN_OP_TYPES_FROM_STRING.end()) {
        return it->second;
    }
    return fallback;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<llama_device> & devices, bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        } else {
            throw std::runtime_error(format("device %s does not support split buffers", ggml_backend_dev_name(dev)));
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg) {
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    return buft_list;
}

struct llama_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;

    std::vector<float> tensor_split_owned;

    // LongHaul tensors and their data in the model files, see TENSOR_LONGHAUL
    struct longhaul_src {
        uint16_t file = UINT16_MAX; // UINT16_MAX if not known (no_alloc)
        size_t   offs = 0;
    };
    std::unordered_map<const ggml_tensor *, longhaul_src> longhaul;
    llama_files longhaul_files;
    std::vector<std::string> longhaul_paths; // [longhaul_files], empty if not known
};

bool llama_prec_policy::apply(ggml_tensor * res, const ggml_tensor * w) const {
    if (!res) {
        return false;
    }

    const ggml_tensor * key = w ? w : res->src[0];
    if (!key) {
        return false;
    }

    bool applied = false;

    const auto it_acc = prec_acc.find(key);
    if (it_acc != prec_acc.end()) {
        applied |= ggml_prec_set_acc(res, it_acc->second);
    }

    const auto it_src1 = prec_src1.find(key);
    if (it_src1 != prec_src1.end()) {
        applied |= ggml_prec_set_src(res, it_src1->second, 1);
    }

    return applied;
}

void llama_prec_policy::load(llama_model_loader & ml, const llama_model & model) {
    std::vector<std::string> tensor_names;
    if (!ml.get_arr(LLM_KV_GENERAL_TENSOR_EXTRA_NAME, tensor_names, false)) {
        return;
    }

    const gguf_context * ctx = ml.metadata;
    const std::string key = ml.llm_kv(LLM_KV_GENERAL_TENSOR_EXTRA_PREC_A4);
    const int kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, kid) != GGUF_TYPE_BOOL) {
        throw std::runtime_error(format("%s must be a bool array", key.c_str()));
    }

    const size_t n_values = gguf_get_arr_n(ctx, kid);
    if (n_values != tensor_names.size()) {
        throw std::runtime_error(format(
            "%s tensor/value length mismatch (%zu vs %zu)",
            key.c_str(), tensor_names.size(), n_values));
    }

    // tensors that can not use 4-bit activations, keep src1 at higher precision
    const int8_t * values = (const int8_t *) gguf_get_arr_data(ctx, kid);
    std::unordered_set<std::string> want;
    for (size_t i = 0; i < n_values; ++i) {
        if (values[i] == 0) {
            want.insert(tensor_names[i]);
        }
    }

    // resolve names to tensor pointers
    for (const auto & [name, w] : model.tensors_by_name) {
        if (want.count(name)) {
            prec_src1.emplace(w, GGML_PREC_Q8);
        }
    }
}

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    if (params.tensor_split != nullptr) {
        // llama_model_params stores tensor_split as a borrowed pointer, but the model
        // may need it later for tensor-parallel KV-cache split metadata.
        pimpl->tensor_split_owned.assign(params.tensor_split, params.tensor_split + llama_max_devices());
        this->params.tensor_split = pimpl->tensor_split_owned.data();
    }
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}

void llama_model_base::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model_base::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // prism.hadamard folded weights need the activation transform, which the Inkling 2 graphs do not apply
    if (gguf_find_key(ml.metadata, "prism.hadamard.version") >= 0) {
        throw std::runtime_error("prism.hadamard folded weights are not supported");
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out_impl, false);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,     false);
    ml.get_key(LLM_KV_POOLING_TYPE,            hparams.pooling_type,    false);
    ml.get_key(LLM_KV_CLASSIFIER_POOLING_TYPE, hparams.pooling_type_cls, false);
    ml.get_key(LLM_KV_CLASSIFIER_FREE_GUIDANCE, hparams.classifier_free_guidance, false);
    std::string act_cls;
    if (ml.get_key(LLM_KV_CLASSIFIER_ACTIVATION, act_cls, false)) {
        const auto it = LLM_CLS_ACT_TYPES_FROM_STRING.find(act_cls);
        GGML_ASSERT(it != LLM_CLS_ACT_TYPES_FROM_STRING.end() && "unsupported classifier activation");
        hparams.act_cls = it->second;
    }
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer_all);
    GGML_ASSERT(hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,    hparams.n_layer_nextn,   false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all);
    // Qwen 3.5 and bailingmoe3 load their MTP block and skip it, the other archs have none
    if (hparams.n_layer_nextn > 0 && arch != LLM_ARCH_QWEN35 && arch != LLM_ARCH_QWEN35MOE && arch != LLM_ARCH_BAILINGMOE3) {
        throw std::runtime_error("NextN/MTP layers are not supported");
    }
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    std::fill(hparams.n_expert_used_arr.begin(), hparams.n_expert_used_arr.end(), 0);
    ml.get_key_or_arr(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    // models may route a different number of experts per layer, so validate the maximum
    uint32_t n_expert_used_max = hparams.n_expert_used_max();

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(n_expert_used_max <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(n_expert_used_max > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(n_expert_used_max == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
    }

    std::fill(hparams.n_head_arr.begin(),       hparams.n_head_arr.end(),       0);
    std::fill(hparams.n_head_kv_arr.begin(),    hparams.n_head_kv_arr.end(),    0);
    std::fill(hparams.n_ff_arr.begin(),         hparams.n_ff_arr.end(),         0);
    std::fill(hparams.n_ff_exp_arr.begin(),     hparams.n_ff_exp_arr.end(),     0);

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.swiglu_clamp_exp.begin(),   hparams.swiglu_clamp_exp.end(),   0.0f);
    std::fill(hparams.swiglu_clamp_shexp.begin(), hparams.swiglu_clamp_shexp.end(), 0.0f);
    std::fill(hparams.rope_pattern.begin(),  hparams.rope_pattern.end(), 1);
    std::fill(hparams.is_swa_impl.begin(),   hparams.is_swa_impl.end(), 0);
    std::fill(hparams.is_recr_impl.begin(),  hparams.is_recr_impl.end(),  llm_arch_is_recurrent(ml.get_arch()) ? 1 : 0);
    std::iota(hparams.kv_source_impl.begin(), hparams.kv_source_impl.end(), 0);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer_all, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer_all, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    std::string rope_section_order("tyxz");
    ml.get_key(LLM_KV_ROPE_SECTION_ORDER, rope_section_order, false);
    hparams.rope_section_order = llama_rope_section_order_from_string(rope_section_order);
    if (hparams.rope_section_order == LLAMA_ROPE_SECTION_ORDER_UNSPECIFIED) {
        throw std::runtime_error("unknown rope section order: " + rope_section_order);
    }

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot_full = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);

    } else {
        hparams.n_rot_full = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // for decision models
    ml.get_arr(LLM_KV_DECISION_LABELS, decision_labels, false);

    // per-arch hparams
    load_arch_hparams(ml);

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    pimpl->ftype = ml.ftype;

    hparams.rope_type = llama_model_rope_type(this);

    if (hparams.rope_section_order != LLAMA_ROPE_SECTION_ORDER_TYXZ && hparams.n_pos_per_embd() != 4) {
        throw std::runtime_error("rope section order " + llama_rope_section_order_name(hparams.rope_section_order) + " requires M-RoPE");
    }
}

void llama_model_base::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

void llama_model_base::load_prec_policy(llama_model_loader & ml) {
    if (hparams.expert_router_f32) {
        for (const auto & layer : layers) {
            prec_policy.set_acc(layer.ffn_gate_inp, GGML_PREC_F32);
        }
    }

    if (getenv("LLAMA_MOE_AMAX_SKIP") != nullptr) {
        LLAMA_LOG_WARN("LLAMA_MOE_AMAX_SKIP is set - the MoE amax rescaling is disabled\n");

        if (llm_arch_mmid_down_bf16(arch)) {
            for (const auto & layer : layers) {
                if (layer.ffn_down_exps) {
                    prec_policy.set_src1(layer.ffn_down_exps, GGML_PREC_BF16);
                }
            }
        }
    }

    // per-tensor activation precision policy
    prec_policy.load(ml, *this);
}

bool llama_model_base::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const bool use_mlock      = params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    const auto & tensor_split = params.tensor_split;

    const int n_layer_all = hparams.n_layer_all;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    this->ml = &ml; // to be used by create_tensor() and load_arch_tensors()

    if (ml.use_mmap && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
            if (!props.caps.mmap_support) {
                ml.use_mmap = false;
                break;
            }
        }
    }

    // resolve AUTO on systems without mmap support (e.g. iGPUs): fall back to OFF unless mmap is in use; see #28160, #31943
    if (ml.lazy.mode == LLAMA_LAZY_MODE_AUTO && !ml.use_mmap) {
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
            if (!props.caps.mmap_support) {
                ml.lazy.mode = LLAMA_LAZY_MODE_OFF;
                break;
            }
        }
    }

    const char * load_mode_name = params.load_mode == LLAMA_LOAD_MODE_AUTO
        ? llama_load_mode_name(ml.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE)
        : llama_load_mode_name(params.load_mode);

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (load_mode = %s)\n",
        __func__, load_mode_name);

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] = split_sum != 0.0f ? splits[i] / split_sum : float(i + 1) / n_devices();
    }

    // the splits are over the layers that hold weights, a cache slot that aliases another layer (looped
    // architectures) goes to the device of that layer so that its KV stays next to the weights it uses
    const int n_layer_off = n_layer_offload();

    const int i_gpu_start = std::max(n_layer_off + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_off + 1);

    // the expert split: every layer and its KV on main_gpu, the output on the last device, the experts in the LongHaul cache of each device
    const bool split_expert = split_mode == LLAMA_SPLIT_MODE_EXPERT;
    if (split_expert) {
        if (devices.empty()) {
            throw std::runtime_error("split_mode = expert needs at least one GPU");
        }
        if (n_gpu_layers < n_layer_off + 1) {
            throw std::runtime_error("split_mode = expert needs all layers on the GPU (n_gpu_layers)");
        }
        if (params.tensor_buft_overrides != nullptr && params.tensor_buft_overrides[0].pattern != nullptr) {
            throw std::runtime_error("split_mode = expert does not support tensor buffer type overrides");
        }
    }

    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (split_expert) {
            auto * dev = il < n_layer_all ? devices.at(params.main_gpu).dev : devices.back().dev;
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
            return {dev, &pimpl->gpu_buft_list.at(dev)};
        }
        const int ip = il < n_layer_all ? layer_phys(il) : n_layer_off;
        if (ip < i_gpu_start || (ip - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(ip - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);

    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);

        const int64_t n_expert = hparams.n_expert;

        if (n_expert > 0 && hparams.n_expert_used_max() == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        layers.resize(n_layer_all);

        // call the per-model loading function
        load_arch_tensors(ml);

        // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
        // this avoids having to add scale loading to every architecture
        for (int i = 0; i < n_layer_all; ++i) {
            auto & layer = layers[i];

            if (hparams.has_kv(i)) {
                layer.k_cache_s = create_tensor(tn(LLM_TENSOR_ATTN_K, "k_scale", i), {1}, TENSOR_NOT_REQUIRED);
                layer.v_cache_s = create_tensor(tn(LLM_TENSOR_ATTN_V, "v_scale", i), {1}, TENSOR_NOT_REQUIRED);

                if ((layer.k_cache_s && layer.k_cache_s->type != GGML_TYPE_F32) ||
                    (layer.v_cache_s && layer.v_cache_s->type != GGML_TYPE_F32)) {
                    throw std::runtime_error(format("KV cache scales for layer %d must be F32", i));
                }
            }

            // attention weight scales (per-tensor, shape {1})
            if (!layer.wq_s && layer.wq) {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_s && layer.wk) {
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_s && layer.wv) {
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_s && layer.wo) {
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_s && layer.wqkv) {
                layer.wqkv_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_s && layer.wqkv_gate) {
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // dense FFN weight scales (per-tensor, shape {1})
            if (!layer.ffn_gate_s && layer.ffn_gate) {
                layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_s && layer.ffn_down) {
                layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_s && layer.ffn_up) {
                layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // MoE expert weight scales (per-expert, shape {n_expert})
            if (!layer.ffn_gate_exps_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
                if (layer.ffn_up_exps_s && layer.ffn_gate_up_exps) {
                    throw std::runtime_error(format("ffn_up_exps.scale and ffn_gate_up_exps.weight both exist for layer %d", i));
                }
            }
            if (!layer.ffn_up_exps_s && layer.ffn_gate_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }

            // gated delta net and shared expert scales (qwen35)
            struct w_scale_slots { ggml_tensor * w; llm_tensor t; ggml_tensor ** s; ggml_tensor ** in_s; };
            for (const w_scale_slots & ws : {
                    w_scale_slots { layer.ssm_out,        LLM_TENSOR_SSM_OUT,        &layer.ssm_out_s,        &layer.ssm_out_in_s        },
                    w_scale_slots { layer.ssm_alpha,      LLM_TENSOR_SSM_ALPHA,      &layer.ssm_alpha_s,      &layer.ssm_alpha_in_s      },
                    w_scale_slots { layer.ssm_beta,       LLM_TENSOR_SSM_BETA,       &layer.ssm_beta_s,       &layer.ssm_beta_in_s       },
                    w_scale_slots { layer.ffn_gate_shexp, LLM_TENSOR_FFN_GATE_SHEXP, &layer.ffn_gate_shexp_s, &layer.ffn_gate_shexp_in_s },
                    w_scale_slots { layer.ffn_up_shexp,   LLM_TENSOR_FFN_UP_SHEXP,   &layer.ffn_up_shexp_s,   &layer.ffn_up_shexp_in_s   },
                    w_scale_slots { layer.ffn_down_shexp, LLM_TENSOR_FFN_DOWN_SHEXP, &layer.ffn_down_shexp_s, &layer.ffn_down_shexp_in_s }}) {
                if (ws.w && !*ws.s) {
                    *ws.s = create_tensor(tn(ws.t, "scale", i), {1}, TENSOR_NOT_REQUIRED);
                }
                if (ws.w && !*ws.in_s) {
                    *ws.in_s = create_tensor(tn(ws.t, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
                }
            }

            // input scales
            if (!layer.wq_in_s && layer.wq) {
                layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_in_s && layer.wk) {
                layer.wk_in_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_in_s && layer.wv) {
                layer.wv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_in_s && layer.wo) {
                layer.wo_in_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_in_s && layer.wqkv) {
                layer.wqkv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_in_s && layer.wqkv_gate) {
                layer.wqkv_gate_in_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_in_s && layer.ffn_gate) {
                layer.ffn_gate_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_in_s && layer.ffn_down) {
                layer.ffn_down_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_in_s && layer.ffn_up) {
                layer.ffn_up_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_exps_in_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_in_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_in_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
        }
        // output scales
        if (output) {
            // weight scale
            if (!output_s && (ml.get_weight(tn(LLM_TENSOR_OUTPUT, "scale").str().c_str()) ||
                              gguf_find_tensor(ml.metadata, tn(LLM_TENSOR_OUTPUT, "scale").str().c_str()) >= 0)) {
                output_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"), {1}, TENSOR_NOT_REQUIRED);
            }
            // input scale
            if (!output_in_s && (ml.get_weight(tn(LLM_TENSOR_OUTPUT, "input_scale").str().c_str()) ||
                                 gguf_find_tensor(ml.metadata, tn(LLM_TENSOR_OUTPUT, "input_scale").str().c_str()) >= 0)) {
                output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED);
            }
        }
        post_arch_tensors(ml);
    }

    ml.done_getting_tensors();

    if (output && tok_embd &&
            strcmp(output->name, tok_embd->name) == 0 &&
            (output_s || output_in_s)) {
        throw std::runtime_error("output scales require a separate output weight");
    }
    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // LongHaul tensors stay on disk, record where their data is
    {
        size_t longhaul_bytes = 0;
        for (const auto & [name, cur] : tensors_by_name) {
            if (!ml.lazy.is_longhaul(cur)) {
                continue;
            }
            impl::longhaul_src src;
            if (const auto * w = ml.get_weight(name.c_str())) {
                src.file = w->idx;
                src.offs = w->offs;
            }
            pimpl->longhaul.emplace(cur, src);
            longhaul_bytes += ggml_nbytes(cur);
        }
        if (!pimpl->longhaul.empty()) {
            if (has_tensor_overrides()) {
                LLAMA_LOG_WARN("%s: LongHaul tensors ignore the tensor buffer type overrides\n", __func__);
            }
            LLAMA_LOG_INFO("%s: LongHaul: %zu expert tensors, %.2f GiB kept on disk\n", __func__,
                pimpl->longhaul.size(), longhaul_bytes/1024.0/1024.0/1024.0);
        }
    }
    if (split_expert && pimpl->longhaul.empty()) {
        throw std::runtime_error("split_mode = expert needs a model with its experts on disk (LongHaul)");
    }

    // map each weight to its "<name>.scale" tensor: shape {1} for a 2D weight, {n_expert} for a 3D weight
    {
        auto ends_with = [](const std::string & str, const std::string & suffix) {
            return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        std::unordered_map<std::string, ggml_tensor *> weights;
        for (const auto & [name, cur] : tensors_by_name) {
            if (ends_with(name, ".weight")) {
                weights.emplace(name.substr(0, name.size() - 7), cur);
            }
        }
        for (const auto & [name, s] : tensors_by_name) {
            if (!ends_with(name, ".scale")) {
                continue;
            }
            const auto it = weights.find(name.substr(0, name.size() - 6));
            if (it == weights.end()) {
                continue;
            }
            ggml_tensor * w = it->second;
            const bool ok = w->ne[3] == 1 && ggml_nelements(s) == w->ne[2] && s->ne[0] == w->ne[2];
            if (ok) {
                w_scales.emplace(w, s);
            } else {
                LLAMA_LOG_DEBUG("%s: '%s' is not a weight scale for '%s', skipping\n", __func__, name.c_str(), w->name);
            }
        }
    }

    load_prec_policy(ml);

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    // note: the buffers are owned by pimpl->ctxs_bufs, this is a raw-pointer view used to load the data
    std::vector<std::pair<ggml_context *, std::vector<ggml_backend_buffer_t>>> ctx_bufs;
    ctx_bufs.reserve(ml.ctx_map.size());

    // reserve hint: each context creates one buffer per file, or more when its tensors are spread
    //     across disjoint ranges of a file (see get_mapping_ranges)
    pimpl->ctxs_bufs.reserve(ml.ctx_map.size() * ml.files.size());

    for (auto & [ctx_key, ctx_ptr] : ml.ctx_map) {
        ggml_backend_buffer_type_t buft = ctx_key.buft;
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;

        // a lazy context is mapped whatever the load mode, but the memory-fit pass maps nothing
        const bool is_lazy_mapped = ctx_key.lazy && !ml.no_alloc;

        if ((ml.use_mmap || is_lazy_mapped) && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            const size_t max_size = ggml_get_max_tensor_size(ctx);

            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // map the mmap regions that contain the tensors of this context
                // this is important for metal with apple silicon: a backend buffer pins everything it
                //     wraps, so the regions owned by the other contexts (lazy tensors read on demand
                //     from disk, MoE tensors kept on the CPU, etc) are left out and stay out of RAM
                // this also allows using partial offloading when the model size exceeds the metal buffer
                //     size, but not the RAM size
                void * addr = ml.mappings.at(idx)->addr();

                for (const auto & range : ml.get_mapping_ranges(idx, ctx)) {
                    ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + range.first, range.last - range.first, max_size);
                    if (buf == nullptr) {
                        throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                    }
                    bufs.emplace_back(buf);
                }
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            // lock the buffers that hold the tensors: a multi_buffer has no base, and its parts are pageable memory when pinning fails
            if (use_mlock && !ml.no_alloc && ggml_backend_buffer_is_host(buf)) {
                std::vector<ggml_backend_buffer_t> lock_bufs;
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    if (t->buffer != nullptr && std::find(lock_bufs.begin(), lock_bufs.end(), t->buffer) == lock_bufs.end()) {
                        lock_bufs.push_back(t->buffer);
                    }
                }
                for (ggml_backend_buffer_t lock_buf : lock_bufs) {
                    pimpl->mlock_bufs.emplace_back(new llama_mlock);
                    auto & mlock_buf = pimpl->mlock_bufs.back();
                    mlock_buf->init   (ggml_backend_buffer_get_base(lock_buf));
                    mlock_buf->grow_to(ggml_backend_buffer_get_size(lock_buf));
                }
            }
            bufs.emplace_back(buf);
        }

        std::vector<ggml_backend_buffer_t> buf_ptrs;
        buf_ptrs.reserve(bufs.size());

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            buf_ptrs.emplace_back(buf.get());
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_bufs.emplace_back(ctx, std::move(buf_ptrs));
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, n_layer_off);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = n_layer_off + 1;
        const int max_offloadable_layers       = n_layer_off + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // without mmap, load non-host buffers first: their tensors go through a staging buffer, which is cheapest while the fewest weights are resident
    if (!ml.use_mmap) {
        std::stable_partition(ctx_bufs.begin(), ctx_bufs.end(), [](const auto & x) {
            const auto & bufs = x.second;
            return !bufs.empty() && !ggml_backend_buffer_is_host(bufs[0]);
        });
    }

    // load tensor data
    for (auto & [ctx, bufs] : ctx_bufs) {
        if (!ml.load_all_data(ctx, bufs, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    // the LongHaul expert cache reads the experts from the files, bypassing the OS file cache
    if (!pimpl->longhaul.empty()) {
        for (auto & file : ml.files) {
            if (file->has_direct_io()) {
                throw std::runtime_error("LongHaul does not support direct I/O, use the default load mode");
            }
            file->set_no_cache();
        }
        pimpl->longhaul_files = std::move(ml.files);
        pimpl->longhaul_paths = std::move(ml.file_paths);
        pimpl->longhaul_paths.resize(pimpl->longhaul_files.size());
    }


    return true;
}

ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    // tensor parallelism splits the experts over the devices like any weight, so they are loaded, not kept on disk
    if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        flags &= ~TENSOR_LONGHAUL;
    }
    const buft_list_t * buft_list_layer = nullptr;
    if (tn.bid != -1) {
        // blocks that are not model layers (e.g. the blocks of a head) go with the output
        buft_list_layer = (size_t) tn.bid < pimpl->dev_layer.size() ? pimpl->dev_layer.at(tn.bid).buft_list : pimpl->dev_output.buft_list;
    }
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

llama_ftype llama_model::ftype() const {
    return pimpl->ftype;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

ggml_tensor * llama_model::get_w_scale(const ggml_tensor * w) const {
    const auto it = w_scales.find(w);
    return it == w_scales.end() ? nullptr : it->second;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

const float * llama_model::tensor_split() const {
    return params.tensor_split;
}

int llama_model::n_layer_offload() const {
    int n = 0;
    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        n = std::max(n, layer_phys(il) + 1);
    }
    return n;
}

uint32_t llama_model::n_gpu_layers() const {
    // note: plus 1 for the "output" layer
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;
}

llama_tp_attn llama_model::tp_attn() const {
    // the policy in effect: some archs cannot split attention
    return llm_arch_sm_tensor_splits_attn(arch) ? params.tp_attn : LLAMA_TP_ATTN_MIRROR;
}

llama_split_mode llama_model::split_mode() const {
    return params.split_mode;
}

bool llama_model::no_alloc() const {
    return params.no_alloc;
}

bool llama_model::has_longhaul() const {
    return !pimpl->longhaul.empty();
}

bool llama_model::is_longhaul(const ggml_tensor * t) const {
    return pimpl->longhaul.count(t) > 0;
}

void llama_model::longhaul_read(const ggml_tensor * t, void * dst, size_t offs, size_t size) const {
    const auto it = pimpl->longhaul.find(t);
    if (it == pimpl->longhaul.end() || it->second.file >= pimpl->longhaul_files.size()) {
        throw std::runtime_error(format("tensor %s has no LongHaul data", t->name));
    }
    if (offs > ggml_nbytes(t) || size > ggml_nbytes(t) - offs) {
        throw std::runtime_error(format("LongHaul read of %zu bytes at %zu is out of the bounds of %s", size, offs, t->name));
    }
    pimpl->longhaul_files[it->second.file]->read_raw_at(dst, size, it->second.offs + offs);
}

bool llama_model::longhaul_src(const ggml_tensor * t, size_t & file, size_t & offs) const {
    const auto it = pimpl->longhaul.find(t);
    if (it == pimpl->longhaul.end() || it->second.file >= pimpl->longhaul_files.size()) {
        return false;
    }
    file = it->second.file;
    offs = it->second.offs;
    return true;
}

const std::string & llama_model::longhaul_path(size_t file) const {
    return pimpl->longhaul_paths.at(file);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (no_alloc()) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<int32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<int32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch                  = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only            = %d\n",     __func__, hparams.vocab_only);
    LLAMA_LOG_INFO("%s: no_alloc              = %d\n",     __func__, no_alloc());

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train           = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd_inp            = %u\n",     __func__, hparams.n_embd_inp());
        LLAMA_LOG_INFO("%s: n_embd                = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_embd_out            = %u\n",     __func__, hparams.n_embd_out());
        LLAMA_LOG_INFO("%s: n_layer               = %u\n",     __func__, hparams.n_layer());
        LLAMA_LOG_INFO("%s: n_layer_all           = %u\n",     __func__, hparams.n_layer_all);
        LLAMA_LOG_INFO("%s: n_head                = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_rot                 = %u\n",     __func__, hparams.n_rot_full);
        LLAMA_LOG_INFO("%s: n_swa                 = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any            = %u\n",     __func__, hparams.is_swa_any());
        LLAMA_LOG_INFO("%s: n_embd_head_k         = %u\n",     __func__, hparams.n_embd_head_k_full);
        LLAMA_LOG_INFO("%s: n_embd_head_v         = %u\n",     __func__, hparams.n_embd_head_v_full);
        LLAMA_LOG_INFO("%s: n_gqa                 = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_logit_scale         = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: f_attn_scale          = %.1e\n",   __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: f_attn_value_scale    = %.4f\n",   __func__, hparams.f_attn_value_scale);
        LLAMA_LOG_INFO("%s: n_ff                  = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_expert              = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used         = %u\n",     __func__, hparams.n_expert_used());
        LLAMA_LOG_INFO("%s: n_expert_groups       = %d\n",     __func__, hparams.n_expert_groups);
        LLAMA_LOG_INFO("%s: n_group_used          = %d\n",     __func__, hparams.n_group_used);
        if (arch == LLM_ARCH_BAILINGMOE3) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
            LLAMA_LOG_INFO("%s: n_layer_nextn         = %d\n",     __func__, hparams.n_layer_nextn);
        }
        LLAMA_LOG_INFO("%s: causal attn           = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type          = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type             = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling          = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train       = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train      = %g\n",     __func__, hparams.rope_freq_scale_train);
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            LLAMA_LOG_INFO("%s: freq_base_swa         = %.1f\n",   __func__, hparams.rope_freq_base_train_swa);
            LLAMA_LOG_INFO("%s: freq_scale_swa        = %g\n",     __func__, hparams.rope_freq_scale_train_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_k_swa     = %u\n",     __func__, hparams.n_embd_head_k_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_v_swa     = %u\n",     __func__, hparams.n_embd_head_v_swa);
            LLAMA_LOG_INFO("%s: n_rot_swa             = %u\n",     __func__, hparams.n_rot_swa);
        }
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_finetuned        = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        if (hparams.rope_section_order != LLAMA_ROPE_SECTION_ORDER_TYXZ) {
            LLAMA_LOG_INFO("%s: rope section order    = %s\n", __func__, llama_rope_section_order_name(hparams.rope_section_order).c_str());
        }
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out             = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (const auto & label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, i++, label.c_str());
            }
        }

        if (arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE) {
            LLAMA_LOG_INFO("%s: ssm_d_conv            = %u\n",     __func__, hparams.ssm_d_conv);
            LLAMA_LOG_INFO("%s: ssm_d_inner           = %u\n",     __func__, hparams.ssm_d_inner);
            LLAMA_LOG_INFO("%s: ssm_d_state           = %u\n",     __func__, hparams.ssm_d_state);
            LLAMA_LOG_INFO("%s: ssm_dt_rank           = %u\n",     __func__, hparams.ssm_dt_rank);
            LLAMA_LOG_INFO("%s: ssm_n_group           = %u\n",     __func__, hparams.ssm_n_group);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %u\n",     __func__, hparams.n_ff_shexp);
            LLAMA_LOG_INFO("%s: n_layer_nextn         = %u (not run)\n", __func__, hparams.n_layer_nextn);
        }

        LLAMA_LOG_INFO("%s: model type            = %s\n",     __func__, type_name().c_str());
        if (pimpl->n_elements >= 1e12) {
            LLAMA_LOG_INFO("%s: model params          = %.2f T\n", __func__, pimpl->n_elements*1e-12);
        } else if (pimpl->n_elements >= 1e9) {
            LLAMA_LOG_INFO("%s: model params          = %.2f B\n", __func__, pimpl->n_elements*1e-9);
        } else if (pimpl->n_elements >= 1e6) {
            LLAMA_LOG_INFO("%s: model params          = %.2f M\n", __func__, pimpl->n_elements*1e-6);
        } else {
            LLAMA_LOG_INFO("%s: model params          = %.2f K\n", __func__, pimpl->n_elements*1e-3);
        }

        // general kv
        LLAMA_LOG_INFO("%s: general.name          = %s\n",    __func__, name.c_str());

    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

void llama_model::log_tensor_split() const {
    const size_t n_devices = get_split_state_ud.n_devices;
    if (n_devices == 0) {
        return;
    }
    if (params.tp_attn != tp_attn()) {
        LLAMA_LOG_WARN("%s: %s cannot split attention, every device holds and computes all of it (--tp-attn mirror)\n",
            __func__, llm_arch_name(arch));
    }

    // sums the slices of one tensor class over all layers, as the rounding moves between the devices per layer
    // n_first: the size of the split dimension in the first layer, step: the FFN rounding of that tensor type
    auto get_share = [&](const std::vector<std::string> & names, const bool per_layer, int64_t & n_first, int64_t & step) -> std::string {
        std::vector<int64_t> ne(n_devices, 0);
        bool found    = false;
        bool mirrored = false;
        n_first = 0;
        step    = 0;
        const uint32_t n_iter = per_layer ? hparams.n_layer() : 1;
        for (uint32_t il = 0; il < n_iter; il++) {
            for (const std::string & name : names) {
                const std::string full = per_layer ? format("blk.%u.%s", il, name.c_str()) : name;
                const ggml_tensor * t = get_tensor(full.c_str());
                if (t == nullptr) {
                    continue;
                }
                const ggml_backend_meta_split_state ss = llama_meta_device_get_split_state(t, const_cast<void *>(static_cast<const void *>(&get_split_state_ud)));
                found = true;
                if (ss.axis < 0 || ss.axis >= GGML_MAX_DIMS) {
                    mirrored = true;
                    break;
                }
                if (n_first == 0) {
                    n_first = t->ne[ss.axis];
                    step    = std::lcm((int64_t) ggml_blck_size(t->type), (int64_t) 128);
                }
                for (size_t j = 0; j < n_devices; j++) {
                    for (size_t is = 0; is < ss.n_segments; is++) {
                        ne[j] += ss.ne[is*n_devices + j] * ss.nr[is];
                    }
                }
                break;
            }
        }
        if (!found) {
            return "";
        }
        if (mirrored) {
            return "mirrored";
        }
        int64_t total = 0;
        for (const int64_t x : ne) {
            total += x;
        }
        std::string ret;
        for (size_t j = 0; j < n_devices; j++) {
            ret += format("%s%.0f", j == 0 ? "" : "/", total == 0 ? 0.0 : 100.0*ne[j]/total);
        }
        return ret;
    };

    std::string msg;
    auto add = [&](const char * label, const std::string & share, const std::string & extra = "") {
        if (!share.empty()) {
            msg += format("%s%s %s%s", msg.empty() ? "" : ", ", label, share.c_str(), extra.c_str());
        }
    };
    int64_t n = 0;
    int64_t step = 0;
    const std::string experts = get_share({ "ffn_down_exps.weight", "ffn_down.weight" }, true, n, step);
    add("experts", experts, experts == "mirrored" ? "" : format(" (n_ff %lld, step %lld)", (long long) n, (long long) step));
    const std::string shexp = get_share({ "ffn_down_shexp.weight" }, true, n, step);
    add("shared experts", shexp, shexp == "mirrored" ? "" : format(" (n_ff %lld, step %lld)", (long long) n, (long long) step));
    add("attention", get_share({ "attn_output.weight", "ssm_out.weight" }, true, n, step));
    add("output", get_share({ "output.weight" }, false, n, step));
    LLAMA_LOG_INFO("%s: tensor split: %s\n", __func__, msg.c_str());
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

llama_rope_params llama_model::get_rope_params(const llama_cparams & cparams, int il) const {
    llama_rope_params res;

    res.n_rot       = hparams.has_rope(il) ? hparams.n_rot(il) : 0;
    // MLA: the cached K row is [kv_lora, k_pe], only k_pe is rotated
    res.n_offs      = hparams.n_lora_kv > 0 ? hparams.n_embd_head_k(il) - hparams.n_rot(il) : 0;
    res.type        = hparams.rope_type;
    res.n_ctx_orig  = cparams.n_ctx_orig_yarn;
    res.freq_base   = get_rope_freq_base (cparams, il);
    res.freq_scale  = get_rope_freq_scale(cparams, il);
    res.ext_factor  = cparams.yarn_ext_factor;
    res.attn_factor = cparams.yarn_attn_factor;
    res.beta_fast   = cparams.yarn_beta_fast;
    res.beta_slow   = cparams.yarn_beta_slow;
    res.factors     = get_rope_factors(cparams, il);

    return res;
}

llama_rope_params llama_model::get_rope_params_indexer(const llama_cparams & cparams, int il) const {
    llama_rope_params res = get_rope_params(cparams, il);

    res.n_offs     = 0;
    res.type       = LLAMA_ROPE_TYPE_NEOX;
    res.freq_base  = cparams.rope_freq_base;
    res.freq_scale = cparams.rope_freq_scale;
    res.factors    = nullptr;

    return res;
}

float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & /*cparams*/, int il) const {
    return layers[il].rope_freqs;
}

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    // Qwen 3.5: each layer has either attention or gated delta net state, the MTP block has neither
    // bailingmoe3: KDA layers keep recurrent state, the other layers an MLA (K-only) cache
    if (arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_BAILINGMOE3) {
        return new llama_memory_hybrid(
            /* model             */ *this,
            /* no_alloc          */ params.no_alloc,
            /* attn_type_k       */ params.type_k,
            /* attn_type_v       */ params.type_v,
            /* attn_v_trans      */ !cparams.flash_attn,
            /* attn_kv_size      */ cparams.n_ctx_seq,
            /* attn_n_swa        */ hparams.n_swa,
            /* attn_swa_type     */ hparams.swa_type,
            /* recurrent_type_k  */ GGML_TYPE_F32,
            /* recurrent_type_v  */ GGML_TYPE_F32,
            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
            /* n_seq_max         */ cparams.n_seq_max,
            /* n_rs_seq          */ cparams.n_rs_seq,
            /* rs_replay         */ cparams.rs_replay,
            /* offload           */ cparams.offload_kqv,
            /* unified           */ cparams.kv_unified,
            /* filter_attn       */ [&](uint32_t il) { return il < hparams.n_layer() && !hparams.is_recr(il); },
            /* filter_recr       */ [&](uint32_t il) { return il < hparams.n_layer() &&  hparams.is_recr(il); });
    }
    // Inkling 2: every layer has attention and short convolution state, some layers use a sliding window
    if (llm_arch_is_hybrid(arch)) {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

        return new llama_memory_hybrid_iswa(
            /* model             */ *this,
            /* no_alloc          */ params.no_alloc,
            /* attn_type_k       */ params.type_k,
            /* attn_type_v       */ params.type_v,
            /* attn_v_trans      */ !cparams.flash_attn,
            /* attn_swa_full     */ params.swa_full,
            /* attn_kv_size      */ cparams.n_ctx_seq,
            /* attn_n_ubatch     */ cparams.n_ubatch,
            /* recurrent_type_r  */ GGML_TYPE_F32,
            /* recurrent_type_s  */ GGML_TYPE_F32,
            /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
            /* n_seq_max         */ cparams.n_seq_max,
            /* n_rs_seq          */ cparams.n_rs_seq,
            /* offload           */ cparams.offload_kqv,
            /* unified           */ cparams.kv_unified,
            /* filter_attn       */ [](uint32_t) { return true; },
            /* filter_recr       */ [](uint32_t) { return true; });
    }

    // DSpark draft: attention only, optionally with sliding-window layers
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(hparams.is_swa_any());

        return new llama_kv_cache_iswa(
                *this,
                params.no_alloc,
                params.type_k,
                params.type_v,
                !cparams.flash_attn,
                cparams.offload_kqv,
                params.swa_full,
                cparams.kv_unified,
                cparams.n_ctx_seq,
                cparams.n_seq_max,
                cparams.n_ubatch,
                nullptr,
                nullptr,
                nullptr,
                nullptr);
    }

    GGML_ASSERT(!hparams.is_swa_any());

    return new llama_kv_cache(
            *this,
            params.no_alloc,
            hparams,
            params.type_k,
            params.type_v,
            !cparams.flash_attn,
            cparams.offload_kqv,
            cparams.kv_unified,
            cparams.n_ctx_seq,
            cparams.n_seq_max,
            hparams.n_swa,
            hparams.swa_type,
            nullptr,
            nullptr,
            nullptr,
            nullptr);
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b);

    // add the LM head (if the context outputs logits)
    llm->build_logits();

    // add backend sampling layers (if any)
    llm->build_sampling();

    llm->build_layer_inp_dev();

    // with -sm tensor the servers compare the expert ids at the next all-reduce, after the whole sub-graph,
    // so their memory must not be reused before that; outputs are never reused
    if (split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        ggml_cgraph * gf = llm->res->get_gf();
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op == GGML_OP_MUL_MAT_ID) {
                ggml_tensor * ids = node->src[2];
                ggml_set_output(ids->view_src != nullptr ? ids->view_src : ids);
            }
        }
    }

    llm->res->set_outputs(params);

    return llm->res->get_gf();
}


//
// interface implementation
//

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.load_mode                   =*/ LLAMA_LOAD_MODE_AUTO,
        /*.lazy_mode                   =*/ LLAMA_LAZY_MODE_AUTO,
        /*.tp_attn                     =*/ LLAMA_TP_ATTN_SPLIT,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
    };

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_embd_inp(const llama_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t llama_model_n_embd_out(const llama_model * model) {
    return model->hparams.n_embd_out();
}

int32_t llama_model_n_layer_offload(const llama_model * model) {
    return model->n_layer_offload();
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer();
}

llama_dflash_variant llama_model_dflash_variant(const llama_model * model) {
    const auto * dflash = dynamic_cast<const llama_model_dflash *>(model);
    if (dflash == nullptr) {
        return LLAMA_DFLASH_VARIANT_NONE;
    }

    // only DSpark drafts load, see llama_model_dflash::load_arch_hparams
    return LLAMA_DFLASH_VARIANT_DSPARK;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    return model->hparams.n_swa;
}

uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

const char * llama_model_decision_label(const struct llama_model * model, uint32_t i) {
    if (i < model->decision_labels.size()) {
        return model->decision_labels[i].c_str();
    }

    return nullptr;
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_CLIP:
        case LLM_ARCH_INKLING2:
        case LLM_ARCH_INKLING2_LOOP:
            return LLAMA_ROPE_TYPE_NONE;

        // M-RoPE with interleaved sections
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return LLAMA_ROPE_TYPE_IMROPE;

        // the dense DSpark backbone uses NeoX RoPE
        case LLM_ARCH_DFLASH:
            // drafts for M-RoPE targets carry rope sections and follow the target's temporal dim
            if (const auto & s = model->hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
                return LLAMA_ROPE_TYPE_MROPE;
            }
            return LLAMA_ROPE_TYPE_NEOX;

        // the MLA weights rotate adjacent pairs, MROPE would pair (i, i + n_rot/2)
        // VL files carry mrope sections, but text positions are equal on all axes, so NORM gives the same result
        case LLM_ARCH_BAILINGMOE3:
            return LLAMA_ROPE_TYPE_NORM;

        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

llama_ftype llama_model_ftype(const llama_model * model) {
    return model->ftype();
}

uint64_t llama_model_size(const llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_DFLASH:    return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const llama_model * model) {
    switch (model->arch) {
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    GGML_UNUSED(model);
    return LLAMA_TOKEN_NULL;
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}

int32_t llama_model_n_expert(const struct llama_model * model) {
    return model->hparams.n_expert;
}

bool llama_model_has_longhaul(const struct llama_model * model) {
    return model->has_longhaul();
}

int32_t llama_model_n_devices(const struct llama_model * model) {
    return (int32_t)model->devices.size();
}

ggml_backend_dev_t llama_model_dev_layer(const struct llama_model * model, int32_t il) {
    return model->dev_layer(il);
}

int32_t llama_model_layer_phys(const struct llama_model * model, int32_t il) {
    return model->layer_phys(il);
}

ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i) {
    if (i < 0 || i >= (int)model->devices.size()) {
        return nullptr;
    }
    return model->devices[i].dev;
}

void llama_model_set_attn_rot(struct llama_model * model, bool value) {
    model->attn_rot = value;
}

//
// llama_model_base
//

llama_model_base::llama_model_base(const struct llama_model_params & params) : llama_model(params), model(this), tn(model->arch),
    TENSOR_DUPLICATED     (llama_model_loader::TENSOR_DUPLICATED),
    TENSOR_NOT_REQUIRED   (llama_model_loader::TENSOR_NOT_REQUIRED),
    TENSOR_SKIP           (llama_model_loader::TENSOR_SKIP),
    TENSOR_SKIP_IF_VIRTUAL(llama_model_loader::TENSOR_SKIP_IF_VIRTUAL),
    TENSOR_ALLOW_RESHAPE  (llama_model_loader::TENSOR_ALLOW_RESHAPE),
    TENSOR_READ_LAZY      (llama_model_loader::TENSOR_READ_LAZY),
    TENSOR_LONGHAUL       (llama_model_loader::TENSOR_LONGHAUL) {}

ggml_tensor * llama_model_base::create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    GGML_ASSERT(ml != nullptr);
    return create_tensor(*ml, tn, ne, flags);
}

void llama_model_base::create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_, int flags) {
    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,    "weight", bid), {n_embd_, n_ff_,     n_expert_}, skip);
        create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,      "weight", bid), {n_embd_, n_ff_,     n_expert_}, skip);
        return;
    }

    // the merged tensor is optional, but stays on disk like the separate ones (TENSOR_LONGHAUL)
    layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED | (flags & TENSOR_LONGHAUL));
    if (layer.ffn_gate_up_exps == nullptr) {
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
    }
}

void llama_model_base::create_tensor_qkv(llama_layer & layer, int bid,
        int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
        int flags) {
    const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;

    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   bid), {n_embd_qkv},          skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", bid), {n_embd_, n_embd_q_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", bid), {n_embd_, n_embd_k_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", bid), {n_embd_, n_embd_v_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias",   bid), {n_embd_q_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias",   bid), {n_embd_k_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias",   bid), {n_embd_v_},           skip);
        return;
    }

    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    if (layer.wqkv) {
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", bid), {n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
        if (!layer.wqkv_b) {
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
        }
    } else {
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", bid), {n_embd_, n_embd_q_}, flags);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", bid), {n_embd_, n_embd_k_}, flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", bid), {n_embd_, n_embd_v_}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
    }
}

void llama_model_base::load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first) {
    if (ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, false)) {
        return;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern, false);
    hparams.set_swa_pattern(n_pattern, dense_first);
}

const int32_t * llama_model_target_layer_ids(const struct llama_model * model) {
    const auto & v = model->target_layer_ids;
    return v.empty() ? nullptr : v.data();
}

uint32_t llama_model_target_layer_ids_n(const struct llama_model * model) {
    return (uint32_t) model->target_layer_ids.size();
}

// read rows [row0, row0 + n_rows) of a 2D tensor as F32
static void llama_tensor_rows_to_f32(const ggml_tensor * tensor, int64_t row0, int64_t n_rows, float * out) {
    const size_t   nelements = (size_t) n_rows * tensor->ne[0];
    const size_t   offset    = (size_t) row0 * tensor->nb[1];
    const size_t   nbytes    = ggml_row_size(tensor->type, tensor->ne[0]) * n_rows;

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out, offset, nbytes);
        return;
    }

    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(tensor, buf.data(), offset, nbytes);

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), out, nelements);
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) buf.data(), out, nelements);
    } else if (ggml_is_quantized(tensor->type) && traits->to_float != nullptr) {
        traits->to_float(buf.data(), out, nelements);
    } else {
        GGML_ABORT("unsupported tensor type for dequantization: %s", ggml_type_name(tensor->type));
    }
}

uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out) {
    if (model->vocab.n_tokens() == 0 || model->tok_embd == nullptr) {
        return 0;
    }

    const ggml_tensor * tensor = model->tok_embd;
    const size_t nelements = ggml_nelements(tensor);
    GGML_ASSERT(nelements <= UINT32_MAX); // for the return type

    if (out == nullptr) {
        return (uint32_t) nelements;
    }

    llama_tensor_rows_to_f32(tensor, 0, tensor->ne[1], out);

    return (uint32_t) nelements;
}

uint32_t llama_model_n_tokens_delay(const struct llama_model * model) {
    GGML_UNUSED(model);
    return 0;
}

bool llama_model_classifier_free_guidance(const struct llama_model * model) {
    return model->hparams.classifier_free_guidance;
}

uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out) {
    if (model->vocab.n_tokens() == 0 || model->tok_embd == nullptr || tokens == nullptr || out == nullptr) {
        return 0;
    }

    const ggml_tensor * tensor = model->tok_embd;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= tensor->ne[1]) {
            return 0;
        }
        llama_tensor_rows_to_f32(tensor, tokens[i], 1, out + (size_t) i * tensor->ne[0]);
    }

    return n_tokens * (uint32_t) tensor->ne[0];
}
