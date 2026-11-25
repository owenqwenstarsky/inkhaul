#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llama_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

// fills the values of one tensor, the name carries whatever the property needs
typedef void (*tensor_filler)(std::vector<float> & dst, const char * name, const tensor_data_params & params);

// a pool of normal draws, generated once and reused for every model
static std::vector<float> g_normal_pool;
static const size_t NORMAL_POOL_SIZE = 16 * 1024 * 1024;

// fill the pool once; each tensor copies a slice of it
static void init_normal_pool(const size_t seed, const float stdev) {
    g_normal_pool.resize(NORMAL_POOL_SIZE);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, stdev);
    for (float & v : g_normal_pool) {
        v = dist(gen);
    }
}

// a normal draw keyed on a name, so that tensors which share a draw ask for the same key
static void fill_normal(std::vector<float> & dst, const char * key, const tensor_data_params &) {
    const size_t n = dst.size();
    const size_t pool_size = g_normal_pool.size();
    const size_t start = std::hash<std::string>{}(key) % pool_size;

    // copy a cyclic slice of the pool, so a tensor larger than the pool still fills
    size_t i = 0;
    size_t pos = start;
    while (i < n) {
        const size_t chunk = std::min(pool_size - pos, n - i);
        std::memcpy(dst.data() + i, g_normal_pool.data() + pos, chunk * sizeof(float));
        i += chunk;
        pos = 0;
    }
}

// Mamba A must be negative (state decay)
static void fill_ssm_a(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = -fabsf(v);
    }
}

// RoPE factors divide the frequencies, a value close to 0 gives angles too large for F32 sin/cos to stay accurate
static void fill_rope_factors(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = 1.0f + fabsf(v);
    }
}

// FFN act scales divide the activation, a value close to 0 overflows the F16 staging of the Metal mul-mat kernel
static void fill_ffn_act(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = copysignf(1.0f + fabsf(v), v);
    }
}

// a norm preserving residual gate holds alpha^2 + beta^2 == 1, and the conversion resolves both from one raw weight
// the alpha and the beta of a block share that weight, so the key drops the suffix and keeps the block
static void fill_resid_gate(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    const char * suffix = strstr(name, "_resid_");
    const bool is_beta = strstr(suffix, "_resid_beta") != nullptr;
    fill_normal(dst, std::string(name, suffix - name).c_str(), params);
    for (float & v : dst) {
        const float beta = 1.0f / (1.0f + expf(-v));
        v = is_beta ? beta : sqrtf(1.0f - beta*beta);
    }
}

// a random router bias can tie within the noise between the backends, a ramp keeps the top-k selection the same
static void fill_exp_probs_b(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = 2.0f*i;
    }
}

// a tensor whose values hold a property of their own, matched on a substring of its name, first match wins
static const struct { const char * match; tensor_filler fill; } TENSOR_FILLERS[] = {
    { "ssm_a",        fill_ssm_a        },
    { "_resid_",      fill_resid_gate   },
    { "exp_probs_b",  fill_exp_probs_b  },
    { "ffn.act",      fill_ffn_act      },
    { "rope_freqs",   fill_rope_factors },
    { "rope_factors", fill_rope_factors },
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;

    tensor_filler fill = fill_normal;
    for (const auto & filler : TENSOR_FILLERS) {
        if (strstr(tensor->name, filler.match) != nullptr) {
            fill = filler.fill;
            break;
        }
    }

    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else if (ggml_is_quantized(tensor->type)) {
        std::vector<uint8_t> tmpq(ggml_nbytes(tensor));
        ggml_quantize_chunk(tensor->type, tmp.data(), tmpq.data(), 0, ggml_nrows(tensor), tensor->ne[0], nullptr);
        ggml_backend_tensor_set(tensor, tmpq.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --kv-shift               Check K-shift against a decode at the shifted positions (CPU only)\n");
    LOG("  --rpc                    Check LongHaul over an RPC server on a thread of this process (127.0.0.1)\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// writes arch-resolved GGUF keys into the fixture context, the test-side equivalent of llama_model_saver
struct kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }

    template <typename T>
    void add_kv(enum llm_kv key, const std::vector<T> & value) {
        if (value.empty()) {
            return;
        }
        if (std::is_same<T, uint32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT32, value.data(), value.size());
        } else if (std::is_same<T, int32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_INT32, value.data(), value.size());
        } else if (std::is_same<T, uint64_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT64, value.data(), value.size());
        } else if (std::is_same<T, float>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_FLOAT32, value.data(), value.size());
        } else {
            GGML_ABORT("fatal error");
        }
    }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), tmp.data(), tmp.size());
    }
};

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const uint32_t n_layer_min = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {
        n_embd  = 128;
        n_head  = 4;
        n_ff    = 24;
        n_layer = 3;
    } else if (arch == LLM_ARCH_BAILINGMOE3) {
        n_embd  = 128;
        n_ff    = 192;
        n_layer = 3; // the dense lead layer, then MoE layers
    }

    uint32_t n_head_kv = n_head;
    n_layer = std::max(n_layer, n_layer_min);
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));


    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv);

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);

    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));



    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));










    if (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {
        const bool loop = arch == LLM_ARCH_INKLING2_LOOP;
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_LENGTH,   uint32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_EXTENT,   uint32_t(16));
        ms.add_kv(LLM_KV_ATTENTION_QK_CLIP,           0.25f);
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_CLIP,     0.1f);
        ms.add_kv(LLM_KV_ATTENTION_LOG_SCALE_FLOOR,   uint32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_LOG_SCALE_ALPHA,   0.1f);
        ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(2));
        if (loop) {
            // prelude 0, core 1, coda 2; the loader adds slot 3 for the second pass over the core
            ms.add_kv(LLM_KV_LOOP_START,     uint32_t(1));
            ms.add_kv(LLM_KV_LOOP_END,       uint32_t(2));
            ms.add_kv(LLM_KV_LOOP_COUNT,     uint32_t(2));
            ms.add_kv(LLM_KV_LOOP_GATE_RANK, uint32_t(8));
            ms.add_kv(LLM_KV_LOOP_VARYING,   false);
        }
        ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(4));
        ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab + 2);
        ms.add_kv(LLM_KV_VOCAB_SIZE_UNPADDED,         n_vocab);
        ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        0.2f);
        ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 0.3f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,        8.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     30.0f);
        ms.add_kv(LLM_KV_EXPERT_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,           uint32_t(2));
        std::vector<bool> pattern(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            pattern[i] = i % 2 == 0;
        }
        std::vector<uint8_t> values(pattern.begin(), pattern.end());
        gguf_set_arr_data(ret.get(), llama_arch_kv_name(arch, LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN).c_str(), GGUF_TYPE_BOOL, values.data(), values.size());
    }

    if (arch == LLM_ARCH_BAILINGMOE3) {
        // the trunk alternates KDA (no KV head) and MLA (one latent KV head) layers, then one NextN (MTP) layer
        const uint32_t n_block = n_layer + 1;
        std::vector<uint32_t> n_head_kv_per_layer;
        std::vector<float>    clamp_exp;
        std::vector<float>    clamp_shexp;
        for (uint32_t il = 0; il < n_block; il++) {
            n_head_kv_per_layer.push_back(il % 2 == 1 || il == n_layer ? 1 : 0);
            // small limits, so that the clamps bite on the activations of the random weights
            clamp_exp  .push_back(il % 2 == 1 ? 0.05f : 0.0f);
            clamp_shexp.push_back(il % 2 == 1 ? 0.07f : 0.0f);
        }
        ms.add_kv(LLM_KV_BLOCK_COUNT,                n_block);
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,       uint32_t(1));
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,    n_head_kv_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(128));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,      uint32_t(0)); // Ling 3 has no q LoRA
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        // VL files carry mrope sections, the text model ropes with NORM anyway
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,    std::vector<uint32_t>({8, 12, 12, 0}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           clamp_exp);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP,         clamp_shexp);
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));

        uint32_t gating_func = 2;
        switch (arch) {
            default:
                break;
        }

        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         gating_func); // sigmoid : softmax : sqrtsoftplus
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));

        if (arch == LLM_ARCH_BAILINGMOE3) {
            // group-limited routing, as Ling 3 (8 groups, 4 used)
            ms.add_kv(LLM_KV_EXPERT_COUNT,            uint32_t(8));
            ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,      uint32_t(4));
            ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT, uint32_t(2));
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    2.5f);
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
        }
    }




    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_ROUTER_HIDDEN_SIZE,        n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    // the inkling2_loop core boundaries must be multiples of the block size
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(arch == LLM_ARCH_INKLING2_LOOP ? 1 : 12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::string get_tmp_gguf_path() {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() / ("test-llama-archs-" + std::to_string(++id) + ".gguf")).string();
}

// the model comes from one of: the fixture (gguf_ctx), a file, or the files of a split model (splits)
static llama_model_ptr get_model(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr,
        const std::vector<std::string> & splits = {}, const float * tensor_split = nullptr) {
    GGML_ASSERT((gguf_ctx != nullptr) + (file != nullptr) + !splits.empty() == 1);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;
    model_params.tensor_split = tensor_split;

    tensor_data_params tensor_params = { seed, stdev };
    std::vector<const char *> split_paths;
    for (const std::string & path : splits) {
        split_paths.push_back(path.c_str());
    }
    llama_model_ptr model(
        gguf_ctx != nullptr ? llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        file     != nullptr ? llama_model_load_from_file_ptr(file, model_params) :
                              llama_model_load_from_splits(split_paths.data(), split_paths.size(), model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    return model;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t longhaul_cache_size = 0,
        const std::vector<std::string> & splits = {}, const float * tensor_split = nullptr) {
    llama_model_ptr model = get_model(gguf_ctx, file, seed, stdev, devs, split_mode, tensor_buft_overrides, splits, tensor_split);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_ubatch = 64;
    // 0: a file-backed model keeps its experts in the mapped file as the reference, see LLAMA_LONGHAUL_RESIDENT in main
    ctx_params.longhaul_cache_size = longhaul_cache_size;

    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

// speculative decoding: each step decodes a token and 0 to n_draft_max drafts, then rejects the drafts
// the logits of the token must match a decode without drafts
static double check_draft_rollback(llama_model * model, const std::vector<llama_token> & tokens, const std::vector<float> & logits_ref) {
    const uint32_t n_vocab     = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_tokens    = tokens.size();
    const uint32_t n_draft_max = 2;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 0;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_ubatch        = 64;
    ctx_params.n_rs_seq        = n_draft_max;

    llama_context_ptr lctx(llama_init_from_model(model, ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }

    common_batch batch(lctx.get());
    for (uint32_t pos = 0; pos < n_tokens/2; pos++) {
        batch.add(tokens[pos], pos, 0, false);
    }
    if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> logits;
    std::vector<float> ref;
    for (uint32_t pos = n_tokens/2; pos < n_tokens; pos++) {
        batch.clear();
        batch.add(tokens[pos], pos, 0, true);
        const uint32_t n_draft = pos % (n_draft_max + 1);
        for (uint32_t i = 1; i <= n_draft; i++) {
            batch.add((tokens[pos] + i) % n_vocab, pos + i, 0, true);
        }
        if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            throw std::runtime_error("failed to decode batch");
        }

        const float * logits_ith = llama_get_logits_ith(lctx.get(), 0);
        logits.insert(logits.end(), logits_ith, logits_ith + n_vocab);
        ref.insert(ref.end(), logits_ref.begin() + (size_t) pos*n_vocab, logits_ref.begin() + (size_t) (pos + 1)*n_vocab);

        if (!llama_memory_seq_rm(llama_get_memory(lctx.get()), 0, pos + 1, -1)) {
            throw std::runtime_error("failed to reject the drafts");
        }
    }

    return nmse(ref, logits);
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_INKLING2:
        case LLM_ARCH_INKLING2_LOOP:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_BAILINGMOE3:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    return true;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_arch_supports_save(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            llama_model_ptr model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            const std::string path = dir + "/" + llama_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: '%s %s' -> %s ...", __func__, llama_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model.get(), path.c_str());
            const std::uintmax_t size = std::filesystem::file_size(path);
            LOG_CNT(" (%.2f MB)\n", (double) size / (1024.0 * 1024.0));
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
            continue;
        }
        dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
        max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------------|\n");
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_DFLASH) {
            continue;
        }

        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                // the KDA safe gate defaults to true
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llama_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                bool skip = !arch_supported(arch);
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens);
                    }
                    test_executed = true;
                    model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode);
                    logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens);
                    const double nmse_val = nmse(logits_cpu, logits_dev);
                    snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                    status_nmse = "\033[1;32mOK\033[0m";
                    if (nmse_val > 1e-4) {
                        test_ok = false;
                        status_nmse = "\033[1;31mFAIL\033[0m";
                    }

                    // chunked decode matches a single batch only with causal attention over a memory
                    llama_context * lctx_dev = model_and_ctx_dev.second.get();
                    if (llama_get_memory(lctx_dev) != nullptr) {
                        std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                        std::mt19937 gen(seed);
                        std::normal_distribution<float> dis(0.0f, stdev);
                        for (float & v : embd_mixed) {
                            v = dis(gen);
                        }
                        std::vector<float> logits_mixed;
                        std::vector<float> logits_chunks;
                        if (llama_arch_supports_mixed_batch(arch)) {
                            if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                throw std::runtime_error("failed to decode mixed batch");
                            }
                            const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                            snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                            status_mixed = "\033[1;32mOK\033[0m";
                            if (nmse_mixed > 1e-4) {
                                test_ok = false;
                                status_mixed = "\033[1;31mFAIL\033[0m";
                            }
                        } else {
                            // must be rejected as an invalid batch, mute the expected error log
                            ud.verbosity = LOG_LEVEL_OUTPUT;
                            const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                            ud.verbosity = verbosity;
                            if (err != -1) {
                                test_ok = false;
                                status_mixed = "\033[1;31mFAIL\033[0m";
                            }
                        }
                    }

                    // runs after the mixed batch check, as it leaves the context with non-causal attention
                    if (!check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                        if (test_ok) {
                            status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                        }
                        test_ok = false;
                    }

                    // rejected drafts roll the recurrent state back
                    if (llm_arch_supports_rs_rollback(arch)) {
                        const double nmse_rollback = check_draft_rollback(model_and_ctx_dev.first.get(), tokens, logits_dev);
                        if (!(nmse_rollback <= 1e-4)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (rollback)";
                            }
                            test_ok = false;
                        }
                    }

                    if (llama_arch_supports_save(arch)) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        const std::string path = get_tmp_gguf_path();
                        llama_model_save_to_file(model_and_ctx_dev.first.get(), path.c_str());
                        FILE * file = fopen(path.c_str(), "rb");
                        GGML_ASSERT(file != nullptr);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode);
                            const std::vector<float> logits_roundtrip = get_logits(
                                model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                            for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                                if (logits_roundtrip[i] != logits_dev[i]) {
                                    test_ok = false;
                                    status_roundtrip = "\033[1;31mFAIL\033[0m";
                                    break;
                                }
                            }
                        }

                        fclose(file);
                        std::filesystem::remove(path);
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_mixed.c_str(), mixed_str);
            }
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

// sets an environment variable, nullptr removes it
static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// the test hook of llama_context: with a cache size of 0, a file-backed model keeps its experts in the mapped file
static void set_longhaul_resident(bool resident) {
    set_env("LLAMA_LONGHAUL_RESIDENT", resident ? "1" : nullptr);
}

// rewrites a GGUF file: edit changes the KV pairs and may add tensors, map turns each tensor into the tensors of the new file (none drops it)
// the new tensors are created in the context given to both, which holds twice the data of the file plus extra bytes
using gguf_edit_fn = std::function<void(gguf_context * gguf, ggml_context * ctx)>;
using gguf_map_fn  = std::function<void(ggml_context * ctx, ggml_tensor * t, std::vector<ggml_tensor *> & out)>;

static void rewrite_gguf(const std::string & src, const std::string & dst, const gguf_edit_fn & edit, const gguf_map_fn & map, size_t extra = 0) {
    ggml_context * ctx_in_raw = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_in_raw };
    gguf_context_ptr in(gguf_init_from_file(src.c_str(), params));
    GGML_ASSERT(in);
    ggml_context_ptr ctx_in(ctx_in_raw);

    const int64_t n_tensors = gguf_get_n_tensors(in.get());
    size_t size = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        size += gguf_get_tensor_size(in.get(), i);
    }
    ggml_init_params ctx_params = { /*.mem_size =*/ 2*size + extra + 4*(n_tensors + 16)*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context_ptr ctx_out(ggml_init(ctx_params));
    GGML_ASSERT(ctx_out);

    gguf_context_ptr out(gguf_init_empty());
    gguf_set_kv(out.get(), in.get());
    edit(out.get(), ctx_out.get());

    std::vector<ggml_tensor *> res;
    for (int64_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = ggml_get_tensor(ctx_in.get(), gguf_get_tensor_name(in.get(), i));
        GGML_ASSERT(t != nullptr);
        res.clear();
        map(ctx_out.get(), t, res);
        for (ggml_tensor * r : res) {
            gguf_add_tensor(out.get(), r);
        }
    }
    GGML_ASSERT(gguf_write_to_file(out.get(), dst.c_str(), false));
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// the expert tensors of the layers of a file, grouped like the LongHaul cache groups them: by the types and sizes of the expert tensors
struct expert_group {
    std::vector<ggml_type> types;
    std::vector<size_t>    sizes;      // of one expert, per expert tensor
    size_t                 host_bytes = 0;
    int32_t                n_layers   = 0;
};

static std::vector<expert_group> get_expert_groups(const std::string & path, const llm_arch arch, uint32_t & n_expert) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr meta(gguf_init_from_file(path.c_str(), params));
    GGML_ASSERT(meta);
    auto get_u32 = [&](llm_kv kv) -> uint32_t {
        const int64_t kid = gguf_find_key(meta.get(), llama_arch_kv_name(arch, kv).c_str());
        return kid < 0 ? 0 : gguf_get_val_u32(meta.get(), kid);
    };
    n_expert = get_u32(LLM_KV_EXPERT_COUNT);
    // the MTP block is not run, the cache does not hold its experts
    const uint32_t n_layer = get_u32(LLM_KV_BLOCK_COUNT) - get_u32(LLM_KV_NEXTN_PREDICT_LAYERS);

    std::vector<expert_group> groups;
    for (uint32_t il = 0; il < n_layer; il++) {
        expert_group cur;
        for (const char * name : { "ffn_gate_up_exps", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" }) {
            const std::string tname = "blk." + std::to_string(il) + "." + name + ".weight";
            const int64_t tid = gguf_find_tensor(meta.get(), tname.c_str());
            if (tid >= 0) {
                cur.types.push_back(gguf_get_tensor_type(meta.get(), tid));
                cur.sizes.push_back(gguf_get_tensor_size(meta.get(), tid)/n_expert);
            }
        }
        if (cur.sizes.empty()) {
            continue;
        }
        auto it = std::find_if(groups.begin(), groups.end(), [&](const expert_group & g) { return g.types == cur.types && g.sizes == cur.sizes; });
        if (it == groups.end()) {
            groups.push_back(cur);
            it = groups.end() - 1;
        }
        for (size_t size : cur.sizes) {
            it->host_bytes += size*n_expert;
        }
        it->n_layers++;
    }
    return groups;
}

// the expert bytes the cache allocates for n_slots slots of a group: one extra slot per expert tensor, padded to the alignment
static size_t get_alloc_size(const expert_group & g, const size_t alignment, const int32_t n_slots) {
    size_t res = 0;
    for (size_t s : g.sizes) {
        res += GGML_PAD(s*(n_slots + 1), alignment);
    }
    return res;
}

// the bytes that give every group the slots of a ubatch of n_ubatch tokens (at most the experts of one layer)
static size_t get_reserve(const std::vector<expert_group> & groups, const uint32_t n_expert, const uint32_t n_expert_used, const size_t alignment, const uint32_t n_ubatch) {
    size_t res = 0;
    for (const expert_group & g : groups) {
        res += get_alloc_size(g, alignment, (int32_t) std::min(n_ubatch*n_expert_used, n_expert));
    }
    return res;
}

// the slots of each group for a cache size, as the cache splits it on one device: every group first gets the slots of the
// largest ubatch that fits in half of the cache, the rest is split by the size of the experts of the groups
// empty below the reserve of one token, the cache does not load
static std::vector<int32_t> get_group_slots(const std::vector<expert_group> & groups, const uint32_t n_expert, const uint32_t n_expert_used, const size_t alignment, const size_t size) {
    if (get_reserve(groups, n_expert, n_expert_used, alignment, 1) > size) {
        return {};
    }
    const uint32_t n_max = (n_expert + n_expert_used - 1)/n_expert_used;
    uint32_t n_ubatch = 1;
    while (n_ubatch < n_max && get_reserve(groups, n_expert, n_expert_used, alignment, n_ubatch + 1) <= size/2) {
        n_ubatch++;
    }
    const size_t rest = size - std::min(size, get_reserve(groups, n_expert, n_expert_used, alignment, n_ubatch));
    size_t host_bytes = 0;
    for (const expert_group & g : groups) {
        host_bytes += g.host_bytes;
    }
    std::vector<int32_t> res;
    for (const expert_group & g : groups) {
        int32_t n_slots = (int32_t) std::min(n_ubatch*n_expert_used, n_expert);
        const size_t  budget    = get_alloc_size(g, alignment, n_slots) + (size_t) ((double) rest*g.host_bytes/host_bytes);
        const int32_t max_slots = g.n_layers*(int32_t) n_expert;
        while (n_slots < max_slots && get_alloc_size(g, alignment, n_slots + 1) <= budget) {
            n_slots++;
        }
        res.push_back(n_slots);
    }
    return res;
}

// the ubatch of a context with a cache of this size: cut to the tokens whose experts fit in the group with the fewest slots,
// unless every group holds a whole layer (64 is the n_ubatch of the test contexts)
static uint32_t get_expected_ubatch(const std::vector<expert_group> & groups, const uint32_t n_expert, const uint32_t n_expert_used, const size_t alignment, const size_t size) {
    uint32_t res = 64;
    for (int32_t n_slots : get_group_slots(groups, n_expert, n_expert_used, alignment, size)) {
        if (n_slots < (int32_t) n_expert) {
            res = std::min<uint32_t>(res, n_slots/n_expert_used);
        }
    }
    return res;
}

// the smallest cache size that gives every group at least n_slots slots
static size_t get_cache_size(const std::vector<expert_group> & groups, const uint32_t n_expert, const uint32_t n_expert_used, const size_t alignment, const int32_t n_slots) {
    // the groups cannot get more than the cache, so the search starts at the sum of their banks
    size_t size = 0;
    for (const expert_group & g : groups) {
        size += get_alloc_size(g, alignment, n_slots);
    }
    auto fits = [&](size_t sz) {
        const std::vector<int32_t> slots = get_group_slots(groups, n_expert, n_expert_used, alignment, sz);
        return !slots.empty() && std::all_of(slots.begin(), slots.end(), [&](int32_t n) { return n >= n_slots; });
    };
    // with one group the bound fits, with more the share of a group can need more: double the step until it fits, then bisect
    size_t lo = size;
    size_t step = 1;
    while (!fits(lo + step - 1)) {
        lo += step;
        step *= 2;
    }
    size_t hi = lo + step - 1;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo)/2;
        if (fits(mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return hi;
}

// the first GPU (or integrated GPU) other than skip, of the named backend device if name is not nullptr
static ggml_backend_dev_t get_gpu(const char * name, ggml_backend_dev_t skip) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) && dev != skip &&
                (name == nullptr || strcmp(name, ggml_backend_dev_name(dev)) == 0)) {
            return dev;
        }
    }
    return nullptr;
}

// LongHaul with the layers split over two GPUs: each GPU caches the experts of its layers in its own banks
// size_dev is the budget that the GPU with the smallest share of a split gets, the logits must match the reference
// a small model can have all its layers on one GPU in some splits, but at least one split must use both
static bool test_longhaul_two_gpus(const std::string & label, const std::string & path, const size_t seed, const float stdev,
        ggml_backend_dev_t gpu, ggml_backend_dev_t gpu2, const size_t size_dev,
        const std::vector<llama_token> & tokens, const std::vector<float> & logits_ref) {
    struct split_case {
        const char * name;
        float        ts[2];
        size_t       size;
    };
    const split_case cases[] = {
        { "even split",           { 1.0f, 1.0f }, 2*size_dev       },
        { "split 1/3",            { 1.0f, 3.0f }, 4*size_dev       },
        { "even split, 512 MiB",  { 1.0f, 1.0f }, 512ull*1024*1024 },
    };
    bool all_ok = true;
    bool spanned = false;
    for (const split_case & sc : cases) {
        FILE * file = fopen(path.c_str(), "rb");
        GGML_ASSERT(file != nullptr);
        bool ok = true;
        std::string detail;
        try {
            auto model_and_ctx = get_model_and_ctx(nullptr, file, seed, stdev, { gpu, gpu2 }, LLAMA_SPLIT_MODE_LAYER, nullptr, sc.size, {}, sc.ts);
            const llama_model * model = model_and_ctx.first.get();

            // both GPUs hold layers, and a slot that aliases another layer is on the GPU of that layer
            int n_dev[2] = { 0, 0 };
            for (int32_t il = 0; il < llama_model_n_layer(model); il++) {
                ggml_backend_dev_t dev = llama_model_dev_layer(model, il);
                n_dev[dev == gpu2]++;
                if (dev != llama_model_dev_layer(model, llama_model_layer_phys(model, il))) {
                    ok = false;
                    detail = "layer " + std::to_string(il) + " is not on the device of its weights";
                }
            }
            spanned = spanned || (n_dev[0] > 0 && n_dev[1] > 0);
            detail += (detail.empty() ? "" : ", ") + std::string("layers ") + std::to_string(n_dev[0]) + "/" + std::to_string(n_dev[1]);

            const std::vector<float> logits = get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
            const double nmse_val = nmse(logits_ref, logits);
            char buf[32];
            snprintf(buf, sizeof(buf), "NMSE %.2e", nmse_val);
            detail += detail.empty() ? buf : std::string(", ") + buf;
            ok = ok && nmse_val <= 1e-4;
        } catch (const std::exception & err) {
            ok = false;
            detail = err.what();
        }
        fclose(file);
        LOG("LongHaul %s, two GPUs, %s: %s (%s)\n", label.c_str(), sc.name, ok ? "OK" : "FAIL", detail.c_str());
        all_ok = all_ok && ok;
    }
    if (!spanned) {
        LOG("LongHaul %s, two GPUs: FAIL (no split puts layers on both GPUs)\n", label.c_str());
    }
    return all_ok && spanned;
}

// a file for the LongHaul test, rewritten from the file saved from the in-memory model
struct longhaul_variant {
    const char * name;
    bool split_gate_up; // ffn_gate_exps and ffn_up_exps instead of ffn_gate_up_exps, as in the unsloth files
    bool mixed_down;    // ffn_down_exps of layer 1 in F16, so the cache builds two groups, as in the unsloth files
    bool mtp;           // an MTP block after the trunk, loaded but not run
};

static void write_longhaul_variant(const std::string & src, const std::string & dst, const llm_arch arch, const longhaul_variant & v, const uint32_t n_layer) {
    auto new_tensor = [](ggml_context * ctx, const std::string & name, ggml_type type, const ggml_tensor * like, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, like->ne[0], ne1, like->ne[2]);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto replace = [](std::string str, const std::string & from, const std::string & to) {
        return str.replace(str.find(from), from.size(), to);
    };
    const std::string blk_mtp = "blk." + std::to_string(n_layer) + ".";

    auto edit = [&](gguf_context * gguf, ggml_context * ctx) {
        if (!v.mtp) {
            return;
        }
        gguf_set_val_u32(gguf, llama_arch_kv_name(arch, LLM_KV_NEXTN_PREDICT_LAYERS).c_str(), 1);
        const int64_t n_embd = gguf_get_val_u32(gguf, gguf_find_key(gguf, llama_arch_kv_name(arch, LLM_KV_EMBEDDING_LENGTH).c_str()));
        const struct { const char * name; int64_t ne0; int64_t ne1; } nextn[] = {
            { "nextn.eh_proj.weight",          2*n_embd, n_embd },
            { "nextn.enorm.weight",            n_embd,   1      },
            { "nextn.hnorm.weight",            n_embd,   1      },
            { "nextn.shared_head_norm.weight", n_embd,   1      },
        };
        for (const auto & nt : nextn) {
            ggml_tensor * t = nt.ne1 == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nt.ne0) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nt.ne0, nt.ne1);
            ggml_set_name(t, (blk_mtp + nt.name).c_str());
            float * data = (float *) t->data;
            for (int64_t i = 0; i < ggml_nelements(t); i++) {
                data[i] = 1.0f;
            }
            gguf_add_tensor(gguf, t);
        }
    };

    auto map = [&](ggml_context * ctx, ggml_tensor * t, std::vector<ggml_tensor *> & out) {
        const std::string name = t->name;
        // the scales of the MTP block are not loaded, the block is skipped
        if (v.mtp && name.rfind(blk_mtp, 0) == 0 && ends_with(name, "scale")) {
            return;
        }
        if (v.split_gate_up && name.find("ffn_gate_up_exps.weight") != std::string::npos) {
            GGML_ASSERT(t->type == GGML_TYPE_F32);
            const int64_t n_ff = t->ne[1]/2;
            ggml_tensor * gate = new_tensor(ctx, replace(name, "ffn_gate_up_exps", "ffn_gate_exps"), t->type, t, n_ff);
            ggml_tensor * up   = new_tensor(ctx, replace(name, "ffn_gate_up_exps", "ffn_up_exps"),   t->type, t, n_ff);
            for (int64_t e = 0; e < t->ne[2]; e++) {
                const char * src_e = (const char *) t->data + e*t->nb[2];
                memcpy((char *) gate->data + e*gate->nb[2], src_e,                n_ff*t->nb[1]);
                memcpy((char *) up->data   + e*up->nb[2],   src_e + n_ff*t->nb[1], n_ff*t->nb[1]);
            }
            out.push_back(gate);
            out.push_back(up);
            return;
        }
        if (v.split_gate_up && name.find("ffn_gate_up_exps.") != std::string::npos) {
            // a scale of the merged tensor scales both halves
            for (const char * part : { "ffn_gate_exps", "ffn_up_exps" }) {
                ggml_tensor * s = ggml_dup_tensor(ctx, t);
                ggml_set_name(s, replace(name, "ffn_gate_up_exps", part).c_str());
                memcpy(s->data, t->data, ggml_nbytes(t));
                out.push_back(s);
            }
            return;
        }
        if (v.mixed_down && name == "blk.1.ffn_down_exps.weight") {
            GGML_ASSERT(t->type == GGML_TYPE_F32);
            ggml_tensor * down = new_tensor(ctx, name, GGML_TYPE_F16, t, t->ne[1]);
            ggml_fp32_to_fp16_row((const float *) t->data, (ggml_fp16_t *) down->data, ggml_nelements(t));
            out.push_back(down);
            return;
        }
        out.push_back(t);
    };

    rewrite_gguf(src, dst, edit, map, 16*1024*1024);
}

// LongHaul: a file-backed MoE model keeps its experts on disk and reads them into a cache on the GPU
// the logits must match the same file with its experts in the mapped file, for caches from the smallest (evictions in every layer) to one that holds all experts
static int test_longhaul(const std::string & arch_filter, const size_t seed, const float stdev, const char * target_backend) {
    ggml_backend_dev_t gpu  = get_gpu(target_backend, nullptr);
    // a second GPU, for the layers split over two GPUs (GGML_CUDA_DEVICES=2 or GGML_METAL_DEVICES=2 emulate one)
    ggml_backend_dev_t gpu2 = get_gpu(nullptr, gpu);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);
    const uint32_t n_expert      = 16;
    const uint32_t n_expert_used = 2;

    const longhaul_variant base       = { "",                         false, false, false };
    const longhaul_variant split      = { ", gate/up split",          true,  false, false };
    const longhaul_variant split_down = { ", gate/up split, F16 down", true,  true,  false };
    const longhaul_variant mtp        = { ", MTP block",              false, false, true  };

    // type_exps: GGML_TYPE_COUNT is the default F32 experts
    struct arch_variants {
        llm_arch                      arch;
        ggml_type                     type_exps;
        std::vector<longhaul_variant> variants;
    };
    std::vector<arch_variants> archs;
    for (const llm_arch arch : { LLM_ARCH_INKLING2, LLM_ARCH_INKLING2_LOOP }) {
        for (const ggml_type type_exps : { GGML_TYPE_COUNT, GGML_TYPE_Q8_0, GGML_TYPE_NVFP4 }) {
            archs.push_back({ arch, type_exps, { base } });
        }
    }
    archs.push_back({ LLM_ARCH_QWEN35MOE, GGML_TYPE_COUNT, { base, split, split_down, mtp } });

    bool all_ok = true;
    for (const arch_variants & av : archs) {
        const llm_arch  arch      = av.arch;
        const ggml_type type_exps = av.type_exps;
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        const std::string name = std::string(llama_arch_name(arch)) + (type_exps == GGML_TYPE_COUNT ? "" : std::string(" ") + ggml_type_name(type_exps));
        if (gpu == nullptr) {
            LOG("LongHaul %s: SKIP (no GPU)\n", name.c_str());
            continue;
        }
        // more experts than used per token, so that the small caches evict
        auto get_gguf_ctx_longhaul = [&](uint32_t n_layer_min) {
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, true, n_layer_min);
            gguf_set_val_u32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EXPERT_COUNT).c_str(), n_expert);
            gguf_set_val_u32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EXPERT_USED_COUNT).c_str(), n_expert_used);
            if (type_exps != GGML_TYPE_COUNT) {
                // quantized experts need rows that are a multiple of the block size
                gguf_set_val_u32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EXPERT_FEED_FORWARD_LENGTH).c_str(), 64);
                const uint32_t n_layer_ctx = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_BLOCK_COUNT).c_str()));
                for (uint32_t il = 0; il < n_layer_ctx; il++) {
                    ggml_tensor t;
                    memset(&t, 0, sizeof(ggml_tensor));
                    t.type = type_exps;
                    ggml_format_name(&t, "blk.%" PRIu32 ".ffn_gate_up_exps.weight", il);
                    gguf_add_tensor(gguf_ctx.get(), &t);
                    ggml_format_name(&t, "blk.%" PRIu32 ".ffn_down_exps.weight", il);
                    gguf_add_tensor(gguf_ctx.get(), &t);
                }
            }
            return gguf_ctx;
        };
        gguf_context_ptr gguf_ctx = get_gguf_ctx_longhaul(0);
        const uint32_t n_layer = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_BLOCK_COUNT).c_str()));

        // the in-memory model, the files without an F16 tensor must match it
        const std::vector<float> logits_mem = [&]() {
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, { gpu });
            return get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
        }();

        for (const longhaul_variant & v : av.variants) {
            const std::string label = name + v.name;

            // the MTP block is one more layer, of full attention like the one the MTP block of a Qwen 3.5 file has
            // the tensors of the trunk are drawn by name, so they are the same as the ones of the in-memory model
            const std::string path_saved = get_tmp_gguf_path();
            {
                gguf_context_ptr gguf_save = v.mtp ? get_gguf_ctx_longhaul(n_layer + 1) : get_gguf_ctx_longhaul(0);
                if (v.mtp) {
                    // the trunk as the full attention interval of 2 gives it, then the MTP block
                    std::vector<uint32_t> recr(n_layer + 1, 0);
                    for (uint32_t il = 0; il < n_layer; il++) {
                        recr[il] = (il + 1) % 2 != 0;
                    }
                    gguf_set_arr_data(gguf_save.get(), llama_arch_kv_name(arch, LLM_KV_ATTENTION_RECURRENT_LAYERS).c_str(), GGUF_TYPE_UINT32, recr.data(), recr.size());
                }
                llama_model_ptr model_cpu = get_model(gguf_save.get(), nullptr, seed, stdev, {});
                llama_model_save_with_meta(model_cpu.get(), gguf_save.get(), path_saved.c_str());
            }
            const std::string path = get_tmp_gguf_path();
            write_longhaul_variant(path_saved, path, arch, v, n_layer);
            std::filesystem::remove(path_saved);

            uint32_t n_expert_file = 0;
            const std::vector<expert_group> groups = get_expert_groups(path, arch, n_expert_file);
            GGML_ASSERT(n_expert_file == n_expert);
            GGML_ASSERT(groups.size() == (v.mixed_down ? 2u : 1u));
            const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(gpu));
            auto cache_size = [&](int32_t n_slots) { return get_cache_size(groups, n_expert, n_expert_used, alignment, n_slots); };

            // the reference: the experts in the mapped file (cache size 0 with the test hook)
            std::vector<float> logits_ref;
            {
                FILE * file = fopen(path.c_str(), "rb");
                GGML_ASSERT(file != nullptr);
                bool ok = true;
                std::string detail;
                try {
                    auto model_and_ctx = get_model_and_ctx(nullptr, file, seed, stdev, { gpu });
                    logits_ref = get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
                    if (!v.mixed_down) {
                        const double nmse_val = nmse(logits_mem, logits_ref);
                        char buf[32];
                        snprintf(buf, sizeof(buf), "NMSE %.2e", nmse_val);
                        detail = buf;
                        ok = nmse_val <= 1e-4;
                    }
                } catch (const std::exception & err) {
                    ok = false;
                    detail = err.what();
                }
                fclose(file);
                if (!v.mixed_down) {
                    LOG("LongHaul %s, file vs. in-memory model: %s (%s)\n", label.c_str(), ok ? "OK" : "FAIL", detail.c_str());
                }
                all_ok = all_ok && ok;
                if (logits_ref.empty()) {
                    std::filesystem::remove(path);
                    continue;
                }
            }

            // n_expert_used slots cut the ubatch to 1 token, 5 slots to 2 tokens, n_expert slots hold one layer, 512 MiB all experts
            // the slots are per group, the default number of I/O threads except for the cases that name one
            struct cache_case {
                std::string  name;
                size_t       size;
                const char * io_threads; // LLAMA_LONGHAUL_IO_THREADS, nullptr = default
            };
            const size_t size_min = cache_size(n_expert_used);
            std::vector<cache_case> cases = {
                { std::to_string(n_expert_used) + " slots", size_min,             nullptr },
                { "5 slots",                                cache_size(5),        nullptr },
                { std::to_string(n_expert) + " slots",      cache_size(n_expert), nullptr },
                { "512 MiB",                                512ull*1024*1024,     nullptr },
            };
            if (&v == &av.variants.front()) {
                cases.push_back({ std::to_string(n_expert_used) + " slots, 1 I/O thread", size_min, "1" });
                cases.push_back({ std::to_string(n_expert) + " slots, 16 I/O threads", cache_size(n_expert), "16" });
            }
            for (const cache_case & cc : cases) {
                const uint32_t n_ubatch_exp = get_expected_ubatch(groups, n_expert, n_expert_used, alignment, cc.size);

                set_env("LLAMA_LONGHAUL_IO_THREADS", cc.io_threads);
                FILE * file = fopen(path.c_str(), "rb");
                GGML_ASSERT(file != nullptr);
                bool ok = true;
                std::string detail;
                try {
                    auto model_and_ctx = get_model_and_ctx(nullptr, file, seed, stdev, { gpu }, LLAMA_SPLIT_MODE_LAYER, nullptr, cc.size);
                    const uint32_t n_ubatch = llama_n_ubatch(model_and_ctx.second.get());
                    if (n_ubatch != n_ubatch_exp) {
                        ok = false;
                        detail = "n_ubatch = " + std::to_string(n_ubatch) + ", expected " + std::to_string(n_ubatch_exp);
                    }
                    const std::vector<float> logits = get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
                    const double nmse_val = nmse(logits_ref, logits);
                    char buf[32];
                    snprintf(buf, sizeof(buf), "NMSE %.2e", nmse_val);
                    detail += detail.empty() ? buf : std::string(", ") + buf;
                    ok = ok && nmse_val <= 1e-4;
                } catch (const std::exception & err) {
                    ok = false;
                    detail = err.what();
                }
                fclose(file);
                set_env("LLAMA_LONGHAUL_IO_THREADS", nullptr);
                LOG("LongHaul %s, cache %s: %s (%s)\n", label.c_str(), cc.name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
                all_ok = all_ok && ok;
            }

            if (gpu2 != nullptr) {
                all_ok = test_longhaul_two_gpus(label, path, seed, stdev, gpu, gpu2, cache_size(5), tokens, logits_ref) && all_ok;
            }

            // a cache that cannot hold the experts of one token, no cache without the test hook, and no GPU are errors
            if (&v == &av.variants.front() && type_exps == GGML_TYPE_COUNT) {
                struct error_case {
                    const char *                    name;
                    size_t                          size;
                    std::vector<ggml_backend_dev_t> devs;
                };
                std::vector<error_case> errors = {
                    { "cache too small", size_min - 1,         { gpu } },
                    { "no cache",        0,                    { gpu } },
                    { "no GPU",          cache_size(n_expert), {} },
                };
                set_longhaul_resident(false);
                for (const error_case & ec : errors) {
                    FILE * file = fopen(path.c_str(), "rb");
                    GGML_ASSERT(file != nullptr);
                    bool failed = false;
                    try {
                        get_model_and_ctx(nullptr, file, seed, stdev, ec.devs, LLAMA_SPLIT_MODE_LAYER, nullptr, ec.size);
                    } catch (const std::exception &) {
                        failed = true;
                    }
                    fclose(file);
                    LOG("LongHaul %s, %s: %s\n", label.c_str(), ec.name, failed ? "OK" : "FAIL (no error)");
                    all_ok = all_ok && failed;
                }
                set_longhaul_resident(true);
            }

            std::filesystem::remove(path);
        }
    }
    return all_ok ? 0 : 1;
}

// a GGUF file with its tensor data in memory
struct gguf_file {
    gguf_context_ptr gguf;
    ggml_context_ptr ctx;

    explicit gguf_file(const std::string & path) {
        ggml_context * ctx_data = nullptr;
        gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };
        gguf.reset(gguf_init_from_file(path.c_str(), params));
        ctx.reset(ctx_data);
        GGML_ASSERT(gguf && ctx);
    }

    // the model tensors, ctx also holds the data blob
    std::vector<ggml_tensor *> tensors() const {
        std::vector<ggml_tensor *> ret;
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf.get()); i++) {
            ret.push_back(ggml_get_tensor(ctx.get(), gguf_get_tensor_name(gguf.get(), i)));
        }
        return ret;
    }
};

// adds a NextN (MTP) layer to a saved bailingmoe3 model, as the converter writes it: the weights of block il_src copied
// to block il_dst, and the NextN tensors (the fixture skips the layer, the saved model only has its optional KV scales)
static void add_nextn_layer(const std::string & path, uint32_t il_src, uint32_t il_dst, int64_t n_embd) {
    gguf_file in(path);
    gguf_context_ptr out(gguf_init_empty());
    gguf_set_kv(out.get(), in.gguf.get());

    const std::string prefix_src = "blk." + std::to_string(il_src) + ".";
    const std::string prefix_dst = "blk." + std::to_string(il_dst) + ".";

    size_t size = 4*ggml_tensor_overhead() + 8*n_embd*n_embd*sizeof(float);
    for (ggml_tensor * t : in.tensors()) {
        size += ggml_tensor_overhead() + ggml_nbytes(t);
    }
    ggml_init_params params = { /*.mem_size =*/ size, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context_ptr ctx(ggml_init(params));

    for (ggml_tensor * t : in.tensors()) {
        gguf_add_tensor(out.get(), t);
        const std::string name = ggml_get_name(t);
        const bool weight = name.size() > 7 && (name.compare(name.size() - 7, 7, ".weight") == 0 || name.compare(name.size() - 5, 5, ".bias") == 0);
        if (name.rfind(prefix_src, 0) == 0 && weight) {
            ggml_tensor * copy = ggml_dup_tensor(ctx.get(), t);
            memcpy(copy->data, t->data, ggml_nbytes(t));
            ggml_set_name(copy, (prefix_dst + name.substr(prefix_src.size())).c_str());
            gguf_add_tensor(out.get(), copy);
        }
    }
    for (const char * name : { "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "layer_output_norm.weight" }) {
        const bool proj = strcmp(name, "nextn.eh_proj.weight") == 0;
        ggml_tensor * t = proj ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2*n_embd, n_embd) : ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_embd);
        memset(t->data, 0, ggml_nbytes(t));
        ggml_set_name(t, (prefix_dst + name).c_str());
        gguf_add_tensor(out.get(), t);
    }
    GGML_ASSERT(gguf_write_to_file(out.get(), path.c_str(), false));
}

// writes a model as 2 split files, the tensors alternate between the files, so the experts of a layer are in both
static std::vector<std::string> write_splits(const std::string & path) {
    gguf_file in(path);
    const uint16_t n_split = 2;
    std::vector<gguf_context_ptr> outs;
    std::vector<std::string> paths;
    for (uint16_t i = 0; i < n_split; i++) {
        char split_path[512];
        llama_split_path(split_path, sizeof(split_path), path.c_str(), i, n_split);
        paths.emplace_back(split_path);
        outs.emplace_back(gguf_init_empty());
        if (i == 0) {
            gguf_set_kv(outs[i].get(), in.gguf.get());
        }
        gguf_set_val_u16(outs[i].get(), "split.no",            i);
        gguf_set_val_u16(outs[i].get(), "split.count",         n_split);
        gguf_set_val_i32(outs[i].get(), "split.tensors.count", gguf_get_n_tensors(in.gguf.get()));
    }
    size_t it = 0;
    for (ggml_tensor * t : in.tensors()) {
        gguf_add_tensor(outs[it++ % n_split].get(), t);
    }
    for (uint16_t i = 0; i < n_split; i++) {
        GGML_ASSERT(gguf_write_to_file(outs[i].get(), paths[i].c_str(), false));
    }
    return paths;
}

// decodes a prompt of n_prompt tokens, then the rest in batches of 1 to 3 tokens
static std::vector<float> get_logits_steps(llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, uint32_t n_prompt) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_tokens = tokens.size();
    llama_memory_clear(llama_get_memory(lctx), true);
    common_batch batch(lctx);

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    uint32_t n_step = 1;
    for (uint32_t i0 = 0; i0 < n_tokens; ) {
        const uint32_t i1 = std::min(n_tokens, i0 == 0 ? n_prompt : i0 + n_step);
        n_step = n_step % 3 + 1;
        batch.clear();
        for (uint32_t pos = i0; pos < i1; pos++) {
            batch.add(tokens[pos], pos, 0, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            throw std::runtime_error("failed to decode batch");
        }
        for (uint32_t i = 0; i < i1 - i0; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
        i0 = i1;
    }
    return ret;
}

// LongHaul with bailingmoe3: a dense lead layer, separate gate, up and down experts, a NextN layer that stays on disk,
// files with two expert layouts and split files; the logits must match the in-memory model, also when decoding in small batches
static int test_longhaul_bailingmoe3(const std::string & arch_filter, const size_t seed, const float stdev, const char * target_backend) {
    const llm_arch arch = LLM_ARCH_BAILINGMOE3;
    if (!arch_matches(arch_filter, arch)) {
        return 0;
    }
    ggml_backend_dev_t gpu  = get_gpu(target_backend, nullptr);
    ggml_backend_dev_t gpu2 = get_gpu(nullptr, gpu);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);
    // more experts than used per token, so that the small caches evict
    const uint32_t n_expert      = 16;
    const uint32_t n_expert_used = 2;

    struct variant {
        ggml_type type_exps;   // GGML_TYPE_COUNT: the default F32 experts
        uint32_t  n_layer_min;
        bool      mixed;       // the last layer keeps F32 experts, a group of its own in the cache
    };
    // a dense lead layer and 3 MoE layers (MLA, KDA, MLA)
    std::vector<variant> variants;
    for (const ggml_type type_exps : { GGML_TYPE_COUNT, GGML_TYPE_Q8_0, GGML_TYPE_NVFP4 }) {
        variants.push_back({ type_exps, 4, type_exps != GGML_TYPE_COUNT });
    }
    // 40 cached MoE layers, for the graph size with the cache lookups
    variants.push_back({ GGML_TYPE_COUNT, 41, false });

    bool all_ok = true;
    for (const variant & v : variants) {
        const ggml_type type_exps = v.type_exps;
        std::string name = std::string(llama_arch_name(arch)) + (type_exps == GGML_TYPE_COUNT ? "" : std::string(" ") + ggml_type_name(type_exps));
        if (v.mixed) {
            name += " + F32";
        }
        if (v.n_layer_min > 4) {
            name += " " + std::to_string(v.n_layer_min) + " layers";
        }
        if (gpu == nullptr) {
            LOG("LongHaul %s: SKIP (no GPU)\n", name.c_str());
            continue;
        }
        gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, true, v.n_layer_min);
        gguf_set_val_u32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EXPERT_COUNT).c_str(),      n_expert);
        gguf_set_val_u32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EXPERT_USED_COUNT).c_str(), n_expert_used);

        const uint32_t n_block = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_BLOCK_COUNT).c_str()));
        const uint32_t n_layer = n_block - 1; // without the NextN layer

        if (type_exps != GGML_TYPE_COUNT) {
            // quantized experts need rows that are a multiple of the block size
            gguf_set_val_u32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EXPERT_FEED_FORWARD_LENGTH).c_str(), 64);
            for (uint32_t il = 0; il < n_layer; il++) {
                if (v.mixed && il == n_layer - 1) {
                    continue;
                }
                for (const char * e : { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" }) {
                    ggml_tensor t;
                    memset(&t, 0, sizeof(ggml_tensor));
                    t.type = type_exps;
                    ggml_format_name(&t, "blk.%" PRIu32 ".%s.weight", il, e);
                    gguf_add_tensor(gguf_ctx.get(), &t);
                }
            }
        }

        const std::string path = get_tmp_gguf_path();
        {
            llama_model_ptr model_cpu = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            llama_model_save_with_meta(model_cpu.get(), gguf_ctx.get(), path.c_str());
        }
        // blk.1 is an MLA MoE layer, as the NextN layer of Ling 3
        const uint32_t n_embd = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EMBEDDING_LENGTH).c_str()));
        add_nextn_layer(path, 1, n_layer, n_embd);

        uint32_t n_expert_file = 0;
        const std::vector<expert_group> groups = get_expert_groups(path, arch, n_expert_file);
        GGML_ASSERT(n_expert_file == n_expert);
        GGML_ASSERT(groups.size() == (v.mixed ? 2u : 1u));
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(gpu));
        auto cache_size = [&](int32_t n_slots) { return get_cache_size(groups, n_expert, n_expert_used, alignment, n_slots); };
        auto reserve    = [&](uint32_t n_ubatch) { return get_reserve(groups, n_expert, n_expert_used, alignment, n_ubatch); };

        const std::vector<float> logits_ref = [&]() {
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, { gpu });
            return get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
        }();

        // 2 experts used per token: 2 and 5 slots cut the ubatch to 1 and 2 tokens, 16 slots hold one layer, 512 MiB all experts
        // the default number of I/O threads, except for the cases that name one
        struct cache_case {
            const char * name;
            size_t       size;
            const char * io_threads; // LLAMA_LONGHAUL_IO_THREADS, nullptr = default
        };
        std::vector<cache_case> cases;
        if (v.mixed) {
            // a group of one layer gets the slots of the ubatch as the others, the smallest cache holds one token in each
            cases = {
                { "reserve(1)",     reserve(1),       nullptr },
                { "2 x reserve(4)", 2*reserve(4),     nullptr },
                { "512 MiB",        512ull*1024*1024, nullptr },
            };
            GGML_ASSERT(get_expected_ubatch(groups, n_expert, n_expert_used, alignment, reserve(1)) == 1);
            GGML_ASSERT(get_expected_ubatch(groups, n_expert, n_expert_used, alignment, 2*reserve(4)) >= 4);
        } else if (v.n_layer_min > 4) {
            cases = {
                { "5 slots", cache_size(5),    nullptr },
                { "512 MiB", 512ull*1024*1024, nullptr },
            };
        } else {
            cases = {
                { "2 slots",                  cache_size(2),    nullptr },
                { "2 slots, 1 I/O thread",    cache_size(2),    "1"     },
                { "5 slots",                  cache_size(5),    nullptr },
                { "16 slots",                 cache_size(16),   nullptr },
                { "16 slots, 16 I/O threads", cache_size(16),   "16"    },
                { "512 MiB",                  512ull*1024*1024, nullptr },
            };
        }
        for (const cache_case & cc : cases) {
            const uint32_t n_ubatch_exp = get_expected_ubatch(groups, n_expert, n_expert_used, alignment, cc.size);
            set_env("LLAMA_LONGHAUL_IO_THREADS", cc.io_threads);
            FILE * file = fopen(path.c_str(), "rb");
            GGML_ASSERT(file != nullptr);
            bool ok = true;
            std::string detail;
            try {
                auto model_and_ctx = get_model_and_ctx(nullptr, file, seed, stdev, { gpu }, LLAMA_SPLIT_MODE_LAYER, nullptr, cc.size);
                const uint32_t n_ubatch = llama_n_ubatch(model_and_ctx.second.get());
                if (n_ubatch != n_ubatch_exp) {
                    ok = false;
                    detail = "n_ubatch = " + std::to_string(n_ubatch) + ", expected " + std::to_string(n_ubatch_exp);
                }
                const std::vector<float> logits = get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
                // also decode and small batches after a prompt
                const double nmse_val = std::max(nmse(logits_ref, logits),
                    nmse(logits_ref, get_logits_steps(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens, 100)));
                char buf[32];
                snprintf(buf, sizeof(buf), "NMSE %.2e", nmse_val);
                detail += detail.empty() ? buf : std::string(", ") + buf;
                ok = ok && nmse_val <= 1e-4;
            } catch (const std::exception & err) {
                ok = false;
                detail = err.what();
            }
            fclose(file);
            set_env("LLAMA_LONGHAUL_IO_THREADS", nullptr);
            LOG("LongHaul %s, cache %s: %s (%s)\n", name.c_str(), cc.name, ok ? "OK" : "FAIL", detail.c_str());
            all_ok = all_ok && ok;
        }

        if (gpu2 != nullptr) {
            all_ok = test_longhaul_two_gpus(name, path, seed, stdev, gpu, gpu2, v.mixed ? reserve(2) : cache_size(5), tokens, logits_ref) && all_ok;
        }

        if (v.mixed) {
            // below the reserve of one token, the error names the smallest cache
            std::string log;
            ggml_log_callback log_cb;
            void * log_ud;
            llama_log_get(&log_cb, &log_ud);
            llama_log_set([](ggml_log_level, const char * text, void * ud) { ((std::string *) ud)->append(text); }, &log);
            FILE * file = fopen(path.c_str(), "rb");
            GGML_ASSERT(file != nullptr);
            bool failed = false;
            try {
                get_model_and_ctx(nullptr, file, seed, stdev, { gpu }, LLAMA_SPLIT_MODE_LAYER, nullptr, reserve(1) - 1);
            } catch (const std::exception &) {
                failed = true;
            }
            fclose(file);
            llama_log_set(log_cb, log_ud);
            char min_str[64];
            snprintf(min_str, sizeof(min_str), "use at least %.0f MiB", std::ceil(reserve(1)/1024.0/1024.0));
            const bool ok = failed && log.find(min_str) != std::string::npos;
            LOG("LongHaul %s, cache below reserve(1): %s\n", name.c_str(), ok ? "OK" : "FAIL (no error or no minimum)");
            all_ok = all_ok && ok;
        }

        if (type_exps != GGML_TYPE_COUNT || v.n_layer_min > 4) {
            std::filesystem::remove(path);
            continue;
        }

        // a split model, the experts are read from both files
        {
            const std::vector<std::string> splits = write_splits(path);
            bool ok = true;
            std::string detail;
            try {
                auto model_and_ctx = get_model_and_ctx(nullptr, nullptr, seed, stdev, { gpu }, LLAMA_SPLIT_MODE_LAYER, nullptr, cache_size(5), splits);
                const double nmse_val = nmse(logits_ref, get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens));
                char buf[32];
                snprintf(buf, sizeof(buf), "NMSE %.2e", nmse_val);
                detail = buf;
                ok = nmse_val <= 1e-4;
            } catch (const std::exception & err) {
                ok = false;
                detail = err.what();
            }
            for (const std::string & p : splits) {
                std::filesystem::remove(p);
            }
            LOG("LongHaul %s, 2 split files: %s (%s)\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
            all_ok = all_ok && ok;
        }

        // a cache that cannot hold the experts of one token, no cache without the test hook and no GPU are errors
        struct error_case {
            const char *                    name;
            size_t                          size;
            std::vector<ggml_backend_dev_t> devs;
        };
        std::vector<error_case> errors = {
            { "cache too small", cache_size(n_expert_used) - 1, { gpu } },
            { "no cache",        0,                             { gpu } },
            { "no GPU",          cache_size(n_expert),          {} },
        };
        set_longhaul_resident(false);
        for (const error_case & ec : errors) {
            FILE * file = fopen(path.c_str(), "rb");
            GGML_ASSERT(file != nullptr);
            bool failed = false;
            try {
                get_model_and_ctx(nullptr, file, seed, stdev, ec.devs, LLAMA_SPLIT_MODE_LAYER, nullptr, ec.size);
            } catch (const std::exception &) {
                failed = true;
            }
            fclose(file);
            LOG("LongHaul %s, %s: %s\n", name.c_str(), ec.name, failed ? "OK" : "FAIL (no error)");
            all_ok = all_ok && failed;
        }
        set_longhaul_resident(true);

        std::filesystem::remove(path);
    }
    return all_ok ? 0 : 1;
}

// LongHaul over RPC: an RPC server on a thread of this process (127.0.0.1 only) exposes the GPU
// a file the server has a copy of is read by the server, a file whose copy differs is sent from the client; the logits must match either way
static int test_longhaul_rpc(const std::string & arch_filter, const size_t seed, const float stdev, const char * target_backend) {
    ggml_backend_dev_t gpu = get_gpu(target_backend, nullptr);
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (gpu == nullptr || reg == nullptr) {
        LOG("LongHaul RPC: SKIP (%s)\n", gpu == nullptr ? "no GPU" : "no RPC backend");
        return 0;
    }
    auto start_server_fn  = (decltype(ggml_backend_rpc_start_server) *)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
    auto set_model_dir_fn = (decltype(ggml_backend_rpc_server_set_model_dir) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_server_set_model_dir");
    auto add_server_fn    = (decltype(ggml_backend_rpc_add_server) *)           ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_add_server");
    GGML_ASSERT(start_server_fn && set_model_dir_fn && add_server_fn);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);
    const uint32_t n_expert      = 16;
    const uint32_t n_expert_used = 2;
    const longhaul_variant base  = { "", false, false, false };

    std::random_device rd;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("test-llama-archs-rpc-" + std::to_string(rd()));
    const std::filesystem::path dir_client = dir / "client";
    const std::filesystem::path dir_server = dir / "server";
    std::filesystem::create_directories(dir_client);
    std::filesystem::create_directories(dir_server);

    // per arch: file "a" with an exact copy on the server, file "b" (another name in the metadata) whose server copy has one changed byte
    // the weights come from a pool keyed on the tensor names, so both files have the same weights
    struct rpc_file {
        llm_arch    arch;
        std::string path;
        bool        server_copy; // the server has the same file
    };
    std::vector<rpc_file> files;
    for (const llm_arch arch : { LLM_ARCH_INKLING2, LLM_ARCH_INKLING2_LOOP }) {
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        // "b" first: the logits read by the server must match the logits sent from the client bit for bit
        for (const bool server_copy : { false, true }) {
            const std::string name = std::string(llama_arch_name(arch)) + (server_copy ? "-a" : "-b") + ".gguf";
            const std::string path = (dir_client / name).string();

            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, true, 0);
            gguf_set_val_u32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EXPERT_COUNT).c_str(), n_expert);
            gguf_set_val_u32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_EXPERT_USED_COUNT).c_str(), n_expert_used);
            gguf_set_val_str(gguf_ctx.get(), "general.name", name.c_str());
            const uint32_t n_layer = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_BLOCK_COUNT).c_str()));
            const std::string path_saved = get_tmp_gguf_path();
            {
                llama_model_ptr model_cpu = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
                llama_model_save_with_meta(model_cpu.get(), gguf_ctx.get(), path_saved.c_str());
            }
            write_longhaul_variant(path_saved, path, arch, base, n_layer);
            std::filesystem::remove(path_saved);

            const std::filesystem::path path_server = dir_server / name;
            std::filesystem::copy_file(path, path_server);
            if (!server_copy) {
                // the last byte is in a block of the fingerprint
                std::fstream f(path_server, std::ios::binary | std::ios::in | std::ios::out);
                f.seekg(-1, std::ios::end);
                const char c = (char) (f.get() ^ 0xff);
                f.seekp(-1, std::ios::end);
                f.put(c);
            }
            files.push_back({ arch, path, server_copy });
        }
    }
    if (files.empty()) {
        std::filesystem::remove_all(dir);
        return 0;
    }

    GGML_ASSERT(set_model_dir_fn(dir_server.string().c_str(), 2) == files.size());
    const std::string endpoint = "127.0.0.1:" + std::to_string(40000 + rd() % 20000);
    std::thread([start_server_fn, endpoint, gpu]() mutable {
        start_server_fn(endpoint.c_str(), nullptr, 4, 1, &gpu);
    }).detach();
    ggml_backend_reg_t reg_server = nullptr;
    for (int i = 0; i < 400 && reg_server == nullptr; i++) {
        reg_server = add_server_fn(endpoint.c_str());
        if (reg_server == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
    if (reg_server == nullptr) {
        LOG("LongHaul RPC: FAIL (the RPC server on %s did not start)\n", endpoint.c_str());
        std::filesystem::remove_all(dir);
        return 1;
    }
    ggml_backend_dev_t rpc = ggml_backend_reg_dev_get(reg_server, 0);

    bool all_ok = true;
    std::map<std::pair<llm_arch, size_t>, std::vector<float>> logits_sent;
    for (const rpc_file & rf : files) {
        const std::string label = std::string(llama_arch_name(rf.arch)) + (rf.server_copy ? ", server copy" : ", changed server copy");

        // the reference: the experts in the mapped file, on the local GPU
        std::vector<float> logits_ref;
        {
            FILE * file = fopen(rf.path.c_str(), "rb");
            GGML_ASSERT(file != nullptr);
            auto model_and_ctx = get_model_and_ctx(nullptr, file, seed, stdev, { gpu });
            logits_ref = get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
            fclose(file);
        }

        uint32_t n_expert_file = 0;
        const std::vector<expert_group> groups = get_expert_groups(rf.path, rf.arch, n_expert_file);
        const size_t alignment = std::max(ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(rpc)), ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(gpu)));
        const size_t size_5 = get_cache_size(groups, n_expert, n_expert_used, alignment, 5);

        struct rpc_case {
            const char *                    name;
            std::vector<ggml_backend_dev_t> devs;
            std::vector<float>              ts;
            size_t                          size;
        };
        const rpc_case cases[] = {
            { "RPC, 5 slots",            { rpc },      {},           size_5           },
            { "RPC, 512 MiB",            { rpc },      {},           512ull*1024*1024 },
            { "RPC + GPU 1/3, 5 slots",  { rpc, gpu }, { 1.0f, 3.0f }, 4*size_5       },
        };
        for (size_t ic = 0; ic < sizeof(cases)/sizeof(cases[0]); ic++) {
            const rpc_case & rc = cases[ic];
            std::string log;
            ggml_log_callback log_cb;
            void * log_ud;
            llama_log_get(&log_cb, &log_ud);
            llama_log_set([](ggml_log_level, const char * text, void * ud) { ((std::string *) ud)->append(text); }, &log);
            bool ok = true;
            std::string detail;
            try {
                auto model_and_ctx = get_model_and_ctx(nullptr, nullptr, seed, stdev, rc.devs, LLAMA_SPLIT_MODE_LAYER, nullptr, rc.size,
                    { rf.path }, rc.ts.empty() ? nullptr : rc.ts.data());
                const std::vector<float> logits = get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
                const double nmse_val = nmse(logits_ref, logits);
                char buf[32];
                snprintf(buf, sizeof(buf), "NMSE %.2e", nmse_val);
                detail = buf;
                ok = nmse_val <= 1e-4;
                if (!rf.server_copy) {
                    logits_sent[{ rf.arch, ic }] = logits;
                } else if (logits != logits_sent[{ rf.arch, ic }]) {
                    ok = false;
                    detail += ", differs from the experts sent from the client";
                }
            } catch (const std::exception & err) {
                ok = false;
                detail = err.what();
            }
            llama_log_set(log_cb, log_ud);

            // the server reads the experts only from an exact copy
            const bool read_by_server = log.find("read by the server") != std::string::npos;
            const bool sent           = log.find("has no copy of") != std::string::npos;
            if (read_by_server != rf.server_copy || sent == rf.server_copy) {
                ok = false;
                detail += rf.server_copy ? ", not read by the server" : ", not sent from the client";
            }
            LOG("LongHaul %s, %s: %s (%s)\n", label.c_str(), rc.name, ok ? "OK" : "FAIL", detail.c_str());
            all_ok = all_ok && ok;
        }
    }

    std::filesystem::remove_all(dir);
    return all_ok ? 0 : 1;
}

// Ling 3 rotates adjacent pairs: a bailingmoe3 model ropes with NORM (one position per token) even with the mrope sections of the VL files
static int test_rope_type(const std::string & arch_filter, const size_t seed, const float stdev) {
    if (!arch_matches(arch_filter, LLM_ARCH_BAILINGMOE3)) {
        return 0;
    }
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_BAILINGMOE3, true);
    llama_model_ptr model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
    const bool ok = llama_model_rope_type(model.get()) == LLAMA_ROPE_TYPE_NORM;
    LOG("Rope type bailingmoe3: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

struct kv_shift_move {
    llama_pos p0;
    llama_pos p1;
    llama_pos delta;
};

static void decode_range(llama_context * lctx, const llama_token * tokens, uint32_t n_tokens, llama_pos pos0, llama_seq_id seq_id, bool output) {
    common_batch batch(lctx);
    for (uint32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], pos0 + i, seq_id, output || i == n_tokens - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }
}

// the raw attention scores of each layer are linear in K, unlike the logits they do not depend on how sharp the attention is
struct kq_capture {
    bool active = false;
    std::vector<std::pair<std::string, std::vector<float>>> data;
};

// the scores are compared cell by cell, so the cells of the probe must match: prefix and probe fit in the SWA window,
// otherwise the probe can reuse a masked SWA cell in one run and not in the other
static bool kq_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    kq_capture * cap = (kq_capture *) user_data;
    const char * name = ggml_get_name(t);
    const bool want = cap->active && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        (strncmp(name, "kq-", 3) == 0 || strncmp(name, "indexer_kq-", 11) == 0 || strncmp(name, "indexer_score-", 14) == 0);
    if (ask) {
        return want;
    }
    if (want) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        auto it = std::find_if(cap->data.begin(), cap->data.end(), [&](const auto & e) { return e.first == name; });
        if (it == cap->data.end()) {
            cap->data.emplace_back(name, std::move(buf));
        } else {
            it->second = std::move(buf);
        }
    }
    return true;
}

struct kq_diff {
    double      worst = 0.0;
    std::string first_bad; // first tensor above the threshold, later layers can inherit its error through Q
    std::string bad;       // all tensors above the threshold, with their nmse
};

static kq_diff compare_kq(const std::vector<std::pair<std::string, std::vector<float>>> & ref,
                          const std::vector<std::pair<std::string, std::vector<float>>> & cur, double thold) {
    if (ref.size() != cur.size()) {
        throw std::runtime_error("kq tensor count mismatch");
    }
    kq_diff res;
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i].first != cur[i].first || ref[i].second.size() != cur[i].second.size()) {
            throw std::runtime_error("kq tensor mismatch: " + ref[i].first);
        }
        double mse_a_b = 0.0;
        double mse_a_0 = 0.0;
        for (size_t j = 0; j < ref[i].second.size(); j++) {
            const double a = ref[i].second[j];
            const double b = cur[i].second[j];
            // masked rows may hold -inf in both
            if (!std::isfinite(a) && a == b) {
                continue;
            }
            mse_a_b += (a - b)*(a - b);
            mse_a_0 += a*a;
        }
        const double val = mse_a_b == 0.0 ? 0.0 : mse_a_0 == 0.0 ? INFINITY : mse_a_b/mse_a_0;
        res.worst = std::max(res.worst, std::isnan(val) ? INFINITY : val);
        if (!(val <= thold)) {
            if (res.first_bad.empty()) {
                res.first_bad = ref[i].first;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), " %s=%.1e", ref[i].first.c_str(), val);
            res.bad += buf;
        }
    }
    return res;
}

// decodes the prefix at pos_prefix, applies each step of moves with its own K-shift and returns the logits of the probe at pos_probe
static std::vector<float> get_logits_kv_shift(
        llama_model * model, llama_context * lctx, kq_capture & cap, const std::vector<llama_token> & prefix, const std::vector<llama_token> & probe,
        llama_pos pos_prefix, const std::vector<std::vector<kv_shift_move>> & steps, llama_pos pos_probe) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(lctx);

    cap.data.clear();
    llama_memory_clear(mem, true);
    decode_range(lctx, prefix.data(), prefix.size(), pos_prefix, 0, false);

    for (size_t i = 0; i < steps.size(); i++) {
        for (const kv_shift_move & mv : steps[i]) {
            llama_memory_seq_add(mem, 0, mv.p0, mv.p1, mv.delta);
        }
        // there is no public API to apply a pending shift, a decode of a throwaway token in seq 1 does it
        if (i + 1 < steps.size()) {
            decode_range(lctx, probe.data(), 1, 0, 1, false);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }

    cap.active = true;
    decode_range(lctx, probe.data(), probe.size(), pos_probe, 0, true);
    cap.active = false;

    std::vector<float> ret;
    ret.reserve(probe.size()*n_vocab);
    for (uint32_t i = 0; i < probe.size(); i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
    }
    return ret;
}

// a K-shift is correct if decoding at [d, d+n) and shifting by -d gives the same probe logits as decoding at [0, n)
static int test_kv_shift(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    // n_prefix + n_probe stays below the fixture SWA window (n_ctx/8 = 32)
    const uint32_t  n_prefix = 20;
    const uint32_t  n_probe  = 8;
    const llama_pos d        = 64;
    const llama_pos gap      = 4;

    const std::vector<llama_token> tokens = get_tokens(n_prefix + n_probe, 128, seed);
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + n_prefix);
    const std::vector<llama_token> probe (tokens.begin() + n_prefix, tokens.end());

    const std::vector<std::vector<kv_shift_move>> steps_shift  = { { { d, d + (llama_pos) n_prefix, -d } } };
    const std::vector<std::vector<kv_shift_move>> steps_shift2 = { { { d, d + (llama_pos) n_prefix, -d/2 } }, { { d/2, d/2 + (llama_pos) n_prefix, -d/2 } } };
    // control: the second half of the prefix ends up gap positions too far, which tells if the probe is sensitive to K positions at all
    const std::vector<std::vector<kv_shift_move>> steps_ctrl   = { {
        { d,                           d + (llama_pos) n_prefix/2, -d       },
        { d + (llama_pos) n_prefix/2,  d + (llama_pos) n_prefix,   -d + gap },
    } };

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }
    const std::string template_row = "|%" + std::to_string(max_arch_name_length) + "s|%6s|%10s|%10s|%10s|%10s|%s\n";

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_row.c_str(), "Model arch.", "Config", "Shift kq", "Shift2 kq", "Logits", "Control kq", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !arch_supported(arch)) {
            continue;
        }
        if (arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const char * config_name = moe ? "MoE" : "Dense";

            const double thold = 1e-6;

            // at least 4 layers, so that "every 4th layer" patterns (NoPE, full attention) are covered
            const uint32_t n_layer_min = 4;
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe, n_layer_min);
            // different bases, so that a shift using the wrong one is caught
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE).c_str(),     20000.0f);
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE_SWA).c_str(), 10000.0f);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }

            llama_model_ptr model;
            try {
                model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            } catch (const std::exception & err) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;31mERROR\033[0m (model load)");
                n_tests++;
                n_failed++;
                continue;
            }
            // absolute position embeddings are not translation invariant, the comparison does not hold
            if (llama_model_rope_type(model.get()) == LLAMA_ROPE_TYPE_NONE) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;33mSKIP\033[0m (no rope)");
                continue;
            }

            LOG("|%*s|%6s|", (int) max_arch_name_length, llama_arch_name(arch), config_name);
            fflush(stdout);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = 0;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            ctx_params.n_seq_max       = 2;
            ctx_params.kv_unified      = true;
            ctx_params.type_k          = GGML_TYPE_F32;
            ctx_params.type_v          = GGML_TYPE_F32;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            // differs from the SWA default, so that a shift using the wrong scale is caught
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
            ctx_params.rope_freq_scale   = 0.5f;
            kq_capture cap;
            ctx_params.cb_eval           = kq_capture_cb;
            ctx_params.cb_eval_user_data = &cap;

            ud.verbosity = LOG_LEVEL_OUTPUT;
            llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
            ud.verbosity = verbosity;
            if (!lctx) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (context)\n", "", "", "", "");
                continue;
            }
            llama_memory_t mem = llama_get_memory(lctx.get());
            if (mem == nullptr || !llama_memory_can_shift(mem)) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (no shift)\n", "", "", "", "");
                continue;
            }

            std::string status;
            char shift_str[16]  = "";
            char shift2_str[16] = "";
            char logits_str[16] = "";
            char ctrl_str[16]   = "";
            n_tests++;
            try {
                const std::vector<float> logits_ref = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, 0, {}, n_prefix);
                const auto kq_ref = cap.data;
                if (verbosity >= LOG_LEVEL_INFO) {
                    for (const auto & e : kq_ref) {
                        LOG("[%s: %zu] ", e.first.c_str(), e.second.size());
                    }
                }
                const std::vector<float> logits_shift = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift, n_prefix);
                const kq_diff diff_shift = compare_kq(kq_ref, cap.data, thold);
                get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift2, n_prefix);
                const kq_diff diff_shift2 = compare_kq(kq_ref, cap.data, thold);
                const std::vector<float> logits_ctrl = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_ctrl, n_prefix + gap);
                const kq_diff diff_ctrl = compare_kq(kq_ref, cap.data, thold);

                const double nmse_logits = nmse(logits_ref, logits_shift);
                snprintf(shift_str,  sizeof(shift_str),  "%.2e", diff_shift.worst);
                snprintf(shift2_str, sizeof(shift2_str), "%.2e", diff_shift2.worst);
                snprintf(logits_str, sizeof(logits_str), "%.2e", nmse_logits);
                snprintf(ctrl_str,   sizeof(ctrl_str),   "%.2e", diff_ctrl.worst);

                if (kq_ref.empty()) {
                    status = "\033[1;33mSKIP\033[0m (no kq captured)";
                } else if (!diff_shift.first_bad.empty() || !diff_shift2.first_bad.empty()) {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m" + (diff_shift.bad.empty() ? " x2:" + diff_shift2.bad : diff_shift.bad);
                } else if (!(diff_ctrl.worst > 10*thold)) {
                    status = "\033[1;33mOK\033[0m (low sensitivity)";
                } else {
                    status = "\033[1;32mOK\033[0m";
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%10s|%10s|%10s|%10s|%s\n", shift_str, shift2_str, logits_str, ctrl_str, status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    // the reference runs keep the experts of file-backed models in the mapped file, see test_longhaul for the cache
    set_longhaul_resident(true);

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    const char * target_backend = nullptr;
    bool kv_shift = false;
    bool rpc      = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llama_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--kv-shift") == 0) {
            kv_shift = true;
        } else if (strcmp(argv[i], "--rpc") == 0) {
            rpc = true;
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    init_normal_pool(seed, stdev);

    try {
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        if (kv_shift) {
            return test_kv_shift(arch_filter, seed, stdev, verbosity);
        }
        if (rpc) {
            return test_longhaul_rpc(arch_filter, seed, stdev, target_backend);
        }
        int ret = test_backends(arch_filter, seed, stdev, verbosity, target_backend);
        ret = test_rope_type(arch_filter, seed, stdev) != 0 ? 1 : ret;
        ret = test_longhaul(arch_filter, seed, stdev, target_backend) != 0 ? 1 : ret;
        return test_longhaul_bailingmoe3(arch_filter, seed, stdev, target_backend) != 0 ? 1 : ret;
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
