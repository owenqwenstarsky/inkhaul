#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd-impl.h"

#include <algorithm>
#include <array>
#include <map>
#include <vector>
#include <cstdint>
#include <cmath>

enum mtmd_modality {
    MTMD_MODALITY_VISION,
    MTMD_MODALITY_AUDIO,
    MTMD_MODALITY_GEN_AUDIO,
};

// gen-side voice reference (chatterbox): speech tokens, mel-rate features and the speaker vector
struct mtmd_gen_ref {
    std::vector<int32_t> tokens;
    std::vector<float>   feat;
    std::vector<float>   spk;
};

enum ffn_op_type {
    FFN_GELU,
    FFN_GELU_ERF,
    FFN_SILU,
    FFN_GELU_QUICK,
    FFN_RELU_SQR,
};

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};

// the low bits select the algo, Pillow-compatible (matching PIL.Image.resize output)
// RESIZE_ALGO_FLAG_TORCH matches torchvision's antialiased resize of uint8 images on the CPU (int16 weights), bilinear and bicubic only
enum resize_algo {
    RESIZE_ALGO_BILINEAR,
    RESIZE_ALGO_BICUBIC,
    RESIZE_ALGO_LANCZOS,

    RESIZE_ALGO_FLAG_TORCH = 1 << 8,
};

// Padding style for img_tool::resize
//   PAD_NONE    - no padding; direct resize to target dimensions
//   PAD_CEIL    - aspect-preserving pad (default)
//   PAD_NEAREST - aspect-preserving pad with nearest-integer rounding (Pillow byte-parity)
enum pad_style {
    PAD_NONE,
    PAD_CEIL,
    PAD_NEAREST,
};

struct mtmd_hparams {
    int32_t image_size = 0;
    int32_t patch_size = 0;
    int32_t n_embd = 0;
    int32_t n_ff = 0;
    int32_t projection_dim = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    // 0 = derive from n_embd; set when qkv width != n_embd
    int32_t n_embd_head = 0;
    int32_t n_layer = 0;
    int32_t n_merge = 1; // number of patch merges **per-side**

    // for preprocessor
    int32_t image_min_pixels = -1;
    int32_t image_max_pixels = -1;
    resize_algo image_resize_algo = RESIZE_ALGO_BICUBIC;

    float image_mean[3];
    float image_std[3];

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM every time they load the model
    int32_t warmup_image_size = 0;
    int32_t warmup_audio_size = 3000;

    ffn_op_type ffn_op = FFN_GELU;

    float eps = 1e-6;
    float rope_theta = 0.0;

    // audio
    int32_t n_mel_bins = 0; // whisper preprocessor

    // audio-to-mel preprocessor params
    int32_t audio_sample_rate = -1;
    int32_t audio_hop_len     = -1;

    // custom value provided by user, can be undefined if not set
    int32_t custom_image_min_tokens = -1;
    int32_t custom_image_max_tokens = -1;

    void set_limit_image_tokens(int n_tokens_min, int n_tokens_max) {
        const int patch_area = patch_size * patch_size * n_merge * n_merge;
        image_min_pixels = (custom_image_min_tokens > 0 ? custom_image_min_tokens : n_tokens_min) * patch_area;
        image_max_pixels = (custom_image_max_tokens > 0 ? custom_image_max_tokens : n_tokens_max) * patch_area;
        warmup_image_size = static_cast<int>(std::sqrt(image_max_pixels));
    }

    void set_warmup_n_tokens(int n_tokens) {
        int n_tok_per_side = static_cast<int>(std::sqrt(n_tokens));
        GGML_ASSERT(n_tok_per_side * n_tok_per_side == n_tokens && "n_tokens must be n*n");
        // do not warmup with more tokens than the max allowed
        if (custom_image_max_tokens > 0 && n_tokens > custom_image_max_tokens) {
            n_tok_per_side = std::max(1, static_cast<int>(std::sqrt(custom_image_max_tokens)));
        }
        warmup_image_size = n_tok_per_side * patch_size * n_merge;
    }
};

struct mtmd_layer {
    // layernorm 1 (or layer input norm, or pre-attention norm)
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    // attention
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * qkv_w = nullptr;
    ggml_tensor * qkv_b = nullptr;

    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    ggml_tensor * k_norm = nullptr;
    ggml_tensor * q_norm = nullptr;

    ggml_tensor * ff_up_w = nullptr;
    ggml_tensor * ff_up_b = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_gate_b = nullptr;
    ggml_tensor * ff_down_w = nullptr;
    ggml_tensor * ff_down_b = nullptr;

    // layernorm 2 (or pre-FFN norm)
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;

    // layer scale (no bias)
    ggml_tensor * ls_1_w = nullptr;
    ggml_tensor * ls_2_w = nullptr;
};

// the weights and hparams of one modality of an mmproj file, read-only once loaded
struct mtmd_model {
    mtmd_modality modality = MTMD_MODALITY_VISION;
    projector_type proj_type = PROJECTOR_TYPE_UNKNOWN;
    mtmd_hparams hparams;

    // runtime requirements declared by the model, applied by mtmd_subcontext
    int max_nodes        = 8192;
    bool flash_attn_auto = true; // false: MTMD_FLASH_ATTN_TYPE_AUTO resolves to disabled

    // weights
    ggml_context_ptr        ctx_data;
    ggml_backend_buffer_ptr buf;
    std::map<ggml_backend_dev_t, size_t> mem_weights;

    // embeddings
    ggml_tensor * class_embedding = nullptr;
    ggml_tensor * class_position_embedding = nullptr;
    ggml_tensor * patch_embeddings_0 = nullptr;
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * position_embeddings = nullptr;

    ggml_tensor * pre_ln_w = nullptr;
    ggml_tensor * pre_ln_b = nullptr;

    std::vector<mtmd_layer> layers;

    ggml_tensor * post_ln_w = nullptr;
    ggml_tensor * post_ln_b = nullptr;

    // projector
    ggml_tensor * mm_1_w = nullptr;
    ggml_tensor * mm_1_b = nullptr;
    ggml_tensor * mm_2_w = nullptr;
    ggml_tensor * mm_2_b = nullptr;

    // this should be equal to the embedding dimension of the text model
    int n_mmproj_embd() const;

    int n_output_tokens(const mtmd_image_f32 * img) const;

    // for M-RoPE, this will be the number of token positions in X and Y directions
    // for other models, X will be the total number of tokens and Y will be 1
    int n_output_tokens_x(const mtmd_image_f32 * img) const;
    int n_output_tokens_y(const mtmd_image_f32 * img) const;

    // max number of output tokens per image, -1 if not dynamic size
    int image_max_tokens() const;

    // TODO @ngxson : this is no longer correct with mtmd_batch API
    int n_temporal_merge() const;
};

struct mtmd_model_params {
    // custom value provided by user, -1 if not set
    int32_t image_min_tokens = -1;
    int32_t image_max_tokens = -1;
    bool no_alloc = false; // only create the tensors, do not read their data
    mtmd_progress_callback progress_callback = nullptr;
    void * progress_callback_user_data = nullptr;
};

// the modalities of an mmproj file, a missing one is nullptr
struct mtmd_models {
    std::unique_ptr<mtmd_model> v;
    std::unique_ptr<mtmd_model> a;
    std::unique_ptr<mtmd_model> gen_a;
};

// the weights are allocated in buft, throws on error
mtmd_models mtmd_model_load(const char * fname, const mtmd_model_params & params, ggml_backend_buffer_type_t buft);

struct mtmd_model_caps {
    bool has_vision;
    bool has_audio;
    bool has_gen_audio;
};

// read the modalities of an mmproj file without loading its tensors
mtmd_model_caps mtmd_model_get_caps(const char * fname);
