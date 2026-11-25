#include "mtmd-impl.h"
#include "mtmd-model.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cinttypes>
#include <limits>
#include <array>
#include <functional>
#include <float.h>

//
// mtmd_model_loader
//

struct mtmd_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_vision    = false;
    bool has_audio     = false;
    bool has_gen_audio = false;

    mtmd_progress_callback progress_callback = nullptr;
    void * progress_callback_user_data = nullptr;

    mtmd_model_loader(const char * fname,
            bool skip_tensors = false,
            mtmd_progress_callback progress_cb = nullptr,
            void * progress_user_data = nullptr)
        : fname(fname),
          progress_callback(progress_cb),
          progress_callback_user_data(progress_user_data) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(mtmd_format("%s: failed to load mmproj model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n",  __func__, name.c_str());
            LOG_INF("%s: description:  %s\n",  __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n",  __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n",  __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n",  __func__, (int)gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // modalities
        {
            get_bool(KEY_HAS_VISION_ENC,    has_vision,    false);
            get_bool(KEY_HAS_AUDIO_ENC,     has_audio,     false);
            get_bool(KEY_HAS_GEN_AUDIO_ENC, has_gen_audio, false);

            if (has_vision) {
                LOG_INF("%s: has vision encoder\n", __func__);
            }
            if (has_audio) {
                LOG_INF("%s: has audio encoder\n", __func__);
            }
            if (has_gen_audio) {
                LOG_INF("%s: has audio generation (gen) encoder\n", __func__);
            }
        }

        // tensors
        if (!skip_tensors) {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams(mtmd_model & model, mtmd_modality modality) {
        auto & hparams = model.hparams;
        std::string log_ffn_op; // for logging

        // sanity check
        if (modality == MTMD_MODALITY_VISION) {
            GGML_ASSERT(has_vision);
        } else if (modality == MTMD_MODALITY_AUDIO) {
            GGML_ASSERT(has_audio);
        } else if (modality == MTMD_MODALITY_GEN_AUDIO) {
            GGML_ASSERT(has_gen_audio);
        }
        model.modality = modality;

        // projector type
        std::string proj_type;
        {
            // default key
            get_string(KEY_PROJ_TYPE, proj_type, false);

            // for models with mixed modalities
            if (proj_type.empty()) {
                if (modality == MTMD_MODALITY_VISION) {
                    get_string(KEY_VISION_PROJ_TYPE, proj_type, false);
                } else if (modality == MTMD_MODALITY_AUDIO) {
                    get_string(KEY_AUDIO_PROJ_TYPE, proj_type, false);
                } else if (modality == MTMD_MODALITY_GEN_AUDIO) {
                    get_string(KEY_GEN_AUDIO_PROJ_TYPE, proj_type, false);
                } else {
                    GGML_ABORT("unknown modality");
                }
            }

            model.proj_type = projector_type_from_string(proj_type);

            if (model.proj_type == PROJECTOR_TYPE_UNKNOWN) {
                throw std::runtime_error(mtmd_format("%s: unknown projector type: %s\n", __func__, proj_type.c_str()));
            }

            // correct arch for multimodal models (legacy method)
        }

        const bool is_vision    = model.modality == MTMD_MODALITY_VISION;
        const bool is_audio     = model.modality == MTMD_MODALITY_AUDIO;
        const bool is_gen_audio = model.modality == MTMD_MODALITY_GEN_AUDIO;

        // other hparams
        {
            const char * prefix = is_vision ? "vision" : (is_audio ? "audio" : "gen.audio");
            get_u32(mtmd_format(KEY_N_EMBD,         prefix), hparams.n_embd);
            get_u32(mtmd_format(KEY_N_HEAD,         prefix), hparams.n_head);
            get_u32(mtmd_format(KEY_N_EMBD_HEAD,    prefix), hparams.n_embd_head, false);
            get_u32(mtmd_format(KEY_N_FF,           prefix), hparams.n_ff);
            get_u32(mtmd_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
            get_u32(mtmd_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
            get_f32(mtmd_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps);

            // n_head_kv is optional (for GQA), default to n_head
            hparams.n_head_kv = hparams.n_head;
            get_u32(mtmd_format(KEY_N_HEAD_KV, prefix), hparams.n_head_kv, false);

            if (is_vision) {
                get_u32(KEY_IMAGE_SIZE, hparams.image_size);
                get_u32(KEY_PATCH_SIZE, hparams.patch_size);
            } else if (is_audio) {
                get_u32(KEY_A_NUM_MEL_BINS, hparams.n_mel_bins);
                // some hparams are unused, but still need to set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;

            } else if (is_gen_audio) {
                // these are unused, but still need to be set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;

            } else {
                GGML_ASSERT(false && "unknown modality");
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            {
                bool use_gelu = false;
                bool use_silu = false;
                get_bool(KEY_USE_GELU, use_gelu, false);
                get_bool(KEY_USE_SILU, use_silu, false);
                if (use_gelu && use_silu) {
                    throw std::runtime_error(mtmd_format("%s: both use_gelu and use_silu are set to true\n", __func__));
                }
                if (use_gelu) {
                    hparams.ffn_op = FFN_GELU;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = FFN_SILU;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = FFN_GELU_QUICK;
                    log_ffn_op = "gelu_quick";
                }
            }

            if (is_vision) {
                std::vector<float> image_mean;
                std::vector<float> image_std;
                get_arr_f32(KEY_IMAGE_MEAN, image_mean, false);
                get_arr_f32(KEY_IMAGE_STD , image_std, false);
                if (image_mean.size() < 3 || image_std.size() < 3) {
                    throw std::runtime_error(mtmd_format("%s: image_mean/image_std arrays must have at least 3 elements, got %zu and %zu\n", __func__, image_mean.size(), image_std.size()));
                }
                for (int i = 0; i < 3; ++i) {
                    hparams.image_mean[i] = image_mean[i];
                    hparams.image_std[i]  = image_std[i];
                }
            }

            // model-specific params
            switch (model.proj_type) {
                case PROJECTOR_TYPE_INKLING2V:
                    {
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf";
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge);
                        hparams.rope_theta = 10000.0f;
                        hparams.set_limit_image_tokens(1, 4096);
                        hparams.set_warmup_n_tokens(32*32); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_QWEN3VL:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_resize_pad  = PAD_NONE; // the HF processor resizes to the smart_resize size, no padding
                        hparams.image_round_half_even = true; // smart_resize uses Python round()
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        if (hparams.n_merge != 2) {
                            // the temporal merge input and the merger use a 2x2 patch grid
                            throw std::runtime_error(mtmd_format("%s: spatial_merge_size (%d) must be 2\n", __func__, hparams.n_merge));
                        }
                        // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
                        hparams.set_limit_image_tokens(8, 4096);
                        // optional limits of the model, the custom values take precedence
                        if (hparams.custom_image_min_tokens <= 0) {
                            get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels, false);
                        }
                        if (hparams.custom_image_max_tokens <= 0) {
                            get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels, false);
                        }
                        if (hparams.image_min_pixels <= 0 || hparams.image_max_pixels <= 0) {
                            throw std::runtime_error(mtmd_format("%s: image_min_pixels (%d) and image_max_pixels (%d) must be positive\n", __func__, hparams.image_min_pixels, hparams.image_max_pixels));
                        }
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                        const int warn_min_pixels = 1024 * hparams.n_merge * hparams.n_merge * hparams.patch_size * hparams.patch_size;
                        if (hparams.image_min_pixels < warn_min_pixels) {
                            LOG_WRN("%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n", __func__);
                            LOG_WRN("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n", __func__);
                            LOG_WRN("%s: more info: https://github.com/ggml-org/llama.cpp/issues/16842\n\n", __func__);
                        }
                    } break;
                default:
                    throw std::runtime_error(mtmd_format("%s: unknown vision projector type %s\n", __func__, proj_type.c_str()));
            }

            // sanity check
            {
                if (hparams.image_size < 0) {
                    // note: some models having hparams.image_size == 0, which means the image size is dynamic
                    throw std::runtime_error(mtmd_format("%s: image_size (%d) cannot be negative\n", __func__, hparams.image_size));
                }
                if (hparams.image_size > 8192) {
                    // cap prevents int32 overflow in n_patches = (image_size/patch_size)^2
                    throw std::runtime_error(mtmd_format("%s: image_size (%d) is too large (max 8192)\n", __func__, hparams.image_size));
                }
                if (hparams.patch_size <= 0 || hparams.patch_size >= 65536) {
                    throw std::runtime_error(mtmd_format("%s: patch_size (%d) must be positive and less than 65536\n", __func__, hparams.patch_size));
                }
                if (hparams.n_embd <= 0) {
                    throw std::runtime_error(mtmd_format("%s: n_embd (%d) must be greater than 0\n", __func__, hparams.n_embd));
                }
                if (hparams.image_max_pixels < hparams.image_min_pixels) {
                    throw std::runtime_error(mtmd_format("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n", __func__, hparams.image_max_pixels, hparams.image_min_pixels));
                }
                if (hparams.n_merge <= 0 || hparams.n_merge >= 65536) {
                    throw std::runtime_error(mtmd_format("%s: n_merge (%d) must be greater than 0 and less than 65536\n", __func__, hparams.n_merge));
                }
            }

            LOG_INF("%s: projector:          %s\n", __func__, proj_type.c_str());
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            if (is_vision) {
                LOG_INF("\n--- vision hparams ---\n");
                LOG_INF("%s: image_size:         %d\n", __func__, hparams.image_size);
                LOG_INF("%s: patch_size:         %d\n", __func__, hparams.patch_size);
                LOG_INF("%s: n_merge:            %d\n", __func__, hparams.n_merge);
                if (hparams.image_min_pixels > 0) {
                    LOG_INF("%s: image_min_pixels:   %d%s\n", __func__, hparams.image_min_pixels, hparams.custom_image_min_tokens > 0 ? " (custom value)" : "");
                }
                if (hparams.image_max_pixels > 0) {
                    LOG_INF("%s: image_max_pixels:   %d%s\n", __func__, hparams.image_max_pixels, hparams.custom_image_max_tokens > 0 ? " (custom value)" : "");
                }
            } else if (is_audio) {
                LOG_INF("\n--- audio hparams ---\n");
                LOG_INF("%s: n_mel_bins:         %d\n", __func__, hparams.n_mel_bins);
                LOG_INF("%s: audio_sample_rate:  %d\n", __func__, hparams.audio_sample_rate);
                LOG_INF("%s: audio_hop_len:      %d\n", __func__, hparams.audio_hop_len);

                // Validate audio hparams loaded from GGUF metadata
                if (hparams.n_mel_bins <= 0 || hparams.n_mel_bins > 256) {
                    throw std::runtime_error(mtmd_format("%s: n_mel_bins (%d) must be in range [1, 256]\n", __func__, hparams.n_mel_bins));
                }
                if (hparams.audio_sample_rate <= 0 || hparams.audio_hop_len <= 0) {
                    throw std::runtime_error(mtmd_format("%s: audio hparams invalid: sample_rate=%d hop_len=%d\n",
                        __func__, hparams.audio_sample_rate, hparams.audio_hop_len));
                }
            }
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(mtmd_model & model, ggml_backend_buffer_type_t buft, bool no_alloc) {
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;

        auto fin = open_ifstream_binary(fname);
        if (!fin) {
            throw std::runtime_error(mtmd_format("%s: failed to open %s\n", __func__, fname.c_str()));
        }

        // TODO @ngxson : support both audio and video in the future
        const char * prefix = model.modality == MTMD_MODALITY_AUDIO ? "a"
                             : model.modality == MTMD_MODALITY_GEN_AUDIO ? "a.gen.code"
                             : "v";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        model.ctx_data.reset(ggml_init(params));
        if (!model.ctx_data) {
            throw std::runtime_error(mtmd_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        std::unordered_set<std::string> loaded_tensor_names;
        auto get_tensor = [&](const std::string & name, bool required = true) {
            // Each tensor should only be loaded once; duplicates indicate a bug
            if (loaded_tensor_names.count(name)) {
                throw std::runtime_error(mtmd_format("%s: tensor already loaded: %s\n", __func__, name.c_str()));
            }
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(mtmd_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                ggml_tensor * data_tensor = ggml_dup_tensor(model.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                loaded_tensor_names.insert(name);
                cur = data_tensor;
                // add to weight memory counter
                model.mem_weights[ggml_backend_buft_get_device(buft)] += ggml_nbytes(cur);
            }
            return cur;
        };

        model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        model.pre_ln_w = get_tensor(mtmd_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(mtmd_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(mtmd_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(mtmd_format(TN_LN_POST, prefix, "bias"),   false);

        // every vision encoder starts with a patch embedding
        const bool has_patch_embd = model.modality == MTMD_MODALITY_VISION;

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   has_patch_embd);
        model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);

        model.position_embeddings = get_tensor(mtmd_format(TN_POS_EMBD, prefix), false);

        // layers
        const int n_layers_to_load = hparams.n_layer;
        model.layers.resize(n_layers_to_load);
        for (int il = 0; il < n_layers_to_load; ++il) {
            auto & layer = model.layers[il];
            const bool o_w_required = true;
            layer.k_w    = get_tensor(mtmd_format(TN_ATTN_K,      prefix, il, "weight"), false);
            layer.q_w    = get_tensor(mtmd_format(TN_ATTN_Q,      prefix, il, "weight"), false);
            layer.v_w    = get_tensor(mtmd_format(TN_ATTN_V,      prefix, il, "weight"), false);
            layer.o_w    = get_tensor(mtmd_format(TN_ATTN_OUTPUT, prefix, il, "weight"), o_w_required);
            layer.qkv_w  = get_tensor(mtmd_format(TN_ATTN_QKV,    prefix, il, "weight"), false);
            layer.k_norm = get_tensor(mtmd_format(TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = get_tensor(mtmd_format(TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = get_tensor(mtmd_format(TN_LN_1,        prefix, il, "weight"), false);
            layer.ln_2_w = get_tensor(mtmd_format(TN_LN_2,        prefix, il, "weight"), false);
            layer.ls_1_w        = get_tensor(mtmd_format(TN_LS_1,         prefix, il, "weight"), false); // no bias
            layer.ls_2_w        = get_tensor(mtmd_format(TN_LS_2,         prefix, il, "weight"), false); // no bias

            layer.k_b    = get_tensor(mtmd_format(TN_ATTN_K,      prefix, il, "bias"), false);
            layer.q_b    = get_tensor(mtmd_format(TN_ATTN_Q,      prefix, il, "bias"), false);
            layer.v_b    = get_tensor(mtmd_format(TN_ATTN_V,      prefix, il, "bias"), false);
            layer.o_b    = get_tensor(mtmd_format(TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.qkv_b  = get_tensor(mtmd_format(TN_ATTN_QKV,    prefix, il, "bias"), false);
            layer.ln_1_b = get_tensor(mtmd_format(TN_LN_1,        prefix, il, "bias"), false);
            layer.ln_2_b = get_tensor(mtmd_format(TN_LN_2,        prefix, il, "bias"), false);

            // ffn
            layer.ff_up_w   = get_tensor(mtmd_format(TN_FFN_UP,   prefix, il, "weight"));
            layer.ff_up_b   = get_tensor(mtmd_format(TN_FFN_UP,   prefix, il, "bias"),   false);
            layer.ff_gate_w = get_tensor(mtmd_format(TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = get_tensor(mtmd_format(TN_FFN_GATE, prefix, il, "bias"),   false);
            layer.ff_down_w = get_tensor(mtmd_format(TN_FFN_DOWN, prefix, il, "weight"));
            layer.ff_down_b = get_tensor(mtmd_format(TN_FFN_DOWN, prefix, il, "bias"),   false);

            // qwen3vl deepstack layer
            layer.deepstack_norm_w = get_tensor(mtmd_format(TN_DEEPSTACK_NORM, il, "weight"), false);
            layer.deepstack_norm_b = get_tensor(mtmd_format(TN_DEEPSTACK_NORM, il, "bias"), false);
            layer.deepstack_fc1_w  = get_tensor(mtmd_format(TN_DEEPSTACK_FC1,  il, "weight"), false);
            layer.deepstack_fc1_b  = get_tensor(mtmd_format(TN_DEEPSTACK_FC1,  il, "bias"), false);
            layer.deepstack_fc2_w  = get_tensor(mtmd_format(TN_DEEPSTACK_FC2,  il, "weight"), false);
            layer.deepstack_fc2_b  = get_tensor(mtmd_format(TN_DEEPSTACK_FC2,  il, "bias"), false);
            if (layer.has_deepstack()) {
                model.n_deepstack_layers++;
            }
        }

        switch (model.proj_type) {
            case PROJECTOR_TYPE_INKLING2V:
                {
                    // these come from the generic path above, where they are optional
                    if (!model.class_embedding || !model.position_embeddings || !model.pre_ln_w || !model.pre_ln_b) {
                        throw std::runtime_error(mtmd_format("%s: missing class_embd, position_embd or pre_ln\n", __func__));
                    }
                    model.class_position_embedding = get_tensor(TN_CLASS_POS_EMBD);
                    model.mm_1_w = get_tensor(mtmd_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(mtmd_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(mtmd_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(mtmd_format(TN_LLAVA_PROJ, 2, "bias"), false);
                } break;
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    // these come from the generic path above, where they are optional
                    if (!model.patch_embeddings_1 || !model.patch_bias || !model.position_embeddings || model.class_embedding) {
                        throw std::runtime_error(mtmd_format("%s: missing patch_embd.weight.1, patch_embd.bias or position_embd, or unexpected class_embd\n", __func__));
                    }
                    // the fused qkv is split at n_embd, the M-RoPE sections are d_head/4 each
                    if (hparams.n_head <= 0 || hparams.n_embd % (hparams.n_head * 4) != 0 ||
                            (hparams.n_embd_head != 0 && hparams.n_embd_head * hparams.n_head != hparams.n_embd)) {
                        throw std::runtime_error(mtmd_format("%s: n_embd (%d) must be n_head (%d) x head_dim, with head_dim a multiple of 4\n", __func__, hparams.n_embd, hparams.n_head));
                    }
                    for (const auto & layer : model.layers) {
                        if (!layer.qkv_w || !layer.qkv_b) {
                            throw std::runtime_error(mtmd_format("%s: missing fused attn_qkv weight or bias\n", __func__));
                        }
                    }
                    model.mm_0_w = get_tensor(mtmd_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(mtmd_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(mtmd_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(mtmd_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            default:
                GGML_ASSERT(false && "unknown projector type");
        }

        // load data
        {
            std::vector<uint8_t> read_buf;

            // start loading event
            if (progress_callback){
                progress_callback(0.0, progress_callback_user_data);
            }

            // compute total tensor data size for progress reporting
            size_t total_data_size = 0;
            for (auto & t : tensors_to_load) {
                total_data_size += ggml_nbytes(t);
            }

            // alloc memory and offload data
            model.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(model.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // read the weight from file
            if (!no_alloc) {
                size_t data_loaded = 0;
                for (auto & t : tensors_to_load) {
                    ggml_tensor * cur = ggml_get_tensor(model.ctx_data.get(), t->name);
                    GGML_ASSERT(cur && "tensor not found in ctx_data");
                    auto it_off = tensor_offset.find(t->name);
                    GGML_ASSERT(it_off != tensor_offset.end() && "no offset for tensor");
                    const size_t offset = it_off->second;
                    fin.seekg(offset, std::ios::beg);
                    if (!fin) {
                        throw std::runtime_error(mtmd_format("%s: failed to seek for tensor %s\n", __func__, t->name));
                    }
                    size_t num_bytes = ggml_nbytes(cur);
                    if (ggml_backend_buft_is_host(buft)) {
                        // for the CPU and Metal backend, we can read directly into the tensor
                        fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                    } else {
                        // read into a temporary buffer first, then copy to device memory
                        read_buf.resize(num_bytes);
                        fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                        ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                    }
                    data_loaded += num_bytes;
                    if (progress_callback && total_data_size > 0) {
                        const float progress = (float)data_loaded / (float)total_data_size;
                        if (!progress_callback(progress, progress_callback_user_data)) {
                            throw std::runtime_error(mtmd_format("%s: model loading cancelled by progress_callback\n", __func__));
                        }
                    }
                }
                LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
            } else {
                LOG_DBG("%s: no_alloc is set, skipping tensor data loading (%zu tensors)\n", __func__, tensors_to_load.size());
            }
            fin.close();
        }

    }

    void get_bool(const std::string & key, bool & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        const uint32_t val = gguf_get_val_u32(ctx_gguf.get(), i);
        // sanity check
        if (val > (uint32_t) INT32_MAX) {
            throw std::runtime_error(mtmd_format("%s: value %u for key '%s' exceeds INT32_MAX\n",
                __func__, val, key.c_str()));
        }
        output = (int) val;
    }

    void get_f32(const std::string & key, float & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_arr_f32(const std::string & key, std::vector<float> & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_ARRAY) {
            throw std::runtime_error(mtmd_format("%s: key '%s' is not an array\n", __func__, key.c_str()));
        }
        const auto type = gguf_get_arr_type(ctx_gguf.get(), i);
        if (type != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error(mtmd_format("%s: array '%s' has type %d, expected %d (GGUF_TYPE_FLOAT32)\n", __func__, key.c_str(), type, GGUF_TYPE_FLOAT32));
        }
        const size_t n = gguf_get_arr_n(ctx_gguf.get(), i);
        if (n > (size_t) std::numeric_limits<int>::max()) {
            throw std::runtime_error(mtmd_format("%s: array '%s' is too large (%zu elements)\n", __func__, key.c_str(), n));
        }
        output.resize(n);
        const float * values = (const float *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (size_t j = 0; j < n; ++j) {
            output[j] = values[j];
        }
    }

    void get_string(const std::string & key, std::string & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }
};

//
// loading
//

mtmd_models mtmd_model_load(const char * fname, const mtmd_model_params & params, ggml_backend_buffer_type_t buft) {
    mtmd_model_loader loader(fname,
        /* skip_tensors */ false,
        params.progress_callback,
        params.progress_callback_user_data);

    auto load = [&](mtmd_modality modality) {
        auto model = std::make_unique<mtmd_model>();
        model->hparams.custom_image_min_tokens = params.image_min_tokens;
        model->hparams.custom_image_max_tokens = params.image_max_tokens;
        loader.load_hparams(*model, modality);
        loader.load_tensors(*model, buft, params.no_alloc);
        return model;
    };

    mtmd_models res;
    if (loader.has_vision) {
        res.v = load(MTMD_MODALITY_VISION);
    }

    if (loader.has_audio) {
        res.a = load(MTMD_MODALITY_AUDIO);
    }

    if (loader.has_gen_audio) {
        res.gen_a = load(MTMD_MODALITY_GEN_AUDIO);
    }

    return res;
}

mtmd_model_caps mtmd_model_get_caps(const char * fname) {
    mtmd_model_caps res;
    mtmd_model_loader loader(fname, /* skip_tensors= */ true);
    res.has_vision    = loader.has_vision;
    res.has_audio     = loader.has_audio;
    res.has_gen_audio = loader.has_gen_audio;
    return res;
}

//
// mtmd_model
//

int mtmd_model::image_max_tokens() const {
    if (hparams.image_max_pixels <= 0) {
        return -1;
    }
    const int patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
    return hparams.image_max_pixels / patch_area;
}

int mtmd_model::n_output_tokens_x(const mtmd_image_f32 * img) const {
    const auto & params = hparams;
    const int n_total = n_output_tokens(img);
    const auto & proj = proj_type;
    switch (proj) {
        case PROJECTOR_TYPE_QWEN3VL:
            return (img->nx() / params.patch_size) / 2;
        case PROJECTOR_TYPE_INKLING2V:
            return img->nx() / (params.patch_size * params.n_merge);
        default:
            break;
    }
    return n_total;
}

int mtmd_model::n_output_tokens_y(const mtmd_image_f32 * img) const {
    const auto & params = hparams;
    const auto & proj = proj_type;
    switch (proj) {
        case PROJECTOR_TYPE_QWEN3VL:
            return (img->ny() / params.patch_size) / 2;
        case PROJECTOR_TYPE_INKLING2V:
            return img->ny() / (params.patch_size * params.n_merge);
        default:
            break;
    }
    return 1;
}

int mtmd_model::n_output_tokens(const mtmd_image_f32 * img) const {
    const auto & params = hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    int patch_size = params.patch_size;
    int n_patches = (img->nx() / patch_size) * (img->ny() / patch_size);

    projector_type proj = proj_type;

    switch (proj) {
        case PROJECTOR_TYPE_INKLING2V:
            {
                int x_patch = img->nx() / (params.patch_size * params.n_merge);
                int y_patch = img->ny() / (params.patch_size * params.n_merge);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_QWEN3VL:
            {
                // dynamic size (2 conv, so double patch size)
                int x_patch = img->nx() / (params.patch_size * 2);
                int y_patch = img->ny() / (params.patch_size * 2);
                n_patches = x_patch * y_patch;
            } break;
        default:
            GGML_ABORT("unsupported projector type");
    }

    n_patches += (int) mtmd_suffix_rows(img->suffix_type).size();

    return n_patches;
}

int mtmd_model::n_mmproj_embd() const {
    switch (proj_type) {
        case PROJECTOR_TYPE_INKLING2V:
            return mm_2_w->ne[1];
        case PROJECTOR_TYPE_QWEN3VL:
            // main path + deepstack paths
            return mm_1_b->ne[0] * (1 + n_deepstack_layers);
        default:
            GGML_ABORT("Unknown projector type");
    }
}

// TODO @ngxson : this is no longer correct with mtmd_batch API
// this was only meant to be used by qwen-vl-based models, to fuse 2 input images into one (qwen-vl video support)
// this logic should be refactored in near future to distinctly handle "merge frames" and "batching"
int mtmd_model::n_temporal_merge() const {
    switch (proj_type) {
        case PROJECTOR_TYPE_QWEN3VL:
            return 2;
        default:
            return 1;
    }
}
