#pragma once

#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "mtmd.h"

#include <array>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cinttypes>
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <memory>
#include <fstream>
#include <random>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// !!! Internal header, to be used by mtmd only !!!

#define MTMD_INTERNAL_HEADER

#define KEY_NAME                "general.name"
#define KEY_DESCRIPTION         "general.description"
#define KEY_PROJ_TYPE           "clip.projector_type"
#define KEY_HAS_AUDIO_ENC       "clip.has_audio_encoder"
#define KEY_HAS_VISION_ENC      "clip.has_vision_encoder"
#define KEY_HAS_GEN_AUDIO_ENC   "clip.has_gen_audio_encoder"
#define KEY_USE_GELU            "clip.use_gelu"
#define KEY_USE_SILU            "clip.use_silu"

#define KEY_N_EMBD              "clip.%s.embedding_length"
#define KEY_N_FF                "clip.%s.feed_forward_length"
#define KEY_N_BLOCK             "clip.%s.block_count"
#define KEY_PROJ_DIM            "clip.%s.projection_dim"
#define KEY_N_HEAD              "clip.%s.attention.head_count"
#define KEY_N_HEAD_KV           "clip.%s.attention.head_count_kv"
#define KEY_N_EMBD_HEAD         "clip.%s.attention.head_dim"
#define KEY_LAYER_NORM_EPS      "clip.%s.attention.layer_norm_epsilon"

// vision-specific
#define KEY_VISION_PROJ_TYPE        "clip.vision.projector_type" // for models with mixed modalities
#define KEY_IMAGE_SIZE              "clip.vision.image_size"
#define KEY_PATCH_SIZE              "clip.vision.patch_size"
#define KEY_IMAGE_MEAN              "clip.vision.image_mean"
#define KEY_IMAGE_STD               "clip.vision.image_std"
#define KEY_SPATIAL_MERGE_SIZE      "clip.vision.spatial_merge_size"

// audio-specific
#define KEY_AUDIO_PROJ_TYPE        "clip.audio.projector_type" // for models with mixed modalities
#define KEY_A_NUM_MEL_BINS         "clip.audio.num_mel_bins"
// audio generation (gen-audio)-specific
#define KEY_GEN_AUDIO_PROJ_TYPE    "clip.gen.audio.projector_type" // for models with mixed modalities

//
// tensor name constants
//

#define TN_POS_EMBD        "%s.position_embd.weight"
#define TN_CLASS_EMBD      "v.class_embd"
#define TN_CLASS_POS_EMBD  "v.class_position_embd.weight"
#define TN_PATCH_EMBD      "v.patch_embd.weight"  // not rename tensor with ".0" postfix for backward compat
#define TN_PATCH_BIAS      "v.patch_embd.bias"
#define TN_ATTN_QKV        "%s.blk.%d.attn_qkv.%s"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_ATTN_K_NORM     "%s.blk.%d.attn_k_norm.%s"
#define TN_ATTN_Q_NORM     "%s.blk.%d.attn_q_norm.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s" // layer norm
#define TN_LN_2            "%s.blk.%d.ln2.%s" // layer norm
#define TN_LS_1            "%s.blk.%d.ls1.%s"         // layer scale
#define TN_LS_2            "%s.blk.%d.ls2.%s"         // layer scale
#define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"
#define TN_LLAVA_PROJ      "mm.%d.%s"

// align x to upper multiple of n
#define MTMD_ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

enum projector_type {
    PROJECTOR_TYPE_INKLING2V,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    { PROJECTOR_TYPE_INKLING2V,         "inkling2v"},
};

static projector_type projector_type_from_string(const std::string & str) {
    for (const auto & pair : PROJECTOR_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return PROJECTOR_TYPE_UNKNOWN;
}

struct mtmd_image_size {
    int width;
    int height;
    bool operator==(const mtmd_image_size & other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const mtmd_image_size & other) const {
        return !(*this == other);
    }
    int area() const {
        // avoid overflow when computing area
        GGML_ASSERT(width  >= 0 && width  <= 46000);
        GGML_ASSERT(height >= 0 && height <= 46000);
        return width * height;
    }
};

// RGB uint8 image
struct mtmd_image_u8 {
    mtmd_image_size get_size() const {
        return { nx, ny };
    }

    void set_size(mtmd_image_size size, bool is_placeholder) {
        nx = size.width;
        ny = size.height;
        if (is_placeholder) {
            buf.clear();
        } else {
            buf.resize((size_t) nx * (size_t) ny * 3);
        }
    }

    void cpy_buf(const std::vector<uint8_t> & new_buf) {
        buf = new_buf;
    }

    const std::vector<uint8_t> & get_ro_buf() const {
        if (is_placeholder()) {
            throw std::runtime_error("this mtmd_image_u8 is a placeholder");
        }
        return buf;
    }

    // note to contributors: NEVER add a get_rw_buf(), it is a DANGEROUS pattern. always use get_pixel / set_pixel for buffer manipulation

    bool is_placeholder() const {
        return buf.empty();
    }

    std::array<uint8_t, 3> get_pixel(int x, int y) const {
        if (is_placeholder()) {
            // return a dummy value, so that legacy code can still process image without errors
            return { 0, 0, 0 };
        }
        size_t idx = ((size_t) y * (size_t) nx + (size_t) x) * 3;
        return { buf[idx], buf[idx + 1], buf[idx + 2] };
    }

    void set_pixel(int x, int y, const std::array<uint8_t, 3> & rgb) {
        if (is_placeholder()) {
            return; // no-op
        }
        size_t idx = ((size_t) y * (size_t) nx + (size_t) x) * 3;
        buf[idx]     = rgb[0];
        buf[idx + 1] = rgb[1];
        buf[idx + 2] = rgb[2];
    }

    size_t n_elements() const {
        return n_pixels() * 3;
    }

  private:
    std::vector<uint8_t> buf;
    int nx = 0;
    int ny = 0;

    size_t n_pixels() const {
        return (size_t) nx * (size_t) ny;
    }
};

struct mtmd_serialization; // forward declaration

// separators appended after the image tokens of one entry, as rows of v.tok_embd_sep
enum mtmd_suffix_type : int32_t {
    MTMD_SUFFIX_NONE = 0,
    // MiniCPM-V 4.7 tiles
    MTMD_SUFFIX_MINICPMV_OV,       // </image>
    MTMD_SUFFIX_MINICPMV_OV_SLICE, // </image><slice>
    MTMD_SUFFIX_MINICPMV_SLICE,    // </slice><slice>
    MTMD_SUFFIX_MINICPMV_ROW_END,  // </slice>\n<slice>
    MTMD_SUFFIX_MINICPMV_LAST,     // </slice>
    MTMD_SUFFIX_COUNT,
};

// rows of v.tok_embd_sep for each suffix type
// MiniCPM-V 4.7 rows (set by the converter): 0 = </image>, 1 = <slice>, 2 = </slice>, 3 = \n
static inline const std::vector<int> & mtmd_suffix_rows(mtmd_suffix_type type) {
    static const std::vector<int> none;
    static const std::vector<int> minicpmv_ov       = { 0 };
    static const std::vector<int> minicpmv_ov_slice = { 0, 1 };
    static const std::vector<int> minicpmv_slice    = { 2, 1 };
    static const std::vector<int> minicpmv_row_end  = { 2, 3, 1 };
    static const std::vector<int> minicpmv_last     = { 2 };
    switch (type) {
        case MTMD_SUFFIX_NONE:              return none;
        case MTMD_SUFFIX_MINICPMV_OV:       return minicpmv_ov;
        case MTMD_SUFFIX_MINICPMV_OV_SLICE: return minicpmv_ov_slice;
        case MTMD_SUFFIX_MINICPMV_SLICE:    return minicpmv_slice;
        case MTMD_SUFFIX_MINICPMV_ROW_END:  return minicpmv_row_end;
        case MTMD_SUFFIX_MINICPMV_LAST:     return minicpmv_last;
        default: GGML_ABORT("invalid suffix type");
    }
}

// For images, buf.size() == nx*ny*3
//     Memory layout: RGBRGBRGB...
// For seq, buf.size() == nx*ny*3*nt
//     Memory layout: RGBRGB...RGBRGB... (nt times)
// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct mtmd_image_f32 {
    // marks the global view in e.g., DeepSeek-OCR Models
    bool add_viewsep = false;
    // appends a learned newline (or EOI) token after the image
    // no model uses it now (Granite4 Vision moved to anyres), kept for future models
    bool add_newline = false;
    // deepseek4v: number of leading IMAGE_PAD embeddings, aligns IMAGE_START to the LLM compressor ratio
    // depends on the chunk position, set at tokenize time (see mtmd_tokenizer::add_media)
    int32_t lead_pad = 0;
    // audio (vibevoice): leading samples that only feed the causal convs, they produce no embedding
    int32_t audio_n_ctx = 0;
    // separators appended after the image tokens
    mtmd_suffix_type suffix_type = MTMD_SUFFIX_NONE;

    // tile grid of the image group this entry belongs to
    // llava-next "anyres" (Granite4 Vision): the whole grid is encoded and assembled in a single graph
    // MiniCPM-V 4.7: set on the overview entry, the decoder positions of all tiles are derived from it
    struct anyres_info {
        int grid_x = 0; // tiles per row, 0 means the image is not tiled
        int grid_y = 0; // tiles per column
        int orig_nx = 0; // size of the source image, used to drop the padding tokens
        int orig_ny = 0;

        bool is_tiled() const {
            return grid_x > 0 && grid_y > 0;
        }
    };
    anyres_info anyres;

    mtmd_image_size get_size() const {
        return { nx_, ny_ };
    }

    int nx() const { return nx_; }
    int ny() const { return ny_; }

    void set_size(mtmd_image_size size, bool is_placeholder, bool is_audio) {
        nx_ = size.width;
        ny_ = size.height;
        if (is_placeholder) {
            buf.clear();
        } else {
            if (is_audio) {
                buf.resize((size_t) nx_ * (size_t) ny_);
            } else {
                buf.resize((size_t) nx_ * (size_t) ny_ * 3);
            }
        }
    }

    void cpy_buf(const std::vector<float> & new_buf) {
        buf = new_buf;
    }

    void from_u8(const mtmd_image_u8 & img) {
        auto size = img.get_size();
        nx_ = size.width;
        ny_ = size.height;
        if (img.is_placeholder()) {
            buf.clear();
            return; // no-op
        }
        buf.resize(img.n_elements());
        const auto & u8_buf = img.get_ro_buf();
        for (size_t i = 0; i < img.n_elements(); ++i) {
            buf[i] = (float) u8_buf[i] / 255.0f;
        }
    }

    size_t n_elements() const {
        return n_pixels() * 3;
    }

    void normalize(const float mean[3], const float std[3]) {
        if (is_placeholder()) {
            return; // no-op
        }
        for (size_t i = 0; i < n_pixels(); ++i) {
            buf[i * 3 + 0] = (buf[i * 3 + 0] - mean[0]) / std[0];
            buf[i * 3 + 1] = (buf[i * 3 + 1] - mean[1]) / std[1];
            buf[i * 3 + 2] = (buf[i * 3 + 2] - mean[2]) / std[2];
        }
    }

    const std::vector<float> & get_ro_buf() const {
        if (is_placeholder()) {
            throw std::runtime_error("this mtmd_image_f32 is a placeholder");
        }
        return buf;
    }

    // note to contributors: NEVER add a get_rw_buf(), it is a DANGEROUS pattern

    bool is_placeholder() const {
        return buf.empty();
    }

    void serialize(struct mtmd_serialization & ser) const;
    void deserialize(struct mtmd_serialization & ser);

  private:
    std::vector<float> buf;
    int nx_ = 0;
    int ny_ = 0;

    size_t n_pixels() const {
        return (size_t) nx_ * (size_t) ny_;
    }
};

//
// logging
//

static void mtmd_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct mtmd_logger_state {
    ggml_log_callback log_callback;
    void * log_callback_user_data;
};

extern struct mtmd_logger_state g_logger_state;

static void mtmd_log_internal_v(enum ggml_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

static void mtmd_log_internal(enum ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    mtmd_log_internal_v(level, format, args);
    va_end(args);
}

#define LOG_TRC(...) mtmd_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_DBG(...) mtmd_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) mtmd_log_internal(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) mtmd_log_internal(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) mtmd_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_CNT(...) mtmd_log_internal(GGML_LOG_LEVEL_CONT,  __VA_ARGS__)

//
// cpp wrappers
//

struct mtmd_image_f32_batch {
    std::vector<mtmd_image_f32> entries;
    bool is_audio = false;
    bool is_video = false;

    mtmd_image_f32_batch clone() const {
        mtmd_image_f32_batch new_batch{
            /* entries  */ {},
            /* is_audio */ is_audio,
            /* is_video */ is_video,
        };
        new_batch.entries.reserve(entries.size());
        for (const auto & entry : entries) {
            new_batch.entries.emplace_back(entry); // copy
        }
        return new_batch;
    }

    void serialize(struct mtmd_serialization & ser) const;
    void deserialize(struct mtmd_serialization & ser);
};

//
// common utils
//

#ifdef _WIN32
static std::ifstream open_ifstream_binary(const std::string & fname) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
    if (!wlen) {
        throw std::runtime_error("failed to convert filename to UTF-16: " + fname);
    }
    std::vector<wchar_t> wfname(wlen);
    (void)MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wfname.data(), wlen);
    return std::ifstream(wfname.data(), std::ios::binary);
}
#else
static std::ifstream open_ifstream_binary(const std::string & fname) {
    return std::ifstream(fname, std::ios::binary);
}
#endif

std::string mtmd_format(const char * fmt, ...);

// in test-mtmd-impl, we include woth common.h and this file, and these functions are duplicated
// this is a quick fix to avoid compilation errors
#ifndef DIRECTORY_SEPARATOR
static void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

// split string by a `std::string delim` instead of `char delim`
static std::vector<std::string> string_split_str(std::string s, const std::string & delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);
    return tokens;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

// remove when moving to c++20
inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
#endif

//
// gguf utils
//

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const int8_t *)data)[i] != 0 ? "true" : "false";
        default:                return mtmd_format("unknown type %d", type);
    }
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        string_replace_all(val, "\\", "\\\\");
                        string_replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

//
// debugging
//

static void print_tensor_shape(ggml_tensor * t) {
    printf("%s.shape = [", t->name);
    for (int i = 0; i < ggml_n_dims(t); ++i) {
        printf("%" PRId64, t->ne[i]);
        if (i < ggml_n_dims(t) - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

static void print_tensor_data(ggml_tensor * t, uint8_t * data, int64_t n) {
    ggml_type type = t->type;
    int64_t * ne = t->ne;
    size_t * nb = t->nb;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("%s.data: [\n", t->name);
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("     ..., \n");
                i2 = ne[2] - n;
            }
            printf("     [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("      ..., \n");
                    i1 = ne[1] - n;
                }
                printf("      [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    printf("%8.4f", v);
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("     ],\n");
        }
        printf("    ]\n");
    }
}

