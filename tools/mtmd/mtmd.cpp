#include "mtmd-subcontext.h"
#include "models/models.h"
#include "mtmd-impl.h"
#include "mtmd.h"
#include "mtmd-internal.h"
#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "debug/mtmd-debug.h"

#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <map>
#include <mutex>
#include <type_traits>
#include <vector>

std::string mtmd_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

// remember to bump this if the serialization format changes
#define MTMD_SERIALIZATION_VERSION 5

// oldest compat version that can be loaded
#define MTMD_SERIALIZATION_VERSION_MIN 2

struct mtmd_serialization {
    // note: using 64-bit here for future-proofing
    uint64_t version = MTMD_SERIALIZATION_VERSION;
    std::vector<char> data;
    size_t read_pos = 0; // cursor used when reading

    // for writing
    mtmd_serialization(uint64_t version) : version(version) {
        write(version);
    }

    // for reading
    mtmd_serialization(uint64_t version, const char * buf, size_t len) {
        // copy buf to data
        data.assign(buf, buf + len);
        uint64_t ver_in = read<uint64_t>();
        if (ver_in < MTMD_SERIALIZATION_VERSION_MIN || ver_in > version) {
            throw std::runtime_error("version mismatch");
        }
        this->version = ver_in;
    }

    template <typename T>
    void write(T value) {
        static_assert(std::is_trivially_copyable<T>::value && !std::is_same<T, bool>::value,
            "T must be trivially copyable and not bool");
        const char * p = reinterpret_cast<const char *>(&value);
        data.insert(data.end(), p, p + sizeof(T));
    }

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable<T>::value && !std::is_same<T, bool>::value,
            "T must be trivially copyable and not bool");
        if (read_pos + sizeof(T) > data.size()) {
            throw std::runtime_error("read OOB");
        }
        T value;
        std::memcpy(&value, data.data() + read_pos, sizeof(T));
        read_pos += sizeof(T);
        return value;
    }

};

template <>
void mtmd_serialization::write<bool>(bool value) {
    write<uint8_t>(value ? 1 : 0);
}
template <>
bool mtmd_serialization::read<bool>() {
    return read<uint8_t>() != 0;
}

template <>
void mtmd_serialization::write<std::string>(std::string value) {
    write<uint64_t>(value.size());
    data.insert(data.end(), value.begin(), value.end());
}
template <>
std::string mtmd_serialization::read<std::string>() {
    uint64_t len = read<uint64_t>();
    if (read_pos + len > data.size()) {
        throw std::runtime_error("read_string OOB");
    }
    std::string str(data.data() + read_pos, len);
    read_pos += len;
    return str;
}

// only mtmd.cpp needs these, so they're implemented here rather than in mtmd-impl.h
void mtmd_image_f32::serialize(mtmd_serialization & ser) const {
    // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
    // note: buf is intentionally NOT serialized; the loaded mtmd_image_f32 will always be a placeholder
    ser.write(add_viewsep);
    ser.write(add_newline);
    ser.write(lead_pad);
    ser.write((int32_t)suffix_type);
    ser.write((int32_t)anyres.grid_x);
    ser.write((int32_t)anyres.grid_y);
    ser.write((int32_t)anyres.orig_nx);
    ser.write((int32_t)anyres.orig_ny);
    ser.write((int32_t)nx_);
    ser.write((int32_t)ny_);
    ser.write(audio_n_ctx);
}
void mtmd_image_f32::deserialize(mtmd_serialization & ser) {
    add_viewsep = ser.read<bool>();
    add_newline = ser.read<bool>();
    lead_pad = ser.read<int32_t>();
    if (ser.version >= 3) {
        const int32_t suffix_raw = ser.read<int32_t>();
        if (suffix_raw < 0 || suffix_raw >= MTMD_SUFFIX_COUNT) {
            throw std::runtime_error("invalid suffix type");
        }
        suffix_type = (mtmd_suffix_type)suffix_raw;
        anyres.grid_x = ser.read<int32_t>();
        anyres.grid_y = ser.read<int32_t>();
        anyres.orig_nx = ser.read<int32_t>();
        anyres.orig_ny = ser.read<int32_t>();
    }
    nx_ = ser.read<int32_t>();
    ny_ = ser.read<int32_t>();
    if (ser.version >= 5) {
        audio_n_ctx = ser.read<int32_t>();
        if (audio_n_ctx < 0 || audio_n_ctx > nx_) {
            throw std::runtime_error("invalid audio_n_ctx");
        }
    }
    buf.clear(); // always a placeholder after loading
}

void mtmd_image_f32_batch::serialize(mtmd_serialization & ser) const {
    // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
    ser.write(is_audio);
    ser.write<uint64_t>(entries.size());
    for (const auto & entry : entries) {
        entry.serialize(ser);
    }
}
void mtmd_image_f32_batch::deserialize(mtmd_serialization & ser) {
    is_audio = ser.read<bool>();
    uint64_t n = ser.read<uint64_t>();
    constexpr size_t min_entry_bytes = sizeof(uint8_t) * 2 + sizeof(int32_t) * 2;
    if (n > (ser.data.size() - ser.read_pos) / min_entry_bytes) {
        throw std::runtime_error("entries count exceeds buffer size");
    }
    entries.clear();
    entries.reserve(n);
    for (uint64_t i = 0; i < n; i++) {
        mtmd_image_f32 entry;
        entry.deserialize(ser);
        entries.push_back(std::move(entry));
    }
}

// for still image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3 bytes
//
// for audio bitmap: nx = sample count, ny = 1, layout is F32 F32 F32 ...
// length of data must be nx * sizeof(float) bytes
struct mtmd_bitmap {
    uint32_t nx = 0;
    uint32_t ny = 0;
    std::string id; // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
    bool is_audio = false; // true if the bitmap is audio
    bool is_voice = false;
    std::shared_ptr<const mtmd_gen_ref> gen_ref; // gen-side reference of a predefined voice, if any
    bool mergeable = false; // [QWEN_VIDEO] set only on frames of the same video
    bool is_video  = false; // see mtmd_bitmap_set_is_video()
    std::string transcript; // see mtmd_bitmap_set_transcript()

    // lazy-loaded bitmap
    mtmd_bitmap_lazy_callback lazy_callback = nullptr;
    void * lazy_user_data = nullptr;

    mtmd_bitmap(const unsigned char * data, uint32_t nx, uint32_t ny)
        : nx(nx), ny(ny), is_audio(false) {
        if (data) {
            size_t data_size = (size_t)nx * ny * 3;
            this->data.resize(data_size);
            std::memcpy(this->data.data(), data, data_size);
        }
    }

    mtmd_bitmap(const unsigned char * data, uint32_t n_samples)
        : nx(n_samples), ny(1), is_audio(true) {
        if (data) {
            size_t data_size = (size_t)nx * sizeof(float);
            this->data.resize(data_size);
            std::memcpy(this->data.data(), data, data_size);
        }
    }

    const std::vector<unsigned char> & get_ro_buf() const {
        return data;
    }

    bool is_placeholder() const {
        return data.empty();
    }

    size_t n_bytes() const {
        return data.size();
    }

    bool is_video_frame() const {
        return mergeable && is_video;
    }

    bool can_merge_with(const mtmd_bitmap & other) const {
        // [QWEN_VIDEO] can (temporal) merge if both are images with same size
        return mergeable && other.mergeable
            && !is_audio && !other.is_audio
            && nx == other.nx && ny == other.ny;
    }

  private:
    std::vector<unsigned char> data;
};

// position indexing for decoder model
enum mtmd_pos_type {
    MTMD_POS_TYPE_NORMAL,    // number of positions equals to number of tokens
    MTMD_POS_TYPE_MROPE,     // qwen-vl mrope style, each image takes max(t,h,w) position indexes
    MTMD_POS_TYPE_HUNYUANVL, // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
    MTMD_POS_TYPE_SENSENOVA_U1, // U1 uses zero-based H/W and one temporal position per image
    MTMD_POS_TYPE_CANVAS,    // MiniCPM-V 4.7: overview + slices in one chunk, sharing one 2D canvas (see mtmd_image_tokens::canvas_tile_grid)
    MTMD_POS_TYPE_COUNT,     // for validation
};

struct mtmd_image_tokens {
    uint32_t nx = 0; // number of tokens in x direction
    uint32_t ny = 0; // number of tokens in y direction
    mtmd_pos_type pos = MTMD_POS_TYPE_NORMAL;
    uint32_t image_idx = 0; // 0-based position of this image among image chunks in the prompt(used by pos == MTMD_POS_TYPE_HUNYUANVL)
    uint32_t n_temporal_merge = 1; // for qwen-vl style temporal merge
    uint32_t n_tokens() const {
        if (pos == MTMD_POS_TYPE_HUNYUANVL) {
            // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
            return (nx + 1) * ny + 2;
        }
        if (pos == MTMD_POS_TYPE_CANVAS) {
            uint32_t n = 0;
            for (size_t k = 0; k < batch_f32.entries.size(); ++k) {
                const auto [gw, gh] = canvas_tile_grid(k);
                n += gw * gh + (uint32_t) mtmd_suffix_rows(batch_f32.entries[k].suffix_type).size();
            }
            return n;
        }
        uint32_t nz = batch_f32.entries.size();
        if (n_temporal_merge > 1) {
            // [QWEN_VIDEO] this logic is quite ugly, it's mostly to make qwen-vl temporal merge work, can be improved in the future
            // TODO: simplify this by repeating the last frame until it fits the temporal merge
            if (nz % n_temporal_merge != 0) {
                nz = nz / n_temporal_merge + 1;
            } else {
                nz = nz / n_temporal_merge;
            }
        }
        return nx * ny * nz;
    }
    mtmd_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    // true if one of entries in batch_f32 is a placeholder
    bool is_placeholder() const {
        for (const auto & entry : batch_f32.entries) {
            if (entry.is_placeholder()) {
                return true;
            }
        }
        return false;
    }

    // MTMD_POS_TYPE_CANVAS: entries are [overview, slices row by row], nx/ny is the token grid of the last entry
    // returns the token grid (w, h) of entry k, scaled from its pixel size
    std::pair<uint32_t, uint32_t> canvas_tile_grid(size_t k) const {
        const auto & ref = batch_f32.entries.back();
        const auto & e   = batch_f32.entries[k];
        return { (uint32_t) e.nx() * nx / ref.nx(), (uint32_t) e.ny() * ny / ref.ny() };
    }

    bool can_batch_with(const mtmd_image_tokens & other) {
        // a canvas chunk holds a whole image group, its layout is not given by nx/ny alone
        return nx == other.nx && ny == other.ny && pos == other.pos && pos != MTMD_POS_TYPE_CANVAS;
    }

    mtmd_image_tokens clone() {
        return mtmd_image_tokens{
            nx,
            ny,
            pos,
            image_idx,
            n_temporal_merge,
            batch_f32.clone(),
            id
        };
    }

    void serialize(mtmd_serialization & ser) const {
        // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
        ser.write(nx);
        ser.write(ny);
        ser.write((uint32_t)pos);
        ser.write(image_idx);
        ser.write(n_temporal_merge);
        ser.write(id);
        batch_f32.serialize(ser);
    }
    void deserialize(mtmd_serialization & ser) {
        nx = ser.read<uint32_t>();
        ny = ser.read<uint32_t>();
        uint32_t pos_raw = ser.read<uint32_t>();
        if (pos_raw >= MTMD_POS_TYPE_COUNT) {
            throw std::runtime_error("invalid pos type");
        }
        pos = (mtmd_pos_type)pos_raw;
        image_idx = ser.read<uint32_t>();
        n_temporal_merge = ser.read<uint32_t>();
        id = ser.read<std::string>();
        batch_f32.deserialize(ser);
    }
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens>;

// the mel of a clip is computed by its first encode, its chunks share the result
using mtmd_audio_source_ptr = std::shared_ptr<mtmd_audio_preproc_out>;

// an audio track split across chunks (e.g. the soundtrack of a video): it is encoded once, as one clip, and every
// chunk takes its rows of it
struct mtmd_audio_track {
    mtmd_image_f32_batch batch_f32; // the mel of the whole track
    mtmd_audio_source_ptr source;
    uint32_t n_tokens = 0;
    std::vector<float> embd;        // filled by the first encode
    std::shared_ptr<const mtmd_gen_ref> gen_ref; // gen-side reference of a predefined voice, if any
};

struct mtmd_audio_tokens {
    uint32_t n_tokens = 0; // number of tokens
    mtmd_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    mtmd_audio_source_ptr source; // if set, batch_f32 has the shape of the mel of chunk source_idx
    uint32_t source_idx = 0;

    // a slice of a track: the rows [row_begin, row_begin + n_tokens) of it, batch_f32 is unused
    std::shared_ptr<mtmd_audio_track> track;
    uint32_t row_begin = 0;

    // true if one of entries in batch_f32 is a placeholder
    bool is_placeholder() const {
        if (source || (track && track->source)) {
            return false;
        }
        if (track && track->batch_f32.entries.empty()) {
            return track->embd.empty(); // a voice: its rows are not serialized
        }
        for (const auto & entry : (track ? track->batch_f32 : batch_f32).entries) {
            if (entry.is_placeholder()) {
                return true;
            }
        }
        return false;
    }

    mtmd_audio_tokens clone() {
        return mtmd_audio_tokens{
            n_tokens,
            batch_f32.clone(),
            id,
            source,
            source_idx,
            track,
            row_begin,
        };
    }

    void serialize(mtmd_serialization & ser) const {
        // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
        ser.write(n_tokens);
        ser.write(id);
        batch_f32.serialize(ser);
        // a slice keeps its own copy of the track
        ser.write(track != nullptr);
        if (track) {
            ser.write(row_begin);
            ser.write(track->n_tokens);
            track->batch_f32.serialize(ser);
        }
    }
    void deserialize(mtmd_serialization & ser) {
        n_tokens = ser.read<uint32_t>();
        id = ser.read<std::string>();
        batch_f32.deserialize(ser);
        track.reset();
        if (ser.version >= 4 && ser.read<bool>()) {
            row_begin = ser.read<uint32_t>();
            track = std::make_shared<mtmd_audio_track>();
            track->n_tokens = ser.read<uint32_t>();
            track->batch_f32.deserialize(ser);
        }
        source.reset();
    }
};
using mtmd_audio_tokens_ptr = std::unique_ptr<mtmd_audio_tokens>;

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    mtmd_image_tokens_ptr tokens_image;
    mtmd_audio_tokens_ptr tokens_audio;

    bool can_batch_with(const mtmd_input_chunk & other) const {
        if (type != other.type) {
            return false;
        }

        if (tokens_image && other.tokens_image) {
            return tokens_image->can_batch_with(*other.tokens_image);
        }

        // TODO: allow batching audio chunks of the same size

        return false;
    }

    bool is_placeholder() const {
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return tokens_image && tokens_image->is_placeholder();
        } else if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            return tokens_audio && tokens_audio->is_placeholder();
        }
        return false;
    }

    void serialize(mtmd_serialization & ser) const {
        // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
        ser.write((uint32_t)type);

        ser.write<uint64_t>(tokens_text.size());
        for (llama_token tok : tokens_text) {
            ser.write((int32_t)tok);
        }

        ser.write(tokens_image != nullptr);
        if (tokens_image) {
            tokens_image->serialize(ser);
        }

        ser.write(tokens_audio != nullptr);
        if (tokens_audio) {
            tokens_audio->serialize(ser);
        }
    }
    void deserialize(mtmd_serialization & ser) {
        uint32_t type_raw = ser.read<uint32_t>();
        if (type_raw >= MTMD_INPUT_CHUNK_TYPE_COUNT) {
            throw std::runtime_error("invalid chunk type");
        }
        type = (mtmd_input_chunk_type)type_raw;

        uint64_t n_tokens_text = ser.read<uint64_t>();
        // reject before resize() so a tiny corrupted/malicious buffer can't force a huge allocation
        if (n_tokens_text > (ser.data.size() - ser.read_pos) / sizeof(int32_t)) {
            throw std::runtime_error("tokens_text length exceeds buffer size");
        }
        tokens_text.resize(n_tokens_text);
        for (uint64_t i = 0; i < n_tokens_text; i++) {
            tokens_text[i] = (llama_token)ser.read<int32_t>();
        }

        if (ser.read<bool>()) {
            tokens_image = std::make_unique<mtmd_image_tokens>();
            tokens_image->deserialize(ser);
        } else {
            tokens_image.reset();
        }

        if (ser.read<bool>()) {
            tokens_audio = std::make_unique<mtmd_audio_tokens>();
            tokens_audio->deserialize(ser);
        } else {
            tokens_audio.reset();
        }

        // catch buffers where the declared type doesn't match which payload is actually present,
        // so a mismatched chunk can't slip through and null-deref/abort later in an accessor
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE && !tokens_image) {
            throw std::runtime_error("type is IMAGE but tokens_image is missing");
        }
        if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO && !tokens_audio) {
            throw std::runtime_error("type is AUDIO but tokens_audio is missing");
        }
    }
};

struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};

struct mtmd_batch {
    mtmd_context * ctx;
    std::vector<const mtmd_input_chunk *> entries;
    std::vector<float> output_embd; // aggregated output embedding for the whole batch
    mtmd_batch(mtmd_context * ctx): ctx(ctx) {}
    int32_t n_tokens() const {
        int32_t n = 0;
        for (const auto * chunk : entries) {
            n += mtmd_input_chunk_get_n_tokens(chunk);
        }
        return n;
    }
};

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum mtmd_slice_tmpl {
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    MTMD_SLICE_TMPL_LLAMA4,
    MTMD_SLICE_TMPL_IDEFICS3,
    MTMD_SLICE_TMPL_LFM2,
    MTMD_SLICE_TMPL_STEP3VL,
};

const char * mtmd_default_marker() {
    return "<__media__>";
}

static mtmd_flash_attn_type mtmd_to_flash_attn_type(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:     return MTMD_FLASH_ATTN_TYPE_AUTO;
        case LLAMA_FLASH_ATTN_TYPE_DISABLED: return MTMD_FLASH_ATTN_TYPE_DISABLED;
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:  return MTMD_FLASH_ATTN_TYPE_ENABLED;
    }
    return MTMD_FLASH_ATTN_TYPE_AUTO;
}

mtmd_context_params mtmd_context_params_default() {
    mtmd_context_params params {
        /* use_gpu           */ true,
        /* device            */ nullptr,
        /* print_timings     */ true,
        /* n_threads         */ 4,
        /* image_marker      */ nullptr,
        /* media_marker      */ mtmd_default_marker(),
        /* flash_attn_type   */ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /* warmup            */ true,
        /* image_min_tokens  */ -1,
        /* image_max_tokens  */ -1,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
        /* batch_max_tokens  */ 1024,
        /* progress_callback */ nullptr,
        /* progress_callback_user_data */ nullptr,
    };
    return params;
}

// a stage of the mmproj that is a libllama text model: its layers are stored under a prefix, with the libllama names
// libllama loads them from a synthetic GGUF: gdst holds the arch hparams, the tensor data is read from the mmproj file
struct mtmd_owned_llama_source {
    std::ifstream f;
    std::map<std::string, std::pair<size_t, size_t>> tensors; // name in the synthetic model -> offset and size in the mmproj
};

static void mtmd_owned_llama_set_tensor_data(ggml_tensor * t, void * user_data) {
    auto * src = (mtmd_owned_llama_source *) user_data;
    std::vector<uint8_t> buf(ggml_nbytes(t), 0);
    const std::string name = t->name;
    auto it = src->tensors.find(name);
    if (it != src->tensors.end()) {
        if (it->second.second != buf.size()) {
            throw std::runtime_error(mtmd_format("unexpected size of %s", t->name));
        }
        src->f.seekg((std::streamoff) it->second.first, std::ios::beg);
        if (!src->f.read((char *) buf.data(), (std::streamsize) buf.size())) {
            throw std::runtime_error(mtmd_format("failed to read %s", t->name));
        }
    } else {
        // optional tensors that libllama creates by itself, give them neutral values (scale = 1, dense_2 = identity, others = 0)
        // any other missing tensor is an error: a weight left at 0 gives wrong output without any error
        // (e.g. a model built from metadata has separate q, k, v and no fused qkv)
        auto ends_with = [&](const char * suffix) {
            const size_t n = strlen(suffix);
            return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
        };
        const bool optional = ends_with("scale") || ends_with(".bias") || name == "dense_2.weight" ||
                              name == "cls.output.weight" || // Qwen3 reranking head, unused by YuE2 NAR
                              name == "rope_freqs.weight" || // granite, voxcpm2 residual LM
                              name == "token_embd.weight" || name == "output.weight"; // inputs are embedding rows, outputs are hidden states
        if (!optional || t->type != GGML_TYPE_F32) {
            throw std::runtime_error(mtmd_format("tensor %s is missing from the mmproj", t->name));
        }
        float * w = (float *) buf.data();
        if (ends_with("scale") || name == "rope_freqs.weight") {
            std::fill(w, w + ggml_nelements(t), 1.0f);
        } else if (name == "dense_2.weight") {
            GGML_ASSERT(t->ne[0] == t->ne[1]);
            for (int64_t i = 0; i < t->ne[0]; i++) {
                w[i * t->ne[0] + i] = 1.0f;
            }
        }
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

// the layers are the tensors "<prefix>blk.*" and "<prefix>output_norm.*" of the mmproj, n_tensors receives their count
static llama_model * mtmd_owned_llama_load(const char * fname, ggml_backend_dev_t device, gguf_context * gdst, const std::string & prefix, size_t & n_tensors) {
    ggml_context * meta = nullptr;
    gguf_init_params gparams = { /*no_alloc =*/ true, /*ctx =*/ &meta };
    gguf_context_ptr gsrc(gguf_init_from_file(fname, gparams));
    ggml_context_ptr meta_ptr(meta);
    if (!gsrc) {
        throw std::runtime_error(mtmd_format("failed to read %s", fname));
    }

    mtmd_owned_llama_source src;
    for (int64_t i = 0; i < gguf_get_n_tensors(gsrc.get()); i++) {
        const std::string name = gguf_get_tensor_name(gsrc.get(), i);
        if (name.rfind(prefix + "blk.", 0) != 0 && name.rfind(prefix + "output_norm.", 0) != 0) {
            continue;
        }
        // the synthetic model only takes the type from here, the shapes come from the hparams
        ggml_tensor t = *ggml_get_tensor(meta, name.c_str());
        ggml_set_name(&t, name.substr(prefix.size()).c_str());
        gguf_add_tensor(gdst, &t);
        src.tensors[t.name] = { gguf_get_data_offset(gsrc.get()) + gguf_get_tensor_offset(gsrc.get(), i), ggml_nbytes(&t) };
    }
    n_tensors = src.tensors.size();
    src.f = open_ifstream_binary(fname);
    if (!src.f) {
        throw std::runtime_error(mtmd_format("failed to open %s", fname));
    }

    llama_model_params mparams = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs;
    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        devs = { device, nullptr };
        mparams.devices      = devs.data();
        mparams.n_gpu_layers = -1;
    } else {
        mparams.n_gpu_layers = 0;
    }
    llama_model * model = llama_model_init_from_user(gdst, mtmd_owned_llama_set_tensor_data, &src, mparams);
    if (!model) {
        throw std::runtime_error(mtmd_format("failed to load the layers under %s", prefix.c_str()));
    }
    return model;
}

struct mtmd_context {
    // must outlive the subcontexts below
    mtmd_models models;
    std::unique_ptr<mtmd_backends> backends; // shared by the subcontexts

    // held by every public function that runs a graph, the internals never lock
    std::mutex compute_mutex;

    std::unique_ptr<mtmd_subcontext> ctx_v; // vision
    std::unique_ptr<mtmd_subcontext> ctx_a; // audio
    std::vector<float> out_embd; // image embedding vector

    // generation context
    std::unique_ptr<mtmd_subcontext> ctx_gen_a; // audio

    // gen-side reference data of the last encoded chunk (chatterbox), see mtmd_gen_state_set_ref()
    mtmd_gen_ref gen_ref;

    bool print_timings;
    int n_threads;
    std::string media_marker;
    const int n_embd_text = -1; // -1 means llm context not provided, skip checking this
    const llama_vocab * vocab = nullptr; // can be nullptr if text_model is not provided
    const llama_model * text_model = nullptr; // for the token embeddings of mtmd_audio_stream
    mtmd_pos_type pos_type;

    // these are not token, but strings used to mark the beginning and end of image/audio embeddings
    std::string img_beg;
    std::string img_end;
    std::string aud_beg;
    std::string aud_end;
    std::string vid_beg; // wraps a whole video (lazy bitmap with is_video)
    std::string vid_end;

    // for llava-uhd style models, we need special tokens in-between slices
    // minicpmv calls them "slices", llama 4 calls them "tiles"
    mtmd_slice_tmpl slice_tmpl = MTMD_SLICE_TMPL_NONE;
    std::vector<llama_token> tok_ov_img_start;  // overview image
    std::vector<llama_token> tok_ov_img_end;    // overview image
    std::vector<llama_token> tok_slices_start;  // start of all slices
    std::vector<llama_token> tok_slices_end;    // end of all slices
    std::vector<llama_token> tok_sli_img_start; // single slice start
    std::vector<llama_token> tok_sli_img_end;   // single slice end
    std::vector<llama_token> tok_sli_img_mid;   // between 2 slices
    std::vector<llama_token> tok_row_end;       // end of row
    bool tok_row_end_trail = false;
    bool ov_img_first      = false;

    // MiniCPM-V 4.6/4.7 prepends an <image_id>N</image_id> tag before <image>
    bool has_prompt_img_id = false;

    // Kimi-K3.1 writes the original image size after img_beg
    bool has_prompt_img_size = false;

    // string template for slice image delimiters with row/col (idefics3)
    std::string sli_img_start_tmpl;

    std::unique_ptr<mtmd_audio_preprocessor> audio_preproc;
    std::unique_ptr<mtmd_image_preprocessor> image_preproc;

    // batching
    int32_t batch_max_tokens;

    // TODO @ngxson : add timings

    mtmd_context(const char * mmproj_fname,
                   const llama_model * text_model,
                   const mtmd_context_params & ctx_params,
                   bool no_alloc = false) :
        print_timings   (ctx_params.print_timings),
        n_threads       (ctx_params.n_threads),
        media_marker    (ctx_params.media_marker),
        n_embd_text     (text_model ? llama_model_n_embd_inp(text_model) : -1),
        vocab           (text_model ? llama_model_get_vocab(text_model) : nullptr),
        text_model      (text_model),
        batch_max_tokens(ctx_params.batch_max_tokens)
    {
        if (ctx_params.image_marker != nullptr) {
            throw std::runtime_error("custom image_marker is not supported anymore, use media_marker instead");
        }

        if (media_marker.empty()) {
            throw std::runtime_error("media_marker must not be empty");
        }

        if (text_model) {
            auto decoder_rope_type = llama_model_rope_type(text_model);
            switch (decoder_rope_type) {
                case LLAMA_ROPE_TYPE_NONE:
                case LLAMA_ROPE_TYPE_NORM:
                case LLAMA_ROPE_TYPE_NEOX:
                    {
                        pos_type = MTMD_POS_TYPE_NORMAL;
                    } break;
                case LLAMA_ROPE_TYPE_MROPE:
                case LLAMA_ROPE_TYPE_IMROPE:
                    {
                        pos_type = MTMD_POS_TYPE_MROPE;
                    } break;
                default:
                    throw std::runtime_error(mtmd_format("unsupported decoder rope type: %d\n", decoder_rope_type));
            }
        }

        ggml_backend_dev_t device = mtmd_select_device(ctx_params.use_gpu, ctx_params.device);

        mtmd_model_params model_params;
        model_params.image_min_tokens            = ctx_params.image_min_tokens;
        model_params.image_max_tokens            = ctx_params.image_max_tokens;
        model_params.no_alloc                    = no_alloc;
        model_params.progress_callback           = ctx_params.progress_callback;
        model_params.progress_callback_user_data = ctx_params.progress_callback_user_data;

        ggml_backend_buffer_type_t buft = device
            ? ggml_backend_dev_buffer_type(device)
            : ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        models = mtmd_model_load(mmproj_fname, model_params, buft);

        // soprano and pocket-tts without voice cloning have no encoder
        if (!models.v && !models.a && !models.gen_a) {
            throw std::runtime_error(mtmd_format("Failed to load mmproj model from %s\n", mmproj_fname));
        }

        int max_nodes = 0;
        for (const auto * m : { models.v.get(), models.a.get(), models.gen_a.get() }) {
            if (m) {
                max_nodes = std::max(max_nodes, m->max_nodes);
            }
        }

        // no_alloc only measures memory, nothing is computed
        backends = std::make_unique<mtmd_backends>(device, no_alloc ? 1 : ctx_params.n_threads, max_nodes,
                                                   ctx_params.cb_eval, ctx_params.cb_eval_user_data);

        // like llama_context::sched_reserve(), each warmup reserves its graph on the shared scheduler
        mtmd_subctx_params sctx_params {
            /* flash_attn_type   */ mtmd_to_flash_attn_type(ctx_params.flash_attn_type),
            /* warmup            */ ctx_params.warmup,
        };
        if (models.v) {
            ctx_v = std::make_unique<mtmd_subcontext>(*models.v, *backends, sctx_params);
        }
        if (models.a) {
            ctx_a = std::make_unique<mtmd_subcontext>(*models.a, *backends, sctx_params);
        }
        if (models.gen_a) {
            ctx_gen_a = std::make_unique<mtmd_subcontext>(*models.gen_a, *backends, sctx_params);
        }
        for (size_t i = 0; i < backends->ptrs.size(); ++i) {
            const size_t size = ggml_backend_sched_get_buffer_size(backends->sched.get(), backends->ptrs[i]);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(backends->bufts[i]), size / 1024.0 / 1024.0);
            }
        }

        // if both vision and audio mmproj are present, we need to validate their n_embd
        if (ctx_v && ctx_a) {
            int n_embd_v = ctx_v->model.n_mmproj_embd();
            int n_embd_a = ctx_a->model.n_mmproj_embd();
            if (n_embd_v != n_embd_a) {
                throw std::runtime_error(mtmd_format(
                    "mismatch between vision and audio mmproj (n_embd_v = %d, n_embd_a = %d)\n",
                    n_embd_v, n_embd_a));
            }
        }

        // since we already validate n_embd of vision and audio mmproj,
        // we can safely assume that they are the same
        // a gen-only mmproj is validated below
        int n_embd_mmproj = ctx_v || ctx_a ? (ctx_v ? ctx_v : ctx_a)->model.n_mmproj_embd() : n_embd_text;
        if (n_embd_text > 0 && n_embd_text != n_embd_mmproj) {
            throw std::runtime_error(mtmd_format(
                "mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
                "hint: you may be using wrong mmproj\n",
                n_embd_text, n_embd_mmproj));
        }
        if (ctx_gen_a) {
            int n_embd_gen = ctx_gen_a->model.n_mmproj_embd();
            if (n_embd_text > 0 && n_embd_text != n_embd_gen) {
                throw std::runtime_error(mtmd_format(
                    "mismatch between text model (n_embd = %d) and gen-audio mmproj (n_embd = %d)\n"
                    "hint: you may be using wrong mmproj\n",
                    n_embd_text, n_embd_gen));
            }
        }
        if (ctx_v) {
            init_vision();
        }
        if (ctx_a) {
            init_audio();
        }
    }

    void init_vision() {
        GGML_ASSERT(ctx_v != nullptr);
        image_preproc.reset();

        projector_type proj = ctx_v->model.proj_type;

        switch (proj) {
            case PROJECTOR_TYPE_INKLING2V:
                {
                    // the chat template code gives the image its own message block
                    img_beg = "<|content_image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_inkling2v>(models.v.get());
                } break;
            default:
                throw std::runtime_error(mtmd_format("%s: unexpected vision projector type %d\n", __func__, proj));
        }

        GGML_ASSERT(image_preproc != nullptr);
    }

    // no kept projector takes audio input
    [[noreturn]] void init_audio() {
        GGML_ASSERT(ctx_a != nullptr);
        throw std::runtime_error(mtmd_format("%s: unexpected audio projector type %d\n", __func__, ctx_a->model.proj_type));
    }

    // get the subcontext based on chunk type
    mtmd_subcontext * get_subctx(const mtmd_input_chunk * chunk) const {
        if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return ctx_v.get();
        } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            return ctx_a.get();
        }
        GGML_ABORT("unknown chunk type");
    }

    projector_type proj_type_v() const {
        return ctx_v ? ctx_v->model.proj_type : PROJECTOR_TYPE_UNKNOWN;
    }

    projector_type proj_type_a() const {
        return ctx_a ? ctx_a->model.proj_type : PROJECTOR_TYPE_UNKNOWN;
    }

    // no kept projector pairs a video with its soundtrack
    bool video_has_soundtrack() const {
        return false;
    }

    int64_t n_embd_out() const {
        if (ctx_v) {
            return ctx_v->model.n_mmproj_embd();
        } else if (ctx_a) {
            return ctx_a->model.n_mmproj_embd();
        } else {
            throw std::runtime_error("no mmproj model loaded");
        }
    }

private:
    llama_token lookup_token(const std::string & token_text) {
        if (vocab == nullptr) {
            // TODO @ngxson : this case is currently hit by mtmd_get_memory_usage
            // but we should reconsider this if this case is needed in other places in the future
            return LLAMA_TOKEN_NULL;
        }
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_vocab; i++) {
            if (token_to_piece(vocab, i, true) == token_text) {
                return i;
            }
        }
        return LLAMA_TOKEN_NULL;
    }

    std::string token_to_piece(const llama_vocab * vocab, llama_token token, bool special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        } else {
            piece.resize(n_chars);
        }
        return piece;
    }
};

mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
        const struct llama_model * text_model,
        const struct mtmd_context_params ctx_params) {
    try {
        return new mtmd_context(mmproj_fname, text_model, ctx_params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return nullptr;
    }
}

void mtmd_free(mtmd_context * ctx) {
    delete ctx;
}

std::vector<std::vector<const mtmd_bitmap *>> mtmd_group_mergeable_bitmaps(std::vector<mtmd_internal_part> & parts, int n_merge) {
    std::vector<std::vector<const mtmd_bitmap *>> output;
    for (size_t i = 0; i < parts.size(); i++) {
        if (parts[i].bitmap == nullptr) {
            continue; // text part
        }
        const bool has_next = n_merge > 1 && i + 1 < parts.size() && parts[i + 1].bitmap != nullptr;
        if (has_next && parts[i].bitmap->can_merge_with(*parts[i + 1].bitmap)) {
            LOG_DBG("%s: merging 2 frames at part index %zu and %zu\n", __func__, i, i + 1);
            output.push_back({parts[i].bitmap, parts[i + 1].bitmap});
            parts.erase(parts.begin() + i + 1);
            continue;
        }
        LOG_DBG("%s: no merging for part index %zu\n", __func__, i);
        output.push_back({parts[i].bitmap});
    }
    return output;
}

struct mtmd_tokenizer {
    const mtmd_context * ctx;

    std::string input_text; // note: can contain null bytes; do not use c_str()
    bool add_special;
    bool parse_special;
    const llama_vocab * vocab;

    using part = mtmd_internal_part;
    std::vector<part> parts;
    // these will be freed when mtmd_tokenizer finishes
    std::vector<mtmd::bitmap> bm_from_lazy; // TODO @ngxson : refactor, free bm_from_lazy progressively
    std::vector<const char *> text_from_lazy;

    // [LFM3_OMNI] the audio pieces of a video are joined back and encoded as one clip, each piece takes its rows
    struct soundtrack {
        std::vector<const mtmd_bitmap *> pieces;
        std::shared_ptr<mtmd_audio_track> track; // built on the first piece
        size_t   n_samples  = 0; // samples of the pieces added so far
        uint32_t n_assigned = 0; // rows of the pieces added so far
    };
    std::vector<soundtrack> soundtracks;
    std::map<const mtmd_bitmap *, size_t> soundtrack_of; // piece -> index in soundtracks

    mtmd_input_chunks cur;
    uint32_t n_images_added = 0; // 0-based index assigned to the next image chunk

    ~mtmd_tokenizer() {
        // note: mtmd::bitmap is already RAII
        for (auto & str : text_from_lazy) {
            free((void *)str);
        }
    }

    mtmd_tokenizer(const mtmd_context * ctx,
            const mtmd_input_text * text,
            const mtmd_bitmap * const * bmps,
            size_t n_bitmaps) : ctx(ctx) {
        add_special   = text->add_special;
        parse_special = text->parse_special;
        vocab         = ctx->vocab;

        input_text.assign(text->text, text->text_len);

        std::vector<const mtmd_bitmap *> bitmaps(bmps, bmps + n_bitmaps);
        auto parts_str = split_text(input_text, ctx->media_marker);
        size_t i_bm = 0;
        for (const auto & part : parts_str) {
            if (part == ctx->media_marker) {
                if (i_bm >= bitmaps.size()) {
                    throw std::runtime_error(mtmd_format("number of media markers in text (%zu) exceeds number of bitmaps (%zu)", i_bm + 1, bitmaps.size()));
                }
                parts.push_back({"", bitmaps[i_bm++]});
            } else {
                parts.push_back({std::move(part), nullptr, parse_special});
            }
        }

        size_t n_markers = 0;
        for (const auto & part : parts) {
            if (part.bitmap != nullptr) {
                n_markers++;
            }
        }
        if (n_markers != bitmaps.size()) {
            throw std::runtime_error(mtmd_format("number of media markers in text (%zu) does not match number of bitmaps (%zu)", n_markers, bitmaps.size()));
        }

        expand_lazy_bitmaps();
    }

    mtmd_tokenizer(const mtmd_context * ctx,
            const mtmd_input_part * const * input_parts,
            size_t n_parts,
            bool add_special) : ctx(ctx) {
        this->add_special = add_special;
        parse_special = true; // only used for text returned by lazy bitmaps
        vocab         = ctx->vocab;

        for (size_t i = 0; i < n_parts; i++) {
            const mtmd_input_part * p = input_parts[i];
            if (p->text != nullptr) {
                parts.push_back({std::string(p->text->text, p->text->text_len), nullptr, p->text->parse_special});
            } else {
                parts.push_back({"", p->bitmap});
            }
        }

        expand_lazy_bitmaps();
    }

    void expand_lazy_bitmaps() {
        std::vector<part> expanded;
        expanded.reserve(parts.size());
        for (auto & p : parts) {
            if (p.bitmap != nullptr && p.bitmap->lazy_callback) {
                LOG_DBG("%s: expanding lazy bitmap\n", __func__);
                const bool is_video = p.bitmap->is_video;
                if (is_video && !ctx->vid_beg.empty()) {
                    expanded.push_back({ctx->vid_beg, nullptr, true});
                }
                soundtrack st;
                for (size_t i = 0;; i++) {
                    char * out_str = nullptr;
                    mtmd_bitmap * out_bm = nullptr;
                    int res = p.bitmap->lazy_callback(i,
                                    p.bitmap->lazy_user_data,
                                    &out_bm,
                                    &out_str);
                    if (out_bm && out_str) {
                        throw std::runtime_error(mtmd_format("lazy callback cannot return both bitmap and text"));
                    }
                    if (res == 0) {
                        // OK, append the returned chunk; lazy part is not yet added
                        if (out_bm) {
                            auto & ptr = bm_from_lazy.emplace_back(out_bm); // remember to free it later
                            expanded.push_back({"", ptr.ptr.get()});
                            if (is_video && out_bm->is_audio && ctx->video_has_soundtrack()) {
                                st.pieces.push_back(out_bm);
                            }
                            LOG_DBG("%s: lazy callback returned bitmap with dimensions %d x %d\n", __func__, out_bm->nx, out_bm->ny);
                        } else if (out_str) {
                            auto & ptr = text_from_lazy.emplace_back(out_str); // remember to free it later
                            expanded.push_back({ptr, nullptr, parse_special});
                            LOG_DBG("%s: lazy callback returned text: %s\n", __func__, out_str);
                        }
                    } else if (res == -1) {
                        // EOF: lazy part removes itself (not added to expanded)
                        if (is_video && !ctx->vid_end.empty()) {
                            expanded.push_back({ctx->vid_end, nullptr, true});
                        }
                        break;
                    } else if (res == -2) {
                        // error
                        throw std::runtime_error(mtmd_format("lazy callback returned error"));
                    }
                }
                if (!st.pieces.empty()) {
                    for (const auto * piece : st.pieces) {
                        soundtrack_of[piece] = soundtracks.size();
                    }
                    soundtracks.push_back(std::move(st));
                }
            } else {
                expanded.push_back(std::move(p));
            }
        }
        parts = std::move(expanded);
    }

    int32_t tokenize(mtmd_input_chunks * output) {
        cur.entries.clear();

        // [QWEN_VIDEO] handle frame merging for models that support it (i.e. qwen-vl)
        int n_merge_frames = 1;
        if (ctx->ctx_v) {
            n_merge_frames = ctx->ctx_v->model.n_temporal_merge();
            GGML_ASSERT(n_merge_frames <= 2 && "we only support merging maximum 2 images for now; open an issue if this model supports merging more");
        }

        auto merged_bitmaps = mtmd_group_mergeable_bitmaps(parts, n_merge_frames);

        size_t i_bm = 0;
        for (const auto & p : parts) {
            if (p.bitmap != nullptr) {
                if (i_bm >= merged_bitmaps.size()) {
                    LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                            __func__, merged_bitmaps.size(), parts.size() - 1);
                    return 1;
                }
                auto bmps = merged_bitmaps[i_bm++];
                int32_t res = add_media(bmps);
                if (res != 0) {
                    return res;
                }
            } else {
                add_text(p.text, p.parse_special);
            }
        }

        if (vocab != nullptr) {
            if (add_special && llama_vocab_get_add_bos(vocab)) {
                // if first chunk is text, we add BOS token to first text chunk
                // otherwise, create a new text chunk with BOS token
                if (!cur.entries.empty() && cur.entries[0].type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    // add BOS token to the beginning of first text chunk
                    cur.entries[0].tokens_text.insert(cur.entries[0].tokens_text.begin(), llama_vocab_bos(vocab));
                } else {
                    // create a new text chunk with BOS token at the beginning
                    mtmd_input_chunk bos_chunk{
                        MTMD_INPUT_CHUNK_TYPE_TEXT,
                        {llama_vocab_bos(vocab)},
                        nullptr, // image tokens
                        nullptr, // audio tokens
                    };
                    cur.entries.insert(cur.entries.begin(), std::move(bos_chunk));
                }
            }

            if (add_special && llama_vocab_get_add_eos(vocab)) {
                // if last chunk is text, we add EOS token to it
                add_text({llama_vocab_eos(vocab)});
            }
        }

        if (i_bm != merged_bitmaps.size()) {
            LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                    __func__, merged_bitmaps.size(), parts.size() - 1);
            return 1;
        }

        *output = std::move(cur);

        return 0;
    }

    void add_text(const std::string & txt, bool parse_special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        LOG_DBG("%s: %s\n", __func__, txt.c_str());
        auto tokens = mtmd_tokenize_text_internal(vocab, txt, /* add_special */ false, parse_special);
        add_text(tokens);
    }

    void add_text(const std::vector<llama_token> & tokens) {
        if (tokens.empty()) {
            return;
        }
        // if last entry is also a text chunk, add tokens to it instead of creating new chunk
        if (!cur.entries.empty() && cur.entries.back().type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            cur.entries.back().tokens_text.insert(
                                            cur.entries.back().tokens_text.end(),
                                            tokens.begin(),
                                            tokens.end());
        } else {
            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_TEXT,
                tokens,
                nullptr, // image tokens
                nullptr, // audio tokens
            };
            cur.entries.emplace_back(std::move(chunk));
        }
    }

    int32_t add_media(std::vector<const mtmd_bitmap *> & bitmaps) {
        GGML_ASSERT(!bitmaps.empty());

        // note: only one type of media is supported per call, caller should enforce this
        if (bitmaps[0]->is_voice) {
            GGML_ASSERT(bitmaps.size() == 1);
            return add_voice(*bitmaps[0]);
        }
        const bool is_vision = !bitmaps[0]->is_audio;

        if (is_vision) {
            // handle image

            if (!ctx->ctx_v) {
                LOG_ERR("%s: error: model does not support vision input\n", __func__);
                return 2;
            }

            if (!ctx->img_beg.empty()) {
                add_text(ctx->img_beg, true); // add image begin token
            }
            if (ctx->has_prompt_img_size) {
                add_text(mtmd_format("image %ux%u<|media_content|>", bitmaps[0]->nx, bitmaps[0]->ny), true);
            }

            // TODO @ngxson : this is quite hacky because preprocessor only support batch with one single element, that need to be fixed in the future (e.g. by changing the preprocessor interface always take single input)

            mtmd_image_preproc_out preproc_out;

            for (const auto * bmp : bitmaps) {
                // sanity check
                GGML_ASSERT(!bmp->is_audio);
                GGML_ASSERT(ctx->image_preproc != nullptr);
                if (bmp->nx <= 0 || bmp->ny <= 0) {
                    LOG_ERR("%s: error: invalid bitmap dimensions: nx = %d, ny = %d\n",
                            __func__, bmp->nx, bmp->ny);
                    return 2;
                }

                // convert mtmd_bitmap to mtmd_image_u8
                mtmd_image_u8 img_u8;
                img_u8.set_size(
                    {(int)bmp->nx, (int)bmp->ny},
                    bmp->is_placeholder());
                img_u8.cpy_buf(bmp->get_ro_buf());

                // preprocess image
                mtmd_image_preproc_out tmp_preproc_out = bmp->is_video_frame()
                    ? ctx->image_preproc->preprocess_video_frame(img_u8)
                    : ctx->image_preproc->preprocess(img_u8);

                // move entries and grid dimensions to the "global" preproc_out
                for (auto & entry : tmp_preproc_out.entries) {
                    preproc_out.entries.emplace_back(std::move(entry));
                }

                // for llava-uhd style, we need to handle grid too
                // we don't care about overwriting these values for now because the case where bitmaps.size() > 1 is only for frame merging (qwen-vl), not supported by llava-uhd
                if ((tmp_preproc_out.grid_x > 0 && tmp_preproc_out.grid_y > 0)
                        || tmp_preproc_out.has_overview()) {
                    GGML_ASSERT(bitmaps.size() == 1);
                    preproc_out.grid_x = tmp_preproc_out.grid_x;
                    preproc_out.grid_y = tmp_preproc_out.grid_y;
                    preproc_out.overview = std::move(tmp_preproc_out.overview);
                }
            }

            LOG_DBG("%s: preproc_out has %zu entries, grid_x = %d, grid_y = %d, has_overview = %d\n",
                    __func__, preproc_out.entries.size(), preproc_out.grid_x, preproc_out.grid_y,
                    preproc_out.has_overview() ? 1 : 0);

            // handle llava-uhd style preprocessing
            // (output either a grid, or overview-only)
            const bool has_tiling_grid = (preproc_out.grid_x > 0 && preproc_out.grid_y > 0)
                || preproc_out.has_overview();

            if (has_tiling_grid) {
                // [QWEN_VIDEO] we do not support "frame merging" for llama-uhd style, so no batching for now
                GGML_ASSERT(bitmaps.size() == 1);

                const int n_col = preproc_out.grid_x;
                const int n_row = preproc_out.grid_y;

                // split batch into chunks of single images
                auto chunks = split_batch_to_chunk(std::move(preproc_out), bitmaps[0]->id);
                GGML_ASSERT(chunks.size() > 0);

                // NOTE: preproc_out is invalidated after this point, do not use it anymore

                // split_batch_to_chunk must always put the overview image first
                auto ov_chunk = std::move(chunks.front());
                chunks.erase(chunks.begin());

                // add overview image (first)
                if (ctx->ov_img_first) {
                    if (ctx->has_prompt_img_id) {
                        add_text("<image_id>" + std::to_string(n_images_added) + "</image_id>", true);
                    }
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }

                // add slices (or tiles)
                if (!chunks.empty()) {
                    LOG_DBG("%s: adding %d slices (%d rows x %d cols)\n", __func__, (int)chunks.size(), n_row, n_col);
                    GGML_ASSERT((int)chunks.size() == n_row * n_col);
                    add_text(ctx->tok_slices_start);
                    for (int y = 0; y < n_row; y++) {
                        for (int x = 0; x < n_col; x++) {
                            const bool is_last_in_row = (x == n_col - 1);
                            if (!ctx->tok_sli_img_start.empty()) {
                                add_text(ctx->tok_sli_img_start);
                            } else if (!ctx->sli_img_start_tmpl.empty()) {
                                // If using a template to preceed a slice image
                                const size_t sz = std::snprintf(nullptr, 0, ctx->sli_img_start_tmpl.c_str(), y+1, x+1) + 1;
                                std::unique_ptr<char[]> buf(new char[sz]);
                                std::snprintf(buf.get(), sz, ctx->sli_img_start_tmpl.c_str(), y+1, x+1);
                                add_text(std::string(buf.get(), buf.get() + sz - 1), true);
                            }

                            auto & curr_chunk = chunks[y * n_col + x];
                            auto & curr_batch = curr_chunk.tokens_image->batch_f32;
                            if (curr_batch.entries.size() != 1) {
                                throw std::runtime_error(mtmd_format("%s: expect 1 image in batch_f32", __func__));
                            }

                            LOG_DBG("%s: adding slice image at row %d col %d\n", __func__, y, x);
                            cur.entries.emplace_back(std::move(curr_chunk));

                            add_text(ctx->tok_sli_img_end);
                            if (!is_last_in_row) {
                                add_text(ctx->tok_sli_img_mid);
                            }
                        }
                        if ((y != n_row - 1 || ctx->tok_row_end_trail)) {
                            add_text(ctx->tok_row_end);
                        }
                    }
                    add_text(ctx->tok_slices_end);
                }

                // add overview image (last)
                if (!ctx->ov_img_first) {
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }
            } else {

                if (preproc_out.entries.size() == 0) {
                    LOG_ERR("%s: no image tokens produced by preprocessor (ref: https://github.com/ggml-org/llama.cpp/pull/24769)\n", __func__);
                    return 2;
                }


                size_t n_tokens = 0;
                for (auto & e : preproc_out.entries) {
                    n_tokens += ctx->ctx_v->model.n_output_tokens(&e);
                    if (ctx->ctx_v->model.n_temporal_merge() == 2) {
                        // [QWEN_VIDEO] pair input is merged to the same embd, so only count as one image
                        break;
                    }
                }

                mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);

                // [QWEN_VIDEO] improve this in the future
                image_tokens->n_temporal_merge = ctx->ctx_v->model.n_temporal_merge();

                if (mtmd_decode_use_mrope(ctx)) {
                    // for Qwen2VL, we need this information for M-RoPE decoding positions
                    image_tokens->nx = ctx->ctx_v->model.n_output_tokens_x(&preproc_out.entries[0]);
                    image_tokens->ny = ctx->ctx_v->model.n_output_tokens_y(&preproc_out.entries[0]);
                } else {
                    // other models, we only need the total number of tokens
                    image_tokens->nx = n_tokens;
                    image_tokens->ny = 1;
                }
                image_tokens->pos = ctx->pos_type;

                mtmd_image_f32_batch batch_f32;
                batch_f32.is_audio = false;
                batch_f32.is_video = bitmaps[0]->is_video_frame();
                batch_f32.entries = std::move(preproc_out.entries);
                // do NOT use preproc_out from this point on, it's moved

                image_tokens->batch_f32 = std::move(batch_f32);
                image_tokens->id = bitmaps[0]->id; // optional

                LOG_DBG("image_tokens->nx = %d\n", image_tokens->nx);
                LOG_DBG("image_tokens->ny = %d\n", image_tokens->ny);
                LOG_DBG("batch_f32 size = %d\n", (int)image_tokens->batch_f32.entries.size());

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    {}, // text tokens
                    std::move(image_tokens),
                    nullptr, // audio tokens
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->img_end.empty()) {
                add_text(ctx->img_end, true); // add image end token
            }

            // advance image-chunk counter so the next image gets the next XD-RoPE dim-3 slot
            n_images_added++;

        } else {
            // handle audio

            GGML_ASSERT(bitmaps.size() == 1); // no batching support for now
            auto & bitmap = bitmaps[0];

            if (!ctx->ctx_a) {
                LOG_ERR("%s: error: model does not support audio input\n", __func__);
                return 2;
            }

            if (auto it = soundtrack_of.find(bitmap); it != soundtrack_of.end()) {
                return add_soundtrack_piece(soundtracks[it->second], *bitmap);
            }

            if (bitmap->nx == 0) {
                LOG_ERR("%s: error: empty audio data\n", __func__);
                return 2;
            }

            if (!ctx->aud_beg.empty()) {
                add_text(ctx->aud_beg, true); // add audio begin token
            }

            // sanity check
            GGML_ASSERT(ctx->audio_preproc != nullptr);

            // preprocess audio
            auto source = std::make_shared<mtmd_audio_preproc_out>();
            {
                std::vector<float> dummy;
                const float * samples = nullptr;
                size_t n_samples = 0;
                if (bitmap->is_placeholder()) {
                    // TODO @ngxson : skip underlay processing if bitmap is placeholder
                    GGML_ASSERT(bitmap->ny == 1);

                    dummy.resize(bitmap->nx);
                    samples = dummy.data();
                    n_samples = dummy.size();
                } else {
                    const auto & buf = bitmap->get_ro_buf();
                    GGML_ASSERT(buf.size() > sizeof(float));
                    GGML_ASSERT(buf.size() % sizeof(float) == 0);

                    samples = (const float *)buf.data();
                    n_samples = buf.size() / sizeof(float);
                }
                if (!ctx->audio_preproc) {
                    LOG_ERR("%s: this model takes audio input only via mtmd_audio_stream\n", __func__);
                    return 2;
                }
                *source = ctx->audio_preproc->preprocess(samples, n_samples);
                if (source->entries.empty()) {
                    LOG_ERR("Unable to preprocess audio\n");
                    return 2;
                }
            }

            // consider each mel_spec as a separate audio chunk
            // TODO: maybe support batching, but this may come with memory cost
            for (size_t i = 0; i < source->entries.size(); i++) {
                mtmd_image_f32 mel_f32 = source->entries[i];

                size_t n_tokens = ctx->ctx_a->model.n_output_tokens(&mel_f32);

                mtmd_image_f32_batch batch_f32;
                batch_f32.is_audio = true;
                batch_f32.entries.push_back(std::move(mel_f32));

                mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
                audio_tokens->n_tokens = n_tokens;
                audio_tokens->batch_f32 = std::move(batch_f32);
                audio_tokens->id = bitmap->id; // optional
                if (source->needs_compute()) {
                    audio_tokens->source = source;
                    audio_tokens->source_idx = (uint32_t) i;
                }

                LOG_DBG("audio_tokens->n_tokens = %d\n", audio_tokens->n_tokens);

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    {}, // text tokens
                    nullptr, // image tokens
                    std::move(audio_tokens),
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->aud_end.empty()) {
                add_text(ctx->aud_end, true); // add audio end token
            }
        }

        return 0;
    }

    int32_t add_voice(const mtmd_bitmap & bitmap) {
        if (bitmap.ny == 0) {
            LOG_ERR("%s: error: voice \"%s\" holds no embeddings\n", __func__, bitmap.id.c_str());
            return 2;
        }
        auto track = std::make_shared<mtmd_audio_track>();
        track->n_tokens = bitmap.ny;
        const auto & buf = bitmap.get_ro_buf();
        track->embd.assign((const float *) buf.data(), (const float *) (buf.data() + buf.size()));
        track->gen_ref = bitmap.gen_ref;

        mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
        audio_tokens->n_tokens = bitmap.ny;
        audio_tokens->id       = bitmap.id;
        audio_tokens->track    = std::move(track);

        cur.entries.emplace_back(mtmd_input_chunk{
            MTMD_INPUT_CHUNK_TYPE_AUDIO,
            {}, // text tokens
            nullptr, // image tokens
            std::move(audio_tokens),
        });
        return 0;
    }

    // a piece of a soundtrack goes in without the audio markers, it lies inside the span of its video
    int32_t add_soundtrack_piece(soundtrack & st, const mtmd_bitmap & bitmap) {
        if (!st.track) {
            std::vector<float> samples;
            for (const auto * piece : st.pieces) {
                if (piece->is_placeholder()) {
                    samples.resize(samples.size() + piece->nx, 0.0f);
                } else {
                    const float * data = (const float *) piece->get_ro_buf().data();
                    samples.insert(samples.end(), data, data + piece->nx);
                }
            }
            if (!ctx->audio_preproc) {
                LOG_ERR("%s: unable to preprocess the soundtrack\n", __func__);
                return 2;
            }
            auto source = std::make_shared<mtmd_audio_preproc_out>(ctx->audio_preproc->preprocess(samples.data(), samples.size()));
            if (source->entries.size() != 1) {
                LOG_ERR("%s: unable to preprocess the soundtrack\n", __func__);
                return 2;
            }
            mtmd_image_f32 mel_f32 = source->entries[0];
            st.track = std::make_shared<mtmd_audio_track>();
            st.track->n_tokens = ctx->ctx_a->model.n_output_tokens(&mel_f32);
            if (source->needs_compute()) {
                st.track->source = source;
            }
            st.track->batch_f32.is_audio = true;
            st.track->batch_f32.entries.push_back(std::move(mel_f32));
        }

        // rows of [begin, end): round(sample / samples per row), the last piece takes the rest of the track
        const double samples_per_row = 8.0 * ctx->ctx_a->model.hparams.audio_hop_len; // one row per 8 mel frames
        const uint32_t n_tokens = st.track->n_tokens;
        st.n_samples += bitmap.nx;
        const uint32_t begin = st.n_assigned;
        const uint32_t end   = &bitmap == st.pieces.back() ? n_tokens
                             : std::clamp((uint32_t) std::nearbyint(st.n_samples / samples_per_row), begin, n_tokens);
        st.n_assigned = end;
        if (end == begin) {
            return 0;
        }

        mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
        audio_tokens->n_tokens  = end - begin;
        audio_tokens->track     = st.track;
        audio_tokens->row_begin = begin;
        audio_tokens->id        = bitmap.id;
        cur.entries.emplace_back(mtmd_input_chunk{
            MTMD_INPUT_CHUNK_TYPE_AUDIO,
            {}, // text tokens
            nullptr, // image tokens
            std::move(audio_tokens),
        });
        return 0;
    }

    // MiniCPM-V 4.7: the overview and all slices go in one chunk, the graph appends the separators after each tile:
    //   [ov] </image><slice> [S00] </slice><slice> [S01] </slice>\n<slice> [S10] </slice><slice> [S11] </slice>
    void add_canvas_chunk(mtmd_image_preproc_out && preproc_out, const std::string & id) {
        const int n_col = preproc_out.grid_x;
        const int n_row = preproc_out.grid_y;
        auto & slices = preproc_out.entries;
        GGML_ASSERT(preproc_out.has_overview());
        GGML_ASSERT((int) slices.size() == n_col * n_row);

        auto & ov = preproc_out.overview;
        ov.suffix_type = MTMD_SUFFIX_MINICPMV_OV;
        if (!slices.empty()) {
            ov.suffix_type = MTMD_SUFFIX_MINICPMV_OV_SLICE;
            ov.anyres.grid_x = n_col;
            ov.anyres.grid_y = n_row;
        }
        for (int y = 0; y < n_row; y++) {
            for (int x = 0; x < n_col; x++) {
                auto & suffix = slices[y * n_col + x].suffix_type;
                if (y == n_row - 1 && x == n_col - 1) {
                    suffix = MTMD_SUFFIX_MINICPMV_LAST;
                } else if (x == n_col - 1) {
                    suffix = MTMD_SUFFIX_MINICPMV_ROW_END;
                } else {
                    suffix = MTMD_SUFFIX_MINICPMV_SLICE;
                }
            }
        }

        mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
        image_tokens->pos = MTMD_POS_TYPE_CANVAS;
        image_tokens->id  = id;
        auto & entries = image_tokens->batch_f32.entries;
        entries.push_back(std::move(ov));
        for (auto & slice : slices) {
            entries.push_back(std::move(slice));
        }
        // token grid of the last entry, the grids of the other entries are scaled from it
        image_tokens->nx = ctx->ctx_v->model.n_output_tokens_x(&entries.back());
        image_tokens->ny = ctx->ctx_v->model.n_output_tokens_y(&entries.back());

        size_t n_tokens = 0;
        for (const auto & entry : entries) {
            n_tokens += ctx->ctx_v->model.n_output_tokens(&entry);
        }
        GGML_ASSERT(n_tokens == image_tokens->n_tokens());

        mtmd_input_chunk chunk{
            MTMD_INPUT_CHUNK_TYPE_IMAGE,
            {}, // text tokens
            std::move(image_tokens),
            nullptr, // audio tokens
        };
        cur.entries.emplace_back(std::move(chunk));
    }

    std::vector<mtmd_input_chunk> split_batch_to_chunk(mtmd_image_preproc_out && preproc_out, const std::string & id) {
        std::vector<mtmd_input_chunk> chunks;

        auto process_chunk = [&](mtmd_image_f32 && img) {
            mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
            image_tokens->nx = ctx->ctx_v->model.n_output_tokens(&img);
            image_tokens->ny = 1;
            image_tokens->batch_f32.entries.push_back(std::move(img));
            image_tokens->id = id;

            GGML_ASSERT(image_tokens->nx > 0);

            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_IMAGE,
                {}, // text tokens
                std::move(image_tokens),
                nullptr, // audio tokens
            };
            chunks.emplace_back(std::move(chunk));
        };

        // overview image first
        auto & overview = preproc_out.overview;
        if (overview.nx() == 0 || overview.ny() == 0) {
            throw std::runtime_error(mtmd_format("%s: invalid overview image for llava-uhd style preprocessing\n", __func__));
        }
        process_chunk(std::move(preproc_out.overview));

        // then, process slices
        for (auto & entry : preproc_out.entries) {
            if (entry.nx() == 0 || entry.ny() == 0) {
                throw std::runtime_error(mtmd_format("%s: invalid image slice for llava-uhd style preprocessing\n", __func__));
            }
            process_chunk(std::move(entry));
        }

        return chunks;
    }

    // for example: "a <__media__> b <__media__> c" --> "a", "<__media__>", "b", "<__media__>", "c"
    static std::vector<std::string> split_text(const std::string & input, const std::string & delimiter) {
        std::vector<std::string> result;
        if (input.empty()) {
            return result;
        }
        size_t start = 0;
        size_t pos = 0;
        while ((pos = input.find(delimiter, start)) != std::string::npos) {
            if (pos > start) {
                result.push_back(input.substr(start, pos - start));
            }
            result.push_back(delimiter);
            start = pos + delimiter.length();
        }
        if (start < input.length()) {
            result.push_back(input.substr(start));
        }
        return result;
    }

    // copied from common_tokenize
    static std::vector<llama_token> mtmd_tokenize_text_internal(
        const struct llama_vocab * vocab,
               const std::string & text,
                            bool   add_special,
                            bool   parse_special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        // upper limit for the number of tokens
        int n_tokens = text.length() + 2 * add_special;
        std::vector<llama_token> result(n_tokens);
        n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        if (n_tokens == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
        }
        if (n_tokens < 0) {
            result.resize(-n_tokens);
            int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
            GGML_ASSERT(check == -n_tokens);
        } else {
            result.resize(n_tokens);
        }
        return result;
    }
};

int32_t mtmd_tokenize(const mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_text * text,
            const mtmd_bitmap * const * bitmaps,
            size_t n_bitmaps) {
    try {
        mtmd_tokenizer tokenizer(ctx, text, bitmaps, n_bitmaps);
        return tokenizer.tokenize(output);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 2;
    }
}

int32_t mtmd_tokenize_from_parts(const mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_part * const * parts,
            size_t n_parts,
            bool add_special) {
    for (size_t i = 0; i < n_parts; i++) {
        if ((parts[i]->text == nullptr) == (parts[i]->bitmap == nullptr)) {
            LOG_ERR("%s: part %zu must have either text or bitmap set, not both\n", __func__, i);
            return 1;
        }
        if (parts[i]->text != nullptr && parts[i]->text->text == nullptr) {
            LOG_ERR("%s: part %zu has null text pointer\n", __func__, i);
            return 1;
        }
    }
    try {
        mtmd_tokenizer tokenizer(ctx, parts, n_parts, add_special);
        return tokenizer.tokenize(output);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 2;
    }
}

static int32_t mtmd_encode_impl(mtmd_context * ctx, const mtmd_image_tokens * image_tokens, std::vector<float> & out_embd) {
    mtmd_subcontext * sctx = ctx->ctx_v.get();
    if (!sctx) {
        LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
        return 1;
    }

    int n_embd_out = ctx->n_embd_out();
    auto n_tokens_out = image_tokens->n_tokens();
    out_embd.resize((size_t)n_embd_out * n_tokens_out);

    if (image_tokens->is_placeholder()) {
        LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
        return 1;
    }

    if (image_tokens->pos == MTMD_POS_TYPE_CANVAS) {
        // the tiles differ in size, encode them one by one
        size_t offset = 0;
        for (const auto & entry : image_tokens->batch_f32.entries) {
            mtmd_image_f32_batch one;
            one.entries.push_back(entry);
            std::vector<float> embd((size_t) n_embd_out * sctx->model.n_output_tokens(&entry));
            if (!sctx->encode_batch(ctx->n_threads, &one, embd)) {
                return 1;
            }
            GGML_ASSERT(offset + embd.size() <= out_embd.size());
            std::copy(embd.begin(), embd.end(), out_embd.begin() + offset);
            offset += embd.size();
        }
        GGML_ASSERT(offset == out_embd.size());
        return 0;
    }

    bool ok = sctx->encode_batch(ctx->n_threads,
        &image_tokens->batch_f32,
        out_embd);

    return ok ? 0 : 1;
}

// copy of batch_f32, the mel is computed from source when there is one
static bool mtmd_audio_batch_prepare(mtmd_context * ctx, const mtmd_image_f32_batch & batch_f32, mtmd_audio_preproc_out * source, uint32_t idx, mtmd_image_f32_batch & batch) {
    batch = batch_f32.clone();
    if (source) {
        if (!ctx->audio_preproc->compute(*source)) {
            LOG_ERR("%s: unable to compute the mel\n", __func__);
            return false;
        }
        GGML_ASSERT(idx < source->entries.size());
        batch.entries[0] = source->entries[idx];
    }
    return true;
}

static bool mtmd_audio_batch_encode(mtmd_context * ctx, const mtmd_image_f32_batch & batch_f32, mtmd_audio_preproc_out * source, uint32_t idx, std::vector<float> & out_embd) {
    mtmd_image_f32_batch batch;
    return mtmd_audio_batch_prepare(ctx, batch_f32, source, idx, batch) && ctx->ctx_a->encode_batch(ctx->n_threads, &batch, out_embd);
}

static int32_t mtmd_encode_chunk_impl(mtmd_context * ctx, const mtmd_input_chunk * chunk, std::vector<float> & out_embd) {
    ctx->gen_ref = {}; // see mtmd_gen_state_set_ref()
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_WRN("mtmd_encode_chunk has no effect for text chunks\n");
        return 0;
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        if (!ctx->ctx_v) {
            LOG_ERR("%s: model does not support vision input\n", __func__);
            return 1;
        }
        if (chunk->tokens_image == nullptr) {
            LOG_ERR("%s: image tokens are null\n", __func__);
            return 1;
        }
        if (chunk->tokens_image->is_placeholder()) {
            LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
            return 1;
        }
        return mtmd_encode_impl(ctx, chunk->tokens_image.get(), out_embd);
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        if (chunk->tokens_audio == nullptr) {
            LOG_ERR("%s: audio tokens are null\n", __func__);
            return 1;
        }
        if (chunk->tokens_audio->is_placeholder()) {
            LOG_ERR("%s: audio tokens batch is placeholder\n", __func__);
            return 1;
        }
        const auto & track = chunk->tokens_audio->track;
        // a filled track (e.g. a voice) needs no audio encoder
        if (!ctx->ctx_a && (!track || track->embd.empty())) {
            LOG_ERR("%s: model does not support audio input\n", __func__);
            return 1;
        }
        if (track) {
            if (track->embd.empty()) {
                track->embd.resize((size_t) track->n_tokens * ctx->n_embd_out());
                if (!mtmd_audio_batch_encode(ctx, track->batch_f32, track->source.get(), 0, track->embd)) {
                    track->embd.clear();
                    return 1;
                }
            }
            const size_t n_mmproj_embd = track->embd.size() / track->n_tokens;
            const size_t off = (size_t) chunk->tokens_audio->row_begin * n_mmproj_embd;
            out_embd.assign(track->embd.begin() + off,
                            track->embd.begin() + off + (size_t) chunk->tokens_audio->n_tokens * n_mmproj_embd);
            if (track->gen_ref) {
                ctx->gen_ref = *track->gen_ref;
            }
            return 0;
        }
        out_embd.resize((size_t)chunk->tokens_audio->n_tokens * ctx->n_embd_out());
        bool ok = mtmd_audio_batch_encode(ctx, chunk->tokens_audio->batch_f32, chunk->tokens_audio->source.get(), chunk->tokens_audio->source_idx, out_embd);
        return ok ? 0 : 1;
    }

    LOG_ERR("%s: unknown chunk type %d\n", __func__, (int)chunk->type);
    return 1;
}

int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    // this is the non-batching version
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    try {
        return mtmd_encode_chunk_impl(ctx, chunk, ctx->out_embd);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    try {
        return mtmd_encode_impl(ctx, image_tokens, ctx->out_embd);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx->out_embd.data();
}

//
// audio generation: no supported model generates audio, the API is kept for its callers
//

mtmd_gen_audio_info mtmd_gen_audio_get_info(const mtmd_context * /*ctx*/) {
    mtmd_gen_audio_info info{};
    info.type          = MTMD_GEN_AUDIO_TYPE_NONE;
    info.model_variant = "";
    info.n_channels    = 1;
    return info;
}

size_t mtmd_gen_audio_read_tensor(mtmd_context * /*ctx*/, const char * /*name*/, float * /*out*/, size_t /*n_max*/) {
    return 0;
}

const char * mtmd_gen_audio_voice_name(const mtmd_context * /*ctx*/, int32_t /*idx*/) {
    return nullptr;
}

mtmd_bitmap * mtmd_gen_audio_get_voice(const mtmd_context * /*ctx*/, const char * /*name*/) {
    return nullptr;
}

mtmd_gen_inp mtmd_gen_inp_default(const mtmd_context * /*ctx*/) {
    mtmd_gen_inp inp{};
    inp.type  = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
    inp.speed = 1.0f;
    return inp;
}

struct mtmd_gen_state {};

mtmd_gen_state_params mtmd_gen_state_params_default(void) {
    mtmd_gen_state_params params{};
    params.seed = UINT32_MAX;
    return params;
}

mtmd_gen_state * mtmd_gen_state_init(mtmd_context * /*ctx*/, mtmd_gen_state_params /*params*/) {
    LOG_ERR("%s: model does not support audio generation\n", __func__);
    return nullptr;
}

void mtmd_gen_state_free(mtmd_gen_state * state) {
    delete state;
}

void mtmd_gen_state_reset(mtmd_gen_state * /*state*/) {
}

int32_t mtmd_gen_state_set_ref(mtmd_context * /*ctx*/, mtmd_gen_state * /*state*/) {
    LOG_ERR("%s: invalid gen state\n", __func__);
    return 1;
}

const int32_t * mtmd_gen_state_get_ref_codes(const mtmd_gen_state * /*state*/, size_t * n_codes) {
    if (n_codes) {
        *n_codes = 0;
    }
    return nullptr;
}

int32_t mtmd_gen_audio_process(mtmd_context * /*ctx*/, mtmd_gen_state * /*state*/, const struct mtmd_gen_inp * /*inp*/, struct mtmd_gen_out * /*out*/) {
    LOG_ERR("%s: model does not support audio generation\n", __func__);
    return 1;
}

mtmd_batch * mtmd_batch_init(mtmd_context * ctx) {
    return new mtmd_batch(ctx);
}

void mtmd_batch_free(mtmd_batch * batch) {
    if (batch) {
        delete batch;
    }
}

int32_t mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("%s: text chunk is not supported in batch\n", __func__);
        return 1;
    }

    auto * ctx = batch->ctx->get_subctx(chunk);
    if (!ctx) {
        LOG_ERR("%s: model does not support input chunk type %d\n", __func__, (int)chunk->type);
        return 1;
    }

    if (batch->entries.empty()) {
        // batch must have at least one chunk
        batch->entries.push_back(chunk);
        return 0;
    }

    if (!ctx->support_batch) {
        // if no batching support, batch can only have one single chunk
        return 2; // "batch too large" error code
    }

    int32_t new_n_tokens = batch->n_tokens() + (int32_t)mtmd_input_chunk_get_n_tokens(chunk);
    if (new_n_tokens > batch->ctx->batch_max_tokens) {
        return 2; // "batch too large" error code
    }

    auto & first_chunk = batch->entries[0];
    if (first_chunk->can_batch_with(*chunk)) {
        batch->entries.push_back(chunk);
        return 0;
    }

    return 3; // "cannot batch" error code
}

static int32_t mtmd_batch_encode_impl(mtmd_batch * batch) {
    if (batch->entries.empty()) {
        LOG_ERR("%s: batch is empty\n", __func__);
        return 1;
    }
    for (const auto * chunk : batch->entries) {
        if (chunk->is_placeholder()) {
            LOG_ERR("%s: chunk is placeholder\n", __func__);
            return 1;
        }
    }

    // represent the whole batch as one single chunk
    mtmd::input_chunk_ptr batch_chunk(mtmd_input_chunk_copy(batch->entries[0]));
    if (batch_chunk->tokens_image) {
        auto & b0_f32 = batch_chunk->tokens_image->batch_f32;
        // copy all entries from other chunks into the first chunk's batch_f32
        // note: skip first entry because it's already in batch_chunk
        for (size_t ic = 1; ic < batch->entries.size(); ic++) {
            auto & chunk = batch->entries[ic];
            GGML_ASSERT(chunk->tokens_image);
            auto b1_f32 = chunk->tokens_image->batch_f32.clone();
            for (size_t i = 0; i < b1_f32.entries.size(); i++) {
                b0_f32.entries.push_back(std::move(b1_f32.entries[i]));
            }
        }
    } else if (batch_chunk->tokens_audio) {
        auto & b0_f32 = batch_chunk->tokens_audio->batch_f32;
        // copy all entries from other chunks into the first chunk's batch_f32
        // note: skip first entry because it's already in batch_chunk
        for (size_t ic = 1; ic < batch->entries.size(); ic++) {
            auto & chunk = batch->entries[ic];
            GGML_ASSERT(chunk->tokens_audio);
            auto b1_f32 = chunk->tokens_audio->batch_f32.clone();
            for (size_t i = 0; i < b1_f32.entries.size(); i++) {
                b0_f32.entries.push_back(std::move(b1_f32.entries[i]));
            }
        }
    } else {
        LOG_ERR("%s: unsupported chunk type\n", __func__);
        return 1;
    }

    LOG_DBG("%s: encoding batch with %zu entries and total %zu tokens\n",
            __func__, batch->entries.size(), mtmd_input_chunk_get_n_tokens(batch_chunk.get()));
    int32_t res = mtmd_encode_chunk_impl(
        batch->ctx,
        batch_chunk.get(),
        batch->output_embd);
    return res;
}

int32_t mtmd_batch_encode(mtmd_batch * batch) {
    std::lock_guard<std::mutex> lock(batch->ctx->compute_mutex);
    try {
        return mtmd_batch_encode_impl(batch);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_batch_get_output_embd(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    if (batch->output_embd.empty()) {
        LOG_ERR("%s: batch has not been encoded yet\n", __func__);
        return nullptr;
    }
    size_t offset = 0;
    const size_t n_embd = batch->ctx->n_embd_out();
    for (const auto * c : batch->entries) {
        size_t offset_prev = offset;
        size_t n_tokens = mtmd_input_chunk_get_n_tokens(c);
        offset += n_tokens * n_embd;
        GGML_ASSERT(offset_prev <  batch->output_embd.size());
        GGML_ASSERT(offset      <= batch->output_embd.size());
        if (c == chunk) {
            return &batch->output_embd.data()[offset_prev];
        }
    }
    return nullptr; // not found
}

bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    auto proj_type = ctx->proj_type_v();
    if (chunk && chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        proj_type = ctx->proj_type_a();
    }
    switch (proj_type) {
        default:
            return false;
    }
}

bool mtmd_decode_use_mrope(const mtmd_context * ctx) {
    return ctx->pos_type == MTMD_POS_TYPE_MROPE;
}

bool mtmd_support_vision(const mtmd_context * ctx) {
    return ctx->ctx_v != nullptr;
}

bool mtmd_support_audio(const mtmd_context * ctx) {
    return ctx->ctx_a != nullptr;
}

int mtmd_get_audio_sample_rate(const mtmd_context * ctx) {
    if (!ctx->ctx_a) {
        return -1;
    }
    return ctx->ctx_a->model.hparams.audio_sample_rate;
}

// no kept model has a video template of its own
mtmd_video_config mtmd_get_video_config(const mtmd_context * /*ctx*/,
                                        int32_t /*src_width*/, int32_t /*src_height*/,
                                        int32_t /*src_n_frames*/, float /*src_fps*/) {
    return { MTMD_VIDEO_TEMPLATE_DEFAULT, 0.0f, 0, 0, 0 };
}

const char * mtmd_get_marker(const mtmd_context * ctx) {
    return ctx->media_marker.c_str();
}

//
// public API functions
//

// mtmd_bitmap

mtmd_bitmap * mtmd_bitmap_init(uint32_t nx,
                               uint32_t ny,
                               const unsigned char * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap(data, nx, ny);
    return bitmap;
}

mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples,
                                          const float * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap((const unsigned char *)data, n_samples);
    GGML_ASSERT(bitmap->is_audio);
    if (!bitmap->is_placeholder()) {
        GGML_ASSERT(bitmap->get_ro_buf().size() == n_samples * sizeof(float));
    }
    return bitmap;
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap) {
    return bitmap->nx;
}

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap) {
    return bitmap->ny;
}

const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap) {
    if (bitmap->is_placeholder()) {
        return nullptr;
    }
    return bitmap->get_ro_buf().data();
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap) {
    if (bitmap->is_placeholder()) {
        return 0;
    }
    return bitmap->get_ro_buf().size();
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap) {
    return bitmap->is_audio;
}

bool mtmd_bitmap_is_voice(const mtmd_bitmap * bitmap) {
    return bitmap->is_voice;
}

const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap) {
    return bitmap->id.c_str();
}

void mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id) {
    if (id) {
        bitmap->id = std::string(id);
    } else {
        bitmap->id.clear();
    }
}

void mtmd_bitmap_set_mergeable(mtmd_bitmap * bitmap, bool mergeable) {
    bitmap->mergeable = mergeable;
}

void mtmd_bitmap_set_is_video(mtmd_bitmap * bitmap, bool is_video) {
    bitmap->is_video = is_video;
}

void mtmd_bitmap_set_transcript(mtmd_bitmap * bitmap, const char * transcript) {
    if (transcript) {
        bitmap->transcript = std::string(transcript);
    } else {
        bitmap->transcript.clear();
    }
}

mtmd_bitmap * mtmd_bitmap_init_lazy(const mtmd_context * ctx,
                                    const char * id,
                                    void * user_data,
                                    mtmd_bitmap_lazy_callback callback) {
    GGML_UNUSED(ctx); // reserved for future use
    mtmd_bitmap * bitmap = new mtmd_bitmap(nullptr, 0, 0);
    bitmap->lazy_callback = callback;
    bitmap->lazy_user_data = user_data;
    mtmd_bitmap_set_id(bitmap, id);
    return bitmap;
}

void mtmd_bitmap_free(mtmd_bitmap * bitmap) {
    if (bitmap) {
        delete bitmap;
    }
}

// mtmd_input_chunks

mtmd_input_chunks * mtmd_input_chunks_init() {
    return new mtmd_input_chunks;
}

size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks) {
    return chunks->entries.size();
}

const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx) {
    if (idx >= chunks->entries.size()) {
        return nullptr;
    }
    return &chunks->entries[idx];
}

void mtmd_input_chunks_free(mtmd_input_chunks * chunks) {
    if (chunks) {
        delete chunks;
    }
}

// mtmd_input_chunk

enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk) {
    return chunk->type;
}

const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        *n_tokens_output = chunk->tokens_text.size();
        return chunk->tokens_text.data();
    }
    *n_tokens_output = 0;
    return nullptr;
}

const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return chunk->tokens_image.get();
    }
    return nullptr;
}

size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_tokens(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_pos(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

const char * mtmd_input_chunk_get_id(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return chunk->tokens_image->id.c_str();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->id.c_str();
    }
    return nullptr;
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk) {
    mtmd_input_chunk * copy = new mtmd_input_chunk{
        chunk->type,
        chunk->tokens_text,
        nullptr,
        nullptr,
    };
    if (chunk->tokens_image) {
        // copy the image tokens
        copy->tokens_image = mtmd_image_tokens_ptr(new mtmd_image_tokens());
        *copy->tokens_image = chunk->tokens_image->clone();
    }
    if (chunk->tokens_audio) {
        // copy the audio tokens
        copy->tokens_audio = mtmd_audio_tokens_ptr(new mtmd_audio_tokens());
        *copy->tokens_audio = chunk->tokens_audio->clone();
    }
    return copy;
}

void mtmd_input_chunk_free(mtmd_input_chunk * chunk) {
    if (chunk) {
        delete chunk;
    }
}

// returns 0 on success
static int32_t mtmd_input_chunk_save_impl(const mtmd_input_chunk * chunk, std::vector<char> & out_buf) {
    try {
        mtmd_serialization ser(MTMD_SERIALIZATION_VERSION);
        chunk->serialize(ser);
        out_buf = std::move(ser.data);
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return -1;
    }
}

mtmd_input_chunk * mtmd_input_chunk_get_placeholder(const mtmd_input_chunk * chunk) {
    // this is hacky, but still faster than copy the whole batch data
    std::vector<char> buf;
    if (mtmd_input_chunk_save_impl(chunk, buf) != 0) {
        return nullptr;
    }
    return mtmd_input_chunk_load(buf.data(), buf.size());
}

int32_t mtmd_input_chunk_save(const mtmd_input_chunk * chunk, char * out_buf, size_t out_len, size_t * expected_out_len) {
    std::vector<char> buf;
    if (mtmd_input_chunk_save_impl(chunk, buf) != 0) {
        return -1;
    }
    if (expected_out_len) {
        *expected_out_len = buf.size();
    }
    if (!out_buf) {
        // caller is only querying the required size
        return 0;
    }
    if (out_len < buf.size()) {
        LOG_ERR("%s: out_buf is too small, need %zu bytes, got %zu\n", __func__, buf.size(), out_len);
        return -1;
    }
    std::memcpy(out_buf, buf.data(), buf.size());
    return 0;
}

mtmd_input_chunk * mtmd_input_chunk_load(const char * buf, size_t len) {
    try {
        mtmd_serialization ser(MTMD_SERIALIZATION_VERSION, buf, len);
        mtmd::input_chunk_ptr chunk(new mtmd_input_chunk());
        chunk->deserialize(ser);
        return chunk.release();
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
}

// mtmd_image_tokens

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens) {
    return image_tokens->n_tokens();
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens) {
    return image_tokens->nx;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens) {
    return image_tokens->ny;
}

// map a tile coordinate onto the canvas like the reference: round(linspace(0, canvas - 1, grid)), round() breaks ties to even
static uint32_t mtmd_canvas_scale(uint32_t coord, uint32_t grid, uint32_t canvas) {
    if (grid <= 1 || canvas <= 1) {
        return 0;
    }
    const double v = (double) coord * (double) (canvas - 1) / (double) (grid - 1);
    return std::min((uint32_t) std::nearbyint(v), canvas - 1);
}

// MTMD_POS_TYPE_CANVAS: every tile shares the <image> token before the chunk as origin
// the overview is stretched over the whole canvas, each slice fills its own cell; the time component is the origin, in slot z
// a tile takes one position in slot t (the KV cache position), the separators after it take one position each
static mtmd_decoder_pos mtmd_canvas_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    const auto & entries = image_tokens->batch_f32.entries;
    const auto & grid    = entries[0].anyres;
    const uint32_t nx = image_tokens->nx;
    const uint32_t ny = image_tokens->ny;
    const uint32_t canvas_w = grid.is_tiled() ? grid.grid_x * nx : nx;
    const uint32_t canvas_h = grid.is_tiled() ? grid.grid_y * ny : ny;
    const uint32_t base = pos_0 - 1;

    mtmd_decoder_pos pos;
    uint32_t t = pos_0;
    for (size_t k = 0; k < entries.size(); ++k) {
        const auto [gw, gh] = image_tokens->canvas_tile_grid(k);
        if (i < gw * gh) {
            const uint32_t row = i / gw;
            const uint32_t col = i % gw;
            uint32_t h;
            uint32_t w;
            if (k == 0) {
                h = mtmd_canvas_scale(row, gh, canvas_h);
                w = mtmd_canvas_scale(col, gw, canvas_w);
            } else {
                const uint32_t s = k - 1;
                h = (s / grid.grid_x) * ny + row;
                w = (s % grid.grid_x) * nx + col;
            }
            pos.t = t;
            pos.x = base + w;
            pos.y = base + h;
            pos.z = base;
            return pos;
        }
        i -= gw * gh;

        const size_t n_sep = mtmd_suffix_rows(entries[k].suffix_type).size();
        if (i < n_sep) {
            const uint32_t p = t + 1 + i;
            pos.t = p;
            pos.x = p;
            pos.y = p;
            pos.z = p;
            return pos;
        }
        i -= n_sep;
        t += 1 + n_sep;
    }
    GGML_ABORT("token index out of range");
}

mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    mtmd_decoder_pos pos;
    switch (image_tokens->pos) {
        case MTMD_POS_TYPE_MROPE:
            {
                pos.t = pos_0;
                pos.x = pos_0 + (i % image_tokens->nx);
                pos.y = pos_0 + (i / image_tokens->nx);
                pos.z = 0; // unused for now
            } break;
        case MTMD_POS_TYPE_NORMAL:
            {
                pos.t = pos_0 + i;
                pos.x = pos_0 + i;
                pos.y = pos_0 + i;
                pos.z = pos_0 + i;
            } break;
        case MTMD_POS_TYPE_SENSENOVA_U1:
            {
                pos.t = pos_0;
                pos.x = i % image_tokens->nx;
                pos.y = i / image_tokens->nx;
                pos.z = 0;
            } break;
        case MTMD_POS_TYPE_HUNYUANVL:
            {
                // HunyuanVL layout: [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
                // Total = 1 + ny*(nx+1) + 1. BOI and EOI use sequential positions in every dim;
                // content and row-newline tokens use (row, col) with XD-RoPE dim-3 = image_idx.
                const uint32_t nx      = image_tokens->nx;
                const uint32_t n_total = image_tokens->n_tokens();
                if (i == 0) {
                    // BOI
                    pos.t = pos_0 + i;
                    pos.x = pos_0 + i;
                    pos.y = pos_0 + i;
                    pos.z = pos_0 + i;
                } else if (i == n_total - 1) {
                    // EOI
                    pos.t = pos_0 + i;
                    pos.x = pos_0 + i;
                    pos.y = pos_0 + i;
                    pos.z = pos_0 + i;
                } else {
                    // content token at (row, col), or the trailing newline of a row (col == nx)
                    //   section 0 = sequential, section 1 = w(col), section 2 = h(row), section 3 = image_count.
                    // set_position_mrope_2d writes .y -> section 1 and .x -> section 2
                    const uint32_t offset = (uint32_t)i - 1;
                    const uint32_t row    = offset / (nx + 1);
                    const uint32_t col    = offset % (nx + 1);
                    pos.t = pos_0 + i;
                    pos.x = row;
                    pos.y = col;
                    pos.z = image_tokens->image_idx;
                }
            } break;
        case MTMD_POS_TYPE_CANVAS:
            {
                pos = mtmd_canvas_decoder_pos(image_tokens, pos_0, i);
            } break;
        default:
            GGML_ABORT("invalid position type");
    }
    return pos;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens) {
    return image_tokens->id.c_str();
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    switch (image_tokens->pos) {
        case MTMD_POS_TYPE_MROPE:
            return std::max(image_tokens->nx, image_tokens->ny);
        case MTMD_POS_TYPE_NORMAL:
            return image_tokens->n_tokens();
        case MTMD_POS_TYPE_SENSENOVA_U1:
            return 1;
        case MTMD_POS_TYPE_HUNYUANVL:
            // HunyuanVL: the sequential (dim-0) position advances by the full token count
            // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
            return image_tokens->n_tokens();
        case MTMD_POS_TYPE_CANVAS:
            {
                // one position per tile, plus one per separator
                llama_pos n_pos = 0;
                for (const auto & entry : image_tokens->batch_f32.entries) {
                    n_pos += 1 + (llama_pos) mtmd_suffix_rows(entry.suffix_type).size();
                }
                return n_pos;
            }
        default:
            GGML_ABORT("invalid position type");
    }
}

// test function

mtmd_input_chunks * mtmd_test_create_input_chunks() {
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        return nullptr;
    }

    // create a text chunk
    std::vector<llama_token> tokens_text = { 1, 2, 3, 4, 5 };
    mtmd_input_chunk chunk_text{
        MTMD_INPUT_CHUNK_TYPE_TEXT,
        std::move(tokens_text),
        nullptr, // image tokens
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_text));

    // create an image chunk
    mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
    image_tokens->nx = 4;
    image_tokens->ny = 4;
    image_tokens->batch_f32.entries.resize(16);
    image_tokens->id = "image_1";
    mtmd_input_chunk chunk_image{
        MTMD_INPUT_CHUNK_TYPE_IMAGE,
        {}, // text tokens
        std::move(image_tokens),
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_image));

    return chunks;
}

struct mtmd_logger_state g_logger_state = {mtmd_log_callback_default, NULL};

void mtmd_log_set(ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : mtmd_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

struct mtmd_caps mtmd_get_cap_from_file(const char * fname) {
    try {
        auto tmp = mtmd_model_get_caps(fname);
        mtmd_caps cap;
        cap.inp_audio  = tmp.has_audio;
        cap.inp_vision = tmp.has_vision;
        cap.gen_audio  = tmp.has_gen_audio;
        return cap;
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to get capabilities from file '%s': %s\n", __func__, fname, e.what());
        return mtmd_caps{ false, false, false };
    }
}

//
// Debugging API (NOT intended for public use)
//

static void mtmd_debug_encode_impl(mtmd_context * ctx, mtmd_subcontext * sctx, mtmd_image_f32 & image) {
    sctx->debug_output_embeddings = true;
    int n_mmproj_embd = sctx->model.n_mmproj_embd();
    int n_tokens = sctx->model.n_output_tokens(&image);
    std::vector<float> embd_output(n_tokens * n_mmproj_embd, 0.0f);
    bool ok = sctx->encode_image(ctx->n_threads,
        &image,
        embd_output);
    if (!ok) {
        LOG_ERR("%s: failed to encode image\n", __func__);
    }
}

void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    const int img_sz = (int)image.size();
    std::vector<float> img_buf;
    img_buf.reserve(img_sz * img_sz);
    for (const auto & row : image) {
        img_buf.insert(img_buf.end(), row.begin(), row.end());
    }
    mtmd_image_f32 inp_image;
    inp_image.set_size({img_sz, img_sz}, false, false);
    inp_image.cpy_buf(img_buf);
    LOG_INF("%s: created input image with nx=%d, ny=%d\n", __func__, img_sz, img_sz);
    mtmd_debug_encode_impl(ctx, ctx->ctx_v.get(), inp_image);
}

void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (!ctx->ctx_a) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    int n_mel = ctx->ctx_a->model.hparams.n_mel_bins;
    const int audio_nx = (int)input.size();
    std::vector<float> audio_buf(audio_nx * n_mel);
    for (int i = 0; i < audio_nx; i++) {
        for (int j = 0; j < n_mel; j++) {
            audio_buf[j * audio_nx + i] = input[i];
        }
    }
    mtmd_image_f32 inp_audio;
    inp_audio.set_size({audio_nx, n_mel}, false, true);
    inp_audio.cpy_buf(audio_buf);
    LOG_INF("%s: created input audio with nx=%d, ny=%d\n", __func__, audio_nx, n_mel);
    mtmd_debug_encode_impl(ctx, ctx->ctx_a.get(), inp_audio);
}

void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    mtmd_image_u8 img_u8;
    img_u8.set_size({nx, ny}, false);
    img_u8.cpy_buf(rgb_values);
    GGML_ASSERT(ctx->image_preproc != nullptr);
    mtmd_image_preproc_out preproc_out = ctx->image_preproc->preprocess(img_u8);

    mtmd_image_f32_batch batch_f32;
    batch_f32.is_audio = false;
    for (auto & entry : preproc_out.entries) {
        batch_f32.entries.push_back(std::move(entry));
    }

    LOG_INF("%s: preprocessed image to batch_f32 with %d entries\n", __func__, (int)batch_f32.entries.size());
    for (size_t i = 0; i < batch_f32.entries.size(); i++) {
        LOG_INF("%s: entry %zu has nx=%d, ny=%d\n", __func__, i, batch_f32.entries[i].nx(), batch_f32.entries[i].ny());
        // TODO: better way to dump entry content?
    }
}

void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & samples) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (!ctx->ctx_a || !ctx->audio_preproc) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    mtmd_audio_preproc_out out = ctx->audio_preproc->preprocess(samples.data(), samples.size());
    if (out.entries.empty() || !ctx->audio_preproc->compute(out)) {
        LOG_ERR("%s: failed to preprocess audio\n", __func__);
        return;
    }
    LOG_INF("%s: preprocessed audio to %zu mel spec chunks\n", __func__, out.entries.size());
    for (size_t i = 0; i < out.entries.size(); i++) {
        const auto & mel = out.entries[i];
        const auto & data = mel.get_ro_buf();
        LOG_INF("%s: mel spec chunk %zu has n_len=%d, n_mel=%d\n", __func__, i, mel.nx(), mel.ny());

        // dump mel entries: data is stored as [n_mel][n_len] (mel-major)
        for (int m = 0; m < mel.ny(); m++) {
            for (int t = 0; t < mel.nx(); t++) {
                LOG_INF("mel[%zu][m=%d][t=%d] = %f\n", i, m, t, data[(size_t) m * mel.nx() + t]);
            }
        }
    }
}

static void stub_log_callback(enum ggml_log_level, const char *, void *) {
    // do nothing
}

mtmd_memory_usage mtmd_get_memory_usage(const char * mmproj_fname,
                                        struct mtmd_context_params ctx_params) {
    mtmd::context_ptr ctx;
    auto saved_log_callback = g_logger_state.log_callback;
    auto saved_log_user_data = g_logger_state.log_callback_user_data;

    ctx_params.progress_callback = nullptr;

    try {
        mtmd_log_set(stub_log_callback, nullptr); // suppress logging
        ctx.reset(new mtmd_context(mmproj_fname, nullptr, ctx_params, true));
        mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
        // compute buffers are shared, weights are per model
        std::map<ggml_backend_dev_t, size_t> total_mem = ctx->backends->get_mem_compute();
        for (const auto * m : { ctx->models.v.get(), ctx->models.a.get(), ctx->models.gen_a.get() }) {
            if (m) {
                for (auto & [dev, size] : m->mem_weights) {
                    total_mem[dev] += size;
                }
            }
        }
        mtmd_memory_usage res;
        res.backend_mem_usage = std::move(total_mem);
        res.image_max_tokens  = ctx->ctx_v ? ctx->ctx_v->model.image_max_tokens() : -1;
        res.use_non_causal    = ctx->ctx_v ? mtmd_decode_use_non_causal(ctx.get(), nullptr) : false;
        return res;
    } catch (const std::exception & e) {
        mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return {{}, -1, false};
    }
}

//
// realtime audio input: no supported model streams audio, the API is kept for its callers
//

struct mtmd_audio_stream {};

bool mtmd_support_audio_stream(const mtmd_context * /*ctx*/) {
    return false;
}

mtmd_audio_stream_params mtmd_audio_stream_params_default(void) {
    mtmd_audio_stream_params params;
    params.step_ms  = 320;
    params.prompt   = nullptr;
    params.n_prompt = 0;
    return params;
}

mtmd_audio_stream * mtmd_audio_stream_init(mtmd_context * /*ctx*/, mtmd_audio_stream_params /*params*/) {
    LOG_ERR("%s: the model does not support audio streaming\n", __func__);
    return nullptr;
}

void mtmd_audio_stream_free(mtmd_audio_stream * stream) {
    delete stream;
}

int32_t mtmd_audio_stream_push(mtmd_audio_stream * /*stream*/, const float * /*samples*/, size_t /*n_samples*/) {
    return 1;
}

int32_t mtmd_audio_stream_flush(mtmd_audio_stream * /*stream*/, bool /*end*/) {
    return 1;
}

int32_t mtmd_audio_stream_step(mtmd_audio_stream * /*stream*/, llama_token /*last_token*/) {
    return -2;
}

const float * mtmd_audio_stream_get_output_embd(const mtmd_audio_stream * /*stream*/) {
    return nullptr;
}

bool mtmd_audio_stream_is_seq_start(const mtmd_audio_stream * /*stream*/) {
    return false;
}

int64_t mtmd_audio_stream_get_n_rows(const mtmd_audio_stream * /*stream*/) {
    return 0;
}

int64_t mtmd_audio_stream_get_n_rows_flushed(const mtmd_audio_stream * /*stream*/) {
    return 0;
}
