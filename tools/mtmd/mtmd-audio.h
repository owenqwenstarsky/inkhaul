#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd-model.h"

#include <cstdint>
#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

struct mtmd_audio_mel {
    int64_t n_len;
    int64_t n_len_org;
    int64_t n_mel;
    int64_t n_ctx = 0; // see mtmd_image_f32::audio_n_ctx

    std::vector<float> data;
};

struct mtmd_audio_mel_filters {
    int64_t n_mel;
    int64_t n_fft;

    std::vector<float> data;
};

// cache for audio processing, each processor instance owns its own cache
struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    std::vector<float> hann_window;

    mtmd_audio_mel_filters filters;

    void fill_sin_cos_table(uint32_t n);

    void fill_hann_window(uint32_t length, bool periodic);

    // Build mel filterbank matrix [n_mel × n_fft_bins] at runtime.
    // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
    void fill_mel_filterbank_matrix(int64_t n_mel,
                                    int64_t n_fft,
                                    int   sample_rate,               // e.g. 16000
                                    float fmin             = 0.0f,   // e.g. 0.0
                                    float fmax             = -1.0f,  // e.g. sr/2; pass -1 for auto
                                    bool  slaney_area_norm = true,
                                    float scale            = 1.0f,
                                    bool  use_htk          = false
    );
};

// FFT of N real samples -> N complex values (interleaved), in holds N floats, out needs 8 * N floats (scratch)
void mtmd_audio_fft(const mtmd_audio_cache & cache, const float * in, int N, float * out);
// inverse FFT of N complex values (interleaved), scaled by 1 / N; in holds 2 * N floats, out needs 8 * N floats (scratch)
void mtmd_audio_ifft(const mtmd_audio_cache & cache, const float * in, int N, float * out);

// compute one log10-mel frame from n_fft windowed samples, cache must be initialized for n_fft
void mtmd_audio_log_mel_frame(const mtmd_audio_cache & cache, const float * samples, int n_fft, int n_mel, float * out);

struct mtmd_audio_mel_config {
    int64_t n_mel            = 0;
    int64_t n_fft_bins       = 0;
    int32_t hann_window_size = 0;
    int32_t hop_length       = 0;
    int32_t sample_rate      = 0;
    bool    no_padding       = false;
    bool    center_padding   = false;
    float   preemph          = 0.f;
    bool    use_natural_log  = false;
    bool    norm_per_feature = false;
    bool    use_magnitude    = false;  // |X| instead of |X|^2
    float   mel_floor        = 5.960464477539063e-08f;
    bool    mel_floor_add    = false;  // log(x + floor) instead of log(max(x, floor))
    bool    std_eps_after_sqrt = false;  // std + eps instead of sqrt(var + eps)
    int     n_threads        = 1;

    const mtmd_audio_cache * cache = nullptr;
    ggml_backend_sched_t     sched = nullptr;
};

struct mtmd_audio_preproc_out {
    std::vector<mtmd_image_f32> entries; // one per chunk, empty on failure
    std::vector<float> samples;          // input of compute(), empty if entries already hold the data

    bool needs_compute() const {
        return !samples.empty() && (entries.empty() || entries[0].is_placeholder());
    }
};

struct mtmd_audio_preprocessor {
    const mtmd_hparams & hparams;
    mtmd_audio_mel_config mel_config; // filled by initialize() of the preprocessors that use the log-mel graph

    // mel_config.sched must be set before the first compute(), see mtmd_context::init_audio()
    mtmd_audio_preprocessor(const mtmd_model * model): hparams(model->hparams) {}
    mtmd_audio_preprocessor(const mtmd_hparams & hparams, ggml_backend_sched_t sched): hparams(hparams) {
        mel_config.sched = sched;
    }

    // mel_config points to the cache of this instance
    mtmd_audio_preprocessor(const mtmd_audio_preprocessor &) = delete;
    mtmd_audio_preprocessor & operator=(const mtmd_audio_preprocessor &) = delete;

    virtual ~mtmd_audio_preprocessor() = default;
    virtual void initialize() = 0; // NOT thread-safe

    // entries have the final shapes, but no data if needs_compute()
    mtmd_audio_preproc_out preprocess(const float * samples, size_t n_samples) const;

    // run it on the thread that encodes, it uses mel_config.sched
    bool compute(mtmd_audio_preproc_out & out) const;

  protected:
    // shape_only skips the log-mel graph: the shapes are right, the data is zero-filled
    virtual bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const = 0;
};
