#include "mtmd-audio.h"
#include "mtmd-subcontext.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <algorithm>
#include <functional>

// some of the code here is copied from whisper.cpp

constexpr bool DEBUG = false;

void mtmd_audio_cache::fill_sin_cos_table(uint32_t n) {
    sin_vals.resize(n);
    cos_vals.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        double theta = (2 * M_PI * i) / n;
        sin_vals[i]  = sinf(theta);
        cos_vals[i]  = cosf(theta);
    }
}

void mtmd_audio_cache::fill_hann_window(uint32_t length, bool periodic) {
    hann_window.resize(length);
    int offset = periodic ? 0 : -1;
    for (uint32_t i = 0; i < length; i++) {
        hann_window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

void mtmd_audio_cache::fill_mel_filterbank_matrix(int64_t n_mel,
                                                  int64_t n_fft,
                                                  int   sample_rate,
                                                  float fmin,
                                                  float fmax,
                                                  bool  slaney_area_norm,
                                                  float scale,
                                                  bool  use_htk) {
    GGML_ASSERT(n_mel > 0 && n_fft > 1);
    if (fmax <= 0.0f) {
        fmax = 0.5f * sample_rate;
    }

    std::function<double(double)> hz_to_mel;
    std::function<double(double)> mel_to_hz;

    if (use_htk) {
        hz_to_mel = [](const double f_hz) -> double {
            return 2595.0 * log10(1.0 + f_hz / 700.0);
        };
        mel_to_hz = [](const double m) -> double {
            return 700.0 * (pow(10.0, m / 2595.0) - 1.0);
        };
    } else {
        // Slaney scale (matches librosa default)
        const double min_log_hz  = 1000.0;
        const double lin_slope   = 3 / 200.;
        const double min_log_mel = min_log_hz * lin_slope;
        const double log_step    = log(6.4) / 27.0;
        hz_to_mel = [min_log_hz, lin_slope, log_step, min_log_mel](const double f_hz) -> double {
            return (f_hz < min_log_hz) ? f_hz * lin_slope : min_log_mel + log(f_hz / min_log_hz) / log_step;
        };
        mel_to_hz = [min_log_hz, lin_slope, log_step, min_log_mel](const double m) -> double {
            return (m < min_log_mel) ? m / lin_slope : min_log_hz * exp((m - min_log_mel) * log_step);
        };
    }

    // infer N_fft from n_fft_bins
    const double bin_hz_step = double(sample_rate) / double(n_fft);

    // mel grid: n_mel + 2 edges
    const double        m_lo = hz_to_mel(fmin);
    const double        m_hi = hz_to_mel(fmax);
    std::vector<double> mel_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        mel_pts[i] = m_lo + (m_hi - m_lo) * (double(i) / (n_mel + 1));
    }

    // convert to Hz
    std::vector<double> hz_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        hz_pts[i] = mel_to_hz(mel_pts[i]);
    }

    const int64_t n_fft_bins = n_fft / 2 + 1;

    // Validate allocation size
    if ((size_t)n_mel * (size_t)n_fft_bins > SIZE_MAX) {
        GGML_ASSERT(false && "mel filterbank allocation too large");
    }

    // filterbank
    std::vector<float> out((size_t)n_mel * (size_t)n_fft_bins, 0);
    for (int64_t m = 0; m < n_mel; ++m) {
        const double f_left   = hz_pts[m];
        const double f_center = hz_pts[m + 1];
        const double f_right  = hz_pts[m + 2];

        const double denom_l = std::max(1e-30, f_center - f_left);
        const double denom_r = std::max(1e-30, f_right - f_center);
        const double enorm   = slaney_area_norm ? (2.0 / std::max(1e-30, f_right - f_left)) : 1.0;

        for (int k = 0; k < n_fft_bins; ++k) {
            const double f = k * bin_hz_step;
            double       w = 0.0;
            if (f >= f_left && f <= f_center) {
                w = (f - f_left) / denom_l;
            } else if (f > f_center && f <= f_right) {
                w = (f_right - f) / denom_r;
            }
            out[size_t(m) * size_t(n_fft_bins) + size_t(k)] = float(w * enorm * scale);
        }
    }

    filters.n_mel = n_mel;
    filters.n_fft = n_fft;
    filters.data  = std::move(out);

    if (DEBUG) {  // debug
        for (size_t i = 0; i < filters.data.size(); ++i) {
            if (filters.data[i] != 0.0f) {
                printf("filters[%zu] = %f\n", i, filters.data[i] * 1000.0f);
            }
        }
    }
}

// FFT/IFFT of N = n_sub * n_odd points (n_sub a power of two, n_odd odd): batched odd-size DFT of the n_sub subsequences into bit-reversed blocks, then iterative radix-2 stages
// matches the previous recursive implementation for finite inputs (the twiddle-1 multiply of the first stage is skipped when n_odd == 1)
// Template parameters:
//   Inverse: false = FFT with exp(-2 pi i k / N), no scaling
//            true  = IFFT with exp(+2 pi i k / N), scales by 1/N (1/n_odd in the DFT, 0.5 per stage)
//   RealInput: true = input is real-valued (stride 1)
//              false = input is complex-valued (interleaved real/imag, stride 2)
// in is not modified, out needs 2 * N floats + 2 * max(N / 2, n_sub) floats of scratch
template <bool Inverse, bool RealInput>
static void fft_impl(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    GGML_ASSERT(N > 0);
    const int n_sin_cos_vals = cache.sin_vals.size();

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    constexpr float scale = Inverse ? 0.5f : 1.0f;
    constexpr int   n_in  = RealInput ? 1 : 2; // floats per input value

    int n_odd = N;
    while (n_odd % 2 == 0) {
        n_odd /= 2;
    }
    const int n_sub = N / n_odd;

    // j = bitrev(r) for r = 0, 1, ...: add 1 to j starting from its highest bit h
    auto bitrev_next = [](int & j, int h) {
        for (; j & h; h >>= 1) {
            j ^= h;
        }
        j |= h;
    };

    float * scratch = out + 2 * N;
    int     len     = 2 * n_odd; // size of the blocks after the next stage
    if (n_odd > 1) {
        const int   sin_cos_step = n_sin_cos_vals / n_odd;
        const float dft_scale    = Inverse ? (1.0f / n_odd) : 1.0f;
        float * acc = scratch;
        for (int k = 0; k < n_odd; k++) {
            std::fill(acc, acc + 2 * n_sub, 0.0f);
            int idx = 0; // (k * n * sin_cos_step) % n_sin_cos_vals
            for (int n = 0; n < n_odd; n++) {
                const float   cos_val = cache.cos_vals[idx];
                const float   sin_val = cache.sin_vals[idx];
                const float * x       = in + n * n_sub * n_in;
                for (int r = 0; r < n_sub; r++) {
                    if constexpr (RealInput) {
                        acc[2 * r + 0] += x[r] * cos_val;
                        acc[2 * r + 1] += sign * x[r] * sin_val;
                    } else {
                        // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                        acc[2 * r + 0] += x[2 * r + 0] * cos_val - sign * x[2 * r + 1] * sin_val;
                        acc[2 * r + 1] += sign * x[2 * r + 0] * sin_val + x[2 * r + 1] * cos_val;
                    }
                }
                idx += k * sin_cos_step;
                if (idx >= n_sin_cos_vals) {
                    idx -= n_sin_cos_vals;
                }
            }
            for (int r = 0, j = 0; r < n_sub; r++, bitrev_next(j, n_sub / 2)) {
                out[2 * (j * n_odd + k) + 0] = acc[2 * r + 0] * dft_scale;
                out[2 * (j * n_odd + k) + 1] = acc[2 * r + 1] * dft_scale;
            }
        }
    } else if (N == 1) {
        out[0] = in[0];
        out[1] = RealInput ? 0.0f : in[1];
        return;
    } else {
        // size 1 subsequences: the values r and r + N / 2 form block bitrev(r) of the first stage
        const int half_N = N / 2;
        for (int j = 0, r = 0; j < half_N; j++, bitrev_next(r, half_N / 2)) {
            const float * a = in + r * n_in;
            const float * b = in + (r + half_N) * n_in;
            if constexpr (RealInput) {
                out[4 * j + 0] = scale * (a[0] + b[0]);
                out[4 * j + 1] = 0.0f;
                out[4 * j + 2] = scale * (a[0] - b[0]);
                out[4 * j + 3] = 0.0f;
            } else {
                out[4 * j + 0] = scale * (a[0] + b[0]);
                out[4 * j + 1] = scale * (a[1] + b[1]);
                out[4 * j + 2] = scale * (a[0] - b[0]);
                out[4 * j + 3] = scale * (a[1] - b[1]);
            }
        }
        len = 4;
    }

    for (; len <= N; len *= 2) {
        const int half_N       = len / 2;
        const int sin_cos_step = n_sin_cos_vals / len;
        // twiddles of this stage, contiguous
        float * twiddle = scratch;
        for (int k = 0; k < half_N; k++) {
            twiddle[2 * k + 0] = cache.cos_vals[k * sin_cos_step];  // t = 2*M_PI*k/len
            twiddle[2 * k + 1] = sign * cache.sin_vals[k * sin_cos_step];
        }
        for (int i = 0; i < N; i += len) {
            float * even_fft = out + 2 * i;
            float * odd_fft  = even_fft + 2 * half_N;
            for (int k = 0; k < half_N; k++) {
                const float re      = twiddle[2 * k + 0];
                const float im      = twiddle[2 * k + 1];
                const float re_even = even_fft[2 * k + 0];
                const float im_even = even_fft[2 * k + 1];
                const float re_odd  = odd_fft[2 * k + 0];
                const float im_odd  = odd_fft[2 * k + 1];

                even_fft[2 * k + 0] = scale * (re_even + re * re_odd - im * im_odd);
                even_fft[2 * k + 1] = scale * (im_even + re * im_odd + im * re_odd);

                odd_fft[2 * k + 0] = scale * (re_even - re * re_odd + im * im_odd);
                odd_fft[2 * k + 1] = scale * (im_even - re * im_odd - im * re_odd);
            }
        }
    }
}

// Forward FFT for real input (used by mel spectrogram)
static void fft(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    fft_impl<false, true>(cache, in, N, out);
}

// Inverse FFT for complex input
static void ifft(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    fft_impl<true, false>(cache, in, N, out);
}

void mtmd_audio_fft(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    fft(cache, in, N, out);
}

void mtmd_audio_ifft(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    ifft(cache, in, N, out);
}

static void mtmd_audio_fill_entries(std::vector<mtmd_audio_mel> & mels, bool shape_only, std::vector<mtmd_image_f32> & entries) {
    entries.clear();
    for (auto & mel : mels) {
        GGML_ASSERT(mel.n_len <= INT32_MAX && mel.n_len >= 0);
        GGML_ASSERT(mel.n_mel <= INT32_MAX && mel.n_mel >= 0);
        mtmd_image_f32 entry;
        entry.set_size({(int) mel.n_len, (int) mel.n_mel}, shape_only || mel.data.empty(), /* is_audio */ true);
        entry.audio_n_ctx = (int32_t) mel.n_ctx;
        if (!entry.is_placeholder()) {
            entry.cpy_buf(mel.data);
        }
        entries.push_back(std::move(entry));
    }
}

mtmd_audio_preproc_out mtmd_audio_preprocessor::preprocess(const float * samples, size_t n_samples) const {
    mtmd_audio_preproc_out out;
    const bool deferred = mel_config.cache != nullptr;
    std::vector<mtmd_audio_mel> mels;
    if (process(samples, n_samples, mels, deferred)) {
        mtmd_audio_fill_entries(mels, deferred, out.entries);
        if (deferred) {
            out.samples.assign(samples, samples + n_samples);
        }
    }
    return out;
}

bool mtmd_audio_preprocessor::compute(mtmd_audio_preproc_out & out) const {
    if (!out.needs_compute()) {
        return true;
    }
    std::vector<mtmd_audio_mel> mels;
    if (!process(out.samples.data(), out.samples.size(), mels, false)) {
        return false;
    }
    mtmd_audio_fill_entries(mels, false, out.entries);
    out.samples = {};
    return true;
}

void mtmd_audio_log_mel_frame(const mtmd_audio_cache & cache, const float * samples, int n_fft, int n_mel, float * out) {
    const int n_fft_bins = n_fft / 2 + 1;
    GGML_ASSERT((int) cache.hann_window.size() == n_fft);
    GGML_ASSERT(cache.filters.n_mel == n_mel && cache.filters.n_fft == n_fft);

    std::vector<float> fft_in(n_fft * 2, 0.0f);
    std::vector<float> fft_out(n_fft * 2 * 2 * 2);
    for (int j = 0; j < n_fft; j++) {
        fft_in[j] = cache.hann_window[j] * samples[j];
    }
    fft(cache, fft_in.data(), n_fft, fft_out.data());

    for (int j = 0; j < n_fft_bins; j++) {
        fft_out[j] = fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
    }

    for (int j = 0; j < n_mel; j++) {
        double sum = 0.0;
        for (int k = 0; k < n_fft_bins; k++) {
            sum += fft_out[k] * cache.filters.data[(size_t) j * n_fft_bins + k];
        }
        out[j] = log10(std::max(sum, 1e-10));
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
        const float * samples,
        const int     n_samples_in,
        const mtmd_audio_mel_config & params,
        const bool    shape_only,
        mtmd_audio_mel & out) {
    //const int64_t t_start_us = ggml_time_us();

    out.n_len_org = n_samples_in;
    int n_samples = n_samples_in;

    const auto & cache = *params.cache;

    // Hann window
    const float * hann       = cache.hann_window.data();
    const int     frame_size = (params.n_fft_bins - 1) * 2;
    const int     frame_step = params.hop_length;

    // Padding
    std::vector<float> samples_padded;
    if (params.no_padding) {
        // no padding, use samples as-is
        samples_padded = std::vector<float>(samples, samples + n_samples);
        samples = samples_padded.data();
        n_samples = samples_padded.size();
    } else if (params.center_padding) {
        const auto pad_amount = frame_size / 2;
        samples_padded = std::vector<float>(n_samples + 2 * pad_amount, 0);
        std::copy(samples, samples + n_samples, samples_padded.data() + pad_amount);
        samples = samples_padded.data();
        n_samples = samples_padded.size();
    } else {
        // existing padding logic
        int64_t stage_1_pad = params.sample_rate * 30;
        int64_t stage_2_pad = frame_size / 2;
        samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
        std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);
        // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
        std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);
        // reflective pad 200 samples at the beginning of audio
        if (n_samples < stage_2_pad + 1) {
            // TODO: Handle short audio differently or return error
            return false;
        }
        std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

        // expose the padded buffer to downstream FFT and to out.n_len computation
        // mirrors the no_padding and center_padding branches above
        samples   = samples_padded.data();
        n_samples = samples_padded.size();
    }

    // preemphasis
    if (params.preemph) {
        const int   pad_amount = frame_size / 2;
        const float preemph = 0.97f;
        float       prev = samples_padded[pad_amount];
        for (int i = pad_amount + 1; i + pad_amount < n_samples; ++i) {
            float cur = samples_padded[i];
            samples_padded[i] = cur - preemph * prev;
            prev = cur;
        }
    }

    // pad hann window if it's smaller than frame_size
    // TODO: probably unnecessary here? (or better doing it in g_cache?)
    std::vector<float> hann_window_padded;
    if (params.hann_window_size < frame_size) {
        hann_window_padded.resize(frame_size);
        const int padding = (frame_size - params.hann_window_size) / 2;
        std::copy(hann, hann + params.hann_window_size, &hann_window_padded[padding]);
        hann = hann_window_padded.data();
    }

    GGML_ASSERT(params.n_fft_bins > 0);
    GGML_ASSERT(params.hop_length > 0);
    out.n_mel = params.n_mel;
    out.n_len = (n_samples - frame_size) / frame_step + 1;
    // Validate dimensions before allocation to prevent integer overflow
    if (out.n_mel <= 0 || out.n_len <= 0) {
        LOG_ERR("%s: invalid mel dimensions n_mel=%lld n_len=%lld\n", __func__, (long long)out.n_mel, (long long)out.n_len);
        return false;
    }
    const size_t total_size = (size_t)out.n_mel * (size_t)out.n_len;
    if (total_size > SIZE_MAX / sizeof(float)) {
        LOG_ERR("%s: size overflow: n_mel=%lld n_len=%lld\n", __func__, (long long)out.n_mel, (long long)out.n_len);
        return false;
    }
    if (n_samples < frame_size) {
        LOG_ERR("%s: not enough samples after padding\n", __func__);
        return false;
    }
    out.data.resize(total_size);

    if (!shape_only && !mtmd_audio_log_mel(params, samples_padded.data(), hann, out.n_len, out.data.data())) {
        return false;
    }

    const int64_t effective_n_len = n_samples_in / frame_step;
    if (params.norm_per_feature) {
        GGML_ASSERT(effective_n_len > 1);
        for (int64_t i = 0; i < out.n_mel; i++) {
            double mean = 0;
            for (int64_t j = 0; j < effective_n_len; ++j) {
                mean += out.data[(size_t)i * out.n_len + j];
            }
            mean /= effective_n_len;

            double var = 0.0;
            for (int64_t j = 0; j < effective_n_len; ++j) {
                const double value = out.data[(size_t)i * out.n_len + j] - mean;
                var += value * value;
            }
            var /= effective_n_len - 1;  // unbiased
            const double mstd = params.std_eps_after_sqrt ? std::sqrt(var) + 1e-5 : std::sqrt(var + 1e-5);

            for (int64_t j = 0; j < effective_n_len; ++j) {
                auto &value = out.data[(size_t)i * out.n_len + j];
                value        = (value - mean) / mstd;
            }

            // pad the rest with zeros
            for (int64_t j = effective_n_len; j < out.n_len; ++j) {
                out.data[(size_t)i * out.n_len + j] = 0.0;
            }
        }
    } else if (!params.no_padding) {
        // Whisper-style clamping and normalization (NOT used by Gemma4)
        double mmax = -1e20;
        const size_t mel_size = (size_t)out.n_mel * (size_t)out.n_len;
        for (size_t i = 0; i < mel_size; i++) {
            if (out.data[i] > mmax) {
                mmax = out.data[i];
            }
        }

        mmax -= 8.0;

        for (size_t i = 0; i < mel_size; i++) {
            if (out.data[i] < mmax) {
                out.data[i] = mmax;
            }
            out.data[i] = (out.data[i] + 4.0)/4.0;
        }
    }

    // Dump log_mel_spectrogram
    if (DEBUG) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < out.data.size() - 1; i++) {
            outFile << out.data[i] << ", ";
        }
        outFile << out.data[out.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}
