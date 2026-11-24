#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "mtmd-audio.h"
#include "mtmd-subcontext.h"
#include "mtmd-helper.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"
#include "models/models.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)

//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_inkling2v) {
    mtmd_hparams hparams;
    hparams.patch_size = 20;
    hparams.n_merge = 2;
    for (int c = 0; c < 3; ++c) {
        hparams.image_mean[c] = hparams.image_std[c] = 0.5f;
    }
    mtmd_image_preprocessor_inkling2v preproc(hparams);
    for (const auto & size : std::vector<mtmd_image_size>{{1, 1}, {40, 40}, {61, 43}, {83, 45}, {161, 21}}) {
        mtmd_image_u8 image;
        image.set_size(size, false);
        for (int y = 0; y < size.height; ++y) {
            for (int x = 0; x < size.width; ++x) {
                image.set_pixel(x, y, {uint8_t(x % 256), uint8_t(y % 256), 255});
            }
        }
        const auto result = preproc.preprocess(image);
        t.assert_equal("one image", size_t(1), result.entries.size());
        const auto & output = result.entries[0];
        t.assert_equal("width ceil", ((size.width + 39) / 40) * 40, output.nx());
        t.assert_equal("height ceil", ((size.height + 39) / 40) * 40, output.ny());
        const auto & pixels = output.get_ro_buf();
        bool exact = true;
        for (int y = 0; y < output.ny(); ++y) {
            for (int x = 0; x < output.nx(); ++x) {
                const bool pad = x >= size.width || y >= size.height;
                const float rgb[3] = {float(x % 256), float(y % 256), 255.0f};
                for (int c = 0; c < 3; ++c) {
                    const float expected = ((pad ? -1.0f : rgb[c]) - 127.5f) / 127.5f;
                    exact = exact && pixels[3 * (size_t(y) * output.nx() + x) + c] == expected;
                }
            }
        }
        t.assert_true("pixels and negative padding", exact);
        image.set_size(size, true);
        const auto placeholder = preproc.preprocess(image);
        t.assert_true("placeholder", placeholder.entries[0].is_placeholder());
        t.assert_equal("placeholder width", output.nx(), placeholder.entries[0].nx());
        t.assert_equal("placeholder height", output.ny(), placeholder.entries[0].ny());
    }
    // an image above image_max_pixels is resized down, the output stays aligned
    hparams.image_max_pixels = 16 * 40 * 40;
    mtmd_image_u8 large;
    large.set_size({1000, 300}, false);
    large.cpy_buf(std::vector<uint8_t>(1000 * 300 * 3, 17));
    const auto capped = preproc.preprocess(large);
    t.assert_true("capped area", capped.entries[0].nx() * capped.entries[0].ny() <= hparams.image_max_pixels);
    t.assert_equal("capped width aligned", 0, capped.entries[0].nx() % 40);
    t.assert_equal("capped height aligned", 0, capped.entries[0].ny() % 40);
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_internal_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// mtmd_helper video
//

// n_frames source frames, each one starts with its index
struct test_video_source : mtmd_helper_video_seq {
    int32_t n_frames;
    int32_t n_read    = 0;
    int32_t n_skipped = 0;

    test_video_source(int32_t n_frames) : n_frames(n_frames) {}

    bool skip_raw_frame() override {
        if (n_read >= n_frames) {
            return false;
        }
        n_read++;
        n_skipped++;
        return true;
    }

    bool read_raw_frame() override {
        if (n_read >= n_frames) {
            return false;
        }
        frame_buf[0] = static_cast<uint8_t>(n_read & 255);
        frame_buf[1] = static_cast<uint8_t>(n_read >> 8);
        n_read++;
        return true;
    }
};

// text as is, a frame as [source index], an audio bitmap as {n_samples}
static std::string video_sequence_to_str(testing & t, mtmd_helper_video_seq & video) {
    std::string out;
    mtmd_bitmap * bitmap;
    char * text;
    int32_t ret;
    while ((ret = video.read_next(&bitmap, &text)) == 0) {
        if (text) {
            out += text;
            free(text);
        } else if (mtmd_bitmap_is_audio(bitmap)) {
            out += "{" + std::to_string(mtmd_bitmap_get_n_bytes(bitmap) / sizeof(float)) + "}";
            mtmd_bitmap_free(bitmap);
        } else {
            const unsigned char * data = mtmd_bitmap_get_data(bitmap);
            out += "[" + std::to_string(data[0] | data[1] << 8) + "]";
            mtmd_bitmap_free(bitmap);
        }
    }
    t.assert_equal("end of stream", -1, ret);
    return out;
}

// a source that gives every frame: the frames of fps_target are picked by index
MAKE_TEST(test_video_sequence_sample_by_index) {
    const mtmd_video_config cfg = { MTMD_VIDEO_TEMPLATE_DEFAULT, 0.0f, 0, 0, 0 };

    // { source video, frames in the source, expected sequence, expected number of skipped frames }
    const std::vector<std::tuple<mtmd_helper_video_stream_info, int32_t, std::string, int32_t>> cases = {
        { { 64, 64, 8.0f, 16, -1.0f }, 16, "Video:[0][0m0.00s][4][8][0m1.00s][12]", 9 },
        // unknown length: every frame is kept
        { { 64, 64, 8.0f, -1, -1.0f },  3, "Video:[0][0m0.00s][1][2]", 0 },
    };

    for (const auto & [src, n_frames, expected, expected_skipped] : cases) {
        const std::string name = std::to_string(src.n_frames) + " frames";
        test_video_source video(n_frames);
        video.timestamp_interval_ms = 1000;
        video.configure(cfg, src, 2.0f);
        video.sample_by_index(src);
        video.src_alive = true;
        t.assert_equal(name + " sequence", expected, video_sequence_to_str(t, video));
        t.assert_equal(name + " skipped", expected_skipped, video.n_skipped);
    }
}

//
// mtmd_audio
//

MAKE_TEST(test_audio_log_mel_precision) {
    const int n_mel    = 80;
    const int n_fft    = 400;
    const int hop      = 160;
    const int n_frames = 200;

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, 16000);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_mel_config config;
    config.sched      = sched.get();
    config.n_threads  = 2;
    config.n_mel      = n_mel;
    config.n_fft_bins = 1 + n_fft / 2;
    config.hop_length = hop;
    config.cache      = &cache;

    // a loud tone next to a 74 dB weaker one: rounding the DFT inputs to F16 or TF32 loses the weak one
    const double pi = 3.14159265358979323846;
    std::vector<float> samples((size_t) (n_frames - 1) * hop + n_fft);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.5 * std::sin(2 * pi * 200 * i / 16000) + 1e-4 * std::sin(2 * pi * 6000 * i / 16000));
    }

    std::vector<float> out((size_t) n_frames * n_mel);
    if (!t.assert_true("log mel", mtmd_audio_log_mel(config, samples.data(), cache.hann_window.data(), n_frames, out.data()))) {
        return;
    }

    const int n_bins = n_fft / 2 + 1;
    double max_diff = 0.0;
    std::vector<double> power(n_bins);
    for (int i = 0; i < n_frames; i++) {
        for (int k = 0; k < n_bins; k++) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < n_fft; j++) {
                const double x = (double) cache.hann_window[j] * samples[(size_t) i * hop + j];
                re += x * std::cos(2 * pi * j * k / n_fft);
                im -= x * std::sin(2 * pi * j * k / n_fft);
            }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < n_mel; m++) {
            double sum = 0.0;
            for (int k = 0; k < n_bins; k++) {
                sum += power[k] * cache.filters.data[(size_t) m * n_bins + k];
            }
            const double expected = std::log10(std::max(sum, (double) config.mel_floor));
            max_diff = std::max(max_diff, std::fabs(expected - out[(size_t) m * n_frames + i]));
        }
    }
    t.assert_true("max_diff = " + std::to_string(max_diff), max_diff < 0.05);
}

MAKE_TEST(test_audio_log_mel_blocks) {
    const int n_mel    = 80;
    const int n_fft    = 400;
    const int hop      = 160;
    const int n_frames = 8192 + 100; // long audio is split into blocks of 8192 frames

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, 16000);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_mel_config config;
    config.sched       = sched.get();
    config.n_threads   = 2;
    config.n_mel       = n_mel;
    config.n_fft_bins  = 1 + n_fft / 2;
    config.hop_length  = hop;
    config.cache       = &cache;

    std::vector<float> samples((size_t) (n_frames - 1) * hop + n_fft);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.3 * std::sin(0.05 * i + 1e-6 * i * i));
    }

    // frames are independent: the frames around the seam must not change when computed on their own
    // not bit-exact, matmul kernels may round differently depending on the shape
    const int off   = 8192 - 50;
    const int n_sub = n_frames - off;
    std::vector<float> all((size_t) n_frames * n_mel);
    std::vector<float> sub((size_t) n_sub * n_mel);
    const float * window = cache.hann_window.data();
    if (!t.assert_true("all", mtmd_audio_log_mel(config, samples.data(), window, n_frames, all.data())) ||
        !t.assert_true("sub", mtmd_audio_log_mel(config, samples.data() + (size_t) off * hop, window, n_sub, sub.data()))) {
        return;
    }

    float max_diff = 0.0f;
    for (int j = 0; j < n_mel; j++) {
        for (int i = 0; i < n_sub; i++) {
            max_diff = std::max(max_diff, std::fabs(all[(size_t) j * n_frames + off + i] - sub[(size_t) j * n_sub + i]));
        }
    }
    t.assert_true("max_diff = " + std::to_string(max_diff), max_diff < 1e-5f);
}

//
// mtmd_helper_resampler
//

static std::vector<float> make_tone(double freq, int rate, size_t n) {
    const double pi = 3.14159265358979323846;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = (float) (0.5 * std::sin(2 * pi * freq * i / rate));
    }
    return x;
}

// amplitude of the freq component, skipping the edges
static double tone_amplitude(const std::vector<float> & x, double freq, int rate) {
    const double pi = 3.14159265358979323846;
    const size_t skip = rate / 10;
    double re = 0.0, im = 0.0;
    for (size_t i = skip; i < x.size() - skip; i++) {
        re += x[i] * std::cos(2 * pi * freq * i / rate);
        im += x[i] * std::sin(2 * pi * freq * i / rate);
    }
    return 2.0 * std::sqrt(re * re + im * im) / (x.size() - 2 * skip);
}

MAKE_TEST(test_audio_fft) {
    // the sizes in use: powers of two, and an odd part (400 = 16 * 25, 736 = 32 * 23, 1920 = 128 * 15)
    // plus small edge cases: 4 = gather + one stage, 6 = n_odd 3 with n_sub 2, 15 = odd composite with n_sub 1
    const int sizes[] = { 1, 2, 3, 4, 6, 12, 15, 256, 400, 512, 736, 1920 };
    const double pi = 3.14159265358979323846;
    for (int n : sizes) {
        mtmd_audio_cache cache;
        cache.fill_sin_cos_table(n);
        // phase offset so that x[0] != 0 (n = 1 would otherwise compare all zeros)
        std::vector<float> x(2 * n);
        for (int i = 0; i < 2 * n; i++) {
            x[i] = (float) std::sin(0.37 * i + 0.001 * i * i + 0.5);
        }
        std::vector<float> out(8 * n);
        for (bool inv : { false, true }) {
            // input sized exactly as documented so that sanitizers catch any overread
            const std::vector<float> in(x.begin(), x.begin() + (inv ? 2 * n : n));
            if (inv) {
                mtmd_audio_ifft(cache, in.data(), n, out.data());
            } else {
                mtmd_audio_fft(cache, in.data(), n, out.data());
            }
            double max_err = 0.0, max_ref = 0.0;
            for (int k = 0; k < n; k++) {
                double re = 0.0, im = 0.0;
                for (int j = 0; j < n; j++) {
                    const double theta = (inv ? 2 : -2) * pi * (double) ((int64_t) j * k % n) / n;
                    const double a = inv ? x[2 * j] : x[j];
                    const double b = inv ? x[2 * j + 1] : 0.0;
                    re += a * std::cos(theta) - b * std::sin(theta);
                    im += a * std::sin(theta) + b * std::cos(theta);
                }
                if (inv) {
                    re /= n;
                    im /= n;
                }
                max_err = std::max({ max_err, std::fabs(re - out[2 * k]), std::fabs(im - out[2 * k + 1]) });
                max_ref = std::max({ max_ref, std::fabs(re), std::fabs(im) });
            }
            t.assert_true(std::string(inv ? "ifft" : "fft") + " n = " + std::to_string(n) + ", relative error " + std::to_string(max_err / max_ref),
                          max_ref > 0.0 && max_err <= 1e-5 * max_ref);
        }

        // round trip: ifft(fft(x)) == x for real x
        std::vector<float> spec(8 * n);
        mtmd_audio_fft(cache, x.data(), n, spec.data());
        const std::vector<float> spec_in(spec.begin(), spec.begin() + 2 * n);
        mtmd_audio_ifft(cache, spec_in.data(), n, out.data());
        double max_err = 0.0, max_ref = 0.0;
        for (int k = 0; k < n; k++) {
            max_err = std::max({ max_err, (double) std::fabs(out[2 * k] - x[k]), (double) std::fabs(out[2 * k + 1]) });
            max_ref = std::max(max_ref, (double) std::fabs(x[k]));
        }
        t.assert_true("fft round trip n = " + std::to_string(n) + ", relative error " + std::to_string(max_err / max_ref),
                      max_ref > 0.0 && max_err <= 1e-5 * max_ref);
    }
}

static std::vector<float> resample_chunked(mtmd_helper_resampler * rs, const std::vector<float> & in, size_t chunk) {
    std::vector<float> res;
    for (size_t off = 0; off < in.size(); off += chunk) {
        const float * out = nullptr;
        const int64_t n = mtmd_helper_resampler_process(rs, in.data() + off, std::min(chunk, in.size() - off), &out);
        if (n < 0) {
            return {};
        }
        res.insert(res.end(), out, out + n);
    }
    return res;
}

MAKE_TEST(test_helper_resampler) {
    const int rate_out = 16000;
    for (int rate_in : { 8000, 24000, 44100, 48000 }) {
        const std::string name = std::to_string(rate_in) + " -> " + std::to_string(rate_out);
        const std::vector<float> in = make_tone(440, rate_in, (size_t) rate_in * 2);

        mtmd_helper::resampler_ptr rs_all(mtmd_helper_resampler_init(rate_in, rate_out));
        mtmd_helper::resampler_ptr rs_chk(mtmd_helper_resampler_init(rate_in, rate_out));
        if (!t.assert_true(name + " init", rs_all && rs_chk)) {
            continue;
        }
        const std::vector<float> all = resample_chunked(rs_all.get(), in, in.size());
        const std::vector<float> chk = resample_chunked(rs_chk.get(), in, 1234); // odd size, not a multiple of the ratio

        t.assert_equal(name + " n_samples", (size_t) rate_out * 2, all.size());
        t.assert_true(name + " chunked == one-shot", all == chk);

        const double amp = tone_amplitude(all, 440, rate_out);
        t.assert_true(name + " amplitude = " + std::to_string(amp), std::fabs(amp - 0.5) < 0.02);
    }

    // above the output Nyquist: 10 kHz at 48 kHz aliases to 6 kHz at 16 kHz, the low-pass filter must cut it
    {
        mtmd_helper::resampler_ptr rs(mtmd_helper_resampler_init(48000, rate_out));
        const std::vector<float> out = resample_chunked(rs.get(), make_tone(10000, 48000, 48000 * 2), 4800);
        const double db = 20 * std::log10(tone_amplitude(out, 6000, rate_out) / 0.5);
        t.assert_true("aliasing = " + std::to_string(db) + " dB", db < -15); // about -20 dB with lpfOrder = 8, -10 dB with the default
    }

    // same rate: no copy
    {
        mtmd_helper::resampler_ptr rs(mtmd_helper_resampler_init(rate_out, rate_out));
        const std::vector<float> in = make_tone(440, rate_out, 1000);
        const float * out = nullptr;
        const int64_t n = rs ? mtmd_helper_resampler_process(rs.get(), in.data(), in.size(), &out) : -1;
        t.assert_true("passthrough", n == (int64_t) in.size() && out == in.data());
    }

    t.assert_true("invalid rate", mtmd_helper_resampler_init(0, rate_out) == nullptr);
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // the audio preprocessors run on the CPU backend
    ggml_backend_load_all();

    // the video helper logs every frame at debug level
    mtmd_helper_log_set([](ggml_log_level level, const char * text, void *) {
        if (level != GGML_LOG_LEVEL_DEBUG) {
            fputs(text, stderr);
        }
    }, nullptr);

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
