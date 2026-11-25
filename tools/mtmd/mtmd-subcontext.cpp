#include "mtmd-subcontext.h"
#include "mtmd-graph.h"
#include "models/models.h"
#include "mtmd-audio.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

// ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
static void mtmd_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(backend_cpu, n_threads);
        }
    }
}

static std::shared_ptr<ggml_threadpool> mtmd_threadpool_init(int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg || n_threads < 2) {
        return nullptr;
    }
    auto * new_fn  = (decltype(ggml_threadpool_new) *)  ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    if (!new_fn || !free_fn) {
        return nullptr;
    }
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    tpp.poll = 0; // the pool is idle most of the time, don't take CPU away from the text model
    ggml_threadpool_t threadpool = new_fn(&tpp);
    if (!threadpool) {
        throw std::runtime_error("failed to create threadpool");
    }
    return std::shared_ptr<ggml_threadpool>(threadpool, free_fn);
}

ggml_backend_dev_t mtmd_select_device(bool use_gpu, ggml_backend_dev_t device) {
    if (!use_gpu) {
        return nullptr;
    }
    if (device != nullptr) {
        return device;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    return dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

mtmd_backends::mtmd_backends(ggml_backend_dev_t device, int n_threads, int max_nodes,
                             ggml_backend_sched_eval_callback cb_eval, void * cb_eval_user_data) {
    threadpool = mtmd_threadpool_init(n_threads);

    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_cpu) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    if (threadpool) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, threadpool.get());
        }
    }
    if (device != nullptr) {
        backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            ggml_backend_free(backend_cpu);
            throw std::runtime_error(mtmd_format("%s: failed to initialize \"%s\" backend\n",
                                                   __func__, ggml_backend_dev_name(device)));
        }
    }

    if (backend) {
        LOG_INF("%s: using %s backend\n", __func__, ggml_backend_name(backend));
        ptrs.push_back(backend);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend));
    } else {
        backend = backend_cpu;
        LOG_INF("%s: using CPU backend\n", __func__);
    }

    ptrs.push_back(backend_cpu);
    bufts.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

    sched.reset(ggml_backend_sched_new(ptrs.data(), bufts.data(), ptrs.size(), max_nodes, false, true));
    if (cb_eval != nullptr) {
        ggml_backend_sched_set_eval_callback(sched.get(), cb_eval, cb_eval_user_data);
    }
}

mtmd_backends::~mtmd_backends() {
    // before the backends it uses
    sched.reset();
    ggml_backend_free(backend);
    if (backend != backend_cpu) {
        ggml_backend_free(backend_cpu);
    }
}

std::map<ggml_backend_dev_t, size_t> mtmd_backends::get_mem_compute() const {
    std::map<ggml_backend_dev_t, size_t> res;
    for (ggml_backend_t b : ptrs) {
        res[ggml_backend_get_device(b)] += ggml_backend_sched_get_buffer_size(sched.get(), b);
    }
    return res;
}

mtmd_subcontext::mtmd_subcontext(const mtmd_model & model, mtmd_backends & backends, const mtmd_subctx_params & params) :
        model(model),
        backends(backends),
        max_nodes(model.max_nodes),
        flash_attn_type(params.flash_attn_type) {
    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_AUTO && !model.flash_attn_auto) {
        flash_attn_type = MTMD_FLASH_ATTN_TYPE_DISABLED;
    }

    debug_output_embeddings = std::getenv("MTMD_DEBUG_EMBEDDINGS") != nullptr;

    buf_compute_meta.resize(max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false));

    auto batch = get_dummy_batch();
    if (model.modality != MTMD_MODALITY_GEN_AUDIO) {
        // check batching support
        support_batch = mtmd_get_graph_builder(this, batch)->support_batch();
    }
    if (params.warmup) {
        warmup(batch);
    }
}

mtmd_image_f32_batch mtmd_subcontext::get_dummy_batch() const {
    // create a fake batch
    const auto & hparams = model.hparams;
    mtmd_image_f32_batch batch;
    mtmd_image_f32 img;
    if (model.modality == MTMD_MODALITY_VISION) {
        const int sz = hparams.warmup_image_size;
        img.set_size({sz, sz}, false, false);
        LOG_INF("%s: warmup with image size = %d x %d\n", __func__, sz, sz);
    } else if (model.modality == MTMD_MODALITY_GEN_AUDIO) {
        // TODO: improve this, move to per-model impl
        const int n_embd = model.n_mmproj_embd();
        img.set_size({n_embd, 1}, false, true);
        LOG_INF("%s: warmup with one gen-audio step\n", __func__);
    } else {
        if (hparams.n_mel_bins <= 0 || hparams.n_mel_bins > 256) {
            throw std::runtime_error(mtmd_format("%s: invalid n_mel_bins (%d), must be in [1, 256]\n", __func__, hparams.n_mel_bins));
        }
        img.set_size({hparams.warmup_audio_size, hparams.n_mel_bins}, false, false);
        LOG_INF("%s: warmup with audio size = %d\n", __func__, hparams.warmup_audio_size);
    }
    batch.entries.push_back(img);
    return batch;
}

void mtmd_subcontext::warmup(const mtmd_image_f32_batch & batch) {
    support_info_graph info;

    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_AUTO) {
        // try to enable flash attention to see if it's supported
        flash_attn_type = MTMD_FLASH_ATTN_TYPE_ENABLED;
        info = reserve_compute_meta(batch);
        if (!info.fattn && info.fattn_op) {
            auto op = info.fattn_op;
            LOG_WRN("%s: *****************************************************************\n", __func__);
            LOG_WRN("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", __func__, ggml_backend_name(backends.backend));
            LOG_WRN("%s: op params: \n", __func__);
            static auto print_shape = [](const char * fn, const char * name, ggml_tensor * t) {
                LOG_WRN("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", fn,
                        name, ggml_type_name(t->type),
                        t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                        t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
            };
            print_shape(__func__, " dst", op);
            print_shape(__func__, "src0", op->src[0]);
            print_shape(__func__, "src1", op->src[1]);
            print_shape(__func__, "src2", op->src[2]);
            LOG_WRN("%s: please report this on github as an issue\n", __func__);
            LOG_WRN("%s: *****************************************************************\n", __func__);
            flash_attn_type = MTMD_FLASH_ATTN_TYPE_DISABLED;
            reserve_compute_meta(batch);
        }
    } else {
        info = reserve_compute_meta(batch);
        if (!info.fattn && flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
            LOG_WRN("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be degraded)\n", __func__);
        }
    }

    is_allocated = true; // mark buffers as allocated

    LOG_INF("%s: flash attention is %s\n", __func__,
        (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");

    // print ops that are not supported by the GPU backend (if there is one)
    if (backends.backend != backends.backend_cpu) {
        std::vector<support_info_op> unsupported_ops;
        for (const auto & op : info.ops) {
            if (!op.is_accel) {
                unsupported_ops.push_back(op);
            }
        }
        if (!unsupported_ops.empty()) {
            LOG_WRN("%s: *****************************************************************\n", __func__);
            LOG_WRN("%s: WARNING: the mmproj graph uses unsupported operators by the backend\n", __func__);
            LOG_WRN("%s:          the performance will be suboptimal                      \n", __func__);
            LOG_WRN("%s:          list of unsupported ops (backend=%s):\n", __func__, ggml_backend_name(backends.backend));
            for (const auto & op : unsupported_ops) {
                LOG_WRN("%s: %16s: type = %s, ne = [%d %d %d %d]\n", __func__,
                        ggml_op_name(op.op->op),
                        ggml_type_name(op.op->type),
                        op.op->ne[0], op.op->ne[1], op.op->ne[2], op.op->ne[3]);
            }
            LOG_WRN("%s: flash attention is %s\n", __func__,
                (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");
            LOG_WRN("%s: please report this on github as an issue\n", __func__);
            LOG_WRN("%s: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118\n", __func__);
            LOG_WRN("%s: *****************************************************************\n", __func__);
        }
    }
}

mtmd_subcontext::support_info_graph mtmd_subcontext::reserve_compute_meta(const mtmd_image_f32_batch & batch) {
    ggml_cgraph * gf = mtmd_get_graph_builder(this, batch)->build();
    ggml_backend_sched_reserve(backends.sched.get(), gf);

    const int n_splits = ggml_backend_sched_get_n_splits(backends.sched.get());
    const int n_nodes  = ggml_graph_n_nodes(gf);

    LOG_INF("%s: graph splits = %d, nodes = %d\n", __func__,  n_splits, n_nodes);

    support_info_graph res {
        /*.fattn    = */ true,
        /*.fattn_op = */ nullptr,
        /*.ops      = */ {},
    };

    // check op support
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        res.ops.push_back({node, true});
        if (!ggml_backend_supports_op(backends.backend, node)) {
            res.ops.back().is_accel = false;
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                res.fattn    = false;
                res.fattn_op = node;
            }
        }
    }

    return res;
}

bool mtmd_audio_log_mel(const mtmd_audio_mel_config & config, const float * samples, const float * window, int64_t n_frames, float * out) {
    ggml_backend_sched_t sched = config.sched;
    ggml_backend_t backend_cpu = ggml_backend_sched_get_backend(sched, ggml_backend_sched_get_n_backends(sched) - 1);
    mtmd_backend_cpu_set_n_threads(backend_cpu, std::max(1, config.n_threads));

    const auto &  cache      = *config.cache;
    const auto &  filters    = cache.filters.data;
    const int64_t n_fft_bins = config.n_fft_bins;
    const int64_t frame_size = (n_fft_bins - 1) * 2;
    const int64_t n_mel      = config.n_mel;
    GGML_ASSERT(filters.size() == (size_t) (n_mel * n_fft_bins));
    GGML_ASSERT(cache.cos_vals.size() == (size_t) frame_size && cache.sin_vals.size() == (size_t) frame_size);

    // DFT matrix with the window folded in, the first n_fft_bins rows give the real part
    std::vector<float> dft((size_t) 2 * n_fft_bins * frame_size);
    for (int64_t k = 0; k < n_fft_bins; k++) {
        for (int64_t j = 0; j < frame_size; j++) {
            const int64_t q = (j * k) % frame_size;
            dft[k * frame_size + j]                = window[j] * cache.cos_vals[q];
            dft[(k + n_fft_bins) * frame_size + j] = window[j] * cache.sin_vals[q];
        }
    }

    // the spectrum is several times larger than the result, so long audio is processed in blocks
    const int64_t n_frames_max = 8192;

    for (int64_t off = 0; off < n_frames; off += n_frames_max) {
        const int64_t n_cur = std::min(n_frames_max, n_frames - off);

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr(ggml_init(params));
        ggml_context * ctx0 = ctx_ptr.get();

        ggml_tensor * inp_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (n_cur - 1) * config.hop_length + frame_size);
        ggml_tensor * inp_dft     = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, frame_size, 2 * n_fft_bins);
        ggml_tensor * inp_filters = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_fft_bins, n_mel);
        ggml_set_input(inp_samples);
        ggml_set_input(inp_dft);
        ggml_set_input(inp_filters);

        ggml_tensor * cur = ggml_cont(ctx0, ggml_view_2d(ctx0, inp_samples, frame_size, n_cur, config.hop_length * ggml_element_size(inp_samples), 0));
        cur = ggml_mul_mat(ctx0, inp_dft, cur); // [2 * n_fft_bins, n_cur]
        ggml_prec_set_src(cur, GGML_PREC_F32, 1);

        cur = ggml_sqr(ctx0, cur);
        cur = ggml_add(ctx0,
                ggml_view_2d(ctx0, cur, n_fft_bins, n_cur, cur->nb[1], 0),
                ggml_view_2d(ctx0, cur, n_fft_bins, n_cur, cur->nb[1], n_fft_bins * ggml_element_size(cur)));
        if (config.use_magnitude) {
            cur = ggml_sqrt(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, cur, inp_filters); // [n_cur, n_mel]
        ggml_prec_set_src(cur, GGML_PREC_F32, 1);
        cur = config.mel_floor_add
            ? ggml_scale_bias(ctx0, cur, 1.0f, config.mel_floor)
            : ggml_clamp(ctx0, cur, config.mel_floor, FLT_MAX);
        cur = ggml_log(ctx0, cur);
        if (!config.use_natural_log) {
            cur = ggml_scale(ctx0, cur, 1.0f / logf(10.0f));
        }
        ggml_set_output(cur);

        ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, cur);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LOG_ERR("%s: failed to allocate compute graph\n", __func__);
            return false;
        }

        ggml_backend_tensor_set(inp_samples, samples + off * config.hop_length, 0, ggml_nbytes(inp_samples));
        ggml_backend_tensor_set(inp_dft,     dft.data(),                        0, ggml_nbytes(inp_dft));
        ggml_backend_tensor_set(inp_filters, filters.data(),                    0, ggml_nbytes(inp_filters));

        auto status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
            return false;
        }

        for (int64_t j = 0; j < n_mel; j++) {
            ggml_backend_tensor_get(cur, out + j * n_frames + off, j * cur->nb[1], n_cur * ggml_element_size(cur));
        }
    }

    return true;
}

bool mtmd_subcontext::encode_image(int n_threads, const mtmd_image_f32 * img, std::vector<float> & out_vec) {
    mtmd_image_f32_batch imgs;
    mtmd_image_f32 img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return encode_batch(n_threads, &imgs, out_vec);
}

bool mtmd_subcontext::encode_batch(int n_threads, const mtmd_image_f32_batch * imgs_c_ptr, std::vector<float> & out_batch_embd) {
    mtmd_subctx_encode_params params;
    params.imgs = imgs_c_ptr;
    params.n_threads = n_threads;
    params.out_embd = &out_batch_embd;

    return encode(&params);
}

mtmd_subctx_gen_state * mtmd_subcontext::gen_state_init(uint32_t seed) {
    auto st = std::make_unique<mtmd_subctx_gen_state>();
    st->rng.seed(seed == UINT32_MAX ? std::random_device{}() : seed);

    return st.release();
}

// no kept encoder streams its input
mtmd_subctx_enc_state * mtmd_subcontext::enc_state_init(int32_t /*n_frames_max*/) {
    LOG_ERR("%s: the encoder does not support streaming input\n", __func__);
    return nullptr;
}

bool mtmd_subcontext::encode(mtmd_subctx_encode_params * params) {
    const mtmd_image_f32_batch & imgs = *params->imgs;
    int n_batch_cur = imgs.entries.size();

    // [QWEN_VIDEO] for video models, the batch dimension is used as temporal dimension for merged frames
    if (!support_batch && n_batch_cur > model.n_temporal_merge()) {
        LOG_ERR("%s: batch size %d exceeds maximum supported batch/temporal-merge size %d\n", __func__, n_batch_cur, model.n_temporal_merge());
        return false;
    }

    // if buffers are not allocated, we need to do a warmup run to allocate them
    if (!is_allocated) {
        warmup(*params->imgs);
    }

    // build the inference graph
    ggml_backend_sched_reset(backends.sched.get());
    ggml_cgraph * gf = mtmd_get_graph_builder(this, imgs, params)->build();

    // the last node is the embedding tensor, code2wav has no out_embd
    // read it before scheduling: backends may reorder the nodes of graphs with several outputs (Metal)
    ggml_tensor * embeddings = params->out_embd ? ggml_graph_node(gf, -1) : nullptr;

    if (!ggml_backend_sched_alloc_graph(backends.sched.get(), gf)) {
        LOG_ERR("%s: failed to allocate compute graph\n", __func__);
        return false;
    }

    // set inputs
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0].nx();
    const int image_size_height = imgs.entries[0].ny();

    const int patch_size    = hparams.patch_size;
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;

    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, const std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // set input pixel values
    if (!imgs.is_audio) {
        size_t nelem = 0;
        for (const auto & img : imgs.entries) {
            nelem += img.nx() * img.ny() * 3;
        }
        std::vector<float> inp_raw(nelem);

        // layout of data (note: the channel dim is unrolled to better visualize the layout):
        //
        // ┌──W──┐
        // │     H │  channel = R
        // ├─────┤ │
        // │     H │  channel = G
        // ├─────┤ │
        // │     H │  channel = B
        // └─────┘ │
        //   ──────┘ x B

        // IMPORTANT: [QWEN_VIDEO] the batch dim is currently used for temporal dim in Qwen-VL models
        // All entries must have the same spatial size (enforced by can_batch_with() during merging)
        {
            const int nx = imgs.entries[0].nx();
            const int ny = imgs.entries[0].ny();
            const int n  = nx * ny;

            for (int b = 0; b < n_batch_cur; b++) {
                LOG_DBG("%s: copying image %d/%d to input buffer (nx=%d, ny=%d)\n", __func__, b+1, n_batch_cur, nx, ny);
                const auto & buf = imgs.entries[b].get_ro_buf();
                float * batch_entry = inp_raw.data() + b * (3*n);
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        size_t base_src = 3*(y * nx + x);
                        size_t base_dst =    y * nx + x;
                        batch_entry[      base_dst] = buf[base_src    ];
                        batch_entry[1*n + base_dst] = buf[base_src + 1];
                        batch_entry[2*n + base_dst] = buf[base_src + 2];
                    }
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);

    } else if (params->gen_process != MTMD_SUBCTX_GEN_PROCESS_GEN_WAV &&
               params->gen_process != MTMD_SUBCTX_GEN_PROCESS_TTS &&
               params->gen_process != MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE &&
               ggml_graph_get_tensor(gf, "inp_raw") != nullptr) {
        // audio input. GEN_WAV and the chatterbox TTS stages are not here: they take codes or feats, set in the switch below.
        // a GEN_CODE graph may have no hidden-state input (lfm3a_gen feedback embedding)
        GGML_ASSERT(imgs.entries.size() == 1);

        const auto & mel_inp = imgs.entries[0];
        const auto & buf = mel_inp.get_ro_buf();
        const int n_step = mel_inp.nx();
        const int n_mel  = mel_inp.ny();
        GGML_ASSERT((size_t)n_step * n_mel == buf.size());

        set_input_f32("inp_raw", buf);
    }

    // set input per projector
    switch (model.proj_type) {
        case PROJECTOR_TYPE_INKLING2V:
            {
                // CLS is at position 0, the patches are 1-indexed
                std::vector<int32_t> positions_w{0}, positions_h{0};
                for (int y = 0; y < pos_h; ++y) {
                    for (int x = 0; x < pos_w; ++x) {
                        positions_w.push_back(x + 1);
                        positions_h.push_back(y + 1);
                    }
                }
                set_input_i32("pos_w", positions_w);
                set_input_i32("pos_h", positions_h);
            } break;
        case PROJECTOR_TYPE_QWEN3VL:
            {
                const int num_patches = pos_w * pos_h;
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size / merge_ratio;
                const int ph = image_size_height / patch_size / merge_ratio;
                std::vector<int32_t> positions(num_patches * 4);
                int ptr = 0;
                for (int ty = 0; ty < ph; ty++) {
                    for (int tx = 0; tx < pw; tx++) {
                        for (int dy = 0; dy < merge_ratio; dy++) {
                            for (int dx = 0; dx < merge_ratio; dx++) {
                                const int y = ty * merge_ratio + dy;
                                const int x = tx * merge_ratio + dx;
                                positions[                  ptr] = y;
                                positions[    num_patches + ptr] = x;
                                positions[2 * num_patches + ptr] = y;
                                positions[3 * num_patches + ptr] = x;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        default:
            GGML_ABORT("Unknown projector type");
    }

    mtmd_backend_cpu_set_n_threads(backends.backend_cpu, params->n_threads);

    auto status = ggml_backend_sched_graph_compute(backends.sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    if (embeddings != nullptr) {
        // sanity check (assuming that all images in batch have the same number of tokens, so we only check the first one)
        const int n_tokens_out = embeddings->ne[1];
        const int expected_n_tokens_out = model.n_output_tokens(&imgs.entries[0]);
        if (n_tokens_out != expected_n_tokens_out) {
            LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
            GGML_ABORT("Invalid number of output tokens");
        }

        LOG_DBG("%s: output embedding shape [%d, %d, %d]\n", __func__,
            (int)embeddings->ne[0], (int)embeddings->ne[1], (int)embeddings->ne[2]);

        // copy output to user buffer if provided
        // if output is empty, skip the copy
        auto & out_batch_embd = *params->out_embd;
        if (!out_batch_embd.empty()) {
            if (out_batch_embd.size() != (size_t)ggml_nelements(embeddings)) {
                LOG_ERR("%s: output buffer has %zu elements but expected %zu\n", __func__, out_batch_embd.size(), (size_t)ggml_nelements(embeddings));
                GGML_ABORT("Output buffer size mismatch");
            }
            ggml_backend_tensor_get(embeddings, out_batch_embd.data(), 0, ggml_nbytes(embeddings));
        } else {
            LOG_WRN("%s: output buffer is empty, skipping copy\n", __func__);
        }
    }

    //
    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
    //

    if (debug_output_embeddings && embeddings != nullptr) {
        const int64_t n_embd = embeddings->ne[0];
        const int64_t n_tokens = embeddings->ne[1];
        std::vector<float> emb_data(ggml_nelements(embeddings));
        ggml_backend_tensor_get(embeddings, emb_data.data(), 0, ggml_nbytes(embeddings));

        LOG_INF("\n=== MTMD_DEBUG_EMBEDDINGS ===\n");
        LOG_INF("Shape: [%lld, %lld]\n", (long long)n_embd, (long long)n_tokens);

        // TEMP debugging (parity validation), will be removed before merge
        // when the env var holds a path, dump the raw data: [int32 n_tokens][int32 n_embd][f32 data]
        const char * dump_path = std::getenv("MTMD_DEBUG_EMBEDDINGS");
        if (dump_path && strcmp(dump_path, "1") != 0) {
            FILE * f = fopen(dump_path, "wb");
            if (f) {
                const int32_t hdr[2] = { (int32_t)n_tokens, (int32_t)n_embd };
                fwrite(hdr, sizeof(hdr), 1, f);
                fwrite(emb_data.data(), sizeof(float), emb_data.size(), f);
                fclose(f);
            }
        }

        // Print first few values of first token
        LOG_INF("Token 0 (first 16 values): ");
        for (int i = 0; i < std::min((int64_t)16, n_embd); i++) {
            LOG_INF("%.6f ", emb_data[i]);
        }
        LOG_INF("\n");

        // Print last few values of first token
        if (n_embd > 16) {
            LOG_INF("Token 0 (last 16 values):  ");
            for (int64_t i = n_embd - 16; i < n_embd; i++) {
                LOG_INF("%.6f ", emb_data[i]);
            }
            LOG_INF("\n");
        }

        // Compute and print statistics
        float sum = 0.0f, sum_sq = 0.0f, min_val = emb_data[0], max_val = emb_data[0];
        for (size_t i = 0; i < emb_data.size(); i++) {
            sum += emb_data[i];
            sum_sq += emb_data[i] * emb_data[i];
            min_val = std::min(min_val, emb_data[i]);
            max_val = std::max(max_val, emb_data[i]);
        }
        float mean = sum / emb_data.size();
        float variance = (sum_sq / emb_data.size()) - (mean * mean);
        LOG_INF("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n",
                mean, sqrtf(variance), min_val, max_val, sum);
        LOG_INF("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n");
    }

    return true;
}
