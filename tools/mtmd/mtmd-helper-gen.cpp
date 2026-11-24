#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"
#include "llama.h"

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

//
// Audio generation helpers
// no supported model generates audio: the API is kept for its callers and reports the missing pipeline
//

struct mtmd_helper_gen_audio {
    llama_context * lctx = nullptr;
};

mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(struct llama_context * lctx, struct mtmd_context * /*mctx*/) {
    auto * ctx = new mtmd_helper_gen_audio();
    ctx->lctx = lctx;
    return ctx;
}

void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx) {
    delete ctx;
}

int32_t mtmd_helper_gen_audio_max_frames(const mtmd_helper_gen_audio * /*ctx*/) {
    return 0;
}

void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * /*ctx*/) {
}

struct mtmd_helper_gen_audio_inp mtmd_helper_gen_audio_inp_default(void) {
    mtmd_helper_gen_audio_inp inp{};
    inp.top_k      = 50;
    inp.top_p      = 1.0f;
    inp.temp       = -1.0f;      // model default
    inp.seed       = UINT32_MAX; // random
    inp.speed      = 1.0f;
    inp.loudness   = 0.0f;
    inp.cfg_scale  = 0.0f;       // model default
    inp.flow_steps = 0;          // model default
    inp.out_type   = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    return inp;
}

int32_t mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio * /*ctx*/, const mtmd_helper_gen_audio_inp * /*inp*/) {
    LOG_ERR("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n");
    return 1;
}

int32_t mtmd_helper_gen_audio_step_prompt(mtmd_helper_gen_audio * /*ctx*/, int32_t /*n_batch*/) {
    return -1;
}

bool mtmd_helper_gen_audio_next_is_text(mtmd_helper_gen_audio * /*ctx*/) {
    return false;
}

const llama_logit_bias * mtmd_helper_gen_audio_get_logit_bias(mtmd_helper_gen_audio * /*ctx*/, size_t * n_bias) {
    *n_bias = 0;
    return nullptr;
}

int32_t mtmd_helper_audio_stream_get_logit_bias(const mtmd_context * /*mctx*/, const llama_model * /*model*/,
                                                llama_logit_bias * /*out*/, int32_t /*n_max*/) {
    return 0;
}

int32_t mtmd_helper_gen_audio_step_gen(mtmd_helper_gen_audio * /*ctx*/, llama_token /*sampled*/,
                                       const float * /*h_state_in*/, const float ** h_state_out,
                                       bool * out_stop) {
    if (h_state_out) {
        *h_state_out = nullptr;
    }
    if (out_stop) {
        *out_stop = false;
    }
    return 1;
}

int32_t mtmd_helper_gen_audio_get_output(mtmd_helper_gen_audio * /*ctx*/, int32_t * /*out_sample_rate*/,
                                         const char ** /*out_data*/, size_t * /*out_data_len*/, int64_t * /*out_n_samples*/) {
    return 1;
}

int32_t mtmd_helper_gen_audio_flush(mtmd_helper_gen_audio * /*ctx*/) {
    return 1;
}
