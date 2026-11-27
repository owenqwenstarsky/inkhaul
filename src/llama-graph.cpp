#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        bool force_f16 = false) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention and the lightning indexer require an f16 mask
    const auto type = (cparams.flash_attn || force_f16) ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static bool can_reuse_attn_mask(
        const llm_attn_mask & attn_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const uint32_t n_kv_rows = cparams.fused_kv_rows ? mctx->get_n_kv_rows() : 0;
    const uint32_t n_slices  = llama_kv_cache::get_n_kv_slices(ubatch);
    const auto     n_tokens  = ubatch.n_tokens;

    // the current ubatch uses rows only when rows are enabled and it can split into streams
    const bool use_rows = n_kv_rows > 0 && n_slices > 0;

    // the previous mask and the current ubatch must agree on the layout
    if (use_rows != (attn_mask.rows != nullptr)) {
        return false;
    }

    if (!use_rows) {
        return can_reuse_kq_mask(attn_mask.raw, mctx, ubatch, cparams);
    }

    bool res = true;
    res &= attn_mask.n_kv == mctx->get_n_kv();
    res &= attn_mask.rows->ne[0] == n_kv_rows;
    res &= attn_mask.rows->ne[1] == n_slices;
    res &= attn_mask.raw->ne[0] == n_kv_rows;
    res &= attn_mask.raw->ne[1] == n_tokens/n_slices;
    res &= attn_mask.raw->ne[3] == n_slices;

    return res;
}

// KQ mask over the cell lists of kv_rows, or over the whole cache if kv_rows is not used
static llm_attn_mask build_attn_inp_attn_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    llm_attn_mask attn_mask;

    attn_mask.n_kv = mctx->get_n_kv();

    const uint32_t n_kv_rows = cparams.fused_kv_rows ? mctx->get_n_kv_rows() : 0;

    if (n_kv_rows == 0) {
        attn_mask.rows = nullptr;
        attn_mask.raw = build_attn_inp_kq_mask(ctx, mctx, ubatch, cparams);
    } else {
        const uint32_t n_slices = llama_kv_cache::get_n_kv_slices(ubatch);

        attn_mask.rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv_rows, n_slices);
        ggml_set_input(attn_mask.rows);
        ggml_set_name(attn_mask.rows, "attn_inp_kv_rows");

        attn_mask.raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv_rows, ubatch.n_tokens/n_slices, 1, n_slices);
        ggml_set_input(attn_mask.raw);
        ggml_set_name(attn_mask.raw, "attn_inp_kq_mask");
    }

    return attn_mask;
}

// impl

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd_row && embd_rows) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd_rows, ubatch->embd_row, 0, n_tokens*ggml_element_size(embd_rows));
    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    if (ubatch->is_mixed() && embd) {
        GGML_ASSERT(mixed_tokens && mixed_slots && mixed_embd && "mixed token/embd ubatch is not supported here");

        std::vector<int32_t> ids;
        std::vector<int64_t> slots;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (!ubatch->type[i]) {
                ids.push_back(ubatch->token[i]);
                slots.push_back(i);
            }
        }
        GGML_ASSERT((int64_t) ids.size() == mixed_tokens->ne[0]);
        GGML_ASSERT(n_embd == mixed_embd->ne[0]);

        ggml_backend_tensor_set(mixed_tokens, ids.data(),    0, ggml_nbytes(mixed_tokens));
        ggml_backend_tensor_set(mixed_slots,  slots.data(),  0, ggml_nbytes(mixed_slots));
        ggml_backend_tensor_set(mixed_embd,   ubatch->embd,  0, ggml_nbytes(mixed_embd));
    }

    if (scale_rows) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> data(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            data[i] = is_embd ? 1.0f : scale_tok;
        }
        ggml_backend_tensor_set(scale_rows, data.data(), 0, ggml_nbytes(scale_rows));
    }
}

// number of token rows of the mixed path, a non-mixed ubatch is sized for the worst case
static int64_t llm_graph_n_tok_rows(const llama_ubatch & ubatch) {
    if (!ubatch.is_mixed()) {
        return ubatch.n_tokens;
    }
    int64_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n += !ubatch.type[i];
    }
    return n;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
    res &= (!scale_rows) || scale_rows->ne[1] == params.ubatch.n_tokens;
    res &= (!params.ubatch.embd_row) || (embd_rows && embd_rows->ne[0] == params.ubatch.n_tokens);

    return res;
}



void llm_graph_input_batch_idxs::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ubatch->data && ubatch->data->batch_idxs.size() == (size_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(idxs->buffer));

    int64_t * data = (int64_t *) idxs->data;
    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = ubatch->data->batch_idxs[i];
    }
}

bool llm_graph_input_batch_idxs::can_reuse(const llm_graph_params & params) {
    return idxs->ne[0] == params.ubatch.n_tokens;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        const bool has_embd = ubatch->is_mixed() || ubatch->token == nullptr;
        if (rope_section_order == LLAMA_ROPE_SECTION_ORDER_TYXZ || !has_embd) {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
            return;
        }

        // input is always [t, y, x, z]
        // token entries are expanded by the batch to [p, p, p, 0]
        GGML_ASSERT(n_pos_per_embd == 4);

        // slot index per section, slots are t = 0, y = 1, x = 2, z = 3
        std::array<int64_t, 4> slot_of_section = { 0, 1, 2, 3 };
        switch (rope_section_order) {
            case LLAMA_ROPE_SECTION_ORDER_TYXZ: break;
            case LLAMA_ROPE_SECTION_ORDER_ZYXT: slot_of_section = { 3, 1, 2, 0 }; break;
            default: GGML_ABORT("unsupported rope section order");
        }

        std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = ubatch->is_mixed() ? ubatch->type[i] != 0 : true;
            for (int64_t s = 0; s < 4; ++s) {
                const int64_t slot = is_embd ? slot_of_section[s] : s;
                pos_data[s*n_tokens + i] = ubatch->pos[slot*n_tokens + i];
            }
        }
        ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}





void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_moe_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(ggml_backend_buffer_is_host(ids->buffer));

    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];

    // seed from the position: new ids for each step, also when the graph is rebuilt
    std::mt19937 rng(ubatch->pos ? (uint32_t) ubatch->pos[0] : 0);

    std::vector<int32_t> perm(n_expert);
    std::iota(perm.begin(), perm.end(), 0);

    int32_t * data = (int32_t *) ids->data;

    // partial Fisher-Yates: n_used distinct experts per token
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int64_t j = 0; j < n_used; ++j) {
            std::swap(perm[j], perm[j + rng() % (n_expert - j)]);
            data[i*n_used + j] = perm[j];
        }
    }
}

bool llm_graph_input_moe_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= ids->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && cparams.causal_attn)
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }

    set_input_rep(mctx, ubatch);
}

void llm_graph_input_rs::set_input_rep(const llama_memory_recurrent_context * mctx, const llama_ubatch * ubatch) {
    if (!has_rep) {
        return;
    }

    const int64_t n_seqs = ubatch->n_seqs;

    int32_t * reps = (int32_t *) rep_n->data;
    int32_t * cols = (int32_t *) rep_cols->data;

    GGML_ASSERT(ggml_backend_buffer_is_host(rep_n->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(rep_cols->buffer));

    for (int64_t s = 0; s < n_seqs; ++s) {
        const int64_t n_rep = mctx->get_rs_rep(s);
        GGML_ASSERT(n_rep <= n_pend);

        reps[s] = n_rep;

        // the conv history is [window | pending]: the window after the kept tokens
        for (int64_t c = 0; c < n_conv; ++c) {
            cols[s*n_conv + c] = n_rep + c;
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}


template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    // the idxs are left unallocated when the KV cache is bypassed during training
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs, ubatch);
    }
    if (self_v_idxs && self_v_idxs->buffer) {
        mctx->set_input_v_idxs(self_v_idxs, ubatch);
    }

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_attn_mask.raw && self_attn_mask.raw->buffer) {
        mctx->set_input_attn_mask(self_attn_mask, ubatch, cparams.causal_attn);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_attn_mask(self_attn_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_attn_mask(self_attn_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_attn_mask(self_attn_mask, mctx, params.ubatch, params.cparams);

    return res;
}












void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    mctx->get_base()->set_input_attn_mask(self_attn_mask, ubatch, cparams.causal_attn);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    mctx->get_swa()->set_input_attn_mask(self_attn_mask_swa, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_attn_mask.raw && self_attn_mask.raw->buffer) {
        res &= can_reuse_attn_mask(self_attn_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_attn_mask_swa.raw && self_attn_mask_swa.raw->buffer) {
        res &= can_reuse_attn_mask(self_attn_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}



static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}




static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}


static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}



static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}












void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    mctx->get_attn()->set_input_attn_mask(inp_attn->self_attn_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }

    inp_rs->set_input_rep(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_attn_mask(inp_attn->self_attn_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_attn_mask(inp_attn->self_attn_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }

    inp_rs->set_input_rep(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_attn_mask(inp_attn->self_attn_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa_rel::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_attn_mask.raw && inp_attn->self_attn_mask.raw->buffer) {
        attn_ctx->get_base()->set_input_attn_mask(inp_attn->self_attn_mask, ubatch, cparams.causal_attn);
    }

    if (self_rel_pos->buffer) {
        attn_ctx->get_base()->set_input_relative_pos(self_rel_pos, ubatch, extent);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_attn_mask_swa.raw && inp_attn->self_attn_mask_swa.raw->buffer) {
        attn_ctx->get_swa()->set_input_attn_mask(inp_attn->self_attn_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_rel_pos_swa->buffer) {
        attn_ctx->get_swa()->set_input_relative_pos(self_rel_pos_swa, ubatch, extent_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }

    inp_rs->set_input_rep(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid_iswa_rel::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // the masks are always dense, see build_inp_mem_hybrid_iswa_rel()

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(inp_attn->self_attn_mask.raw, attn_ctx->get_base(), params.ubatch, params.cparams);

    res &= self_rel_pos->ne[0] == attn_ctx->get_base()->get_n_kv();
    res &= self_rel_pos->ne[1] == params.ubatch.n_tokens;

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(inp_attn->self_attn_mask_swa.raw, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= self_rel_pos_swa->ne[0] == attn_ctx->get_swa()->get_n_kv();
    res &= self_rel_pos_swa->ne[1] == params.ubatch.n_tokens;

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::set_max_nodes(int64_t max_nodes) {
    this->max_nodes = max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();
    t_sampling_all.clear();
    t_moe_ids.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();

    const size_t meta_size = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    if (meta_size != buf_compute_meta_size) {
        // not make_unique: it would zero the whole reserve and make its pages resident
        buf_compute_meta.reset(new uint8_t[meta_size]);
        buf_compute_meta_size = meta_size;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta_size,
        /*.mem_buffer =*/ buf_compute_meta.get(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size() && !params.cparams.embeddings_layer_inp_dev; ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    // the chains are read on the host after compute, so keep every one alive
    for (auto * tensor : t_sampling_all) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}


void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    model            (*params.model),
    hparams          (model.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    prec_policy      (model.prec_policy),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (hparams.n_expert_used()),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    backend_kv       (params.backend_kv),
    loras            (params.loras),
    mctx             (params.mctx),
    moe_cache        (params.moe_cache),
    layer_inp_dev    (params.layer_inp_dev),
    layer_inp_src    (params.layer_inp_src),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}




static ggml_tensor * build_rope_impl(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * pos, const llama_hparams & hparams, const llama_rope_params & rp) {
    if (rp.n_rot == 0) {
        return cur;
    }

    if (rp.type & GGML_ROPE_TYPE_MROPE) {
        int sections[GGML_MROPE_SECTIONS];
        std::copy(hparams.rope_sections.begin(), hparams.rope_sections.end(), sections);

        return ggml_rope_multi(ctx, cur, pos, rp.factors,
                rp.n_rot, sections, rp.type, rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
                rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
    }

    return ggml_rope_ext(ctx, cur, pos, rp.factors,
            rp.n_rot, rp.type, rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
            rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
}

ggml_tensor * llm_graph_context::build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) const {
    return build_rope_impl(ctx0, cur, pos, hparams, model.get_rope_params(cparams, il));
}


ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    if (!w_s) {
        w_s = model.get_w_scale(w);
    }

    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }


    prec_policy.apply(res, w);

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

llm_graph_amax_src1 llm_graph_context::build_amax_src1_f16(
          ggml_tensor * cur) const {
    llm_graph_amax_src1 result = {
        /*.res     =*/ cur,
        /*.scale_s =*/ nullptr,
    };

    const int64_t n_embd = cur->ne[0];

    // amax = max|cur| along the rows for each [expert, token] -> [1, n_expert, n_tokens]
    ggml_tensor * ab = ggml_abs(ctx0, cur);
    ggml_tensor * amax = ggml_pool_1d(ctx0, ab, GGML_OP_POOL_MAX, n_embd, n_embd, 0);

    // ref: https://github.com/ggml-org/llama.cpp/pull/26223
    constexpr float k_log2_e = 1.4426950408889634f; // 1 / ln(2)
    constexpr float k_ln_2   = 0.6931471805599453f;

    ggml_tensor * e;
    // note: this can technically be:
    // e = ggml_scale(ctx0, amax, 1.0f/32768.0f);
    // but we leave x32 headroom for the F16 sums of 32 values in Q8 activation blocks
    e = ggml_scale(ctx0, amax, 1.0f/1024.0f);
    e = ggml_log  (ctx0, e);
    e = ggml_scale(ctx0, e, k_log2_e);
    e = ggml_ceil (ctx0, e);
    e = ggml_clamp(ctx0, e, 0.0f, 127.0f);
    e = ggml_scale(ctx0, e, k_ln_2);
    e = ggml_exp  (ctx0, e);

    result = {
        /*.res     =*/ ggml_div(ctx0, cur, e),
        /*.scale_s =*/ e,
    };

    return result;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * slots,
              int32_t   cache_dev) const {
    ggml_tensor * w_s = model.get_w_scale(w);
    // the experts in the MoE cache are selected by their slots
    ggml_tensor * res = slots == nullptr ?
        ggml_mul_mat_id(ctx0, w, cur, ids) :
        ggml_mul_mat_id(ctx0, moe_cache->get_experts(w, cache_dev), cur, slots);

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    // after the NVFP4 default, so that the policy can request a higher precision
    prec_policy.apply(res, w);

    if (slots != nullptr) {
        moe_cache->check_op(w, res, cache_dev);
    }

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s;
        if (ggml_is_contiguous(ids)) {
            // one gather by all the ids, without a copy of the scales per token
            s = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, w_s, 1, n_expert), ggml_reshape_1d(ctx0, ids, ggml_nelements(ids)));
            s = ggml_reshape_3d(ctx0, s, 1, ids->ne[0], ids->ne[1]);
        } else {
            s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, ids);
        }
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    return build_qkv(layer, cur,
            n_embd_head, n_head,
            n_embd_head, n_head_kv,
            n_embd_head, n_head_kv,
            il);
}

llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape) const {
    const int64_t n_embd_q = n_embd_head_q * n_head_q;
    const int64_t n_embd_k = n_embd_head_k * n_head_k;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        } else if (layer.wq_b && layer.wk_b && layer.wv_b) {
            // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
            ggml_tensor * qkv_b = ggml_concat(ctx0, ggml_concat(ctx0, layer.wq_b, layer.wk_b, 0), layer.wv_b, 0);
            qkv = ggml_add(ctx0, qkv, qkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (reshape) {
            Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head_q, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_q), qkv->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_k, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_k), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_v, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        } else {
            Qcur = ggml_view_2d(ctx0, qkv, n_embd_q, n_tokens, qkv->nb[1], 0);
            Kcur = ggml_view_2d(ctx0, qkv, n_embd_k, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_2d(ctx0, qkv, n_embd_head_v * n_head_v, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        }
        if (!reshape) {
            Qcur = ggml_cont(ctx0, Qcur);
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);
        }
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        if (reshape) {
            cb(Qcur, "Qcur", il);
        }
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            if (reshape) {
                cb(Qcur, "Qcur", il);
            }
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        if (reshape) {
            cb(Kcur, "Kcur", il);
        }
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            if (reshape) {
                cb(Kcur, "Kcur", il);
            }
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        if (reshape) {
            cb(Vcur, "Vcur", il);
        }
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            if (reshape) {
                cb(Vcur, "Vcur", il);
            }
        }
        if (reshape) {
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_q, n_head_q, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_k, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_v, n_tokens);
        }
    }

    if (reshape) {
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
    }

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    ggml_tensor * tmp = up ? build_lora_mm(up, cur, up_s) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp, gate_s);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur, gate_s);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR && il >= 0 && hparams.swiglu_clamp_shexp[il] > 1e-6f) {
                // clamped swiglu (bailingmoe3): silu(gate) is clamped from above, up on both sides
                const float limit = hparams.swiglu_clamp_shexp[il];
                tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                cb(tmp, "ffn_up_clamped", il);
                cur = ggml_clamp(ctx0, ggml_silu(ctx0, cur), -INFINITY, limit);
                cb(cur, "ffn_silu_clamped", il);
                cur = ggml_mul(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu_limited", il);
                type_gate = LLM_FFN_SEQ;
            } else if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_GELU_ERF:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_GEGLU_ERF:
            {
                cur = ggml_geglu_erf(ctx0, cur);
                cb(cur, "ffn_geglu_erf", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        case LLM_FFN_SITU:
            GGML_ABORT("not yet supported");
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur, down_s);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * selected_experts_in,
         ggml_tensor * post_exp_norm,
               float   post_exp_norm_eps) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        selected_experts_in,
        post_exp_norm,
        post_exp_norm_eps
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * selected_experts_in,
         ggml_tensor * post_exp_norm,
               float   post_exp_norm_eps) const {
    const llm_arch arch = model.arch;
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        if (hparams.expert_weights_norm_eps > 0.0f) {
            weights_sum = ggml_scale_bias(ctx0, weights_sum, 1.0f, hparams.expert_weights_norm_eps);
        } else {
            // Avoid division by zero, clamp to smallest number representable by F16
            weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        }
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    // benchmark only: the expert GEMMs use random ids, the router above is not changed
    if (cparams.moe_random_routing && n_expert == hparams.n_expert && n_expert_used <= hparams.n_expert_used_max() && n_tokens <= ubatch.n_tokens) {
        // fused top-k kernels write the router ids together with the weights, so keep the ids alive until here
        // the empty get_rows is skipped by the backends, it only adds this dependency
        ggml_build_forward_expand(gf, ggml_get_rows(ctx0, weights, ggml_view_2d(ctx0, selected_experts, 0, n_tokens, selected_experts->nb[1], 0)));

        selected_experts = build_inp_moe_ids(n_expert_used, n_tokens);

        // host experts: compute the ids on the device, as the router ids are, because the MoE cache and the expert copy read them there
        if (down_exps->buffer && ggml_backend_buffer_is_host(down_exps->buffer)) {
            selected_experts = ggml_cont(ctx0, selected_experts);
        }
    }

    // compute the expert input before the slot lookup (the router can read a different tensor)
    ggml_build_forward_expand(gf, cur);

    // the experts on disk are read from the LongHaul cache (see llama_moe_cache::build_lookup)
    const llama_moe_cache_lookup cache = moe_cache == nullptr ? llama_moe_cache_lookup{} :
        moe_cache->build_lookup(*this, { il, up_exps, gate_exps, down_exps, gate_up_exps }, selected_experts);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    // the experts selected by mm_ids, up to the down projection, from the MoE cache of device cache_dev if mm_ids are slots
    // ids are the selected experts, for the biases and scales
    auto build_experts = [&](ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * mm_ids, int32_t cache_dev) {
        ggml_tensor * up = nullptr;

        if (gate_up_exps) {
            // merged gate_up path: one mul_mat_id, then split into gate and up views
            ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, ids, mm_ids, cache_dev); // [n_ff*2, n_expert_used, n_tokens]
            cb(gate_up, "ffn_moe_gate_up", il);

            if (gate_up_exps_b) {
                gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, ids);
                cb(gate_up, "ffn_moe_gate_up_biased", il);
            }

            const int64_t n_ff = gate_up->ne[0] / 2;
            cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            cb(cur, "ffn_moe_gate", il);
            up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
            cb(up, "ffn_moe_up", il);
        } else {
            // separate gate and up path
            up = build_lora_mm_id(up_exps, cur, ids, mm_ids, cache_dev); // [n_ff, n_expert_used, n_tokens]
            cb(up, "ffn_moe_up", il);

            if (up_exps_b) {
                up = ggml_add_id(ctx0, up, up_exps_b, ids);
                cb(up, "ffn_moe_up_biased", il);
            }

            if (gate_exps) {
                cur = build_lora_mm_id(gate_exps, cur, ids, mm_ids, cache_dev); // [n_ff, n_expert_used, n_tokens]
                cb(cur, "ffn_moe_gate", il);
            } else {
                cur = up;
            }

            if (gate_exps_b) {
                cur = ggml_add_id(ctx0, cur, gate_exps_b, ids);
                cb(cur, "ffn_moe_gate_biased", il);
            }
        }

        const bool has_gate = gate_exps || gate_up_exps;

        switch (type_op) {
            case LLM_FFN_SILU:
                if (has_gate && il >= 0 && hparams.swiglu_clamp_exp[il] > 1e-6f) {
                    // clamped swiglu (bailingmoe3), see build_ffn
                    const float limit = hparams.swiglu_clamp_exp[il];
                    up = ggml_clamp(ctx0, up, -limit, limit);
                    cb(up, "ffn_moe_up_clamped", il);
                    cur = ggml_clamp(ctx0, ggml_silu(ctx0, cur), -INFINITY, limit);
                    cb(cur, "ffn_moe_silu_clamped", il);
                    cur = ggml_mul(ctx0, cur, up);
                    cb(cur, "ffn_moe_swiglu_limited", il);
                } else if (has_gate) {
                    cur = ggml_swiglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_swiglu", il);
                } else {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_moe_silu", il);
                } break;
            case LLM_FFN_SITU:
                {
                    // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                    GGML_ASSERT(has_gate);
                    const float beta = hparams.situ_beta;
                    const float lb   = hparams.situ_linear_beta;

                    ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                    act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                    if (lb > 0.0f) {
                        up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                    }
                    cur = ggml_mul(ctx0, act, up);
                    cb(cur, "ffn_moe_situ", il);
                } break;
            case LLM_FFN_GELU:
                if (has_gate) {
                    cur = ggml_geglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_geglu", il);
                } else {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_moe_gelu", il);
                } break;
            case LLM_FFN_SWIGLU_OAI_MOE:
                {
                    // TODO: move to hparams?
                    constexpr float alpha = 1.702f;
                    constexpr float limit = 7.0f;
                    cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                    cb(cur, "ffn_moe_swiglu_oai", il);
                } break;
            case LLM_FFN_RELU:
                if (has_gate) {
                    cur = ggml_reglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_reglu", il);
                } else {
                    cur = ggml_relu(ctx0, cur);
                    cb(cur, "ffn_moe_relu", il);
                } break;
            case LLM_FFN_RELU_SQR:
                if (has_gate) {
                    // TODO: add support for gated squared relu
                    GGML_ABORT("fatal error: gated squared relu not implemented");
                } else {
                    cur = ggml_relu(ctx0, cur);
                    cur = ggml_sqr(ctx0, cur);
                    cb(cur, "ffn_moe_relu_sqr", il);
                } break;
            default:
                GGML_ABORT("fatal error");
        }

        // TODO: this graph-level rescaling should only be applied when the corresponding backend does not support
        //       ggml_prec_set_src(experts, GGML_PREC_BF16, 1);
        //       the amax op cannot be split across devices, so these architectures are denied -sm tensor [TAG_MOE_AMAX_SCALE]
        //       ref: https://github.com/ggml-org/llama.cpp/pull/30656#issuecomment-6269597922
        const bool use_amax = llm_arch_mmid_down_bf16(arch) && prec_policy.get_src1(down_exps) != GGML_PREC_BF16;

        llm_graph_amax_src1 amax_src1 = { nullptr, nullptr };

        if (use_amax) {
            amax_src1 = build_amax_src1_f16(cur);
            cur = amax_src1.res;
        }

        ggml_tensor * experts = build_lora_mm_id(down_exps, cur, ids, mm_ids, cache_dev); // [n_embd, n_expert_used, n_tokens]
        cb(experts, "ffn_moe_down", il);

        if (amax_src1.scale_s) {
            experts = ggml_mul(ctx0, experts, amax_src1.scale_s);
        }

        return experts;
    };

    // the weighted sum of the outputs of the selected experts, [n_embd, n_tokens]
    auto sum_experts = [&](ggml_tensor * experts) {
        // an expert output can be scaled by a norm weight, then normalized, before the router weights apply
        if (post_exp_norm) {
            experts = ggml_mul(ctx0, experts, post_exp_norm);
            experts = ggml_rms_norm(ctx0, experts, post_exp_norm_eps);
            cb(experts, "ffn_moe_post_exp_norm", il);
        }

        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);

        ggml_build_forward_expand(gf, experts);

        ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

        assert(n_expert_used > 0);

        // order the views before the adds
        // Use per-layer n_expert_used to bound the graph even during warmup (avoids
        // the large-add-nodes issue for uniform arches; for Puzzle the per-layer
        // value is correct). ref: https://github.com/ggml-org/llama.cpp/pull/14753
        const uint32_t n_expert_used_il = hparams.n_expert_used(il);
        for (uint32_t i = 0; i < n_expert_used_il; ++i) {
            cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

            ggml_build_forward_expand(gf, cur_experts[i]);
        }

        // aggregate experts
        ggml_tensor * moe_out = cur_experts[0];

        for (uint32_t i = 1; i < n_expert_used_il; ++i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

            ggml_build_forward_expand(gf, moe_out);
        }

        if (n_expert_used_il == 1) {
            // avoid returning a non-contiguous tensor
            moe_out = ggml_cont(ctx0, moe_out);
        }
        return moe_out;
    };

    ggml_tensor * moe_out = nullptr;

    if (cache.parts.size() <= 1) {
        const llama_moe_cache_part part = cache.parts.empty() ? llama_moe_cache_part{} : cache.parts[0];
        ggml_tensor * experts = build_experts(cur, selected_experts, part.slots, part.dev);

        if (down_exps_b) {
            experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
            cb(experts, "ffn_moe_down_biased", il);
        }

        moe_out = sum_experts(experts);
    } else {
        // expert split: each device computes the selected experts it owns (the others have a slot of -1 and give zeros) and sums
        // them, so that only [n_embd, n_tokens] crosses to the home backend, which adds the sums of the devices
        // per-expert scales keep a skipped row at zero, a bias would not
        // the devices other than the home device come first, so that their work is queued before the home device computes its part
        GGML_ASSERT(!up_exps_b && !gate_exps_b && !down_exps_b && !gate_up_exps_b && "the expert split does not support expert biases");
        std::vector<ggml_tensor *> part_outs;
        for (const llama_moe_cache_part & part : cache.parts) {
            const int n_nodes_prev = ggml_graph_n_nodes(gf);

            ggml_tensor * part_out = sum_experts(build_experts(cur, selected_experts, part.slots, part.dev));
            ggml_build_forward_expand(gf, part_out);

            // the new nodes run on the device of the part, except the slot lookup, which is already on the home backend
            ggml_backend_t backend = moe_cache->backend(part.dev);
            for (int i = n_nodes_prev; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                if (node->view_src == nullptr && ggml_backend_sched_get_tensor_backend(sched, node) == nullptr) {
                    ggml_backend_sched_set_tensor_backend(sched, node, backend);
                }
            }

            part_outs.push_back(part_out);
        }

        // the home backend adds the sums after all parts, in a split of its own: the inputs of a split are copied when it starts,
        // so the home device queues its part before it waits for the sums of the other devices
        moe_out = part_outs[0];
        for (size_t ip = 1; ip < part_outs.size(); ++ip) {
            moe_out = ggml_add(ctx0, moe_out, part_outs[ip]);
            ggml_backend_sched_set_tensor_backend(sched, moe_out, moe_cache->home());
            if (ip == 1) {
                ggml_backend_sched_set_split_start(sched, moe_out);
            }
            ggml_build_forward_expand(gf, moe_out);
        }
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd, float tok_scale) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    // mixed path (ubatch.is_mixed()): set_rows the token rows into a copy of the embd rows, with its own inputs as select branches must not share tensors
    // TODO: use inp->tokens and inp->embd once ggml_build_forward_select allows it
    const bool has_mixed = llm_arch_supports_mixed_batch(model.arch) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT;

    const int64_t n_tok_rows = has_mixed ? llm_graph_n_tok_rows(ubatch) : 0;

    // we have 3 standard paths to produce the input embeddings for the first layer:
    // - embd0: extract from the token embeddings weight (`tok_embd`) using the input token ids
    // - embd1: pass raw embeddings, skipping the `tok_embd`
    // - embd2: mixed path of both tokens ids + raw embeddings (if supported)
    ggml_tensor * embd0 = nullptr;
    ggml_tensor * embd1 = nullptr;
    ggml_tensor * embd2 = nullptr;

    // construct the input tensors
    {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);

        if (has_mixed) {
            inp->mixed_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok_rows);
            cb(inp->mixed_tokens, "inp_mixed_tokens", -1);
            ggml_set_input(inp->mixed_tokens);
        }
    }

    // the embeddings placeholders for the 3 paths
    // we use ggml_build_forward_order to make the GET_ROWS ops stick at the beginning of the compute graph
    // this way the embeddings remain in the host buffer, and the GET_ROWS run before any other computations
    {
        embd0 = ggml_get_rows(ctx0, tok_embd, inp->tokens);
        ggml_build_forward_order(gf, embd0);

        embd1 = inp->embd;

        if (has_mixed) {
            embd2 = ggml_get_rows(ctx0, tok_embd, inp->mixed_tokens);
            ggml_build_forward_order(gf, embd2);
        }
    }

    // helper for extracting token embeddings with lora and padding
    // TODO: when lora is active, this is likely going to cause issues similar to https://github.com/ggml-org/llama.cpp/pull/30160
    //       need to add lora tests and refactor the logic to make the lora GET_ROWS go at the front of the graph
    auto build_tok = [&](ggml_tensor * cur, ggml_tensor * ids) {
        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, ids)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }

        return cur;
    };

    // select one of the 3 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 3> inps = {};

    // token embeddings path (ubatch.token != nullptr)
    inps[0] = build_tok(embd0, inp->tokens);

    // vector embeddings path (ubatch.embd != nullptr)
    inps[1] = embd1;

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    if (has_mixed) {
        inp->mixed_slots = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok_rows);
        cb(inp->mixed_slots, "inp_mixed_slots", -1);
        ggml_set_input(inp->mixed_slots);

        inp->mixed_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->mixed_embd, "inp_mixed_embd", -1);
        ggml_set_input(inp->mixed_embd);

        // note: set_rows writes into its destination, so it gets a copy of the input
        ggml_tensor * embd_mixed = build_tok(embd2, inp->mixed_tokens);
        inps[2] = ggml_set_rows(ctx0, ggml_dup(ctx0, inp->mixed_embd), embd_mixed, inp->mixed_slots);

        assert(ggml_are_same_shape (inps[0], inps[2]));
        assert(ggml_are_same_stride(inps[0], inps[2]));
    }

    const int idx = ubatch.is_mixed() ? 2 : ubatch.token ? 0 : 1;

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), has_mixed ? 3 : 2, idx);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    if (hparams.f_embedding_scale != 0.0f) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    // scale the token rows only, applied after the select so that the graph is the same for any batch contents
    inp->scale_tok = tok_scale;
    if (inp->scale_tok != 1.0f) {
        inp->scale_rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, ubatch.n_tokens);
        cb(inp->scale_rows, "inp_scale_rows", -1);
        ggml_set_input(inp->scale_rows);

        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_mul(ctx0, cur, inp->scale_rows);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}


ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd(), hparams.rope_section_order);

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);
    cb(cur, "inp_pos", -1);

    res->add_input(std::move(inp));

    return cur;
}


ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);
    ggml_set_name(cur, "out_ids");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "mean");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, model.arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "cls");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_moe_ids(int64_t n_used, int64_t n_rows) const {
    auto & ts = res->t_moe_ids;

    if (ts.empty()) {
        auto inp = std::make_unique<llm_graph_input_moe_ids>(hparams.n_expert);

        auto & cur = inp->ids;

        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, hparams.n_expert_used_max(), n_tokens);
        ggml_set_input(cur);
        cb(cur, "inp_moe_ids", -1);

        ts.push_back(cur);

        res->add_input(std::move(inp));
    }

    // one tensor per shape: each distinct tensor is copied to the device on each ubatch
    for (auto * t : ts) {
        if (t->ne[0] == n_used && t->ne[1] == n_rows) {
            return t;
        }
    }

    ts.push_back(ggml_view_2d(ctx0, ts[0], n_used, n_rows, ts[0]->nb[1], 0));

    return ts.back();
}





void llm_graph_context::pin_kv(ggml_tensor * t, int il) const {
    if (backend_kv != nullptr && il >= 0 && model.kv_remote(il)) {
        ggml_backend_sched_set_tensor_backend(sched, t, backend_kv);
    }
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * kv_rows,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
             int64_t   n_kv_max,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = kv_rows ? kv_rows->ne[1] : k->ne[3];

    // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
    // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
    // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] read another head's queries for streams s >= 1.
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2]*(q->ne[2]/n_stream), 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    // kv_gpu: the nodes from the KV cache to the attention output run on the KV device, before the MLA weights
    auto pin = [&](ggml_tensor * t) {
        pin_kv(t, il);
        return t;
    };

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    GGML_ASSERT(use_flash_attn || kv_rows == nullptr);
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = pin(ggml_cast(ctx0, k, GGML_TYPE_F16));
        }

        if (v->type == GGML_TYPE_F32) {
            v = pin(ggml_cast(ctx0, v, GGML_TYPE_F16));
        }

        if (kv_rows) {
            cur = ggml_flash_attn_ext_rows(ctx0, q, k, v, kq_mask, kv_rows, kq_scale, 0.0f, 0.0f);
        } else {
            cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        }
        pin(cur);
        res->add_fused_node({kv_rows ? LLM_FUSED_OP_FLASH_ATTN_KV_ROWS : LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
        ggml_prec_set_acc(cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = pin(ggml_mul_mat(ctx0, k, q));
        cb(kq, "kq", il);

        ggml_prec_set_acc(kq, GGML_PREC_BF16);

        if (kq_b) {
            kq = pin(ggml_add(ctx0, kq, kq_b));
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = pin(ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f));
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = pin(ggml_cont(ctx0, ggml_transpose(ctx0, v)));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = pin(ggml_mul_mat(ctx0, v, kq));
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        if (!v_mla) {
            pin(cur);
        }

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}


static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_attn_mask = build_attn_inp_attn_mask(ctx0, mctx_cur, ubatch, cparams);
    }

    inp->self_k_rot = mctx_cur->get_k_rot();
    inp->self_v_rot = mctx_cur->get_v_rot();

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

static void build_attn_scale_kv_inputs(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        ggml_tensor * & q,
        ggml_tensor * & k,
        ggml_tensor * & v,
        int32_t il) {
    if (ggml_tensor * scale = mctx->get_k_scale(il)) {
        // Store K/s and move its dequant scale to Q: (Q*s) dot (K/s) = Q dot K.
        q = ggml_mul(ctx, q, scale);
        if (k) {
            k = ggml_div(ctx, k, scale);
        }
    }

    if (v) {
        // Store V/s here and restore its dequant scale after the weighted sum.
        ggml_tensor * scale = mctx->get_v_scale(il);
        if (scale) {
            v = ggml_div(ctx, v, scale);
        }
    }
}

static ggml_tensor * build_attn_scale_kv_output(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        ggml_tensor * cur,
        int32_t il) {
    if (ggml_tensor * scale = mctx->get_v_scale(il)) {
        // P*(V/s)*s = P*V.
        cur = ggml_mul(ctx, cur, scale);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot[il]) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot[il]);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot[il]);
    }

    if (inp->self_v_rot[il]) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot[il]);
    }

    const auto * mctx_cur = inp->mctx;

    build_attn_scale_kv_inputs(ctx0, mctx_cur, q_cur, k_cur, v_cur, il);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    ggml_tensor * q = q_cur;
    ggml_tensor * k;
    ggml_tensor * v;

    if (cparams.training) {
        GGML_ASSERT(mctx_cur->get_n_kv() == n_tokens);

        k = k_cur;
        v = v_cur;
    } else {
        {
            const auto & k_idxs = inp->get_k_idxs();
            const auto & v_idxs = inp->get_v_idxs();

            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
        }

        k = mctx_cur->get_k(ctx0, il);
        v = mctx_cur->get_v(ctx0, il);
    }

    const auto & kq_mask = inp->self_attn_mask.raw;
    const auto & kv_rows = inp->self_attn_mask.rows;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il);
    cur = build_attn_scale_kv_output(ctx0, mctx_cur, cur, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot[il]) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot[il]);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}




static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_attn_mask = build_attn_inp_attn_mask(ctx0, mctx_cur, ubatch, cparams);
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->self_attn_mask.raw;
    const auto & kv_rows = inp->self_attn_mask.rows;

    // MLA: V is the latent part at the start of the cached K row
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// the iswa build_attn with the KQ mask and kv_rows given by the caller, which has to keep them in sync
static ggml_tensor * build_attn_iswa_impl(
        const llm_graph_context & gctx,
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * kq_mask,
        ggml_tensor * kv_rows,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) {
    ggml_context * ctx0 = gctx.ctx0;
    ggml_cgraph  * gf   = gctx.gf;

    const auto & hparams = gctx.hparams;
    const auto & cparams = gctx.cparams;

    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa[il] : inp->self_k_rot[il];
    auto * v_rot = is_swa ? inp->self_v_rot_swa[il] : inp->self_v_rot[il];

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    build_attn_scale_kv_inputs(ctx0, mctx_cur, q_cur, k_cur, v_cur, il);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    // whole seq fits into batch in training mode
    const bool use_kv_cur = cparams.training && k_cur && v_cur;
    if (use_kv_cur) {
        GGML_ASSERT(mctx_cur->get_n_kv() == gctx.n_tokens);
    }

    const bool store_kv = !use_kv_cur;

    // optionally store to KV cache
    if (store_kv && k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (store_kv && v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kv_cur ? k_cur : mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kv_cur ? v_cur : mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = gctx.build_attn_mha(q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il);
    cur = build_attn_scale_kv_output(ctx0, mctx_cur, cur, il);
    gctx.cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = gctx.build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->self_attn_mask_swa.raw  : inp->self_attn_mask.raw;
    const auto & kv_rows = is_swa ? inp->self_attn_mask_swa.rows : inp->self_attn_mask.rows;

    return build_attn_iswa_impl(*this, inp, wo, wo_b, wo_s, q_cur, k_cur, v_cur, kq_b, kq_mask, kv_rows, sinks, v_mla, kq_scale, il);
}








// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_attn_mask = build_attn_inp_attn_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_attn_mask_swa = build_attn_inp_attn_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
    }

    inp->self_k_rot = mctx_cur->get_base()->get_k_rot();
    inp->self_v_rot = mctx_cur->get_base()->get_v_rot();

    inp->self_k_rot_swa = mctx_cur->get_swa()->get_k_rot();
    inp->self_v_rot_swa = mctx_cur->get_swa()->get_v_rot();

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}



ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy,
        ggml_tensor * state_copy_main,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_fill_inplace(ctx0, state_zero, 0.0f));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // one gather of the states i0..n_rs (ubatch states then extra states), sized by n_rs so the reserve covers every split
    // {state_size, rs_size} -> {state_size, n_rs - i0}
    const int64_t i0 = n_rs - state_copy->ne[0];

    ggml_tensor * states_all = ggml_get_rows(ctx0, states, state_copy);

    ggml_tensor * output_states = get_state_rows ?
        get_state_rows(ctx0, states, state_copy_main) :
        ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_view_2d(ctx0, states_all, state_size, n_rs - n_seqs, states_all->nb[1], (n_seqs - i0)*states_all->nb[1]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);
    ggml_set_name(inp->s_copy, "rs_s_copy");

    inp->s_copy_main = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_tail = ggml_view_1d(ctx0, inp->s_copy, n_rs - 1, inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    inp->has_rep = mctx_cur->get_rs_replay();
    if (inp->has_rep) {
        // the conv window of the model: the delta-net conv
        const int64_t n_conv_r = hparams.ssm_d_conv - 1;

        inp->n_conv = n_conv_r;
        inp->n_pend = cparams.n_rs_seq + 1;

        inp->rep_cols = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, inp->n_conv, n_seqs);
        ggml_set_input(inp->rep_cols);
        ggml_set_name(inp->rep_cols, "rs_rep_cols");

        const auto win = [&](int64_t n) {
            return n == inp->n_conv ? inp->rep_cols : ggml_view_2d(ctx0, inp->rep_cols, n, n_seqs, inp->rep_cols->nb[1], 0);
        };

        inp->rep_cols_r = win(n_conv_r);

        inp->rep_n = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seqs);
        ggml_set_input(inp->rep_n);
        ggml_set_name(inp->rep_n, "rs_rep_n");
    }

    return inp;
}


ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first
    // state: it still holds the n_rs - n_seqs extra states and copies no state of a single sequence ubatch
    ggml_tensor * state_copy = get_state_rows ? inp->s_copy_tail : inp->s_copy;

    return build_rs(s, state_copy, inp->s_copy_main, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

void llm_graph_context::build_conv_state_store(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
        ggml_tensor * conv_x,
            int64_t   n_cols,
             size_t   offs) const {
    const auto * mctx_cur = inp->mctx;

    const int64_t n        = conv_x->ne[0] - n_cols; // n_seq_tokens
    const int64_t channels = conv_x->ne[1];
    const int64_t n_seqs   = conv_x->ne[2];

    const size_t row_size  = s->nb[1];
    const size_t cell_offs = mctx_cur->get_head()*row_size + offs;

    if (cparams.n_rs_seq == 0) {
        ggml_tensor * last = ggml_view_3d(ctx0, conv_x,
                n_cols, channels, n_seqs,
                conv_x->nb[1], conv_x->nb[2],
                ggml_row_size(conv_x->type, n));

        ggml_tensor * cell = ggml_view_2d(ctx0, s,
                n_cols*channels, n_seqs,
                row_size, cell_offs);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, cell));
        return;
    }

    // [TAG_RECURRENT_ROLLBACK_SPLITS]
    // this logic assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are inside
    //   the same ubatch, which `split_equal()` guarantees via its n_keep_tail argument
    // a rollback of k tokens reads slot k, and the memory allows k < n only (see rs_max), so slots from n on are left unwritten
    const int64_t K  = (int64_t) cparams.n_rs_seq + 1;
    const int64_t Kw = std::min(n, K);

    // each conv layer builds its own slot order {K-1, ..., 0}, on the device of the layer, and takes its tail
    // {Kw-1, ..., 0}; the size K never depends on the ubatch, so the reserved graph covers them all
    ggml_tensor * s_slot_rev = ggml_arange(ctx0, 0.0f, (float) K, 1.0f);
    s_slot_rev = ggml_scale_bias(ctx0, s_slot_rev, -1.0f, (float) (K - 1));
    s_slot_rev = ggml_cast(ctx0, s_slot_rev, GGML_TYPE_I32);
    cb(s_slot_rev, "s_slot_rev", -1);

    ggml_tensor * slot_order = ggml_view_1d(ctx0, s_slot_rev, Kw, ggml_row_size(GGML_TYPE_I32, K - Kw));

    // window w starts at column n - (Kw - 1) + w (ascending time) and is rollback slot Kw - 1 - w,
    // consecutive windows start one element apart, so they overlap
    ggml_tensor * windows = ggml_view_4d(ctx0, conv_x,
            n_cols, Kw, channels, n_seqs,
            ggml_element_size(conv_x), conv_x->nb[1], conv_x->nb[2],
            ggml_row_size(conv_x->type, n - (Kw - 1)));

    // [n_cols, slot, channel, seq] view of the cells of this ubatch
    ggml_tensor * slots = ggml_view_4d(ctx0, s,
            n_cols, K, channels, n_seqs,
            (size_t) mctx_cur->get_size()*row_size, ggml_row_size(s->type, n_cols), row_size,
            cell_offs);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, slots, windows, slot_order));
}






llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, hparams, cparams, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl    (ctx0, ubatch, hparams, cparams, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa_rel * llm_graph_context::build_inp_mem_hybrid_iswa_rel(
        uint32_t extent,
        uint32_t extent_swa) const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_recr());

    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    // dense masks: the bias is gathered by the distance to each cell of the K/V view
    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_attn_mask.raw = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_attn_mask_swa.raw = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
    }

    inp_attn->self_k_rot = attn_ctx->get_base()->get_k_rot();
    inp_attn->self_v_rot = attn_ctx->get_base()->get_v_rot();

    inp_attn->self_k_rot_swa = attn_ctx->get_swa()->get_k_rot();
    inp_attn->self_v_rot_swa = attn_ctx->get_swa()->get_v_rot();

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa_rel>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur, extent, extent_swa);

    inp->self_rel_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, attn_ctx->get_base()->get_n_kv(), ubatch.n_tokens);
    ggml_set_input(inp->self_rel_pos);
    ggml_set_name(inp->self_rel_pos, "attn_inp_rel_pos");

    inp->self_rel_pos_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, attn_ctx->get_swa()->get_n_kv(), ubatch.n_tokens);
    ggml_set_input(inp->self_rel_pos_swa);
    ggml_set_name(inp->self_rel_pos_swa, "attn_inp_rel_pos_swa");

    return (llm_graph_input_mem_hybrid_iswa_rel *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_mem_hybrid_iswa_rel * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * inp_attn = inp->get_attn();

    const auto & attn_mask = is_swa ? inp_attn->self_attn_mask_swa : inp_attn->self_attn_mask;

    // TODO: support kv_rows - fill the distances over the kv_rows cell lists and pass the rows to build_attn_mha,
    //       needs a kv_rows flash attention with a per-head mask (CUDA has none)
    GGML_ASSERT(attn_mask.rows == nullptr);

    // a mask per head, in the dense layout of the input mask, which flash attention takes in F16
    // kq_b can be a permuted view, with kv_gpu it is made contiguous on the KV device
    ggml_tensor * kq_mask = attn_mask.raw;
    if (cparams.flash_attn && kq_mask->type == GGML_TYPE_F16) {
        // the mask holds only 0 and -inf, so an F16 add gives the same result as an F32 add and a cast
        kq_b = ggml_cast(ctx0, kq_b, GGML_TYPE_F16);
        pin_kv(kq_b, il);
        kq_mask = ggml_add(ctx0, kq_b, kq_mask);
    } else {
        if (kq_mask->type != GGML_TYPE_F32) {
            kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        }
        kq_b = ggml_cont(ctx0, kq_b);
        pin_kv(kq_b, il);
        kq_mask = ggml_add(ctx0, kq_b, kq_mask);
        if (cparams.flash_attn) {
            kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
        }
    }
    cb(kq_mask, "kq_mask_rel_pos", il);

    return build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, kq_mask, nullptr, sinks, nullptr, kq_scale, il);
}

void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b) const {
    if (!cparams.embeddings) {
        // a graph without logits still computes t_embd, so that it is never empty
        if (res->t_logits == nullptr && res->t_embd != nullptr) {
            ggml_build_forward_expand(gf, res->t_embd);
        }
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (hparams.pooling_type_cls == LLAMA_POOLING_TYPE_MEAN) {
                    // modern bert with classifier_pooling = "mean" builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_unary(ctx0, cur, hparams.act_cls);
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_logits() const {
    // a pooled embeddings context outputs one vector per sequence, so its graph skips the LM head
    if (cparams.embeddings && pooling_type != LLAMA_POOLING_TYPE_NONE) {
        res->t_logits = nullptr;
    }

    // the models declare t_logits without expanding it, so the LM head is computed only when it is expanded here
    if (res->t_logits != nullptr) {
        ggml_build_forward_expand(gf, res->t_logits);
    }
}

void llm_graph_context::build_layer_inp_dev() const {
    if (!cparams.embeddings_layer_inp_dev || layer_inp_dev == nullptr) {
        return;
    }

    auto inp = std::make_unique<llm_graph_input_batch_idxs>();

    inp->idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
    ggml_set_input(inp->idxs);

    // row layout: [layer_0 | layer_1 | ...] in ascending layer id order
    int64_t col = 0;
    for (size_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }

        ggml_tensor * src = res->t_layer_inp[il];
        GGML_ASSERT(src != nullptr && "layer input tensor is null");
        GGML_ASSERT(src->ne[0] == n_embd);

        ggml_tensor * dst = ggml_view_2d(ctx0, layer_inp_dev, n_embd, layer_inp_dev->ne[1], layer_inp_dev->nb[1], col*n_embd*ggml_element_size(layer_inp_dev));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, dst, ggml_reshape_2d(ctx0, src, n_embd, n_tokens), inp->idxs));
        col++;
    }
    GGML_ASSERT(col*n_embd == layer_inp_dev->ne[0]);

    res->add_input(std::move(inp));
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
        }
    }

    res->t_sampled.resize(n_rows, nullptr);
    res->t_sampled_probs.resize(n_rows, nullptr);
    res->t_sampled_logits.resize(n_rows, nullptr);
    res->t_candidates.resize(n_rows, nullptr);

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // the padding row gives the chains without a row of the ubatch a valid input, even with no output at all
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    static const std::vector<uint32_t> no_rows;

    // every sampler builds n_outputs_max_per_seq chains, like the reserve, so the graph keeps its topology
    // whatever rows the ubatch outputs: a chain without a row works on the first row and is not selected
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        const auto & rows = it != sampling_rows.end() ? it->second : no_rows;

        for (uint32_t i = 0; i < cparams.n_outputs_max_per_seq; ++i) {
            const bool     active = i < rows.size();
            const uint32_t row    = active ? rows[i] : 0;
            const int      i_out  = active ? 1       : 0;

            ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row * logits_t->nb[1]);
            ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);

            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            if (data.sampled     != nullptr) { res->t_sampling_all.push_back(data.sampled);     }
            if (data.probs       != nullptr) { res->t_sampling_all.push_back(data.probs);       }
            if (data.logits      != nullptr) { res->t_sampling_all.push_back(data.logits);      }
            if (data.candidates  != nullptr) { res->t_sampling_all.push_back(data.candidates);  }

            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[row] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[row] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[row] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[row] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
        }
    }

    // TODO: Call backend_accept after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

