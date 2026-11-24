#include "mtmd-graph.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

//
// mtmd_graph
//

mtmd_graph::mtmd_graph(mtmd_subcontext * ctx, const mtmd_image_f32 & img) :
        model(ctx->model),
        hparams(model.hparams),
        proj_type(ctx->model.proj_type),
        img(img),
        patch_size(hparams.patch_size),
        n_patches_x(img.nx() / patch_size),
        n_patches_y(img.ny() / patch_size),
        n_patches(n_patches_x * n_patches_y),
        n_embd(hparams.n_embd),
        n_head(hparams.n_head),
        n_head_kv(hparams.n_head_kv),
        d_head(hparams.n_embd_head > 0 ? hparams.n_embd_head : (n_head > 0 ? n_embd / n_head : 0)),
        n_layer(hparams.n_layer),
        n_mmproj_embd(ctx->model.n_mmproj_embd()),
        eps(hparams.eps),
        kq_scale(d_head > 0 ? 1.0f / sqrtf((float)d_head) : 0.0f),
        flash_attn_type(ctx->flash_attn_type) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ctx0_ptr.reset(ggml_init(params));
    ctx0 = ctx0_ptr.get();
    gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
}

mtmd_graph::mtmd_graph(const mtmd_graph & parent) :
        model(parent.model),
        hparams(parent.hparams),
        proj_type(parent.proj_type),
        img(parent.img),
        patch_size(parent.patch_size),
        n_patches_x(parent.n_patches_x),
        n_patches_y(parent.n_patches_y),
        n_patches(parent.n_patches),
        n_embd(parent.n_embd),
        n_head(parent.n_head),
        n_head_kv(parent.n_head_kv),
        d_head(parent.d_head),
        n_layer(parent.n_layer),
        n_mmproj_embd(parent.n_mmproj_embd),
        eps(parent.eps),
        kq_scale(parent.kq_scale),
        flash_attn_type(parent.flash_attn_type) {
    // reuse from parent
    ctx0 = parent.ctx0;
    gf   = parent.gf;
}

ggml_tensor * mtmd_graph::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    return ggml_mul_mat(ctx0, w, x);
}

void mtmd_graph::cb(ggml_tensor * cur, const char * name, int il) const {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
}

// siglip2 naflex
ggml_tensor * mtmd_graph::resize_position_embeddings(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny() / patch_size;
    const int width        = img.nx() / patch_size;
    const uint32_t mode    = interpolation_mode;
    const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);

    GGML_ASSERT(pos_embd);

    if (height == n_per_side && width == n_per_side) {
        return pos_embd;
    }

    pos_embd = ggml_reshape_3d(ctx0, pos_embd, n_embd, n_per_side, n_per_side);  // -> (n_embd, n_per_side, n_per_side)
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
    pos_embd = ggml_interpolate(ctx0, pos_embd, width, height, n_embd, 1, mode); // -> (width, height, n_embd)
    pos_embd = ggml_permute(ctx0, pos_embd, 1, 2, 0, 3);                         // -> (n_embd, width, height)
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);             // -> (n_embd, width * height)

    return pos_embd;
}

// build vision transformer (ViT) cgraph
// this function should cover most of the models
// if your model has specific features, you should probably duplicate this function
ggml_tensor * mtmd_graph::build_vit(
            ggml_tensor * inp,
            int64_t n_pos,
            norm_type norm_t,
            ffn_op_type ffn_t,
            ggml_tensor * learned_pos_embd,
            std::function<ggml_tensor *(ggml_tensor *, const mtmd_layer &)> add_pos,
            const build_vit_opts & opts
        ) {
    // batch dim: inp is [n_embd, n_pos, B]
    const int64_t B = inp->ne[2];

    if (learned_pos_embd) {
        inp = ggml_add(ctx0, inp, learned_pos_embd);
        cb(inp, "pos_embed", -1);
    }

    // flatten batch; unflatten again in attention
    inp = ggml_reshape_2d(ctx0, inp, n_embd, n_pos * B);

    ggml_tensor * inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        ggml_tensor * attn_mask = opts.attn_mask;
        if (opts.attn_mask_layers.size() > (size_t) il) {
            attn_mask = opts.attn_mask_layers[il];
        }

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "layer_inp_normed", il);

        // self-attention
        {
            ggml_tensor * Qcur = nullptr;
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            if (layer.qkv_w != nullptr) {
                // fused qkv
                cur = build_mm(layer.qkv_w, cur);
                if (layer.qkv_b != nullptr) {
                    cur = ggml_add(ctx0, cur, layer.qkv_b);
                }

                // Q/K/V as [d_head, n_head, n_pos, B], the batch stride is cur->nb[1]*n_pos.
                Qcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B,
                /* nb1    */ ggml_row_size(cur->type, d_head),
                /* nb2    */ cur->nb[1],
                /* nb3    */ cur->nb[1] * n_pos,
                /* offset */ 0);

                Kcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B,
                /* nb1    */ ggml_row_size(cur->type, d_head),
                /* nb2    */ cur->nb[1],
                /* nb3    */ cur->nb[1] * n_pos,
                /* offset */ ggml_row_size(cur->type, n_head * d_head));

                Vcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B,
                /* nb1    */ ggml_row_size(cur->type, d_head),
                /* nb2    */ cur->nb[1],
                /* nb3    */ cur->nb[1] * n_pos,
                /* offset */ ggml_row_size(cur->type, 2 * n_head * d_head));

                if (layer.q_norm) {
                    GGML_ASSERT(layer.q_norm->ne[0] == Qcur->ne[0]);
                    Qcur = build_norm(Qcur, layer.q_norm, nullptr, norm_t, eps, il);
                    cb(Qcur, "Qcur_norm", il);
                }

                if (layer.k_norm) {
                    GGML_ASSERT(layer.k_norm->ne[0] == Kcur->ne[0]);
                    Kcur = build_norm(Kcur, layer.k_norm, nullptr, norm_t, eps, il);
                    cb(Kcur, "Kcur_norm", il);
                }

            } else {
                // separate q, k, v
                Qcur = build_mm(layer.q_w, cur);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                Kcur = build_mm(layer.k_w, cur);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                Vcur = build_mm(layer.v_w, cur);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                // if true, norm must be applied after reshaping to (d_head, n_head, n_pos)
                bool norm_per_head = layer.q_norm && layer.q_norm->ne[0] == d_head;

                if (!norm_per_head) {
                    if (layer.q_norm) {
                        Qcur = build_norm(Qcur, layer.q_norm, nullptr, norm_t, eps, il);
                        cb(Qcur, "Qcur_norm", il);
                    }
                    if (layer.k_norm) {
                        Kcur = build_norm(Kcur, layer.k_norm, nullptr, norm_t, eps, il);
                        cb(Kcur, "Kcur_norm", il);
                    }
                }

                Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head,    n_pos, B);
                Kcur = ggml_reshape_4d(ctx0, Kcur, d_head, n_head_kv, n_pos, B);
                Vcur = ggml_reshape_4d(ctx0, Vcur, d_head, n_head_kv, n_pos, B);

                if (norm_per_head) {
                    if (layer.q_norm) {
                        Qcur = build_norm(Qcur, layer.q_norm, nullptr, norm_t, eps, il);
                        cb(Qcur, "Qcur_norm_per_head", il);
                    }
                    if (layer.k_norm) {
                        Kcur = build_norm(Kcur, layer.k_norm, nullptr, norm_t, eps, il);
                        cb(Kcur, "Kcur_norm_per_head", il);
                    }
                }
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (add_pos) {
                Qcur = add_pos(Qcur, layer);
                Kcur = add_pos(Kcur, layer);
                cb(Qcur, "Qcur_pos", il);
                cb(Kcur, "Kcur_pos", il);
            }

            // build_attn returns a flat 2D [n_embd, n_pos*B]
            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (layer.ls_1_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            cb(cur, "attn_out_scaled", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        if (opts.callback_layer_out) {
            opts.callback_layer_out(cur, il);
        }

        cb(cur, "ffn_inp", il);

        // layernorm2 (pre-ffn norm)
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            ffn_t, il);

        cb(cur, "ffn_out", il);

        if (layer.ls_2_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            cb(cur, "ffn_out_scaled", il);
        }

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w && !opts.skip_post_ln) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
    }

    // restore the batch dim
    GGML_ASSERT(inpL->ne[1] % B == 0);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, inpL->ne[1] / B, B);
    return inpL;
}

// build the input after conv2d (inp_raw --> patches)
// returns tensor with shape [n_embd, n_patches]
ggml_tensor * mtmd_graph::build_inp() {
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }
    return inp;
}

ggml_tensor * mtmd_graph::build_inp_raw(int channels) {
    ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img.nx(), img.ny(), channels, n_batch);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    return inp_raw;
}

ggml_tensor * mtmd_graph::build_inp_attn_mask(int64_t n_kv, int64_t n_q, int idx) {
    const ggml_type type = flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx0, type, n_kv, n_q);
    ggml_set_name(mask, get_attn_mask_name(idx).c_str());
    ggml_set_input(mask);
    return mask;
}

ggml_tensor * mtmd_graph::build_norm(
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb,
        norm_type type,
        float norm_eps,
        int il) const {

    cur = type == NORM_TYPE_RMS
        ? ggml_rms_norm(ctx0, cur, norm_eps)
        : ggml_norm(ctx0, cur, norm_eps);

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        cb(cur, "norm_w", il);
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
        cb(cur, "norm_b", il);
    }

    return cur;
}

ggml_tensor * mtmd_graph::build_ffn(
        ggml_tensor * cur,
        ggml_tensor * up,
        ggml_tensor * up_b,
        ggml_tensor * gate,
        ggml_tensor * gate_b,
        ggml_tensor * down,
        ggml_tensor * down_b,
        ffn_op_type type_op,
        int il) const {

    ggml_tensor * tmp = up ? build_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        cur = build_mm(gate, cur);
        cb(cur, "ffn_gate", il);

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    // we only support parallel ffn for now
    switch (type_op) {
        case FFN_SILU:
            if (gate) {
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case FFN_GELU:
            if (gate) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
            } break;
        case FFN_GELU_ERF:
            if (gate) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
            } break;
        case FFN_GELU_QUICK:
            if (gate) {
                cur = ggml_geglu_quick_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_quick", il);
            } else {
                cur = ggml_gelu_quick(ctx0, cur);
                cb(cur, "ffn_gelu_quick", il);
            } break;
        case FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_relu_sqr", il);
            } break;
    }

    if (down) {
        cur = build_mm(down, cur);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    return cur;
}

ggml_tensor * mtmd_graph::build_attn(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale,
        int il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // the order is fixed without the compute flag, so an unselected branch stays out of the compute set
    ggml_build_forward_order(gf, q_cur);
    ggml_build_forward_order(gf, k_cur);
    ggml_build_forward_order(gf, v_cur);

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    //cb(k, "k", il);

    ggml_tensor * cur;

    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);

        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        // mask must be f16 here, use build_inp_attn_mask()
        GGML_ASSERT(!kq_mask || kq_mask->type == GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        ggml_prec_set_acc(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    } else {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        // TODO: this most likely requires GGML_PREC_BF16, but will enable it later to avoid scope-creep
        // ggml_prec_set_acc(kq, GGML_PREC_BF16);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_mm(wo, cur);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// implementation of the 2D RoPE without adding a new op in ggml
ggml_tensor * mtmd_graph::build_rope_2d(
    ggml_context * ctx0,
    ggml_tensor * cur,
    ggml_tensor * pos_a, // first half
    ggml_tensor * pos_b, // second half
    const float freq_base,
    const bool interleave_freq
) {
    const int64_t n_dim = cur->ne[0];

    // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
    // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
    // first half of cur will use 1e-0, 1e-2 (even)
    // second half of cur will use 1e-1, 1e-3 (odd)
    // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
    //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
    // then for the second half, we use freq_scale to shift the inv_freq
    //  ^ why? replace (2i) with (2i+1) in the above equation
    const float freq_scale_odd = interleave_freq
                                ? std::pow(freq_base, (float)-2/n_dim)
                                : 1.0;

    // first half, dims [0, n_dim/2)
    cur = ggml_rope_ext(
        ctx0,
        cur,
        pos_a,      // positions
        nullptr,    // freq factors
        n_dim/2,    // n_dims
        0, 0, freq_base,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f
    );

    // second half, dims [n_dim/2, n_dim)
    cur = ggml_rope_ext(
        ctx0,
        cur,
        pos_b,      // positions
        nullptr,    // freq factors
        n_dim/2,    // n_dims
        0, 0, freq_base,
        freq_scale_odd,
        0.0f, 1.0f, 0.0f, 0.0f
    );
    cur = ggml_rope_set_offset(cur, n_dim/2);

    return cur;
}

// aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
// support dynamic resolution
ggml_tensor * mtmd_graph::build_patch_merge_permute(ggml_tensor * cur, int scale_factor) {
    GGML_ASSERT(scale_factor > 1);

    const int n_embd = cur->ne[0];
    int width  = img.nx() / patch_size;
    int height = img.ny() / patch_size;

    // pad width and height to factor
    const int64_t pad_width  = MTMD_ALIGN(width,  scale_factor) - width;
    const int64_t pad_height = MTMD_ALIGN(height, scale_factor) - height;
    cur = ggml_reshape_3d(ctx0, cur, n_embd, width, height);
    if (pad_width || pad_height) {
        cur     = ggml_pad(ctx0, cur, 0, pad_width, pad_height, 0);
        width  += pad_width;
        height += pad_height;
    }

    // unshuffle h
    cur = ggml_reshape_3d(ctx0, cur, n_embd * scale_factor, width / scale_factor, height);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    // unshuffle w
    cur = ggml_cont_3d(ctx0, cur, n_embd * scale_factor * scale_factor, height / scale_factor, width / scale_factor);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    cur = ggml_cont_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cb(cur, "pixel_shuffle", -1);

    return cur;
}

std::unique_ptr<mtmd_graph> mtmd_get_graph_builder(mtmd_subcontext * ctx, const mtmd_image_f32_batch & imgs,
                                                   const mtmd_subctx_encode_params * /*params*/) {
    const mtmd_image_f32 & img = imgs.entries[0];
    std::unique_ptr<mtmd_graph> builder;

    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_INKLING2V:
            {
                builder = std::make_unique<mtmd_graph_inkling2v>(ctx, img);
            } break;
        default:
            GGML_ABORT("missing cgraph builder");
    }

    builder->img_batch = &imgs;

    // TODO [QWEN_VIDEO]: improve this in the future
    builder->n_batch = imgs.entries.size();

    return builder;
}
