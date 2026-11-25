from __future__ import annotations

from typing import Sequence

from .constants import MODEL_ARCH, MODEL_TENSOR, MODEL_TENSORS, TENSOR_NAMES


class TensorNameMap:
    mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Token embeddings
        MODEL_TENSOR.TOKEN_EMBD: (
            "gpt_neox.embed_in",                         # gptneox
            "transformer.wte",                           # gpt2 gpt-j mpt refact qwen dbrx jais exaone
            "transformer.word_embeddings",               # falcon
            "word_embeddings",                           # bloom
            "model.embed_tokens",                        # llama-hf nemotron olmoe olmo2 rwkv6qwen2 glm4-0414 plamo2 granite-hybrid
            "embed_tokens",                              # embeddinggemma
            "tok_embeddings",                            # llama-pth
            "embeddings.word_embeddings",                # bert nomic-bert
            "embeddings.tok_embeddings",                 # modern-bert
            "wte",                                       # gpt2
            "transformer.embd.wte",                      # phi2
            "model.tok_embeddings",                      # internlm2
            "model.embedding",                           # mamba-qbert
            "backbone.embedding",                        # mamba
            "backbone.embeddings",                       # mamba-hf
            "transformer.in_out_embed",                  # Grok
            "embedding.word_embeddings",                 # chatglm
            "transformer.token_embeddings",              # openelm
            "shared",                                    # t5
            "rwkv.embeddings",                           # rwkv6
            "model.embeddings",                          # rwkv7
            "model.word_embeddings",                     # bailingmoe
            "encoder",                                   # neobert
            "model.transformer.wte",                     # llada
            "embed_tokens",                              # qwen3-embedding
            "model.embed",                               # talkie
            "embedding",                                 # metis16
        ),

        # Normalization of token embeddings
        MODEL_TENSOR.TOKEN_EMBD_NORM: (
            "word_embeddings_layernorm",  # bloom
            "embeddings.LayerNorm",       # bert
            "embeddings.norm",            # modern-bert
            "emb_ln",                     # nomic-bert
            "transformer.norm",           # openelm
            "rwkv.blocks.0.pre_ln",       # rwkv
            "rwkv.blocks.0.pre_ln",       # rwkv6
            "model.pre_ln",               # rwkv7
            "model.layers.0.pre_norm",    # rwkv7
            "backbone.norm",              # wavtokenizer
            "model.embedding_norm",       # lfm2
            "model.embed_tokens.embed_norm", # inkling2
            "model.embed_norm",           # inkling2
        ),

        # Output
        MODEL_TENSOR.OUTPUT: (
            "embed_out",                 # gptneox
            "lm_head",                   # gpt2 mpt falcon llama-hf baichuan qwen mamba dbrx jais nemotron exaone olmoe olmo2 phimoe plamo2 llama4
            "output",                    # llama-pth bloom internlm2
            "word_embeddings_for_head",  # persimmon
            "lm_head.linear",            # phi2
            "output_layer",              # chatglm
            "head",                      # rwkv
            "head.out",                  # wavtokenizer
            "model.lm_head",             # dflash
            "model.transformer.ff_out",  # llada
            "head.decoder",              # modern-bert
            "embedding_projection",      # embeddinggemma2
            "model.unembed",             # inkling2
        ),
        # Output norm
        MODEL_TENSOR.OUTPUT_NORM: (
            "gpt_neox.final_layer_norm",               # gptneox
            "transformer.ln_f",                        # gpt2 gpt-j falcon jais exaone
            "model.norm",                              # llama-hf baichuan internlm2 olmoe olmo2 phimoe plamo2
            "norm",                                    # llama-pth
            "transformer.norm_f",                      # mpt dbrx
            "ln_f",                                    # refact bloom qwen gpt2
            "model.final_layernorm",                   # persimmon
            "lm_head.ln",                              # phi2
            "model.norm_f",                            # mamba-qbert
            "backbone.norm_f",                         # mamba
            "transformer.rms_norm",                    # Grok
            "encoder.final_layernorm",                 # chatglm
            "transformer.norm",                        # openelm
            "model.norm",                              # nemotron
            "rwkv.ln_out",                             # rwkv6
            "model.ln_out",                            # rwkv7
            "backbone.final_layer_norm",               # wavtokenizer
            "model.norm",                              # llama4
            "model.transformer.ln_f",                  # llada
            "final_norm",                              # modern-bert
            "model.norm",                              # cogvlm
            "model.embedding_norm",                    # lfm3
        ),

        MODEL_TENSOR.OUTPUT_RES_QUERY: (
            "model.final_input.query",      # inkling2
            "model.attn_res.output_query",  # inkling2
        ),

        # Rope frequencies
        MODEL_TENSOR.ROPE_FREQS: (
            "rope.freqs",  # llama-pth
            "rotary_pos_emb.inv_freq",  # chatglm
        ),

        MODEL_TENSOR.V_ENC_CONV_STEM: (
            "model.vision_tower.timm_model.conv_stem.conv", # gemma3n
        ),
        MODEL_TENSOR.V_ENC_CONV_STEM_NORM: (
            "model.vision_tower.timm_model.conv_stem.bn", # gemma3n
        ),
        MODEL_TENSOR.V_ENC_MSFA_EXP: (
            "model.vision_tower.timm_model.msfa.ffn.pw_exp.conv", # gemma3n
        ),
        MODEL_TENSOR.V_ENC_MSFA_EXP_NORM: (
            "model.vision_tower.timm_model.msfa.ffn.pw_exp.bn", # gemma3n
        ),
        MODEL_TENSOR.V_ENC_MSFA_PROJ: (
            "model.vision_tower.timm_model.msfa.ffn.pw_proj.conv", # gemma3n
        ),
        MODEL_TENSOR.V_ENC_MSFA_PROJ_NORM: (
            "model.vision_tower.timm_model.msfa.ffn.pw_proj.bn", # gemma3n
        ),
        MODEL_TENSOR.V_ENC_MSFA_NORM: (
            "model.vision_tower.timm_model.msfa.norm", # gemma3n
        ),
    }

    block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Attention norm
        MODEL_TENSOR.ATTN_NORM: (
            "layers.{bid}.mixer_connection.norm",                   # metis16
            "model.lilicorr.layers.{bid}.attn_norm",                # lilicorr
            "gpt_neox.layers.{bid}.input_layernorm",                # gptneox
            "transformer.h.{bid}.ln_1",                             # gpt2 gpt-j refact qwen jais exaone
            "transformer.blocks.{bid}.norm_1",                      # mpt
            "transformer.h.{bid}.input_layernorm",                  # falcon7b
            "h.{bid}.input_layernorm",                              # bloom
            "transformer.h.{bid}.ln_mlp",                           # falcon40b
            "model.layers.{bid}.input_layernorm",                   # llama-hf nemotron olmoe phimoe granite-hybrid
            "layers.{bid}.attention_norm",                          # llama-pth
            "model.layers.{bid}.ln1",                               # yi
            "h.{bid}.ln_1",                                         # gpt2
            "transformer.h.{bid}.ln",                               # phi2
            "model.layers.layers.{bid}.norm",                       # plamo
            "model.layers.layers.{bid}.pre_mixer_norm",             # plamo2
            "model.layers.{bid}.attention_norm",                    # internlm2
            "model.layers.{bid}.norm",                              # mamba-qbert
            "backbone.layers.{bid}.norm",                           # mamba
            "transformer.decoder_layer.{bid}.rms_norm",             # Grok
            "model.layers.{bid}.pre_attn_norm",                     # grok-2
            "transformer.blocks.{bid}.norm_attn_norm.norm_1",       # dbrx
            "encoder.layers.{bid}.input_layernorm",                 # chatglm
            "transformer.layers.{bid}.attn_norm",                   # openelm
            "rwkv.blocks.{bid}.ln1",                                # rwkv6
            "model.layers.{bid}.ln1",                               # rwkv7
            "model.layers.{bid}.input_layernorm",                   # llama4
            "layers.{bid}.input_layernorm",                         # embeddinggemma
            "transformer_encoder.{bid}.attention_norm",             # neobert
            "layers.{bid}.attn_norm",                               # modern-bert
            "model.layers.{bid}.operator_norm",                     # lfm2
            "model.transformer.blocks.{bid}.attn_norm",             # llada
            "layers.{bid}.input_layernorm",                         # qwen3-embedding
            "model.layers.{bid}.attention_layernorm",               # apertus
            "model.layers.{bid}.pre_attention_layernorm",           # kormo
            "head.layers.{bid}.norm1",  # laya
            "model.layers.{bid}.attn_norm",                         # inkling2
            "model.layers.{bid}.pre_attention_norm",                # domynedge
        ),

        # Attention query
        MODEL_TENSOR.ATTN_Q: (
            "model.layers.{bid}.self_attn.q_proj",                       # llama-hf nemotron olmoe olmo2 phimoe
            "model.layers.{bid}.attention.q_proj",                       # bailingmoe3
            "layers.{bid}.self_attn.q_proj",                             # embeddinggemma
            "model.layers.{bid}.self_attn.q_proj_no_perm",               # llama-custom
            "layers.{bid}.attention.wq",                                 # llama-pth
            "encoder.layer.{bid}.attention.self.query",                  # bert
            "transformer.layer.{bid}.attention.q_lin",                   # distillbert
            "transformer.h.{bid}.attn.q_proj",                           # gpt-j
            "model.layers.layers.{bid}.self_attn.q_proj",                # plamo
            "model.layers.{bid}.attention.wq",                           # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.query",# Grok
            "transformer.h.{bid}.attn.attention.q_proj",                 # exaone
            "model.layers.{bid}.self_attn.q_proj",                       # llama4
            "model.transformer.blocks.{bid}.q_proj",                     # llada
            "layers.{bid}.self_attn.q_proj",                             # qwen3-embedding
            "backbone.layers.{bid}.mixer.q_proj",                        # nemotron-h
            "model.blocks.{bid}.attn.attn_query",                        # talkie
            "model.layers.{bid}.attn.wq_du",                             # inkling2
        ),

        # Attention key
        MODEL_TENSOR.ATTN_K: (
            "model.layers.{bid}.self_attn.k_proj",                     # llama-hf nemotron olmoe olmo2 phimoe
            "model.layers.{bid}.attention.k_proj",                     # bailingmoe3
            "layers.{bid}.self_attn.k_proj",                           # embeddinggemma
            "model.layers.{bid}.self_attn.k_proj_no_perm",             # llama-custom
            "layers.{bid}.attention.wk",                               # llama-pth
            "encoder.layer.{bid}.attention.self.key",                  # bert
            "transformer.layer.{bid}.attention.k_lin",                 # distillbert
            "transformer.h.{bid}.attn.k_proj",                         # gpt-j
            "transformer.h.{bid}.attn.k",                              # refact
            "model.layers.layers.{bid}.self_attn.k_proj",              # plamo
            "model.layers.{bid}.attention.wk",                         # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.key",# Grok
            "transformer.h.{bid}.attn.attention.k_proj",               # exaone
            "model.layers.{bid}.self_attn.k_proj",                     # llama4
            "model.transformer.blocks.{bid}.k_proj",                   # llada
            "layers.{bid}.self_attn.k_proj",                           # qwen3-embedding
            "backbone.layers.{bid}.mixer.k_proj",                      # nemotron-h
            "model.blocks.{bid}.attn.attn_key",                        # talkie
            "model.layers.{bid}.attn.wk_dv",                           # inkling2
        ),

        # Attention value
        MODEL_TENSOR.ATTN_V: (
            "model.layers.{bid}.self_attn.v_proj",                       # llama-hf nemotron olmoe olmo2 phimoe
            "model.layers.{bid}.attention.v_proj",                       # bailingmoe3
            "layers.{bid}.self_attn.v_proj",                             # embeddinggemma
            "layers.{bid}.attention.wv",                                 # llama-pth
            "encoder.layer.{bid}.attention.self.value",                  # bert
            "transformer.layer.{bid}.attention.v_lin",                   # distillbert
            "transformer.h.{bid}.attn.v_proj",                           # gpt-j
            "transformer.h.{bid}.attn.v",                                # refact
            "model.layers.layers.{bid}.self_attn.v_proj",                # plamo
            "model.layers.{bid}.attention.wv",                           # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.value",# Grok
            "transformer.h.{bid}.attn.attention.v_proj",                 # exaone
            "model.layers.{bid}.self_attn.v_proj",                       # llama4
            "model.transformer.blocks.{bid}.v_proj",                     # llada
            "layers.{bid}.self_attn.v_proj",                             # qwen3-embedding
            "backbone.layers.{bid}.mixer.v_proj",                        # nemotron-h
            "model.blocks.{bid}.attn.attn_value",                        # talkie
            "model.layers.{bid}.attn.wv_dv",                             # inkling2
        ),

        # Attention output
        MODEL_TENSOR.ATTN_OUT: (
            "model.lilicorr.layers.{bid}.attn.out_proj",                    # lilicorr
            "gpt_neox.layers.{bid}.attention.dense",                        # gptneox
            "transformer.h.{bid}.attn.c_proj",                              # gpt2 refact qwen jais
            "transformer.blocks.{bid}.attn.out_proj",                       # mpt
            "transformer.h.{bid}.self_attention.dense",                     # falcon
            "h.{bid}.self_attention.dense",                                 # bloom
            "model.layers.{bid}.self_attn.o_proj",                          # llama-hf nemotron olmoe olmo2 phimoe
            "model.layers.{bid}.attention.o_proj",                          # bailingmoe3
            "model.layers.{bid}.attention.dense",                           # bailingmoe3 MLA
            "layers.{bid}.self_attn.o_proj",                                # embeddinggemma
            "model.layers.{bid}.self_attn.out_proj",                        # lfm2 minimax-01
            "model.layers.{bid}.self_attn.linear_attn",                     # deci
            "layers.{bid}.attention.wo",                                    # llama-pth
            "encoder.layer.{bid}.attention.output.dense",                   # bert
            "layers.{bid}.attn.Wo",                                         # modern-bert
            "transformer.layer.{bid}.attention.out_lin",                    # distillbert
            "transformer.h.{bid}.attn.out_proj",                            # gpt-j
            "model.layers.{bid}.self_attn.dense",                           # persimmon
            "model.layers.{bid}.attention.dense",                           # bailingmoe2
            "h.{bid}.attn.c_proj",                                          # gpt2
            "transformer.h.{bid}.mixer.out_proj",                           # phi2
            "model.layers.layers.{bid}.self_attn.o_proj",                   # plamo
            "model.layers.layers.{bid}.mixer.o_proj",                       # plamo2
            "model.layers.{bid}.attention.wo",                              # internlm2
            "encoder.layers.{bid}.attn.out_proj",                           # nomic-bert
            "encoder.layers.{bid}.mixer.out_proj",                          # jina
            "transformer.decoder_layer.{bid}.multi_head_attention.linear",  # Grok
            "transformer.blocks.{bid}.norm_attn_norm.attn.out_proj",        # dbrx
            "encoder.layers.{bid}.self_attention.dense",                    # chatglm
            "transformer.layers.{bid}.attn.out_proj",                       # openelm
            "transformer.h.{bid}.attn.attention.out_proj",                  # exaone
            "model.layers.{bid}.self_attn.o_proj",                          # llama4
            "transformer_encoder.{bid}.wo",                                 # neobert
            "model.transformer.blocks.{bid}.attn_out",                      # llada
            "layers.{bid}.self_attn.o_proj",                                # qwen3-embedding
            "backbone.layers.{bid}.mixer.o_proj",                           # nemotron-h
            "model.layers.{bid}.self_attn.language_expert_dense",           # cogvlm
            "model.blocks.{bid}.attn.attn_resid",                           # talkie
            "head.layers.{bid}.self_attn.out_proj",  # laya
            "layers.{bid}.mixer.out",                                       # metis16
            "model.layers.{bid}.attn.wo_ud",                                # inkling2
        ),

        MODEL_TENSOR.ATTN_POST_NORM: (
            "model.layers.{bid}.post_attention_layernorm",       # gemma2 olmo2    # ge
            "layers.{bid}.post_attention_layernorm",             # embeddinggemma
            "model.layers.{bid}.post_self_attn_layernorm",       # glm-4-0414
            "model.layers.{bid}.post_attn_norm",                 # kolibri1
            "model.layers.layers.{bid}.post_mixer_norm.weight",  # plamo2
            "model.layers.{bid}.post_operator_norm",             # lfm3
            "model.layers.{bid}.post_attention_norm",            # domynedge
        ),

        MODEL_TENSOR.ATTN_SINKS: (
            "model.layers.{bid}.self_attn.softmax_offset", # flint
            "model.layers.{bid}.self_attn.sinks", # openai-moe
            "model.layers.{bid}.self_attn.attention_sink_bias", # mimov2
            "model.layers.{bid}.self_attn.learnable_sink_param", # hy-v4
        ),

        MODEL_TENSOR.ATTN_GATE: (
            "model.layers.{bid}.self_attn.gate_proj", # afmoe muse-glimmer
            "model.layers.{bid}.linear_attn.in_proj_z",  # qwen3.5
            "layers.{bid}.linear_attn.in_proj_z",        # qwen3.5 text
            "model.layers.{bid}.self_attn.g_proj",    # step3.5 head-wise attention gate
            "model.layers.{bid}.self_attn.output_gate",  # minimax-01
            "model.layers.{bid}.self_attn.linear_gate",  # hy-v4
            "model.layers.{bid}.self_attn.attn_gate_proj",  # k2-horizon
            "model.layers.{bid}.attn.gate_attn",         # inkling2
        ),

        MODEL_TENSOR.ATTN_REL: (
            "model.layers.{bid}.self_attn.r_proj",  # inkling2
            "model.layers.{bid}.attn.wr_du",        # inkling2
        ),

        MODEL_TENSOR.ATTN_REL_BIAS: (
            "model.layers.{bid}.self_attn.rel_logits_proj.proj",  # inkling2
            "model.layers.{bid}.attn.rel_logits_proj.proj",       # inkling2
        ),

        MODEL_TENSOR.ATTN_Q_CLIP: (
            "model.layers.{bid}.self_attn.q_l2_clip.scale",  # inkling2
            "model.layers.{bid}.attn.q_l2_clip.scale",       # inkling2
        ),

        MODEL_TENSOR.ATTN_REL_CLIP: (
            "model.layers.{bid}.self_attn.rel_l2_clip.scale",  # inkling2
            "model.layers.{bid}.attn.rel_l2_clip.scale",       # inkling2
        ),

        MODEL_TENSOR.ATTN_SINK_PROJ: (
            "model.layers.{bid}.self_attn.sink_proj",  # inkling2
            "model.layers.{bid}.attn.attn_sink_proj",  # inkling2
        ),

        MODEL_TENSOR.ATTN_Q_INP_NORM: (
            "model.layers.{bid}.q_input_layernorm",  # inkling2
            "model.layers.{bid}.q_input_norm",       # inkling2
        ),

        MODEL_TENSOR.ATTN_K_INP_NORM: (
            "model.layers.{bid}.k_input_layernorm",  # inkling2
            "model.layers.{bid}.k_input_norm",       # inkling2
        ),

        MODEL_TENSOR.ATTN_V_INP_NORM: (
            "model.layers.{bid}.v_input_layernorm",  # inkling2
            "model.layers.{bid}.v_input_norm",       # inkling2
        ),

        MODEL_TENSOR.ATTN_K_CONV: (
            "model.layers.{bid}.self_attn.k_sconv.conv1d",  # inkling2
            "model.layers.{bid}.attn.k_sconv",              # inkling2
        ),

        MODEL_TENSOR.ATTN_V_CONV: (
            "model.layers.{bid}.self_attn.v_sconv.conv1d",  # inkling2
            "model.layers.{bid}.attn.v_sconv",              # inkling2
        ),

        MODEL_TENSOR.ATTN_K_CONV_PROJ: (
            "model.layers.{bid}.self_attn.k_sconv.weight_proj",  # inkling2
            "model.layers.{bid}.attn.k_sconv.weight_proj",       # inkling2
        ),

        MODEL_TENSOR.ATTN_V_CONV_PROJ: (
            "model.layers.{bid}.self_attn.v_sconv.weight_proj",  # inkling2
            "model.layers.{bid}.attn.v_sconv.weight_proj",       # inkling2
        ),

        MODEL_TENSOR.ATTN_POST_CONV: (
            "model.layers.{bid}.post_attention_convolution.conv1d",  # inkling2
            "model.layers.{bid}.attn_sconv",                         # inkling2
        ),

        MODEL_TENSOR.ATTN_RES_QUERY: (
            "model.layers.{bid}.attention_input.query",  # inkling2
        ),

        MODEL_TENSOR.LOOP_Q_CTRL_DOWN: (
            "model.layers.{bid}.q_controller.down_proj",       # inkling2_loop
            "model.layers.{bid}.controllers.attn.down_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_Q_CTRL_GAIN: (
            "model.layers.{bid}.q_controller.gain_proj",       # inkling2_loop
            "model.layers.{bid}.controllers.attn.gain_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_Q_CTRL_FORGET: (
            "model.layers.{bid}.q_controller.forget_proj",       # inkling2_loop
            "model.layers.{bid}.controllers.attn.forget_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_K_CTRL_DOWN: (
            "model.layers.{bid}.k_controller.down_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.k.down_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_K_CTRL_GAIN: (
            "model.layers.{bid}.k_controller.gain_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.k.gain_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_K_CTRL_FORGET: (
            "model.layers.{bid}.k_controller.forget_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.k.forget_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_V_CTRL_DOWN: (
            "model.layers.{bid}.v_controller.down_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.v.down_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_V_CTRL_GAIN: (
            "model.layers.{bid}.v_controller.gain_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.v.gain_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_V_CTRL_FORGET: (
            "model.layers.{bid}.v_controller.forget_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.v.forget_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_FFN_CTRL_DOWN: (
            "model.layers.{bid}.mlp_controller.down_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.mlp.down_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_FFN_CTRL_GAIN: (
            "model.layers.{bid}.mlp_controller.gain_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.mlp.gain_weight",  # inkling2_loop
        ),

        MODEL_TENSOR.LOOP_FFN_CTRL_FORGET: (
            "model.layers.{bid}.mlp_controller.forget_proj",    # inkling2_loop
            "model.layers.{bid}.controllers.mlp.forget_weight",  # inkling2_loop
        ),

        # Feed-forward norm
        MODEL_TENSOR.FFN_NORM: (
            "layers.{bid}.moe_connection.norm",                              # metis16
            "model.lilicorr.layers.{bid}.mlp_norm",                          # lilicorr
            "model.lilicorr.feature_mlp.{bid}.norm",                         # lilicorr
            "gpt_neox.layers.{bid}.post_attention_layernorm",                # gptneox
            "transformer.h.{bid}.ln_2",                                      # gpt2 refact qwen jais exaone
            "h.{bid}.post_attention_layernorm",                              # bloom
            "transformer.blocks.{bid}.norm_2",                               # mpt
            "model.layers.{bid}.post_attention_layernorm",                   # llama-hf nemotron olmoe phimoe
            "layers.{bid}.ffn_norm",                                         # llama-pth
            "model.layers.{bid}.ln2",                                        # yi
            "h.{bid}.ln_2",                                                  # gpt2
            "model.layers.{bid}.ffn_norm",                                   # internlm2
            "transformer.decoder_layer.{bid}.rms_norm_2",                    # Grok
            "model.layers.{bid}.pre_moe_norm",                               # grok-2
            "encoder.layers.{bid}.post_attention_layernorm",                 # chatglm
            "transformer.layers.{bid}.ffn_norm",                             # openelm
            "model.layers.{bid}.pre_ff_layernorm",                           # jamba granite-hybrid
            "model.layers.{bid}.pre_moe_layernorm",                          # mini-jamba
            "model.layers.{bid}.post_attention_layernorm",                   # llama4
            "transformer_encoder.{bid}.ffn_norm",                            # neobert
            "model.layers.layers.{bid}.pre_mlp_norm",                        # plamo2
            "model.transformer.blocks.{bid}.ff_norm",                        # llada
            "layers.{bid}.post_attention_layernorm",                         # qwen3-embedding
            "model.layers.{bid}.feedforward_layernorm",                      # apertus
            "model.layers.{bid}.pre_mlp_layernorm",                          # kormo
            "layers.{bid}.mlp_norm",                                         # modern-bert
            "head.layers.{bid}.norm2",  # laya
            "model.layers.{bid}.mlp_norm",                                   # inkling2
        ),

        MODEL_TENSOR.FFN_POST_NORM: (
            "model.layers.{bid}.post_feedforward_layernorm",  # gemma2 olmo2
            "layers.{bid}.post_feedforward_layernorm",        # embeddinggemma
            "model.layers.{bid}.post_mlp_layernorm",          # glm-4-0414
            "model.layers.{bid}.post_ffn_norm",               # kolibri1
            "model.layers.layers.{bid}.post_mlp_norm.weight", # plamo2
            "model.layers.{bid}.feed_forward.up_proj",
            "model.layers.{bid}.post_moe_norm",               # grok-2
            "model.layers.{bid}.post_ffn_norm",               # lfm3
            "model.layers.{bid}.post_feedforward_norm",       # domynedge
        ),

        MODEL_TENSOR.FFN_GATE_INP: (
            "layers.{bid}.feed_forward.gate",                   # mixtral
            "model.layers.{bid}.block_sparse_moe.gate",         # mixtral phimoe
            "model.layers.{bid}.mlp.gate",                      # qwen2moe olmoe
            "transformer.decoder_layer.{bid}.router",           # Grok
            "transformer.blocks.{bid}.ffn.router.layer",        # dbrx
            "model.layers.{bid}.block_sparse_moe.router.layer", # granitemoe
            "model.layers.{bid}.block_sparse_moe.router",       # granite_swa
            "model.layers.{bid}.feed_forward.router",           # llama4 jamba
            "encoder.layers.{bid}.mlp.router.layer",            # nomic-bert-moe
            "model.layers.{bid}.mlp.router",                    # openai-moe
            "model.layers.{bid}.mlp.gate.wg",                   # hunyuan
            "model.layers.{bid}.block_sparse_moe.primary_router", # smallthinker
            "model.layers.{bid}.feed_forward.gate",               # lfm2moe
            "model.layers.{bid}.mlp.router.gate",               # afmoe
            "layers.{bid}.gate",                                # mistral-large
            "backbone.layers.{bid}.mixer.gate",                 # nemotron-h-moe
            "model.layers.{bid}.moe.gate",                      # step3.5
            "model.layers.{bid}.router.proj",                   # gemma4
            "layers.{bid}.moe.expert_router",                   # metis16
        ),

        MODEL_TENSOR.FFN_EXP_PROBS_B: (
            "layers.{bid}.moe.selection_bias",                              # metis16
            "model.layers.{bid}.mlp.gate.e_score_correction",               # deepseek-v3 dots1
            "model.layers.{bid}.mlp.gate.e_score_correction_bias",          # muse-spark
            "model.layers.{bid}.mlp.moe_statics.e_score_correction",        # ernie4.5-moe
            "model.layers.{bid}.mlp.gate.expert_bias",                      # bailingmoe2
            "model.layers.{bid}.mlp.expert_bias",                           # afmoe
            "model.layers.{bid}.moe.router.expert_bias",                    # kolibri1
            "model.layers.{bid}.feed_forward.expert_bias",                  # lfm2moe
            "model.layers.{bid}.block_sparse_moe.e_score_correction",       # minimax-m2
            "backbone.layers.{bid}.mixer.gate.e_score_correction",          # nemotron-h-moe
            "model.layers.{bid}.mlp.e_score_correction",                    # exaone-moe
            "model.layers.{bid}.block_sparse_moe.gate.e_score_correction",  # kimi
            "model.layers.{bid}.moe.router_bias",                           # step3.5 expert selection bias
            "model.layers.{bid}.mlp.experts.e_score_correction",       # laguna
        ),

        # Feed-forward up
        MODEL_TENSOR.FFN_UP: (
            "model.lilicorr.layers.{bid}.mlp.up_proj",                # lilicorr
            "model.lilicorr.feature_mlp.{bid}.up_proj",               # lilicorr
            "gpt_neox.layers.{bid}.mlp.dense_h_to_4h",                # gptneox
            "transformer.h.{bid}.mlp.c_fc",                           # gpt2 jais
            "transformer.blocks.{bid}.ffn.up_proj",                   # mpt
            "transformer.h.{bid}.mlp.dense_h_to_4h",                  # falcon
            "h.{bid}.mlp.dense_h_to_4h",                              # bloom
            "model.layers.{bid}.mlp.up_proj",                         # llama-hf refact nemotron olmo2
            "layers.{bid}.mlp.up_proj",                               # embeddinggemma
            "layers.{bid}.feed_forward.w3",                           # llama-pth
            "encoder.layer.{bid}.intermediate.dense",                 # bert
            "layers.{bid}.mlp.Wi",                                    # modern-bert
            "transformer.layer.{bid}.ffn.lin1",                       # distillbert
            "transformer.h.{bid}.mlp.fc_in",                          # gpt-j
            "transformer.h.{bid}.mlp.linear_3",                       # refact
            "model.layers.{bid}.mlp.dense_h_to_4h",                   # persimmon
            "transformer.h.{bid}.mlp.w1",                             # qwen
            "h.{bid}.mlp.c_fc",                                       # gpt2
            "transformer.h.{bid}.mlp.fc1",                            # phi2
            "model.layers.{bid}.mlp.fc1",                             # phi2
            "model.layers.{bid}.mlp.gate_up_proj",                    # phi3 glm-4-0414
            "model.layers.layers.{bid}.mlp.up_proj",                  # plamo
            "model.layers.layers.{bid}.mlp.gate_up_proj",             # plamo2
            "model.layers.{bid}.feed_forward.w3",                     # internlm2
            "encoder.layers.{bid}.mlp.fc11",                          # nomic-bert
            "encoder.layers.{bid}.mlp.fc1",                           # nomic-bert-moe
            "model.layers.{bid}.mlp.c_fc",                            # starcoder2
            "encoder.layer.{bid}.mlp.gated_layers_v",                 # jina-bert-v2 (split up/gate, no longer used)
            "encoder.layer.{bid}.mlp.gated_layers",                   # jina-bert-v2 (GEGLU)
            "encoder.layer.{bid}.mlp.up_gated_layer",                 # jina-v2-code (GEGLU)
            "model.layers.{bid}.residual_mlp.w3",                     # arctic
            "encoder.layers.{bid}.mlp.dense_h_to_4h",                 # chatglm
            "transformer.h.{bid}.mlp.c_fc_1",                         # exaone
            "model.layers.{bid}.feed_forward.up_proj",                # llama4 jamba granite-hybrid
            "model.layers.{bid}.feedforward.up_proj",                 # domynedge
            "transformer_encoder.{bid}.ffn.w12",                      # neobert
            "model.layers.{bid}.block_sparse_moe.up",                 # smallthinker
            "model.transformer.blocks.{bid}.up_proj",                 # llada
            "layers.{bid}.mlp.up_proj",                               # qwen3-embedding
            "backbone.layers.{bid}.mixer.up_proj",                    # nemotron-h
            "model.layers.{bid}.mlp.language_mlp.up_proj",            # cogvlm
            "model.blocks.{bid}.mlp.mlp_linear",                      # talkie
            "head.layers.{bid}.linear1",  # laya
        ),

        MODEL_TENSOR.FFN_UP_EXP: (
            "layers.{bid}.feed_forward.experts.w3",                 # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear_v",         # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.v1",          # dbrx
            "model.layers.{bid}.mlp.experts.up_proj",               # qwen2moe olmoe (merged) ernie4.5-moe, nemotron-h-moe (merged)
            "model.layers.{bid}.block_sparse_moe.experts.w3",       # phimoe (merged)
            "model.layers.{bid}.feed_forward.experts.up_proj",      # llama4
            "encoder.layers.{bid}.mlp.experts.mlp.w1",              # nomic-bert-moe
            "model.layers.{bid}.block_sparse_moe.experts.up", # smallthinker
            "model.layers.{bid}.moe.up_proj",                       # step3.5
        ),

        MODEL_TENSOR.FFN_UP_SHEXP: (
            "model.layers.{bid}.mlp.shared_experts.up_proj",         # deepseek deepseek2
        ),

        # Feed-forward gate
        MODEL_TENSOR.FFN_GATE: (
            "model.layers.{bid}.mlp.gate_proj",               # llama-hf refact olmo2
            "layers.{bid}.mlp.gate_proj",                     # embeddinggemma
            "layers.{bid}.feed_forward.w1",                   # llama-pth
            "transformer.h.{bid}.mlp.w2",                     # qwen
            "transformer.h.{bid}.mlp.c_fc2",                  # jais
            "model.layers.layers.{bid}.mlp.gate_proj",        # plamo
            "model.layers.{bid}.feed_forward.w1",             # internlm2
            "encoder.layers.{bid}.mlp.fc12",                  # nomic-bert
            "encoder.layer.{bid}.mlp.gated_layers_w",         # jina-bert-v2 (split up/gate, no longer used)
            "transformer.h.{bid}.mlp.linear_1",               # refact
            "model.layers.{bid}.residual_mlp.w1",             # arctic
            "transformer.h.{bid}.mlp.c_fc_0",                 # exaone
            "model.layers.{bid}.feed_forward.gate_proj",      # llama4 jamba granite-hybrid
            "model.transformer.blocks.{bid}.ff_proj",         # llada
            "layers.{bid}.mlp.gate_proj",                     # qwen3-embedding
            "model.layers.{bid}.mlp.language_mlp.gate_proj",  # cogvlm
            "model.blocks.{bid}.mlp.mlp_gate",                # talkie
            "model.layers.{bid}.feedforward.gate_proj",       # domynedge
        ),

        MODEL_TENSOR.FFN_GATE_EXP: (
            "layers.{bid}.feed_forward.experts.w1",                     # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear",               # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.w1",              # dbrx
            "model.layers.{bid}.mlp.experts.gate_proj",                 # qwen2moe olmoe (merged) ernie4.5-moe
            "model.layers.{bid}.block_sparse_moe.experts.w1",           # phimoe (merged)
            "model.layers.{bid}.feed_forward.experts.gate_proj",        # llama4
            "model.layers.{bid}.block_sparse_moe.experts.gate",         # smallthinker
            "model.layers.{bid}.moe.gate_proj",                         # step3.5
        ),

        MODEL_TENSOR.FFN_GATE_SHEXP: (
            "model.layers.{bid}.mlp.shared_experts.gate_proj",         # deepseek deepseek2
        ),

        MODEL_TENSOR.FFN_GATE_UP_EXP: (
            "model.layers.{bid}.mlp.experts.gate_up_proj",
            "model.layers.{bid}.experts.gate_up_proj", # gemma4
            "layers.{bid}.moe.local_experts.gate_up",  # metis16
        ),

        MODEL_TENSOR.FFN_POST_CONV: (
            "model.layers.{bid}.post_mlp_convolution.conv1d",  # inkling2
            "model.layers.{bid}.mlp_sconv",                    # inkling2
        ),

        MODEL_TENSOR.FFN_RES_QUERY: (
            "model.layers.{bid}.mlp_input.query",  # inkling2
            "model.attn_res.mlp_queries.{bid}",    # inkling2
        ),

        # Feed-forward down
        MODEL_TENSOR.FFN_DOWN: (
            "model.lilicorr.layers.{bid}.mlp.down_proj",              # lilicorr
            "model.lilicorr.feature_mlp.{bid}.down_proj",             # lilicorr
            "gpt_neox.layers.{bid}.mlp.dense_4h_to_h",                # gptneox
            "transformer.h.{bid}.mlp.c_proj",                         # gpt2 refact qwen jais
            "transformer.blocks.{bid}.ffn.down_proj",                 # mpt
            "transformer.h.{bid}.mlp.dense_4h_to_h",                  # falcon
            "h.{bid}.mlp.dense_4h_to_h",                              # bloom
            "model.layers.{bid}.mlp.down_proj",                       # llama-hf nemotron olmo2
            "layers.{bid}.mlp.down_proj",                             # embeddinggemma
            "layers.{bid}.feed_forward.w2",                           # llama-pth
            "encoder.layer.{bid}.output.dense",                       # bert
            "layers.{bid}.mlp.Wo",                                    # modern-bert
            "transformer.layer.{bid}.ffn.lin2",                       # distillbert
            "transformer.h.{bid}.mlp.fc_out",                         # gpt-j
            "model.layers.{bid}.mlp.dense_4h_to_h",                   # persimmon
            "h.{bid}.mlp.c_proj",                                     # gpt2
            "transformer.h.{bid}.mlp.fc2",                            # phi2
            "model.layers.{bid}.mlp.fc2",                             # phi2
            "model.layers.layers.{bid}.mlp.down_proj",                # plamo
            "model.layers.{bid}.feed_forward.w2",                     # internlm2
            "encoder.layers.{bid}.mlp.fc2",                           # nomic-bert
            "model.layers.{bid}.mlp.c_proj",                          # starcoder2
            "encoder.layer.{bid}.mlp.wo",                             # jina-bert-v2
            "transformer.layers.{bid}.ffn.proj_2",                    # openelm
            "model.layers.{bid}.residual_mlp.w2",                     # arctic
            "encoder.layer.{bid}.mlp.down_layer",                     # jina-bert-v2
            "encoder.layers.{bid}.mlp.dense_4h_to_h",                 # chatglm
            "model.layers.h.{bid}.mlp.c_proj",                        # exaone
            "model.layers.{bid}.feed_forward.down_proj",              # llama4 jamba granite-hybrid
            "model.layers.{bid}.feedforward.down_proj",               # domynedge
            "transformer_encoder.{bid}.ffn.w3",                       # neobert
            "model.layers.{bid}.block_sparse_moe.down",               # smallthinker
            "model.transformer.blocks.{bid}.ff_out",                  # llada
            "layers.{bid}.mlp.down_proj",                             # qwen3-embedding
            "backbone.layers.{bid}.mixer.down_proj",                  # nemotron-h
            "model.layers.{bid}.mlp.language_mlp.down_proj",          # cogvlm
            "model.blocks.{bid}.mlp.mlp_resid",                       # talkie
            "head.layers.{bid}.linear2",  # laya
        ),

        MODEL_TENSOR.FFN_DOWN_EXP: (
            "layers.{bid}.feed_forward.experts.w2",                 # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear_1",         # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.w2",          # dbrx
            "model.layers.{bid}.mlp.experts.down_proj",             # qwen2moe olmoe (merged) ernie4.5-moe nemotron-h-moe (merged)
            "model.layers.{bid}.block_sparse_moe.output_linear",    # granitemoe
            "model.layers.{bid}.block_sparse_moe.experts.w2",       # phimoe (merged)
            "model.layers.{bid}.feed_forward.experts.down_proj",    # llama4
            "encoder.layers.{bid}.mlp.experts.mlp.w2",              # nomic-bert-moe
            "model.layers.{bid}.block_sparse_moe.experts.down",     # smallthinker
            "model.layers.{bid}.moe.down_proj",                     # step3.5
            "model.layers.{bid}.experts.down_proj",                 # gemma4
            "layers.{bid}.moe.local_experts.down",                  # metis16
        ),

        MODEL_TENSOR.FFN_DOWN_SHEXP: (
            "model.layers.{bid}.mlp.shared_experts.down_proj",         # deepseek deepseek2
        ),

        MODEL_TENSOR.ATTN_Q_NORM: (
            "encoder.layers.{bid}.self_attention.q_layernorm",
            "model.layers.{bid}.self_attn.q_layernorm",                       # persimmon
            "model.layers.{bid}.self_attn.query_layernorm",                   # hunyuan
            "model.layers.{bid}.attention.query_layernorm",                   # bailingmoe2
            "model.layers.{bid}.self_attn.q_norm",                            # cohere olmoe chameleon olmo2
            "layers.{bid}.self_attn.q_norm",                                  # embeddinggemma
            "transformer.blocks.{bid}.attn.q_ln",                             # sea-lion
            "encoder.layer.{bid}.attention.self.layer_norm_q",                # jina-bert-v2
            "transformer.layers.{bid}.attn.q_norm",                           # openelm
            "model.layers.layers.{bid}.mixer.q",                              # plamo2
            "model.layers.layers.{bid}.mixer.q_norm",                         # plamo3
            "layers.{bid}.self_attn.q_norm",                                  # qwen3-embedding
            "model.layers.{bid}.attention.query_layernorm",                   # apertus
            "model.blocks.{bid}.attn.head_gain.head_g",                       # talkie
        ),

        MODEL_TENSOR.ATTN_K_NORM: (
            "encoder.layers.{bid}.self_attention.k_layernorm",
            "model.layers.{bid}.self_attn.k_layernorm",                       # persimmon
            "model.layers.{bid}.self_attn.key_layernorm",                     # hunyuan
            "model.layers.{bid}.attention.key_layernorm",                     # bailingmoe2
            "model.layers.{bid}.self_attn.k_norm",                            # cohere olmoe chameleon olmo2
            "layers.{bid}.self_attn.k_norm",                                  # embeddinggemma
            "transformer.blocks.{bid}.attn.k_ln",                             # sea-lion
            "encoder.layer.{bid}.attention.self.layer_norm_k",                # jina-bert-v2
            "transformer.layers.{bid}.attn.k_norm",                           # openelm
            "model.layers.layers.{bid}.mixer.k",                              # plamo2
            "model.layers.layers.{bid}.mixer.k_norm",                         # plamo3
            "layers.{bid}.self_attn.k_norm",                                  # qwen3-embedding
            "model.layers.{bid}.self_attn.k_rope_only_layernorm",             # dots3note
            "model.layers.{bid}.attention.key_layernorm",                     # apertus
            "model.layers.{bid}.attention.k_norm",                            # domynedge
        ),

        MODEL_TENSOR.ROPE_FREQS: (
            "encoder.layers.{bid}.self_attention.rotary_emb.inv_freq",  # persimmon
        ),

        MODEL_TENSOR.LAYER_OUT_NORM: (
            "model.layers.{bid}.final_layernorm",           # bailingmoe2
        ),

        MODEL_TENSOR.LAYER_OUT_SCALE: (
            "model.layers.{bid}.layer_scalar", # gemma4
            "layers.{bid}.layer_scalar", # embeddinggemma2
            "model.blocks.{bid}.embed_skip.a_g", # talkie
        ),

        MODEL_TENSOR.SSM_DT: (
            "model.layers.{bid}.attention.dt_proj",     # bailingmoe3
        ),

        MODEL_TENSOR.SSM_A: (
            "model.layers.{bid}.attention.A_log",     # bailingmoe3
        ),

        MODEL_TENSOR.SSM_NORM: (
            "model.layers.{bid}.attention.o_norm",  # bailingmoe3
        ),

        # Kimi Linear KDA (using SSM_ prefix for consistency)
        MODEL_TENSOR.SSM_CONV1D_Q: (
            "model.layers.{bid}.attention.q_conv1d",
        ),
        MODEL_TENSOR.SSM_CONV1D_K: (
            "model.layers.{bid}.attention.k_conv1d",
        ),
        MODEL_TENSOR.SSM_CONV1D_V: (
            "model.layers.{bid}.attention.v_conv1d",
        ),
        # bailingmoe3 f_proj and g_proj are renamed in the converter
        MODEL_TENSOR.SSM_F_A: (),
        MODEL_TENSOR.SSM_BETA: (
            "model.layers.{bid}.attention.b_proj",       # bailingmoe3
        ),
        MODEL_TENSOR.SSM_G_A: (),

        MODEL_TENSOR.ATTN_Q_A: (
            "model.layers.{bid}.attention.q_a_proj",  # bailingmoe3 (Ling-3.0-tiny)
        ),

        MODEL_TENSOR.ATTN_Q_B: (
            "model.layers.{bid}.attention.q_b_proj",  # bailingmoe3 (Ling-3.0-tiny)
        ),

        MODEL_TENSOR.ATTN_KV_A_MQA: (
            "model.layers.{bid}.attention.kv_a_proj_with_mqa", # bailingmoe3
        ),

        MODEL_TENSOR.ATTN_K_B: (
            "model.layers.{bid}.attention.k_b_proj",  # bailingmoe3
        ),

        MODEL_TENSOR.ATTN_V_B: (
            "model.layers.{bid}.attention.v_b_proj",  # bailingmoe3
        ),

        MODEL_TENSOR.ATTN_Q_A_NORM: (
            "model.layers.{bid}.attention.q_a_layernorm", # bailingmoe3 (Ling-3.0-tiny)
        ),

        MODEL_TENSOR.ATTN_KV_A_NORM: (
            "model.layers.{bid}.attention.kv_a_layernorm", # bailingmoe3
        ),

        # NextN/MTP tensors
        MODEL_TENSOR.NEXTN_EH_PROJ: (
            "model.layers.{bid}.eh_proj",
        ),

        MODEL_TENSOR.NEXTN_ENORM: (
            "model.layers.{bid}.enorm",
        ),

        MODEL_TENSOR.NEXTN_HNORM: (
            "model.layers.{bid}.hnorm",
        ),

        ############################################################################
        # TODO: these do not belong to block_mappings_cfg - move them to mappings_cfg
        MODEL_TENSOR.ENC_OUTPUT_NORM: (
            "encoder.final_layer_norm", # t5
            "layer_norm",               # neobert
            "model.hidden_norm",        # dflash
            "encoder.output_norm_enc",  # dflash (transformers MuseGlimmerAssistant)
        ),

        MODEL_TENSOR.FC: (
            "model.fc",   # dflash
            "encoder.fc", # dflash (transformers MuseGlimmerAssistant)
        ),

        MODEL_TENSOR.DSPARK_MARKOV_W1: (
            "model.markov_head.markov_w1", # dspark
        ),

        MODEL_TENSOR.DSPARK_MARKOV_W2: (
            "model.markov_head.markov_w2", # dspark
        ),

        MODEL_TENSOR.DSPARK_CONF_PROJ: (
            "model.confidence_head.proj", # dspark
        ),

        MODEL_TENSOR.V_MMPROJ: (
            "aligner.w{bid}", # deepseek4v (w1 -> mm.1, w2 -> mm.2)
            "multi_modal_projector.linear_{bid}",
            "model.multi_modal_projector.linear_{bid}", # cohere2v
            "mm_projector.proj.linear_{bid}", # Kimi-K2.5
            "visual.merger.mlp.{bid}", # qwen2vl
            "mlp_AR.linear_{bid}", # PaddleOCR-VL
            "merger.mlp.{bid}",
            "vision_tower.merger.mlp.{bid}", # dots.ocr
            "vision_encoder.adapter.mlp.{bid}", # dots3note
            "vit.perceive.proj.{bid}", # HunyuanVL (proj.0 = conv1, proj.2 = conv2)
        ),

        MODEL_TENSOR.V_ENC_EMBD_CLS: (
            "vision_tower.vision_model.embeddings.class_embedding",
            "model.vision_tower.embeddings.cls_token", # Intern-S1
            "vision_model.class_embedding", # llama 4
            "model.vision.patch_embedding.cls_embedding", # cogvlm
            "vision_model.radio_model.model.patch_generator.cls_token.token", # Nemotron Nano v2 VL
            "vision_model.embeddings.cls_register_token", # Nemotron Omni
            "model.vision_model.embeddings.class_embedding", # Deepseek-OCR
            "model.vision_tower.embeddings.class_embedding", # inkling2
            "model.visual.backbone.embeddings.class_embedding", # inkling2
        ),

        MODEL_TENSOR.V_ENC_EMBD_PATCH: (
            "visual.embeddings.patch_embedding", # mage-vl
            "vision.patch_embed.proj", # deepseek4v
            "model.vision_tower.vision_model.embeddings.patch_embedding", # Granite4Vision
            "vision_tower.vision_model.embeddings.patch_embedding",
            "model.vision_tower.embeddings.patch_embedding", # minicpmv4_6
            "model.vision_tower.embeddings.patch_embeddings.projection", # Intern-S1
            "vpm.embeddings.patch_embedding",
            "model.vision_model.embeddings.patch_embedding", # SmolVLM
            "vit.embeddings.patch_embedding", # HunyuanVL
            "vision_tower.patch_conv", # pixtral-hf
            "vision_encoder.patch_conv", # pixtral
            "vision_model.patch_embedding.linear", # llama 4
            "visual.patch_embed.proj", # qwen2vl
            "vision_tower.patch_embed.proj", # kimi-vl
            "model.vision.patch_embedding.proj", # cogvlm
            "model.vision_model.embeddings.patch_embedding", # Deepseek-OCR CLIP
            "siglip2.vision_model.embeddings.patch_embedding",
            "vision_model.radio_model.model.patch_generator.embedder", # Nemotron Nano v2 VL
            "vision_model.embeddings.patch_projection", # Nemotron Omni
            "model.vision_tower.patch_embedder.input_proj", # gemma4
            "vision_tower.patch_embed.patchifier.proj", # dots.ocr
            "vision_encoder.patch_embed.proj", # dots3note
            "vision_model.conv1", # Step3-VL
            "model.vision_embedder.patch_dense", # gemma4 unified
            "model.vision_tower.patch_embedder.patch_embedding", # muse-glimmer
            "model.visual.backbone.embeddings.patch_embedding", # inkling2
        ),

        MODEL_TENSOR.V_ENC_EMBD_CLS_POS: (
            "model.vision_tower.embeddings.cls_position_embedding", # inkling2
            "model.visual.backbone.embeddings.cls_position_embedding", # inkling2
        ),

        MODEL_TENSOR.V_ENC_EMBD_POS: (
            "model.vision_tower.vision_model.embeddings.position_embedding", # Granite4Vision
            "vision_tower.vision_model.embeddings.position_embedding",
            "model.vision_tower.embeddings.position_embedding", # minicpmv4_6
            "model.vision_tower.embeddings.position_embeddings", # Intern-S1
            "vpm.embeddings.position_embedding",
            "model.vision_model.embeddings.position_embedding", # SmolVLM
            "vit.embeddings.position_embedding", # HunyuanVL
            "vision_model.positional_embedding_vlm", # llama 4
            "vision_tower.patch_embed.pos_emb", # kimi-vl
            "visual.pos_embed", # qwen3vl
            "model.vision.patch_embedding.position_embedding", # cogvlm
            "visual.embeddings.position_embedding", # glm4v
            "vision_model.radio_model.model.patch_generator.pos_embed", # Nemotron Nano v2 VL
            "vision_model.embeddings.position_embedding", # Nemotron Omni
            "model.vision_tower.patch_embedder.position_embedding_table", # gemma4
            "vision_model.positional_embedding", # Step3-VL
            "model.vision_embedder.pos_embedding", # gemma4 unified
            "model.visual.backbone.embeddings.position_embedding", # inkling2
        ),

        MODEL_TENSOR.V_ENC_ATTN_Q: (
            "vision_model.encoder.layer.{bid}.attention.attention.query", # Nemotron Omni
            "model.vision_tower.vision_model.encoder.layers.{bid}.self_attn.q_proj", # Granite4Vision
            "vision_tower.vision_model.encoder.layers.{bid}.self_attn.q_proj",
            "model.vision_tower.encoder.layers.{bid}.self_attn.q_proj", # minicpmv4_6
            "model.vision_tower.encoder.layer.{bid}.attention.q_proj", # Intern-S1
            "vpm.encoder.layers.{bid}.self_attn.q_proj",
            "model.vision_model.encoder.layers.{bid}.self_attn.q_proj", # SmolVLM
            "vit.layers.{bid}.self_attn.q_proj", # HunyuanVL
            "vision_model.model.layers.{bid}.self_attn.q_proj", # llama4
            "vision_tower.transformer.layers.{bid}.attention.q_proj", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.attention.wq", # pixtral
            "model.vision_tower.layers.{bid}.self_attn.q_proj", # lfm3vl
            "visual.blocks.{bid}.attn.q", # qwen2vl, generated
            "vision_tower.encoder.blocks.{bid}.wq", # kimi-vl, generated
            "siglip2.vision_model.encoder.layers.{bid}.self_attn.q_proj", # youtuvl
            "model.vision_model.transformer.layers.{bid}.self_attn.q_proj", # Deepseek-OCR CLIP, generated
            "vision_model.model.layers.{bid}.self_attn.q_proj.linear", # gemma4
            "model.qwen2_model.model.model.layers.{bid}.self_attn.q_proj", # Deepseek-OCR-2 qwen2
            "model.vision_tower.layers.{bid}.attn.q_proj", # muse-glimmer
            "model.visual.backbone.layers.{bid}.self_attn.q_proj", # inkling2
        ),

        MODEL_TENSOR.V_ENC_ATTN_Q_NORM: (
            "vision_encoder.blocks.{bid}.attn.q_norm", # dots3note
            "vision_tower.vision_model.encoder.layers.{bid}.attn.q_norm", # InternVL
            "model.vision_tower.encoder.layer.{bid}.attention.q_norm", # Intern-S1
            "visual.blocks.{bid}.attn.q_norm", # GLM-OCR
            "vision_model.model.layers.{bid}.self_attn.q_norm", # gemma4
            "model.vision_tower.layers.{bid}.self_attn.q_layernorm", # lfm3vl
        ),

        MODEL_TENSOR.V_ENC_ATTN_K: (
            "vision_model.encoder.layer.{bid}.attention.attention.key", # Nemotron Omni
            "model.vision_tower.vision_model.encoder.layers.{bid}.self_attn.k_proj", # Granite4Vision
            "vision_tower.vision_model.encoder.layers.{bid}.self_attn.k_proj",
            "model.vision_tower.encoder.layers.{bid}.self_attn.k_proj", # minicpmv4_6
            "model.vision_tower.encoder.layer.{bid}.attention.k_proj", # Intern-S1
            "vpm.encoder.layers.{bid}.self_attn.k_proj",
            "model.vision_model.encoder.layers.{bid}.self_attn.k_proj", # SmolVLM
            "vit.layers.{bid}.self_attn.k_proj", # HunyuanVL
            "vision_model.model.layers.{bid}.self_attn.k_proj", # llama4
            "vision_tower.transformer.layers.{bid}.attention.k_proj", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.attention.wk", # pixtral
            "model.vision_tower.layers.{bid}.self_attn.k_proj", # lfm3vl
            "visual.blocks.{bid}.attn.k", # qwen2vl, generated
            "vision_tower.encoder.blocks.{bid}.wk", # kimi-vl, generated
            "model.vision_model.transformer.layers.{bid}.self_attn.k_proj", # Deepseek-OCR CLIP, generated
            "siglip2.vision_model.encoder.layers.{bid}.self_attn.k_proj",
            "vision_model.model.layers.{bid}.self_attn.k_proj.linear", # gemma4
            "model.qwen2_model.model.model.layers.{bid}.self_attn.k_proj", # Deepseek-OCR-2 qwen2
            "model.vision_tower.layers.{bid}.attn.k_proj", # muse-glimmer
            "model.visual.backbone.layers.{bid}.self_attn.k_proj", # inkling2
        ),

        MODEL_TENSOR.V_ENC_ATTN_K_NORM: (
            "vision_encoder.blocks.{bid}.attn.k_norm", # dots3note
            "vision_tower.vision_model.encoder.layers.{bid}.attn.k_norm", # InternVL
            "model.vision_tower.encoder.layer.{bid}.attention.k_norm", # Intern-S1
            "visual.blocks.{bid}.attn.k_norm", # GLM-OCR
            "vision_model.model.layers.{bid}.self_attn.k_norm", # gemma4
            "model.vision_tower.layers.{bid}.self_attn.k_layernorm", # lfm3vl
        ),

        MODEL_TENSOR.V_ENC_ATTN_V: (
            "vision_model.encoder.layer.{bid}.attention.attention.value", # Nemotron Omni
            "model.vision_tower.vision_model.encoder.layers.{bid}.self_attn.v_proj", # Granite4Vision
            "vision_tower.vision_model.encoder.layers.{bid}.self_attn.v_proj",
            "model.vision_tower.encoder.layers.{bid}.self_attn.v_proj", # minicpmv4_6
            "model.vision_tower.encoder.layer.{bid}.attention.v_proj", # Intern-S1
            "vpm.encoder.layers.{bid}.self_attn.v_proj",
            "model.vision_model.encoder.layers.{bid}.self_attn.v_proj", # SmolVLM
            "vit.layers.{bid}.self_attn.v_proj", # HunyuanVL
            "vision_model.model.layers.{bid}.self_attn.v_proj", # llama4
            "vision_tower.transformer.layers.{bid}.attention.v_proj", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.attention.wv", # pixtral
            "model.vision_tower.layers.{bid}.self_attn.v_proj", # lfm3vl
            "visual.blocks.{bid}.attn.v", # qwen2vl, generated
            "vision_tower.encoder.blocks.{bid}.wv", # kimi-vl, generated
            "siglip2.vision_model.encoder.layers.{bid}.self_attn.v_proj",
            "model.vision_model.transformer.layers.{bid}.self_attn.v_proj", # Deepseek-OCR CLIP, generated
            "vision_model.model.layers.{bid}.self_attn.v_proj.linear", # gemma4
            "model.qwen2_model.model.model.layers.{bid}.self_attn.v_proj", # Deepseek-OCR-2 qwen2
            "model.vision_tower.layers.{bid}.attn.v_proj", # muse-glimmer
            "model.visual.backbone.layers.{bid}.self_attn.v_proj", # inkling2
        ),

        MODEL_TENSOR.V_ENC_INPUT_NORM: (
            "visual.encoder.layers.{bid}.layer_norm1", # mage-vl
            "vision_model.encoder.layer.{bid}.norm1", # Nemotron Omni
            "vision.blocks.{bid}.norm1", # deepseek4v
            "model.vision_tower.vision_model.encoder.layers.{bid}.layer_norm1", # Granite4Vision
            "vision_tower.vision_model.encoder.layers.{bid}.layer_norm1",
            "model.vision_tower.encoder.layers.{bid}.layer_norm1", # minicpmv4_6
            "vision_tower.vision_model.encoder.layers.{bid}.norm1", # InternVL
            "model.vision_tower.encoder.layer.{bid}.layernorm_before", # Intern-S1
            "vpm.encoder.layers.{bid}.layer_norm1",
            "model.vision_model.encoder.layers.{bid}.layer_norm1", # SmolVLM
            "vit.layers.{bid}.input_layernorm", # HunyuanVL
            "vision_tower.transformer.layers.{bid}.attention_norm", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.attention_norm", # pixtral
            "model.vision_tower.layers.{bid}.operator_norm", # lfm3vl
            "vision_model.model.layers.{bid}.input_layernorm", # llama4, gemma4
            "visual.blocks.{bid}.norm1", # qwen2vl
            "vision_tower.encoder.blocks.{bid}.norm0", # kimi-vl (norm0/norm1)
            "model.vision.transformer.layers.{bid}.input_layernorm", # cogvlm
            "model.vision_model.transformer.layers.{bid}.layer_norm1", # Deepseek-OCR CLIP
            "siglip2.vision_model.encoder.layers.{bid}.layer_norm1",
            "vision_model.radio_model.model.blocks.{bid}.norm1", # Nemotron Nano v2 VL
            "vision_tower.blocks.{bid}.norm1", # dots.ocr
            "vision_encoder.blocks.{bid}.norm_1", # dots3note
            "vision_model.transformer.resblocks.{bid}.ln_1", # Step3-VL
            "model.qwen2_model.model.model.layers.{bid}.input_layernorm", # Deepseek-OCR-2 qwen2
            "model.vision_tower.layers.{bid}.norm1", # muse-glimmer
            "model.vision_tower.layers.{bid}.layernorm_before", # inkling2
            "model.visual.backbone.layers.{bid}.ln_1", # inkling2
        ),

        MODEL_TENSOR.V_ENC_ATTN_O: (
            "visual.encoder.layers.{bid}.self_attn.proj", # mage-vl
            "vision_model.encoder.layer.{bid}.attention.output.dense", # Nemotron Omni
            "vision.blocks.{bid}.attn.wo", # deepseek4v
            "model.vision_tower.vision_model.encoder.layers.{bid}.self_attn.out_proj", # Granite4Vision
            "vision_tower.vision_model.encoder.layers.{bid}.self_attn.out_proj",
            "model.vision_tower.encoder.layers.{bid}.self_attn.out_proj", # minicpmv4_6
            "vision_tower.vision_model.encoder.layers.{bid}.attn.proj", # InternVL
            "model.vision_tower.encoder.layer.{bid}.attention.projection_layer", # Intern-S1
            "vpm.encoder.layers.{bid}.self_attn.out_proj",
            "model.vision_model.encoder.layers.{bid}.self_attn.out_proj", # SmolVLM
            "vit.layers.{bid}.self_attn.o_proj", # HunyuanVL
            "model.vision_model.encoder.layers.{bid}.self_attn.projection_layer", # Janus Pro
            "vision_model.model.layers.{bid}.self_attn.o_proj", # llama4
            "vision_tower.transformer.layers.{bid}.attention.o_proj", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.attention.wo", # pixtral
            "model.vision_tower.layers.{bid}.self_attn.out_proj", # lfm3vl
            "visual.blocks.{bid}.attn.proj", # qwen2vl
            "vision_tower.encoder.blocks.{bid}.wo", # kimi-vl
            "model.vision.transformer.layers.{bid}.attention.dense", # cogvlm
            "model.vision_model.transformer.layers.{bid}.self_attn.out_proj", # Deepseek-OCR CLIP
            "siglip2.vision_model.encoder.layers.{bid}.self_attn.out_proj", # youtuvl
            "vision_model.radio_model.model.blocks.{bid}.attn.proj", # Nemotron Nano v2 VL
            "model.qwen2_model.model.model.layers.{bid}.self_attn.o_proj", # Deepseek-OCR-2 qwen2
            "vision_model.model.layers.{bid}.self_attn.o_proj.linear", # gemma4
            "vision_tower.blocks.{bid}.attn.proj", # dots.ocr
            "vision_encoder.blocks.{bid}.attn.proj", # dots3note
            "vision_model.transformer.resblocks.{bid}.attn.out_proj", # Step3-VL
            "model.vision_tower.layers.{bid}.attn.proj", # muse-glimmer
            "model.visual.backbone.layers.{bid}.self_attn.out_proj", # inkling2
        ),

        MODEL_TENSOR.V_ENC_POST_ATTN_NORM: (
            "visual.encoder.layers.{bid}.layer_norm2", # mage-vl
            "vision_model.encoder.layer.{bid}.norm2", # Nemotron Omni
            "vision.blocks.{bid}.norm2", # deepseek4v
            "model.vision_tower.vision_model.encoder.layers.{bid}.layer_norm2", # Granite4Vision
            "vision_tower.vision_model.encoder.layers.{bid}.layer_norm2",
            "model.vision_tower.encoder.layers.{bid}.layer_norm2", # minicpmv4_6
            "vision_tower.vision_model.encoder.layers.{bid}.norm2", # InternVL
            "model.vision_tower.encoder.layer.{bid}.layernorm_after", # Intern-S1
            "vpm.encoder.layers.{bid}.layer_norm2",
            "model.vision_model.encoder.layers.{bid}.layer_norm2", # SmolVLM
            "vit.layers.{bid}.post_attention_layernorm", # HunyuanVL
            "vision_model.model.layers.{bid}.post_attention_layernorm", # llama4
            "vision_tower.transformer.layers.{bid}.ffn_norm", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.ffn_norm", # pixtral
            "model.vision_tower.layers.{bid}.ffn_norm", # lfm3vl
            "visual.blocks.{bid}.norm2", # qwen2vl
            "vision_tower.encoder.blocks.{bid}.norm1", # kimi-vl (norm0/norm1)
            "model.vision.transformer.layers.{bid}.post_attention_layernorm", # cogvlm
            "model.vision_model.transformer.layers.{bid}.layer_norm2", # Deepseek-OCR CLIP
            "siglip2.vision_model.encoder.layers.{bid}.layer_norm2",
            "vision_model.radio_model.model.blocks.{bid}.norm2", # Nemotron Nano v2 VL
            "vision_model.model.layers.{bid}.pre_feedforward_layernorm", # gemma4
            "vision_tower.blocks.{bid}.norm2", # dots.ocr
            "vision_encoder.blocks.{bid}.norm_2", # dots3note
            "vision_model.transformer.resblocks.{bid}.ln_2", # Step3-VL
            "model.qwen2_model.model.model.layers.{bid}.post_attention_layernorm", # Deepseek-OCR-2 qwen2
            "model.vision_tower.layers.{bid}.norm2", # muse-glimmer
            "model.vision_tower.layers.{bid}.layernorm_after", # inkling2
            "model.visual.backbone.layers.{bid}.ln_2", # inkling2
        ),

        MODEL_TENSOR.V_ENC_FFN_UP: (
            "visual.encoder.layers.{bid}.mlp.fc1", # mage-vl
            "vision_model.encoder.layer.{bid}.mlp.fc1", # Nemotron Omni
            "vision.blocks.{bid}.mlp.w1_up", # deepseek4v (split from fused w1)
            "vision_encoder.blocks.{bid}.mlp.fc3", # dots3note
            "model.vision_tower.vision_model.encoder.layers.{bid}.mlp.fc1", # Granite4Vision
            "vision_tower.vision_model.encoder.layers.{bid}.mlp.fc1",
            "model.vision_tower.encoder.layers.{bid}.mlp.fc1", # minicpmv4_6
            "model.vision_tower.encoder.layer.{bid}.mlp.fc1", # Intern-S1
            "vpm.encoder.layers.{bid}.mlp.fc1",
            "model.vision_model.encoder.layers.{bid}.mlp.fc1", # SmolVLM, gemma3
            "vit.layers.{bid}.mlp.dense_h_to_4h", # HunyuanVL
            "vision_tower.transformer.layers.{bid}.feed_forward.up_proj", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.feed_forward.w3", # pixtral
            "model.vision_tower.layers.{bid}.feed_forward.w3", # lfm3vl
            "vision_model.model.layers.{bid}.mlp.fc1", # llama4
            "visual.blocks.{bid}.mlp.fc1", # qwen2vl
            "visual.blocks.{bid}.mlp.up_proj", # qwen2.5vl
            "visual.blocks.{bid}.mlp.linear_fc1", # qwen3vl
            "vision_tower.encoder.blocks.{bid}.mlp.fc0", # kimi-vl (fc0/fc1)
            "model.vision_model.transformer.layers.{bid}.mlp.fc1", # Deepseek-OCR CLIP
            "model.vision.transformer.layers.{bid}.mlp.fc1", # cogvlm
            "siglip2.vision_model.encoder.layers.{bid}.mlp.fc1",
            "vision_model.radio_model.model.blocks.{bid}.mlp.fc1", # Nemotron Nano v2 VL
            "vision_model.model.layers.{bid}.mlp.up_proj", # gemma4
            "vision_model.transformer.resblocks.{bid}.mlp.c_fc", # Step3-VL
            "model.qwen2_model.model.model.layers.{bid}.mlp.up_proj", # Deepseek-OCR-2 qwen2
            "model.vision_tower.layers.{bid}.mlp.fc1", # muse-glimmer
            "model.visual.backbone.layers.{bid}.mlp.c_fc", # inkling2
        ),

        MODEL_TENSOR.V_ENC_FFN_GATE: (
            "vision.blocks.{bid}.mlp.w1_gate", # deepseek4v (split from fused w1)
            "vision_encoder.blocks.{bid}.mlp.fc1", # dots3note
            "vision_tower.transformer.layers.{bid}.feed_forward.gate_proj", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.feed_forward.w1", # pixtral
            "model.vision_tower.layers.{bid}.feed_forward.w1", # lfm3vl
            "visual.blocks.{bid}.mlp.gate_proj", # qwen2.5vl
            "vision_model.model.layers.{bid}.mlp.gate_proj", # gemma4
            "model.qwen2_model.model.model.layers.{bid}.mlp.gate_proj", # Deepseek-OCR-2 qwen2
        ),

        MODEL_TENSOR.V_ENC_FFN_DOWN: (
            "visual.encoder.layers.{bid}.mlp.fc2", # mage-vl
            "vision_model.encoder.layer.{bid}.mlp.fc2", # Nemotron Omni
            "vision.blocks.{bid}.mlp.w2", # deepseek4v
            "vision_encoder.blocks.{bid}.mlp.fc2", # dots3note
            "model.vision_tower.vision_model.encoder.layers.{bid}.mlp.fc2", # Granite4Vision
            "vision_tower.vision_model.encoder.layers.{bid}.mlp.fc2",
            "model.vision_tower.encoder.layers.{bid}.mlp.fc2", # minicpmv4_6
            "model.vision_tower.encoder.layer.{bid}.mlp.fc2", # Intern-S1
            "vpm.encoder.layers.{bid}.mlp.fc2",
            "model.vision_model.encoder.layers.{bid}.mlp.fc2", # SmolVLM, gemma3
            "vit.layers.{bid}.mlp.dense_4h_to_h", # HunyuanVL
            "vision_tower.transformer.layers.{bid}.feed_forward.down_proj", # pixtral-hf
            "vision_encoder.transformer.layers.{bid}.feed_forward.w2", # pixtral
            "model.vision_tower.layers.{bid}.feed_forward.w2", # lfm3vl
            "vision_model.model.layers.{bid}.mlp.fc2", # llama4
            "visual.blocks.{bid}.mlp.fc2", # qwen2vl
            "visual.blocks.{bid}.mlp.down_proj", # qwen2.5vl
            "visual.blocks.{bid}.mlp.linear_fc2", # qwen3vl
            "vision_tower.encoder.blocks.{bid}.mlp.fc1", # kimi-vl (fc0/fc1)
            "model.vision.transformer.layers.{bid}.mlp.fc2", # cogvlm
            "model.vision_model.transformer.layers.{bid}.mlp.fc2", # Deepseek-OCR CLIP
            "siglip2.vision_model.encoder.layers.{bid}.mlp.fc2",
            "vision_model.radio_model.model.blocks.{bid}.mlp.fc2", # Nemotron Nano v2 VL
            "model.qwen2_model.model.model.layers.{bid}.mlp.down_proj" , # Deepseek-OCR-2 qwen2
            "vision_model.model.layers.{bid}.mlp.down_proj", # gemma4
            "vision_model.transformer.resblocks.{bid}.mlp.c_proj", # Step3-VL
            "model.vision_tower.layers.{bid}.mlp.fc2", # muse-glimmer
            "model.visual.backbone.layers.{bid}.mlp.c_proj", # inkling2
        ),

        MODEL_TENSOR.V_LAYER_SCALE_1: (
            "vision_model.encoder.layer.{bid}.layer_scale1", # Nemotron Omni
            "vision_tower.vision_model.encoder.layers.{bid}.ls1", # InternVL
            "model.vision_tower.encoder.layer.{bid}.lambda_1", # Intern-S1
            "vision_model.transformer.resblocks.{bid}.ls_1", # Step3-VL
            "model.vision_tower.layers.{bid}.lambda_1", # inkling2
            "model.visual.backbone.layers.{bid}.ls_1", # inkling2
        ),

        MODEL_TENSOR.V_LAYER_SCALE_2: (
            "vision_model.encoder.layer.{bid}.layer_scale2", # Nemotron Omni
            "vision_tower.vision_model.encoder.layers.{bid}.ls2", # InternVL
            "model.vision_tower.encoder.layer.{bid}.lambda_2", # Intern-S1
            "vision_model.transformer.resblocks.{bid}.ls_2", # Step3-VL
            "model.vision_tower.layers.{bid}.lambda_2", # inkling2
            "model.visual.backbone.layers.{bid}.ls_2", # inkling2
        ),

        MODEL_TENSOR.V_PRE_NORM: (
            "visual.layernorm_pre", # mage-vl
            "vision_tower.vision_model.pre_layrnorm",
            "vision_tower.ln_pre", # pixtral-hf
            "vision_encoder.ln_pre", # pixtral
            "vision_model.layernorm_pre", # llama4
            "model.vision_model.pre_layrnorm", # Deepseek-OCR CLIP
            "vision_tower.patch_embed.patchifier.norm", # dots.ocr
            "vision_encoder.patch_embed.norm", # dots3note
            "vision_model.ln_pre", # Step3-VL
            "model.vision_tower.ln_pre", # muse-glimmer
            "model.vision_tower.pre_layernorm", # inkling2
            "model.visual.backbone.ln_pre", # inkling2
        ),

        MODEL_TENSOR.V_POST_NORM: (
            "vision.norm", # deepseek4v
            "model.vision_tower.vision_model.post_layernorm", # Granite4Vision
            "vision_tower.vision_model.post_layernorm",
            "model.vision_tower.post_layernorm", # minicpmv4_6
            "model.vision_model.post_layernorm", # SmolVLM
            "vision_model.layernorm_post", # llama4
            "visual.merger.ln_q", # qwen2vl
            "vision_tower.encoder.final_layernorm", # kimi-vl
            "visual.post_layernorm", # glm4v
            "siglip2.vision_model.post_layernorm",
            "model.qwen2_model.model.model.norm", # Deepseek-OCR-2 qwen2
            "model.vision_tower.ln_post", # muse-glimmer
            "vision_projector.vision_final_layernorm", # Nemotron Omni Reasoning V3
        ),

        MODEL_TENSOR.V_DS_NORM: (
            "model.visual.deepstack_merger_list.{bid}.norm", # deepstack in qwen3vl
        ),

        MODEL_TENSOR.V_DS_FC1: (
            "model.visual.deepstack_merger_list.{bid}.linear_fc1", # deepstack in qwen3vl
        ),

        MODEL_TENSOR.V_DS_FC2: (
            "model.visual.deepstack_merger_list.{bid}.linear_fc2", # deepstack in qwen3vl
        ),

        MODEL_TENSOR.V_MERGER_LN1: (
            "model.vision_tower.vit_merger.layer_norm1", # minicpmv4_6
        ),

        MODEL_TENSOR.V_MERGER_ATTN_Q: (
            "model.vision_tower.vit_merger.self_attn.q_proj", # minicpmv4_6
        ),

        MODEL_TENSOR.V_MERGER_ATTN_K: (
            "model.vision_tower.vit_merger.self_attn.k_proj", # minicpmv4_6
        ),

        MODEL_TENSOR.V_MERGER_ATTN_V: (
            "model.vision_tower.vit_merger.self_attn.v_proj", # minicpmv4_6
        ),

        MODEL_TENSOR.V_MERGER_ATTN_O: (
            "model.vision_tower.vit_merger.self_attn.out_proj", # minicpmv4_6
        ),

        MODEL_TENSOR.V_MERGER_DS_LN: (
            "model.vision_tower.vit_merger.pre_norm", # minicpmv4_6
        ),

        MODEL_TENSOR.V_MERGER_DS_UP: (
            "model.vision_tower.vit_merger.linear_1", # minicpmv4_6
        ),

        MODEL_TENSOR.V_MERGER_DS_DOWN: (
            "model.vision_tower.vit_merger.linear_2", # minicpmv4_6
        ),

        MODEL_TENSOR.V_STD_BIAS: (
            "model.vision_tower.std_bias", # gemma4
        ),

        MODEL_TENSOR.V_STD_SCALE: (
            "model.vision_tower.std_scale", # gemma4
        ),
    }

    mapping: dict[str, tuple[MODEL_TENSOR, str]]

    def __init__(self, arch: MODEL_ARCH, n_blocks: int):
        self.mapping = {}
        for tensor, keys in self.mappings_cfg.items():
            if tensor not in MODEL_TENSORS[arch]:
                continue
            tensor_name = TENSOR_NAMES[tensor]
            self.mapping[tensor_name] = (tensor, tensor_name)
            for key in keys:
                self.mapping[key] = (tensor, tensor_name)
        for bid in range(n_blocks):
            for tensor, keys in self.block_mappings_cfg.items():
                if tensor not in MODEL_TENSORS[arch]:
                    continue

                tensor_name = TENSOR_NAMES[tensor].format(bid = bid)
                self.mapping[tensor_name] = (tensor, tensor_name)
                for key in keys:
                    key = key.format(bid = bid)
                    self.mapping[key] = (tensor, tensor_name)

    def get_type_and_name(self, key: str, try_suffixes: Sequence[str] = ()) -> tuple[MODEL_TENSOR, str] | None:
        result = self.mapping.get(key)
        if result is not None:
            return result
        for suffix in try_suffixes:
            if key.endswith(suffix):
                result = self.mapping.get(key[:-len(suffix)])
                if result is not None:
                    return result[0], result[1] + suffix
        return None

    def get_name(self, key: str, try_suffixes: Sequence[str] = ()) -> str | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[1]

    def get_type(self, key: str, try_suffixes: Sequence[str] = ()) -> MODEL_TENSOR | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[0]

    def __getitem__(self, key: str) -> str:
        try:
            return self.mapping[key][1]
        except KeyError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return repr(self.mapping)


def get_tensor_name_map(arch: MODEL_ARCH, n_blocks: int) -> TensorNameMap:
    return TensorNameMap(arch, n_blocks)
