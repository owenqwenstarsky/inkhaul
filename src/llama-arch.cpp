#include "llama-arch.h"

#include "llama-ext.h"
#include "llama-impl.h"

#include <map>
#include <vector>

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_BAILINGMOE3,      "bailingmoe3"      },
    { LLM_ARCH_CLIP,             "clip"             }, // dummy, only used by llama-quantize
    { LLM_ARCH_DFLASH,           "dflash"           },
    { LLM_ARCH_INKLING2,         "inkling2"         },
    { LLM_ARCH_INKLING2_LOOP,    "inkling2_loop"    },
    { LLM_ARCH_QWEN35,           "qwen35"           },
    { LLM_ARCH_QWEN35MOE,        "qwen35moe"        },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};

static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE,                           "general.type"                          },
    { LLM_KV_GENERAL_ARCHITECTURE,                   "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,           "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,                      "general.alignment"                     },
    { LLM_KV_GENERAL_FILE_TYPE,                      "general.file_type"                     },
    { LLM_KV_GENERAL_SAMPLING_SEQUENCE,              "general.sampling.sequence"             },
    { LLM_KV_GENERAL_SAMPLING_TOP_K,                 "general.sampling.top_k"                },
    { LLM_KV_GENERAL_SAMPLING_TOP_P,                 "general.sampling.top_p"                },
    { LLM_KV_GENERAL_SAMPLING_MIN_P,                 "general.sampling.min_p"                },
    { LLM_KV_GENERAL_SAMPLING_XTC_PROBABILITY,       "general.sampling.xtc_probability"      },
    { LLM_KV_GENERAL_SAMPLING_XTC_THRESHOLD,         "general.sampling.xtc_threshold"        },
    { LLM_KV_GENERAL_SAMPLING_TEMP,                  "general.sampling.temp"                 },
    { LLM_KV_GENERAL_SAMPLING_PENALTY_LAST_N,        "general.sampling.penalty_last_n"       },
    { LLM_KV_GENERAL_SAMPLING_PENALTY_REPEAT,        "general.sampling.penalty_repeat"       },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT,              "general.sampling.mirostat"             },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT_TAU,          "general.sampling.mirostat_tau"         },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT_ETA,          "general.sampling.mirostat_eta"         },
    { LLM_KV_GENERAL_NAME,                           "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                         "general.author"                        },
    { LLM_KV_GENERAL_VERSION,                        "general.version"                       },
    { LLM_KV_GENERAL_URL,                            "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,                    "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,                        "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,                     "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,                 "general.source.huggingface.repository" },
    { LLM_KV_GENERAL_TENSOR_EXTRA_NAME,              "general.tensor_extra.name"             },
    { LLM_KV_GENERAL_TENSOR_EXTRA_PREC_A4,           "general.tensor_extra.prec_a4"          },

    { LLM_KV_VOCAB_SIZE,                        "%s.vocab_size"                        },
    { LLM_KV_VOCAB_SIZE_UNPADDED,               "%s.vocab_size_unpadded"               },
    { LLM_KV_CONTEXT_LENGTH,                    "%s.context_length"                    },
    { LLM_KV_EMBEDDING_LENGTH,                  "%s.embedding_length"                  },
    { LLM_KV_EMBEDDING_LENGTH_OUT,              "%s.embedding_length_out"              },
    { LLM_KV_FEATURES_LENGTH,                   "%s.features_length"                   },
    { LLM_KV_BLOCK_COUNT,                       "%s.block_count"                       },
    { LLM_KV_LEADING_DENSE_BLOCK_COUNT,         "%s.leading_dense_block_count"         },
    { LLM_KV_ATTN_RES_BLOCK_SIZE,               "%s.attn_res.block_size"               },
    { LLM_KV_ACTIVATION_SITU_BETA,              "%s.activation.situ_beta"              },
    { LLM_KV_ACTIVATION_SITU_LINEAR_BETA,       "%s.activation.situ_linear_beta"       },
    { LLM_KV_FEED_FORWARD_LENGTH,               "%s.feed_forward_length"               },
    { LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        "%s.expert_feed_forward_length"        },
    { LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, "%s.expert_shared_feed_forward_length" },
    { LLM_KV_USE_PARALLEL_RESIDUAL,             "%s.use_parallel_residual"             },
    { LLM_KV_TENSOR_DATA_LAYOUT,                "%s.tensor_data_layout"                },
    { LLM_KV_EXPERT_COUNT,                      "%s.expert_count"                      },
    { LLM_KV_EXPERT_USED_COUNT,                 "%s.expert_used_count"                 },
    { LLM_KV_EXPERT_SHARED_COUNT,               "%s.expert_shared_count"               },
    { LLM_KV_EXPERT_GROUP_COUNT,                "%s.expert_group_count"                },
    { LLM_KV_EXPERT_GROUP_USED_COUNT,           "%s.expert_group_used_count"           },
    { LLM_KV_EXPERT_WEIGHTS_SCALE,              "%s.expert_weights_scale"              },
    { LLM_KV_EXPERT_WEIGHTS_NORM,               "%s.expert_weights_norm"               },
    { LLM_KV_EXPERT_LATENT_LENGTH,              "%s.expert_latent_length"              },
    { LLM_KV_EXPERT_GATING_FUNC,                "%s.expert_gating_func"                },
    { LLM_KV_EXPERT_GROUP_SCALE,                "%s.expert_group_scale"                },
    { LLM_KV_EXPERTS_PER_GROUP,                 "%s.experts_per_group"                 },
    { LLM_KV_NEXTN_PREDICT_LAYERS,              "%s.nextn_predict_layers"              },
    { LLM_KV_SWIGLU_CLAMP_EXP,                  "%s.swiglu_clamp_exp"                  },
    { LLM_KV_SWIGLU_CLAMP_SHEXP,                "%s.swiglu_clamp_shexp"                },
    { LLM_KV_HIDDEN_ACT,                        "%s.hidden_activation"                 },
    { LLM_KV_POOLING_TYPE,                      "%s.pooling_type"                      },
    { LLM_KV_LOGIT_SCALE,                       "%s.logit_scale"                       },
    { LLM_KV_ROUTER_HIDDEN_SIZE,                "%s.router_hidden_size"                },
    { LLM_KV_FINAL_LOGIT_SOFTCAPPING,           "%s.final_logit_softcapping"           },
    { LLM_KV_TIME_MIX_EXTRA_DIM,                "%s.time_mix_extra_dim"                },
    { LLM_KV_TIME_DECAY_EXTRA_DIM,              "%s.time_decay_extra_dim"              },
    { LLM_KV_RESIDUAL_SCALE,                    "%s.residual_scale"                    },
    { LLM_KV_EMBEDDING_SCALE,                   "%s.embedding_scale"                   },
    { LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         "%s.interleave_moe_layer_step"         },
    { LLM_KV_FULL_ATTENTION_INTERVAL,           "%s.full_attention_interval"           },
    { LLM_KV_LOOP_START,                        "%s.loop.start"                        },
    { LLM_KV_LOOP_END,                          "%s.loop.end"                          },
    { LLM_KV_LOOP_COUNT,                        "%s.loop.count"                        },
    { LLM_KV_LOOP_GATE_RANK,                    "%s.loop.gate_rank"                    },
    { LLM_KV_LOOP_VARYING,                      "%s.loop.varying"                      },

    { LLM_KV_ATTENTION_HEAD_COUNT,                   "%s.attention.head_count"                   },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,                "%s.attention.head_count_kv"                },
    { LLM_KV_ATTENTION_MAX_ALIBI_BIAS,               "%s.attention.max_alibi_bias"               },
    { LLM_KV_ATTENTION_CLAMP_KQV,                    "%s.attention.clamp_kqv"                    },
    { LLM_KV_ATTENTION_KEY_LENGTH,                   "%s.attention.key_length"                   },
    { LLM_KV_ATTENTION_VALUE_LENGTH,                 "%s.attention.value_length"                 },
    { LLM_KV_ATTENTION_KEY_LENGTH_MLA,               "%s.attention.key_length_mla"               },
    { LLM_KV_ATTENTION_VALUE_LENGTH_MLA,             "%s.attention.value_length_mla"             },
    { LLM_KV_ATTENTION_LAYERNORM_EPS,                "%s.attention.layer_norm_epsilon"           },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,            "%s.attention.layer_norm_rms_epsilon"       },
    { LLM_KV_ATTENTION_GROUPNORM_EPS,                "%s.attention.group_norm_epsilon"           },
    { LLM_KV_ATTENTION_GROUPNORM_GROUPS,             "%s.attention.group_norm_groups"            },
    { LLM_KV_ATTENTION_CAUSAL,                       "%s.attention.causal"                       },
    { LLM_KV_ATTENTION_Q_LORA_RANK,                  "%s.attention.q_lora_rank"                  },
    { LLM_KV_ATTENTION_KV_LORA_RANK,                 "%s.attention.kv_lora_rank"                 },
    { LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT,       "%s.attention.relative_buckets_count"       },
    { LLM_KV_ATTENTION_SLIDING_WINDOW,               "%s.attention.sliding_window"               },
    { LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,       "%s.attention.sliding_window_pattern"       },
    { LLM_KV_ATTENTION_ROPE_PATTERN,                 "%s.attention.rope_pattern"                 },

    { LLM_KV_ATTENTION_SCALE,                        "%s.attention.scale"                        },
    { LLM_KV_ATTENTION_VALUE_SCALE,                  "%s.attention.value_scale"                  },
    { LLM_KV_CLASSIFIER_FREE_GUIDANCE,               "%s.classifier_free_guidance"               },
    { LLM_KV_ATTENTION_KEY_LENGTH_SWA,               "%s.attention.key_length_swa"               },
    { LLM_KV_ATTENTION_VALUE_LENGTH_SWA,             "%s.attention.value_length_swa"             },
    { LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,           "%s.attention.indexer.head_count"           },
    { LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,           "%s.attention.indexer.key_length"           },
    { LLM_KV_ATTENTION_INDEXER_TOP_K,                "%s.attention.indexer.top_k"                },
    { LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,           "%s.attention.indexer.block_size"           },
    { LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,         "%s.attention.indexer.local_blocks"         },
    { LLM_KV_ATTENTION_INDEXER_KPOOL,                "%s.attention.indexer.kpool"                },
    { LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL,    "%s.attention.indexer.kpool_select_tail"    },
    { LLM_KV_ATTENTION_SHARED_KV_LAYERS,             "%s.attention.shared_kv_layers"             },
    { LLM_KV_ATTENTION_KV_SOURCE_MAP,                "%s.attention.kv_source_map"                },
    { LLM_KV_ATTENTION_RECURRENT_LAYERS,             "%s.attention.recurrent_layers"             },
    { LLM_KV_ATTENTION_RELATIVE_LENGTH,              "%s.attention.relative_length"              },
    { LLM_KV_ATTENTION_RELATIVE_EXTENT,              "%s.attention.relative_extent"              },
    { LLM_KV_ATTENTION_QK_CLIP,                      "%s.attention.qk_clip"                      },
    { LLM_KV_ATTENTION_RELATIVE_CLIP,                "%s.attention.relative_clip"                },
    { LLM_KV_ATTENTION_LOG_SCALE_FLOOR,              "%s.attention.log_scale_floor"              },
    { LLM_KV_ATTENTION_LOG_SCALE_ALPHA,              "%s.attention.log_scale_alpha"              },

    { LLM_KV_ROPE_DIMENSION_COUNT,           "%s.rope.dimension_count"                 },
    { LLM_KV_ROPE_DIMENSION_COUNT_SWA,       "%s.rope.dimension_count_swa"             },
    { LLM_KV_ROPE_DIMENSION_SECTIONS,        "%s.rope.dimension_sections"              },
    { LLM_KV_ROPE_SECTION_ORDER,             "%s.rope.section_order"                   },
    { LLM_KV_ROPE_FREQ_BASE,                 "%s.rope.freq_base"                       },
    { LLM_KV_ROPE_FREQ_BASE_SWA,             "%s.rope.freq_base_swa"                   },
    { LLM_KV_ROPE_SCALE_LINEAR,              "%s.rope.scale_linear"                    },
    { LLM_KV_ROPE_SCALING_TYPE,              "%s.rope.scaling.type"                    },
    { LLM_KV_ROPE_SCALING_FACTOR,            "%s.rope.scaling.factor"                  },
    { LLM_KV_ROPE_SCALING_ATTN_FACTOR,       "%s.rope.scaling.attn_factor"             },
    { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,      "%s.rope.scaling.original_context_length" },
    { LLM_KV_ROPE_SCALING_FINETUNED,         "%s.rope.scaling.finetuned"               },
    { LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,   "%s.rope.scaling.yarn_ext_factor"         },
    { LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR,  "%s.rope.scaling.yarn_attn_factor"        },
    { LLM_KV_ROPE_SCALING_YARN_BETA_FAST,    "%s.rope.scaling.yarn_beta_fast"          },
    { LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,    "%s.rope.scaling.yarn_beta_slow"          },

    { LLM_KV_SPLIT_NO,            "split.no"            },
    { LLM_KV_SPLIT_COUNT,         "split.count"         },
    { LLM_KV_SPLIT_TENSORS_COUNT, "split.tensors.count" },

    { LLM_KV_SSM_CONV_KERNEL,    "%s.ssm.conv_kernel"    },
    { LLM_KV_SSM_INNER_SIZE,     "%s.ssm.inner_size"     },
    { LLM_KV_SSM_STATE_SIZE,     "%s.ssm.state_size"     },
    { LLM_KV_SSM_TIME_STEP_RANK, "%s.ssm.time_step_rank" },
    { LLM_KV_SSM_GROUP_COUNT,    "%s.ssm.group_count"    },

    { LLM_KV_KDA_HEAD_DIM,         "%s.kda.head_dim"         },
    { LLM_KV_KDA_SAFE_GATE,        "%s.kda.safe_gate"        },
    { LLM_KV_KDA_GATE_LOWER_BOUND, "%s.kda.gate_lower_bound" },

    { LLM_KV_WKV_HEAD_SIZE, "%s.wkv.head_size" },

    { LLM_KV_POSNET_EMBEDDING_LENGTH, "%s.posnet.embedding_length" },
    { LLM_KV_POSNET_BLOCK_COUNT,      "%s.posnet.block_count"      },

    { LLM_KV_CONVNEXT_EMBEDDING_LENGTH, "%s.convnext.embedding_length" },
    { LLM_KV_CONVNEXT_BLOCK_COUNT,      "%s.convnext.block_count"      },

    { LLM_KV_CLASSIFIER_OUTPUT_LABELS, "%s.classifier.output_labels" },
    { LLM_KV_CLASSIFIER_POOLING_TYPE,  "%s.classifier.pooling_type"  },
    { LLM_KV_CLASSIFIER_ACTIVATION,    "%s.classifier.activation"    },

    { LLM_KV_DECISION_LABELS,              "%s.decision.labels"              },

    { LLM_KV_TARGET_LAYERS,         "%s.target_layers"        },

    { LLM_KV_DFLASH_BLOCK_SIZE,       "%s.block_size"       },

    { LLM_KV_SHORTCONV_L_CACHE, "%s.shortconv.l_cache" },
    // sentence-transformers dense modules feature dims
    { LLM_KV_DENSE_2_FEAT_OUT,       "%s.dense_2_feat_out" },
    { LLM_KV_DENSE_3_FEAT_IN,        "%s.dense_3_feat_in"  },

    { LLM_KV_TOKENIZER_MODEL,                    "tokenizer.ggml.model"                    },
    { LLM_KV_TOKENIZER_PRE,                      "tokenizer.ggml.pre"                      },
    { LLM_KV_TOKENIZER_LIST,                     "tokenizer.ggml.tokens"                   },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,               "tokenizer.ggml.token_type"               },
    { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,         "tokenizer.ggml.token_type_count"         },
    { LLM_KV_TOKENIZER_SCORES,                   "tokenizer.ggml.scores"                   },
    { LLM_KV_TOKENIZER_MERGES,                   "tokenizer.ggml.merges"                   },
    { LLM_KV_TOKENIZER_BOS_ID,                   "tokenizer.ggml.bos_token_id"             },
    { LLM_KV_TOKENIZER_EOS_ID,                   "tokenizer.ggml.eos_token_id"             },
    { LLM_KV_TOKENIZER_EOT_ID,                   "tokenizer.ggml.eot_token_id"             },
    { LLM_KV_TOKENIZER_EOM_ID,                   "tokenizer.ggml.eom_token_id"             },
    { LLM_KV_TOKENIZER_UNK_ID,                   "tokenizer.ggml.unknown_token_id"         },
    { LLM_KV_TOKENIZER_SEP_ID,                   "tokenizer.ggml.seperator_token_id"       },
    { LLM_KV_TOKENIZER_PAD_ID,                   "tokenizer.ggml.padding_token_id"         },
    { LLM_KV_TOKENIZER_CLS_ID,                   "tokenizer.ggml.cls_token_id"             },
    { LLM_KV_TOKENIZER_MASK_ID,                  "tokenizer.ggml.mask_token_id"            },
    { LLM_KV_TOKENIZER_ADD_BOS,                  "tokenizer.ggml.add_bos_token"            },
    { LLM_KV_TOKENIZER_ADD_EOS,                  "tokenizer.ggml.add_eos_token"            },
    { LLM_KV_TOKENIZER_ADD_SEP,                  "tokenizer.ggml.add_sep_token"            },
    { LLM_KV_TOKENIZER_ADD_PREFIX,               "tokenizer.ggml.add_space_prefix"         },
    { LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,          "tokenizer.ggml.remove_extra_whitespaces" },
    { LLM_KV_TOKENIZER_CLEAN_SPACES,             "tokenizer.ggml.clean_spaces"             },
    { LLM_KV_TOKENIZER_HF_JSON,                  "tokenizer.huggingface.json"              },
    { LLM_KV_TOKENIZER_RWKV,                     "tokenizer.rwkv.world"                    },
    { LLM_KV_TOKENIZER_CHAT_TEMPLATE,            "tokenizer.chat_template"                 },
    { LLM_KV_TOKENIZER_METASPACE_PREPEND_SCHEME, "tokenizer.ggml.metaspace.prepend_scheme" },
    { LLM_KV_TOKENIZER_FIM_PRE_ID,               "tokenizer.ggml.fim_pre_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SUF_ID,               "tokenizer.ggml.fim_suf_token_id"         },
    { LLM_KV_TOKENIZER_FIM_MID_ID,               "tokenizer.ggml.fim_mid_token_id"         },
    { LLM_KV_TOKENIZER_FIM_PAD_ID,               "tokenizer.ggml.fim_pad_token_id"         },
    { LLM_KV_TOKENIZER_FIM_REP_ID,               "tokenizer.ggml.fim_rep_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SEP_ID,               "tokenizer.ggml.fim_sep_token_id"         },
    { LLM_KV_TOKENIZER_SUPPRESS_TOKENS,          "tokenizer.ggml.suppress_tokens"          },

    { LLM_KV_ADAPTER_TYPE,                    "adapter.type"               },
    { LLM_KV_ADAPTER_LORA_ALPHA,              "adapter.lora.alpha"         },
    { LLM_KV_ADAPTER_LORA_TASK_NAME,          "adapter.lora.task_name"     },
    { LLM_KV_ADAPTER_LORA_PROMPT_PREFIX,      "adapter.lora.prompt_prefix" },
    { LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS, "adapter.alora.invocation_tokens" },

    { LLM_KV_XIELU_ALPHA_N,         "xielu.alpha_n"         },
    { LLM_KV_XIELU_ALPHA_P,         "xielu.alpha_p"         },
    { LLM_KV_XIELU_BETA,            "xielu.beta"            },
    { LLM_KV_XIELU_EPS,             "xielu.eps"             },

    // deprecated
    { LLM_KV_TOKENIZER_PREFIX_ID, "tokenizer.ggml.prefix_token_id" },
    { LLM_KV_TOKENIZER_SUFFIX_ID, "tokenizer.ggml.suffix_token_id" },
    { LLM_KV_TOKENIZER_MIDDLE_ID, "tokenizer.ggml.middle_token_id" },
};

static const std::map<llm_tensor, const char *> LLM_TENSOR_NAMES = {
    { LLM_TENSOR_TOKEN_EMBD,                             "token_embd" },
    { LLM_TENSOR_OUTPUT_NORM,                            "output_norm" },
    { LLM_TENSOR_OUTPUT,                                 "output" },
    { LLM_TENSOR_OUTPUT_RES_QUERY,                       "output_res_query" },
    { LLM_TENSOR_ROPE_FREQS,                             "rope_freqs" },
    { LLM_TENSOR_ATTN_NORM,                              "blk.%d.attn_norm" },
    { LLM_TENSOR_ATTN_Q,                                 "blk.%d.attn_q" },
    { LLM_TENSOR_ATTN_K,                                 "blk.%d.attn_k" },
    { LLM_TENSOR_ATTN_V,                                 "blk.%d.attn_v" },
    { LLM_TENSOR_ATTN_OUT,                               "blk.%d.attn_output" },
    { LLM_TENSOR_FFN_GATE_INP,                           "blk.%d.ffn_gate_inp" },
    { LLM_TENSOR_FFN_NORM,                               "blk.%d.ffn_norm" },
    { LLM_TENSOR_FFN_GATE,                               "blk.%d.ffn_gate" },
    { LLM_TENSOR_FFN_DOWN,                               "blk.%d.ffn_down" },
    { LLM_TENSOR_FFN_UP,                                 "blk.%d.ffn_up" },
    { LLM_TENSOR_FFN_GATE_EXPS,                          "blk.%d.ffn_gate_exps" },
    { LLM_TENSOR_FFN_GATE_UP_EXPS,                       "blk.%d.ffn_gate_up_exps" },
    { LLM_TENSOR_FFN_DOWN_EXPS,                          "blk.%d.ffn_down_exps" },
    { LLM_TENSOR_FFN_UP_EXPS,                            "blk.%d.ffn_up_exps" },
    { LLM_TENSOR_ATTN_POST_NORM,                         "blk.%d.post_attention_norm" },
    { LLM_TENSOR_ATTN_Q_NORM,                            "blk.%d.attn_q_norm" },
    { LLM_TENSOR_ATTN_K_NORM,                            "blk.%d.attn_k_norm" },
    { LLM_TENSOR_ATTN_GATE,                              "blk.%d.attn_gate" },
    { LLM_TENSOR_ATTN_REL,                               "blk.%d.attn_rel" },
    { LLM_TENSOR_ATTN_REL_BIAS,                          "blk.%d.attn_rel_bias" },
    { LLM_TENSOR_ATTN_Q_CLIP,                            "blk.%d.attn_q_clip" },
    { LLM_TENSOR_ATTN_REL_CLIP,                          "blk.%d.attn_rel_clip" },
    { LLM_TENSOR_ATTN_SINK_PROJ,                         "blk.%d.attn_sink_proj" },
    { LLM_TENSOR_ATTN_Q_INP_NORM,                        "blk.%d.attn_q_inp_norm" },
    { LLM_TENSOR_ATTN_K_INP_NORM,                        "blk.%d.attn_k_inp_norm" },
    { LLM_TENSOR_ATTN_V_INP_NORM,                        "blk.%d.attn_v_inp_norm" },
    { LLM_TENSOR_ATTN_K_CONV,                            "blk.%d.attn_k_conv" },
    { LLM_TENSOR_ATTN_V_CONV,                            "blk.%d.attn_v_conv" },
    { LLM_TENSOR_ATTN_K_CONV_PROJ,                       "blk.%d.attn_k_conv_proj" },
    { LLM_TENSOR_ATTN_V_CONV_PROJ,                       "blk.%d.attn_v_conv_proj" },
    { LLM_TENSOR_ATTN_POST_CONV,                         "blk.%d.attn_post_conv" },
    { LLM_TENSOR_ATTN_RES_QUERY,                         "blk.%d.attn_res_query" },
    { LLM_TENSOR_FFN_POST_NORM,                          "blk.%d.post_ffw_norm" },
    { LLM_TENSOR_FFN_EXP_PROBS_B,                        "blk.%d.exp_probs_b" },
    { LLM_TENSOR_FFN_POST_CONV,                          "blk.%d.ffn_post_conv" },
    { LLM_TENSOR_FFN_RES_QUERY,                          "blk.%d.ffn_res_query" },
    { LLM_TENSOR_LOOP_MSG_QUERY,                         "blk.%d.loop_msg_query" },
    { LLM_TENSOR_LOOP_Q_CTRL_DOWN,                       "blk.%d.loop_q_ctrl_down" },
    { LLM_TENSOR_LOOP_Q_CTRL_GAIN,                       "blk.%d.loop_q_ctrl_gain" },
    { LLM_TENSOR_LOOP_Q_CTRL_FORGET,                     "blk.%d.loop_q_ctrl_forget" },
    { LLM_TENSOR_LOOP_K_CTRL_DOWN,                       "blk.%d.loop_k_ctrl_down" },
    { LLM_TENSOR_LOOP_K_CTRL_GAIN,                       "blk.%d.loop_k_ctrl_gain" },
    { LLM_TENSOR_LOOP_K_CTRL_FORGET,                     "blk.%d.loop_k_ctrl_forget" },
    { LLM_TENSOR_LOOP_V_CTRL_DOWN,                       "blk.%d.loop_v_ctrl_down" },
    { LLM_TENSOR_LOOP_V_CTRL_GAIN,                       "blk.%d.loop_v_ctrl_gain" },
    { LLM_TENSOR_LOOP_V_CTRL_FORGET,                     "blk.%d.loop_v_ctrl_forget" },
    { LLM_TENSOR_LOOP_FFN_CTRL_DOWN,                     "blk.%d.loop_ffn_ctrl_down" },
    { LLM_TENSOR_LOOP_FFN_CTRL_GAIN,                     "blk.%d.loop_ffn_ctrl_gain" },
    { LLM_TENSOR_LOOP_FFN_CTRL_FORGET,                   "blk.%d.loop_ffn_ctrl_forget" },
    { LLM_TENSOR_ATTN_QKV,                               "blk.%d.attn_qkv" },
    { LLM_TENSOR_LAYER_OUT_SCALE,                        "blk.%d.layer_output_scale" },
    { LLM_TENSOR_TOKEN_EMBD_NORM,                        "token_embd_norm" },
    { LLM_TENSOR_ENC_OUTPUT_NORM,                        "enc.output_norm" },
    { LLM_TENSOR_ATTN_SINKS,                             "blk.%d.attn_sinks" },
    { LLM_TENSOR_FC,                                     "fc" },
    { LLM_TENSOR_D2T,                                    "d2t" },
    { LLM_TENSOR_DSPARK_MARKOV_W1,                       "markov_w1" },
    { LLM_TENSOR_DSPARK_MARKOV_W2,                       "markov_w2" },
    { LLM_TENSOR_DSPARK_CONF_PROJ,                       "conf_proj" },
    { LLM_TENSOR_DSPARK_JOINT_PROJ,                      "joint_proj" },
    { LLM_TENSOR_ATTN_QKV_GATE,                          "blk.%d.attn_qkv_gate" },
    { LLM_TENSOR_FFN_GATE_INP_SHEXP,                     "blk.%d.ffn_gate_inp_shexp" },
    { LLM_TENSOR_FFN_GATE_SHEXP,                         "blk.%d.ffn_gate_shexp" },
    { LLM_TENSOR_FFN_UP_SHEXP,                           "blk.%d.ffn_up_shexp" },
    { LLM_TENSOR_FFN_DOWN_SHEXP,                         "blk.%d.ffn_down_shexp" },
    { LLM_TENSOR_SSM_A_NOSCAN,                           "blk.%d.ssm_a" },
    { LLM_TENSOR_SSM_CONV1D,                             "blk.%d.ssm_conv1d" },
    { LLM_TENSOR_SSM_DT,                                 "blk.%d.ssm_dt" },
    { LLM_TENSOR_SSM_BETA_ALPHA_CAT,                     "blk.%d.ssm_beta_alpha" },
    { LLM_TENSOR_SSM_BETA,                               "blk.%d.ssm_beta" },
    { LLM_TENSOR_SSM_ALPHA,                              "blk.%d.ssm_alpha" },
    { LLM_TENSOR_SSM_NORM,                               "blk.%d.ssm_norm" },
    { LLM_TENSOR_SSM_OUT,                                "blk.%d.ssm_out" },
    { LLM_TENSOR_NEXTN_EH_PROJ,                          "blk.%d.nextn.eh_proj" },
    { LLM_TENSOR_NEXTN_EMBED_TOKENS,                     "blk.%d.nextn.embed_tokens" },
    { LLM_TENSOR_NEXTN_ENORM,                            "blk.%d.nextn.enorm" },
    { LLM_TENSOR_NEXTN_HNORM,                            "blk.%d.nextn.hnorm" },
    { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,                 "blk.%d.nextn.shared_head_head" },
    { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,                 "blk.%d.nextn.shared_head_norm" },
    { LLM_TENSOR_ATTN_Q_A,                               "blk.%d.attn_q_a" },
    { LLM_TENSOR_ATTN_Q_B,                               "blk.%d.attn_q_b" },
    { LLM_TENSOR_ATTN_KV_A_MQA,                          "blk.%d.attn_kv_a_mqa" },
    { LLM_TENSOR_ATTN_K_B,                               "blk.%d.attn_k_b" },
    { LLM_TENSOR_ATTN_V_B,                               "blk.%d.attn_v_b" },
    { LLM_TENSOR_ATTN_Q_A_NORM,                          "blk.%d.attn_q_a_norm" },
    { LLM_TENSOR_ATTN_KV_A_NORM,                         "blk.%d.attn_kv_a_norm" },
    { LLM_TENSOR_LAYER_OUT_NORM,                         "blk.%d.layer_output_norm" },
    { LLM_TENSOR_SSM_CONV1D_Q,                           "blk.%d.ssm_conv1d_q" },
    { LLM_TENSOR_SSM_CONV1D_K,                           "blk.%d.ssm_conv1d_k" },
    { LLM_TENSOR_SSM_CONV1D_V,                           "blk.%d.ssm_conv1d_v" },
    { LLM_TENSOR_SSM_F_A,                                "blk.%d.ssm_f_a" },
    { LLM_TENSOR_SSM_G_A,                                "blk.%d.ssm_g_a" },
};

// declare information about the model weight tensors:
// - the layer in which the tensor is going to be used. this is needed in order to assign the correct buffer type for the weight
// - the operator which is going to use the weight. this is needed to determine if the respective backend supports the operator
//
// for example, input layers are usually assigned to CPU/host buffer types
//
// a mismatch between the declared information and the actual layer/op in which the tensor is used can lead to sub-optimal
//   assignment of the buffer types and extra overhead during computation
// example: https://github.com/ggml-org/llama.cpp/pull/17548
//
static const std::map<llm_tensor, llm_tensor_info> LLM_TENSOR_INFOS = {
    {LLM_TENSOR_TOKEN_EMBD,                 {LLM_TENSOR_LAYER_INPUT,     GGML_OP_GET_ROWS}},
    {LLM_TENSOR_TOKEN_EMBD_NORM,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},  // do the norms on the first layer (not the input layer)
    {LLM_TENSOR_OUTPUT,                     {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_OUTPUT_NORM,                {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL}},
    {LLM_TENSOR_OUTPUT_RES_QUERY,           {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ENC_OUTPUT_NORM,            {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL}},
    {LLM_TENSOR_ROPE_FREQS,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ROPE}},
    {LLM_TENSOR_ATTN_Q,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_K,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_V,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_QKV,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_OUT,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_GATE,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_REL,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_REL_BIAS,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q_CLIP,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_REL_CLIP,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_SINK_PROJ,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q_INP_NORM,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_K_INP_NORM,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_V_INP_NORM,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_K_CONV,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
    {LLM_TENSOR_ATTN_V_CONV,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
    {LLM_TENSOR_ATTN_K_CONV_PROJ,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_V_CONV_PROJ,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_POST_CONV,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
    {LLM_TENSOR_ATTN_RES_QUERY,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_DOWN,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_UP,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_SINKS,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SCALE}},
    {LLM_TENSOR_FFN_GATE_INP,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_NORM,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_POST_NORM,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_NORM,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_POST_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_Q_NORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_K_NORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_LAYER_OUT_SCALE,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_DOWN_EXPS,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
    {LLM_TENSOR_FFN_GATE_EXPS,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
    {LLM_TENSOR_FFN_UP_EXPS,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
    {LLM_TENSOR_FFN_GATE_UP_EXPS,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
    {LLM_TENSOR_FFN_EXP_PROBS_B,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ADD}},
    {LLM_TENSOR_FFN_POST_CONV,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
    {LLM_TENSOR_FFN_RES_QUERY,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_MSG_QUERY,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_Q_CTRL_DOWN,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_Q_CTRL_GAIN,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_Q_CTRL_FORGET,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_K_CTRL_DOWN,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_K_CTRL_GAIN,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_K_CTRL_FORGET,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_V_CTRL_DOWN,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_V_CTRL_GAIN,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_V_CTRL_FORGET,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_FFN_CTRL_DOWN,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_FFN_CTRL_GAIN,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_LOOP_FFN_CTRL_FORGET,       {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    // dflash
    {LLM_TENSOR_FC,                         {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_D2T,                        {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_SET_ROWS}},
    // dspark
    {LLM_TENSOR_DSPARK_MARKOV_W1,           {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_GET_ROWS}},
    {LLM_TENSOR_DSPARK_MARKOV_W2,           {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DSPARK_CONF_PROJ,           {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DSPARK_JOINT_PROJ,          {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    // qwen35
    {LLM_TENSOR_ATTN_QKV_GATE,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE_INP_SHEXP,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE_SHEXP,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_UP_SHEXP,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_DOWN_SHEXP,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_A_NOSCAN,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}}, // a version of SSM_A used for MUL instead of SSM_SCAN
    {LLM_TENSOR_SSM_CONV1D,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
    {LLM_TENSOR_SSM_DT,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_BETA_ALPHA_CAT,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_BETA,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_ALPHA,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_NORM,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_SSM_OUT,                    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    // the MTP block is not run, its tensors are skipped at load
    {LLM_TENSOR_NEXTN_EH_PROJ,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_NEXTN_EMBED_TOKENS,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_NEXTN_ENORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_NEXTN_HNORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    // bailingmoe3 (with the shared expert, SSM and NextN entries above)
    {LLM_TENSOR_ATTN_Q_A,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q_B,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_KV_A_MQA,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_K_B,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_V_B,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q_A_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_KV_A_NORM,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_LAYER_OUT_NORM,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_SSM_CONV1D_Q,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_SSM_CONV1D_K,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_SSM_CONV1D_V,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_SSM_F_A,                    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_G_A,                    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
};

LLM_KV::LLM_KV(llm_arch arch, const char * suffix) : arch(arch), suffix(suffix) {}

std::string LLM_KV::operator()(llm_kv kv) const {
    std::string name = ::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch));

    if (suffix != nullptr) {
        name += ".";
        name += suffix;
    }

    return name;
}

LLM_TN_IMPL::LLM_TN_IMPL(llm_arch arch, llm_tensor tensor, const char * suffix, int bid, int xid)
    : arch(arch), tensor(tensor), suffix(suffix), bid(bid), xid(xid) {}

std::string LLM_TN_IMPL::str() const {
    if (LLM_TENSOR_NAMES.find(tensor) == LLM_TENSOR_NAMES.end()) {
        GGML_ABORT("unknown tensor name for tensor id %d", static_cast<int>(tensor));
    }

    std::string name = ::format(LLM_TENSOR_NAMES.at(tensor), bid, xid);
    if (suffix != nullptr) {
        name += ".";
        name += suffix;
    }

    return name;
}

std::vector<llm_arch> llm_arch_all() {
    std::vector<llm_arch> ret;
    ret.reserve(LLM_ARCH_NAMES.size());
    for (const auto & [arch, _] : LLM_ARCH_NAMES) {
        ret.push_back(arch);
    }
    return ret;
}

const char * llm_arch_name(llm_arch arch) {
    auto it = LLM_ARCH_NAMES.find(arch);
    if (it == LLM_ARCH_NAMES.end()) {
        return "unknown";
    }
    return it->second;
}

llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

const llm_tensor_info & llm_tensor_info_for(llm_tensor tensor) {
    return LLM_TENSOR_INFOS.at(tensor);
}

bool llm_arch_is_recurrent(const llm_arch & arch) {
    switch (arch) {
        default:
            return false;
    }
}

bool llm_arch_is_hybrid(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_INKLING2:
        case LLM_ARCH_INKLING2_LOOP:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}

bool llm_arch_is_diffusion(const llm_arch & arch) {
    switch (arch) {
        default:
            return false;
    }
}

bool llm_arch_supports_rs_rollback(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_INKLING2:
        case LLM_ARCH_INKLING2_LOOP:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}

// these models pick weights, routing or input meaning per ubatch based on token vs embd input
bool llm_arch_supports_mixed_batch(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_DFLASH:
        case LLM_ARCH_INKLING2:
        case LLM_ARCH_INKLING2_LOOP:
            return false;
        default:
            return true;
    }
}

// archs whose tensors llama_meta_device_get_split_state knows how to split
bool llm_arch_supports_sm_tensor(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}

bool llm_arch_mmid_down_bf16(const llm_arch & arch) {
    switch (arch) {
        default:
            return false;
    }
}

bool llama_arch_supports_save(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_INKLING2_LOOP: // the loop tensors are not saved and the cache slots are expanded at load
            return false;
        default:
            return true;
    }
}

std::vector<llm_arch> llama_arch_all() {
    return llm_arch_all();
}

const char * llama_arch_name(llm_arch arch) {
    return llm_arch_name(arch);
}

llm_arch llama_arch_from_string(const std::string & name) {
    return llm_arch_from_string(name);
}

std::string llama_arch_kv_name(llm_arch arch, llm_kv key) {
    return LLM_KV(arch)(key);
}

bool llama_arch_supports_mixed_batch(llm_arch arch) {
    return llm_arch_supports_mixed_batch(arch);
}

bool llama_arch_supports_sm_tensor(llm_arch arch) {
    return llm_arch_supports_sm_tensor(arch);
}
