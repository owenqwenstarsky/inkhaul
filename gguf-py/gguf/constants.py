from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import Any

#
# constants
#

GGUF_MAGIC             = 0x46554747  # "GGUF"
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_QUANT_VERSION     = 2  # GGML_QNT_VERSION from ggml.h
GGML_MAX_DIMS          = 4  # GGML_MAX_DIMS from ggml.h

#
# metadata keys
#


class Keys:
    class General:
        TYPE                       = "general.type"
        ARCHITECTURE               = "general.architecture"
        QUANTIZATION_VERSION       = "general.quantization_version"
        ALIGNMENT                  = "general.alignment"
        FILE_TYPE                  = "general.file_type"

        # Per-tensor extra options (tensor name array + parallel option arrays, e.g. prec_a4).
        TENSOR_EXTRA_NAME          = "general.tensor_extra.name"
        TENSOR_EXTRA_PREC_A4       = "general.tensor_extra.prec_a4"

        # Recommended Sampler Parameters
        SAMPLING_SEQUENCE           = "general.sampling.sequence"
        SAMPLING_TOP_K              = "general.sampling.top_k"
        SAMPLING_TOP_P              = "general.sampling.top_p"
        SAMPLING_MIN_P              = "general.sampling.min_p"
        SAMPLING_XTC_PROBABILITY    = "general.sampling.xtc_probability"
        SAMPLING_XTC_THRESHOLD      = "general.sampling.xtc_threshold"
        SAMPLING_TEMP               = "general.sampling.temp"
        SAMPLING_PENALTY_LAST_N     = "general.sampling.penalty_last_n"
        SAMPLING_PENALTY_REPEAT     = "general.sampling.penalty_repeat"
        SAMPLING_MIROSTAT           = "general.sampling.mirostat"
        SAMPLING_MIROSTAT_TAU       = "general.sampling.mirostat_tau"
        SAMPLING_MIROSTAT_ETA       = "general.sampling.mirostat_eta"

        # Authorship Metadata
        NAME                       = "general.name"
        AUTHOR                     = "general.author"
        VERSION                    = "general.version"
        ORGANIZATION               = "general.organization"

        FINETUNE                   = "general.finetune"
        BASENAME                   = "general.basename"

        DESCRIPTION                = "general.description"
        QUANTIZED_BY               = "general.quantized_by"

        SIZE_LABEL                 = "general.size_label"

        # Licensing details
        LICENSE                    = "general.license"
        LICENSE_NAME               = "general.license.name"
        LICENSE_LINK               = "general.license.link"

        # Typically represents the converted GGUF repo (Unless native)
        URL                        = "general.url" # Model Website/Paper
        DOI                        = "general.doi"
        UUID                       = "general.uuid"
        REPO_URL                   = "general.repo_url" # Model Source Repository (git/svn/etc...)

        # Model Source during conversion
        SOURCE_URL                 = "general.source.url" # Model Website/Paper
        SOURCE_DOI                 = "general.source.doi"
        SOURCE_UUID                = "general.source.uuid"
        SOURCE_REPO_URL            = "general.source.repo_url" # Model Source Repository (git/svn/etc...)

        # Base Model Source. There can be more than one source if it's a merged
        # model like with 'Mistral-7B-Merge-14-v0.1'. This will assist in
        # tracing linage of models as it is finetuned or merged over time.
        BASE_MODEL_COUNT           = "general.base_model.count"
        BASE_MODEL_NAME            = "general.base_model.{id}.name"
        BASE_MODEL_AUTHOR          = "general.base_model.{id}.author"
        BASE_MODEL_VERSION         = "general.base_model.{id}.version"
        BASE_MODEL_ORGANIZATION    = "general.base_model.{id}.organization"
        BASE_MODEL_DESCRIPTION     = "general.base_model.{id}.description"
        BASE_MODEL_URL             = "general.base_model.{id}.url" # Model Website/Paper
        BASE_MODEL_DOI             = "general.base_model.{id}.doi"
        BASE_MODEL_UUID            = "general.base_model.{id}.uuid"
        BASE_MODEL_REPO_URL        = "general.base_model.{id}.repo_url" # Model Source Repository (git/svn/etc...)

        # Dataset Source
        DATASET_COUNT           = "general.dataset.count"
        DATASET_NAME            = "general.dataset.{id}.name"
        DATASET_AUTHOR          = "general.dataset.{id}.author"
        DATASET_VERSION         = "general.dataset.{id}.version"
        DATASET_ORGANIZATION    = "general.dataset.{id}.organization"
        DATASET_DESCRIPTION     = "general.dataset.{id}.description"
        DATASET_URL             = "general.dataset.{id}.url" # Model Website/Paper
        DATASET_DOI             = "general.dataset.{id}.doi"
        DATASET_UUID            = "general.dataset.{id}.uuid"
        DATASET_REPO_URL        = "general.dataset.{id}.repo_url" # Model Source Repository (git/svn/etc...)

        # Array based KV stores
        TAGS                       = "general.tags"
        LANGUAGES                  = "general.languages"

    class LLM:
        VOCAB_SIZE                        = "{arch}.vocab_size"
        VOCAB_SIZE_UNPADDED               = "{arch}.vocab_size_unpadded"
        CONTEXT_LENGTH                    = "{arch}.context_length"
        EMBEDDING_LENGTH                  = "{arch}.embedding_length"
        EMBEDDING_LENGTH_OUT              = "{arch}.embedding_length_out"
        BLOCK_COUNT                       = "{arch}.block_count"
        FEED_FORWARD_LENGTH               = "{arch}.feed_forward_length"
        EXPERT_FEED_FORWARD_LENGTH        = "{arch}.expert_feed_forward_length"
        EXPERT_SHARED_FEED_FORWARD_LENGTH = "{arch}.expert_shared_feed_forward_length"
        TENSOR_DATA_LAYOUT                = "{arch}.tensor_data_layout"
        EXPERT_COUNT                      = "{arch}.expert_count"
        EXPERT_USED_COUNT                 = "{arch}.expert_used_count"
        EXPERT_GROUP_COUNT                = "{arch}.expert_group_count"
        EXPERT_GROUP_USED_COUNT           = "{arch}.expert_group_used_count"
        EXPERT_WEIGHTS_SCALE              = "{arch}.expert_weights_scale"
        EXPERT_GATING_FUNC                = "{arch}.expert_gating_func"
        NEXTN_PREDICT_LAYERS              = "{arch}.nextn_predict_layers"
        NUM_DEEPSTACK_LAYERS              = "{arch}.n_deepstack_layers"
        POOLING_TYPE                      = "{arch}.pooling_type"
        LOGIT_SCALE                       = "{arch}.logit_scale"
        DECODER_START_TOKEN_ID            = "{arch}.decoder_start_token_id"
        ATTN_LOGIT_SOFTCAPPING            = "{arch}.attn_logit_softcapping"
        FINAL_LOGIT_SOFTCAPPING           = "{arch}.final_logit_softcapping"
        EMBEDDING_SCALE                   = "{arch}.embedding_scale"
        FULL_ATTENTION_INTERVAL           = "{arch}.full_attention_interval"
        EMBD_LENGTH_PER_LAYER_INP         = "{arch}.embedding_length_per_layer_input"
        SWIGLU_CLAMP_EXP                  = "{arch}.swiglu_clamp_exp"
        SWIGLU_CLAMP_SHEXP                = "{arch}.swiglu_clamp_shexp"
        SWIGLU_CLAMP                      = "{arch}.swiglu_clamp"
        HIDDEN_ACT                        = "{arch}.hidden_activation"
        TARGET_LAYERS                     = "{arch}.target_layers"
        BLOCK_SIZE                        = "{arch}.block_size"
        SAMPLE_FROM_ANCHOR                = "{arch}.sample_from_anchor"
        HAS_CONFIDENCE_HEAD               = "{arch}.has_confidence_head"
        CLASSIFIER_FREE_GUIDANCE          = "{arch}.classifier_free_guidance"

    class Attention:
        HEAD_COUNT                   = "{arch}.attention.head_count"
        HEAD_COUNT_KV                = "{arch}.attention.head_count_kv"
        MAX_ALIBI_BIAS               = "{arch}.attention.max_alibi_bias"
        CLAMP_KQV                    = "{arch}.attention.clamp_kqv"
        KEY_LENGTH                   = "{arch}.attention.key_length"
        VALUE_LENGTH                 = "{arch}.attention.value_length"
        LAYERNORM_EPS                = "{arch}.attention.layer_norm_epsilon"
        LAYERNORM_RMS_EPS            = "{arch}.attention.layer_norm_rms_epsilon"
        GROUPNORM_EPS                = "{arch}.attention.group_norm_epsilon"
        GROUPNORM_GROUPS             = "{arch}.attention.group_norm_groups"
        CAUSAL                       = "{arch}.attention.causal"
        REL_BUCKETS_COUNT            = "{arch}.attention.relative_buckets_count"
        SLIDING_WINDOW               = "{arch}.attention.sliding_window"
        SCALE                        = "{arch}.attention.scale"
        VALUE_SCALE                  = "{arch}.attention.value_scale"
        KEY_LENGTH_MLA               = "{arch}.attention.key_length_mla"
        VALUE_LENGTH_MLA             = "{arch}.attention.value_length_mla"
        KEY_LENGTH_SWA               = "{arch}.attention.key_length_swa"
        VALUE_LENGTH_SWA             = "{arch}.attention.value_length_swa"
        SHARED_KV_LAYERS             = "{arch}.attention.shared_kv_layers"
        KV_SOURCE_MAP                = "{arch}.attention.kv_source_map"
        SLIDING_WINDOW_PATTERN       = "{arch}.attention.sliding_window_pattern"
        RECURRENT_LAYERS             = "{arch}.attention.recurrent_layers"
        ROPE_PATTERN                 = "{arch}.attention.rope_pattern"
        RELATIVE_LENGTH              = "{arch}.attention.relative_length"
        RELATIVE_EXTENT              = "{arch}.attention.relative_extent"
        QK_CLIP                      = "{arch}.attention.qk_clip"
        RELATIVE_CLIP                = "{arch}.attention.relative_clip"
        LOG_SCALE_FLOOR              = "{arch}.attention.log_scale_floor"
        LOG_SCALE_ALPHA              = "{arch}.attention.log_scale_alpha"

        class Indexer:
            TOP_K      = "{arch}.attention.indexer.top_k"

    class HyperConnection:
        COUNT                = "{arch}.hyper_connection.count"

    class PerLayerEmbedding:
        LAYERS             = "{arch}.ple.layers"
        NGRAM_SIZE         = "{arch}.ple.ngram_size"
        HEADS_PER_NGRAM    = "{arch}.ple.heads_per_ngram"
        CONV_KERNEL        = "{arch}.ple.conv_kernel"
        LAYER_MULTIPLIERS  = "{arch}.ple.layer_multipliers"
        HEAD_OFFSETS       = "{arch}.ple.head_offsets"
        HEAD_VOCAB_SIZES   = "{arch}.ple.head_vocab_sizes"
        EOS_TOKEN_ID       = "{arch}.ple.eos_token_id"

    class Rope:
        DIMENSION_COUNT           = "{arch}.rope.dimension_count"
        DIMENSION_COUNT_SWA       = "{arch}.rope.dimension_count_swa"
        DIMENSION_SECTIONS        = "{arch}.rope.dimension_sections"
        SECTION_ORDER             = "{arch}.rope.section_order"
        FREQ_BASE                 = "{arch}.rope.freq_base"
        FREQ_BASE_SWA             = "{arch}.rope.freq_base_swa"
        SCALING_TYPE              = "{arch}.rope.scaling.type"
        SCALING_FACTOR            = "{arch}.rope.scaling.factor"
        SCALING_ATTN_FACTOR       = "{arch}.rope.scaling.attn_factor"
        SCALING_ORIG_CTX_LEN      = "{arch}.rope.scaling.original_context_length"
        SCALING_FINETUNED         = "{arch}.rope.scaling.finetuned"
        SCALING_YARN_TRUNCATE     = "{arch}.rope.scaling.yarn_truncate"
        SCALING_YARN_LOG_MUL      = "{arch}.rope.scaling.yarn_log_multiplier"
        SCALING_YARN_EXT_FACTOR   = "{arch}.rope.scaling.yarn_ext_factor"
        SCALING_YARN_ATTN_FACTOR  = "{arch}.rope.scaling.yarn_attn_factor"
        SCALING_YARN_BETA_FAST    = "{arch}.rope.scaling.yarn_beta_fast"
        SCALING_YARN_BETA_SLOW    = "{arch}.rope.scaling.yarn_beta_slow"

    class Activation:
        SITU_BETA        = "{arch}.activation.situ_beta"
        SITU_LINEAR_BETA = "{arch}.activation.situ_linear_beta"

    class AttnRes:
        BLOCK_SIZE = "{arch}.attn_res.block_size"

    class Split:
        LLM_KV_SPLIT_NO            = "split.no"
        LLM_KV_SPLIT_COUNT         = "split.count"
        LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count"

    class Loop:
        START     = "{arch}.loop.start"
        END       = "{arch}.loop.end"
        COUNT     = "{arch}.loop.count"
        GATE_RANK = "{arch}.loop.gate_rank"
        VARYING   = "{arch}.loop.varying"

    class SSM:
        CONV_KERNEL    = "{arch}.ssm.conv_kernel"
        INNER_SIZE     = "{arch}.ssm.inner_size"
        STATE_SIZE     = "{arch}.ssm.state_size"
        TIME_STEP_RANK = "{arch}.ssm.time_step_rank"
        GROUP_COUNT    = "{arch}.ssm.group_count"

    class KDA:
        HEAD_DIM         = "{arch}.kda.head_dim"

    class Classifier:
        OUTPUT_LABELS = "{arch}.classifier.output_labels"
        POOLING_TYPE  = "{arch}.classifier.pooling_type"
        ACTIVATION    = "{arch}.classifier.activation"

    class ShortConv:
        L_CACHE = "{arch}.shortconv.l_cache"

    class Decision:
        # note: single-use-case keys can be hard-coded in cpp code
        LABELS             = "{arch}.decision.labels"  # the text of the token read for each option, in order

    class Tokenizer:
        MODEL                = "tokenizer.ggml.model"
        PRE                  = "tokenizer.ggml.pre"
        LIST                 = "tokenizer.ggml.tokens"
        TOKEN_TYPE           = "tokenizer.ggml.token_type"
        TOKEN_TYPE_COUNT     = "tokenizer.ggml.token_type_count"  # for BERT-style token types
        SCORES               = "tokenizer.ggml.scores"
        MERGES               = "tokenizer.ggml.merges"
        BOS_ID               = "tokenizer.ggml.bos_token_id"
        EOS_ID               = "tokenizer.ggml.eos_token_id"
        EOT_ID               = "tokenizer.ggml.eot_token_id"
        EOM_ID               = "tokenizer.ggml.eom_token_id"
        UNK_ID               = "tokenizer.ggml.unknown_token_id"
        SEP_ID               = "tokenizer.ggml.seperator_token_id"
        PAD_ID               = "tokenizer.ggml.padding_token_id"
        MASK_ID              = "tokenizer.ggml.mask_token_id"
        ADD_BOS              = "tokenizer.ggml.add_bos_token"
        ADD_EOS              = "tokenizer.ggml.add_eos_token"
        ADD_SEP              = "tokenizer.ggml.add_sep_token"
        ADD_PREFIX           = "tokenizer.ggml.add_space_prefix"
        REMOVE_EXTRA_WS      = "tokenizer.ggml.remove_extra_whitespaces"
        CLEAN_SPACES         = "tokenizer.ggml.clean_spaces"
        SUPPRESS_TOKENS      = "tokenizer.ggml.suppress_tokens"
        HF_JSON              = "tokenizer.huggingface.json"
        RWKV                 = "tokenizer.rwkv.world"
        CHAT_TEMPLATE        = "tokenizer.chat_template"
        CHAT_TEMPLATE_N      = "tokenizer.chat_template.{name}"
        CHAT_TEMPLATES       = "tokenizer.chat_templates"
        # Normalizer constants
        NORMALIZER_NFC           = "tokenizer.ggml.normalizer.nfc"
        NORMALIZER_LOWERCASE     = "tokenizer.ggml.normalizer.lowercase"
        NORMALIZER_STRIP_ACCENTS = "tokenizer.ggml.normalizer.strip_accents"
        # Metaspace constants
        METASPACE_PREPEND_SCHEME = "tokenizer.ggml.metaspace.prepend_scheme"
        # FIM/Infill special tokens constants
        FIM_PRE_ID           = "tokenizer.ggml.fim_pre_token_id"
        FIM_SUF_ID           = "tokenizer.ggml.fim_suf_token_id"
        FIM_MID_ID           = "tokenizer.ggml.fim_mid_token_id"
        FIM_PAD_ID           = "tokenizer.ggml.fim_pad_token_id"
        FIM_REP_ID           = "tokenizer.ggml.fim_rep_token_id"
        FIM_SEP_ID           = "tokenizer.ggml.fim_sep_token_id"
        # deprecated:
        PREFIX_ID            = "tokenizer.ggml.prefix_token_id"
        SUFFIX_ID            = "tokenizer.ggml.suffix_token_id"
        MIDDLE_ID            = "tokenizer.ggml.middle_token_id"

    class Adapter:
        TYPE                    = "adapter.type"
        LORA_ALPHA              = "adapter.lora.alpha"
        LORA_TASK_NAME          = "adapter.lora.task_name"
        LORA_PROMPT_PREFIX      = "adapter.lora.prompt_prefix"
        ALORA_INVOCATION_TOKENS = "adapter.alora.invocation_tokens"

    class IMatrix:
        CHUNK_COUNT = "imatrix.chunk_count"
        CHUNK_SIZE  = "imatrix.chunk_size"
        DATASETS    = "imatrix.datasets"

    class Clip:
        PROJECTOR_TYPE        = "clip.projector_type"
        HAS_VISION_ENCODER    = "clip.has_vision_encoder"
        HAS_AUDIO_ENCODER     = "clip.has_audio_encoder"
        HAS_GEN_AUDIO_ENCODER = "clip.has_gen_audio_encoder"

    class ClipVision:
        PROJECTOR_TYPE        = "clip.vision.projector_type" # for mixed modality models
        IMAGE_SIZE            = "clip.vision.image_size"
        PATCH_SIZE            = "clip.vision.patch_size"
        EMBEDDING_LENGTH      = "clip.vision.embedding_length"
        FEED_FORWARD_LENGTH   = "clip.vision.feed_forward_length"
        PROJECTION_DIM        = "clip.vision.projection_dim"
        BLOCK_COUNT           = "clip.vision.block_count"
        IMAGE_MEAN            = "clip.vision.image_mean"
        IMAGE_STD             = "clip.vision.image_std"
        MAX_SLICE_NUMS        = "clip.vision.max_slice_nums"
        SPATIAL_MERGE_SIZE    = "clip.vision.spatial_merge_size"
        USE_GELU              = "clip.use_gelu"
        USE_SILU              = "clip.use_silu"
        IS_DEEPSTACK_LAYERS   = "clip.vision.is_deepstack_layers"
        FEATURE_LAYERS        = "clip.vision.feature_layer" # Granite4 Vision
        IMAGE_GRID_PINPOINTS  = "clip.vision.image_grid_pinpoints" # Granite4 Vision

        class Attention:
            HEAD_COUNT      = "clip.vision.attention.head_count"
            HEAD_COUNT_KV   = "clip.vision.attention.head_count_kv" # used by mimovl (GQA)
            HEAD_DIM        = "clip.vision.attention.head_dim" # set when qkv width != n_embd
            LAYERNORM_EPS   = "clip.vision.attention.layer_norm_epsilon"

    class ClipAudio:
        PROJECTOR_TYPE      = "clip.audio.projector_type" # for mixed modality models
        NUM_MEL_BINS        = "clip.audio.num_mel_bins"
        EMBEDDING_LENGTH    = "clip.audio.embedding_length"
        FEED_FORWARD_LENGTH = "clip.audio.feed_forward_length"
        PROJECTION_DIM      = "clip.audio.projection_dim"
        BLOCK_COUNT         = "clip.audio.block_count"
        FEATURE_LAYERS      = "clip.audio.feature_layer" # Granite Speech Plus

        class Attention:
            HEAD_COUNT      = "clip.audio.attention.head_count"
            HEAD_COUNT_KV   = "clip.audio.attention.head_count_kv"
            HEAD_DIM        = "clip.audio.attention.head_dim" # set when qkv width != n_embd
            LAYERNORM_EPS   = "clip.audio.attention.layer_norm_epsilon"

    class ClipGenAudio:
        PROJECTOR_TYPE      = "clip.gen.audio.projector_type" # for mixed modality models
        # name of the weight variant, for settings that are not in the checkpoint
        MODEL_VARIANT       = "clip.gen.audio.model_variant"
        DEFAULT_VOICE       = "clip.gen.audio.default_voice"

    class Diffusion:
        SHIFT_LOGITS        = "diffusion.shift_logits"


#
# recommended mapping of model tensor names for storage in gguf
#


class GGUFType:
    MODEL   = "model"
    ADAPTER = "adapter"
    IMATRIX = "imatrix"
    MMPROJ  = "mmproj" # dummy, unused for now


class MODEL_ARCH(IntEnum):
    MMPROJ           = auto() # dummy arch for mmproj
    DFLASH           = auto()
    INKLING2         = auto()
    INKLING2_LOOP    = auto()
    QWEN35           = auto()
    QWEN35MOE        = auto()


class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD           = auto()
    TOKEN_EMBD_NORM      = auto()
    OUTPUT               = auto()
    OUTPUT_NORM          = auto()
    OUTPUT_RES_QUERY     = auto()
    ROPE_FREQS           = auto()
    ATTN_Q               = auto()
    ATTN_K               = auto()
    ATTN_V               = auto()
    ATTN_QKV             = auto()
    ATTN_OUT             = auto()
    ATTN_NORM            = auto()
    ATTN_POST_NORM       = auto()
    ATTN_SINKS           = auto()
    ATTN_GATE            = auto()
    ATTN_REL             = auto()
    ATTN_REL_BIAS        = auto()
    ATTN_Q_CLIP          = auto()
    ATTN_REL_CLIP        = auto()
    ATTN_SINK_PROJ       = auto()
    ATTN_Q_INP_NORM      = auto()
    ATTN_K_INP_NORM      = auto()
    ATTN_V_INP_NORM      = auto()
    ATTN_K_CONV          = auto()
    ATTN_V_CONV          = auto()
    ATTN_K_CONV_PROJ     = auto()
    ATTN_V_CONV_PROJ     = auto()
    ATTN_POST_CONV       = auto()
    ATTN_RES_QUERY       = auto()
    FFN_GATE_INP         = auto()
    FFN_GATE_INP_SHEXP   = auto()
    FFN_NORM             = auto()
    FFN_POST_NORM        = auto()
    FFN_GATE             = auto()
    FFN_DOWN             = auto()
    FFN_UP               = auto()
    FFN_GATE_EXP         = auto()
    FFN_DOWN_EXP         = auto()
    FFN_UP_EXP           = auto()
    FFN_GATE_UP_EXP      = auto()
    FFN_GATE_SHEXP       = auto()
    FFN_DOWN_SHEXP       = auto()
    FFN_UP_SHEXP         = auto()
    FFN_EXP_PROBS_B      = auto()
    FFN_POST_CONV        = auto()
    FFN_RES_QUERY        = auto()
    LOOP_MSG_QUERY       = auto()
    LOOP_Q_CTRL_DOWN     = auto()
    LOOP_Q_CTRL_GAIN     = auto()
    LOOP_Q_CTRL_FORGET   = auto()
    LOOP_K_CTRL_DOWN     = auto()
    LOOP_K_CTRL_GAIN     = auto()
    LOOP_K_CTRL_FORGET   = auto()
    LOOP_V_CTRL_DOWN     = auto()
    LOOP_V_CTRL_GAIN     = auto()
    LOOP_V_CTRL_FORGET   = auto()
    LOOP_FFN_CTRL_DOWN   = auto()
    LOOP_FFN_CTRL_GAIN   = auto()
    LOOP_FFN_CTRL_FORGET = auto()
    ATTN_Q_NORM          = auto()
    ATTN_K_NORM          = auto()
    LAYER_OUT_SCALE      = auto()
    ENC_OUTPUT_NORM      = auto()
    SSM_CONV1D           = auto()
    SSM_DT               = auto()
    SSM_A                = auto()
    SSM_NORM             = auto()
    SSM_OUT              = auto()
    SSM_ALPHA            = auto() # qwen3.5
    SSM_BETA             = auto() # qwen3.5
    ATTN_QKV_GATE        = auto() # qwen35, [attn_qkv | attn_gate]
    SSM_BETA_ALPHA_CAT   = auto() # qwen35, [ssm_beta | ssm_alpha]
    # vision
    V_MMPROJ             = auto()
    V_ENC_EMBD_CLS       = auto()
    V_ENC_EMBD_PATCH     = auto()
    V_ENC_EMBD_CLS_POS   = auto()
    V_ENC_EMBD_POS       = auto()
    V_ENC_INPUT_NORM     = auto()
    V_ENC_ATTN_Q         = auto()
    V_ENC_ATTN_Q_NORM    = auto()
    V_ENC_ATTN_K         = auto()
    V_ENC_ATTN_K_NORM    = auto()
    V_ENC_ATTN_V         = auto()
    V_ENC_ATTN_QKV       = auto()
    V_ENC_ATTN_O         = auto()
    V_ENC_POST_ATTN_NORM = auto()
    V_ENC_FFN_UP         = auto()
    V_ENC_FFN_GATE       = auto()
    V_ENC_FFN_DOWN       = auto()
    V_LAYER_SCALE_1      = auto()
    V_LAYER_SCALE_2      = auto()
    V_PRE_NORM           = auto()
    V_POST_NORM          = auto()
    V_ENC_CONV_STEM      = auto() # gemma3n
    V_ENC_CONV_STEM_NORM = auto() # gemma3n
    V_ENC_MSFA_EXP       = auto() # gemma3n
    V_ENC_MSFA_EXP_NORM  = auto() # gemma3n
    V_ENC_MSFA_PROJ      = auto() # gemma3n
    V_ENC_MSFA_PROJ_NORM = auto() # gemma3n
    V_ENC_MSFA_NORM      = auto() # gemma3n
    V_DS_NORM            = auto() # qwen3vl
    V_DS_FC1             = auto() # qwen3vl
    V_DS_FC2             = auto() # qwen3vl
    V_MERGER_LN1         = auto() # minicpmv4_6
    V_MERGER_ATTN_Q      = auto() # minicpmv4_6
    V_MERGER_ATTN_K      = auto() # minicpmv4_6
    V_MERGER_ATTN_V      = auto() # minicpmv4_6
    V_MERGER_ATTN_O      = auto() # minicpmv4_6
    V_MERGER_DS_LN       = auto() # minicpmv4_6
    V_MERGER_DS_UP       = auto() # minicpmv4_6
    V_MERGER_DS_DOWN     = auto() # minicpmv4_6
    V_STD_BIAS           = auto() # gemma4
    V_STD_SCALE          = auto() # gemma4

    # dflash / dspark
    FC                     = auto()  # feature fusion layer
    D2T                    = auto()  # draft to target vocabulary mapping
    DSPARK_MARKOV_W1       = auto()  # markov head: prev-token embed
    DSPARK_MARKOV_W2       = auto()  # markov head: bias projection
    DSPARK_CONF_PROJ       = auto()  # confidence head
    DSPARK_JOINT_PROJ      = auto()  # rnn markov head


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.MMPROJ:           "clip", # dummy arch for mmproj files (tools/mtmd)
    MODEL_ARCH.DFLASH:           "dflash",
    MODEL_ARCH.INKLING2:         "inkling2",
    MODEL_ARCH.INKLING2_LOOP:    "inkling2_loop",
    MODEL_ARCH.QWEN35:           "qwen35",
    MODEL_ARCH.QWEN35MOE:        "qwen35moe",
}


TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:                "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM:           "token_embd_norm",
    MODEL_TENSOR.OUTPUT_NORM:               "output_norm",
    MODEL_TENSOR.OUTPUT:                    "output",
    MODEL_TENSOR.OUTPUT_RES_QUERY:          "output_res_query",
    MODEL_TENSOR.ROPE_FREQS:                "rope_freqs",
    MODEL_TENSOR.ATTN_NORM:                 "blk.{bid}.attn_norm",
    MODEL_TENSOR.ATTN_Q:                    "blk.{bid}.attn_q",
    MODEL_TENSOR.ATTN_K:                    "blk.{bid}.attn_k",
    MODEL_TENSOR.ATTN_V:                    "blk.{bid}.attn_v",
    MODEL_TENSOR.ATTN_QKV:                  "blk.{bid}.attn_qkv",
    MODEL_TENSOR.ATTN_OUT:                  "blk.{bid}.attn_output",
    MODEL_TENSOR.ATTN_SINKS:                "blk.{bid}.attn_sinks",
    MODEL_TENSOR.ATTN_GATE:                 "blk.{bid}.attn_gate",
    MODEL_TENSOR.ATTN_REL:                  "blk.{bid}.attn_rel",
    MODEL_TENSOR.ATTN_REL_BIAS:             "blk.{bid}.attn_rel_bias",
    MODEL_TENSOR.ATTN_Q_CLIP:               "blk.{bid}.attn_q_clip",
    MODEL_TENSOR.ATTN_REL_CLIP:             "blk.{bid}.attn_rel_clip",
    MODEL_TENSOR.ATTN_SINK_PROJ:            "blk.{bid}.attn_sink_proj",
    MODEL_TENSOR.ATTN_Q_INP_NORM:           "blk.{bid}.attn_q_inp_norm",
    MODEL_TENSOR.ATTN_K_INP_NORM:           "blk.{bid}.attn_k_inp_norm",
    MODEL_TENSOR.ATTN_V_INP_NORM:           "blk.{bid}.attn_v_inp_norm",
    MODEL_TENSOR.ATTN_K_CONV:               "blk.{bid}.attn_k_conv",
    MODEL_TENSOR.ATTN_V_CONV:               "blk.{bid}.attn_v_conv",
    MODEL_TENSOR.ATTN_K_CONV_PROJ:          "blk.{bid}.attn_k_conv_proj",
    MODEL_TENSOR.ATTN_V_CONV_PROJ:          "blk.{bid}.attn_v_conv_proj",
    MODEL_TENSOR.ATTN_POST_CONV:            "blk.{bid}.attn_post_conv",
    MODEL_TENSOR.ATTN_RES_QUERY:            "blk.{bid}.attn_res_query",
    MODEL_TENSOR.ATTN_Q_NORM:               "blk.{bid}.attn_q_norm",
    MODEL_TENSOR.ATTN_K_NORM:               "blk.{bid}.attn_k_norm",
    MODEL_TENSOR.ATTN_POST_NORM:            "blk.{bid}.post_attention_norm",
    MODEL_TENSOR.FFN_GATE_INP:              "blk.{bid}.ffn_gate_inp",
    MODEL_TENSOR.FFN_GATE_INP_SHEXP:        "blk.{bid}.ffn_gate_inp_shexp",
    MODEL_TENSOR.FFN_NORM:                  "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_POST_NORM:             "blk.{bid}.post_ffw_norm",
    MODEL_TENSOR.FFN_GATE:                  "blk.{bid}.ffn_gate",
    MODEL_TENSOR.FFN_DOWN:                  "blk.{bid}.ffn_down",
    MODEL_TENSOR.FFN_UP:                    "blk.{bid}.ffn_up",
    MODEL_TENSOR.FFN_GATE_EXP:              "blk.{bid}.ffn_gate_exps",
    MODEL_TENSOR.FFN_DOWN_EXP:              "blk.{bid}.ffn_down_exps",
    MODEL_TENSOR.FFN_UP_EXP:                "blk.{bid}.ffn_up_exps",
    MODEL_TENSOR.FFN_GATE_UP_EXP:           "blk.{bid}.ffn_gate_up_exps",
    MODEL_TENSOR.FFN_GATE_SHEXP:            "blk.{bid}.ffn_gate_shexp",
    MODEL_TENSOR.FFN_DOWN_SHEXP:            "blk.{bid}.ffn_down_shexp",
    MODEL_TENSOR.FFN_UP_SHEXP:              "blk.{bid}.ffn_up_shexp",
    MODEL_TENSOR.FFN_EXP_PROBS_B:           "blk.{bid}.exp_probs_b",
    MODEL_TENSOR.FFN_POST_CONV:             "blk.{bid}.ffn_post_conv",
    MODEL_TENSOR.FFN_RES_QUERY:             "blk.{bid}.ffn_res_query",
    MODEL_TENSOR.LOOP_MSG_QUERY:            "blk.{bid}.loop_msg_query",
    MODEL_TENSOR.LOOP_Q_CTRL_DOWN:          "blk.{bid}.loop_q_ctrl_down",
    MODEL_TENSOR.LOOP_Q_CTRL_GAIN:          "blk.{bid}.loop_q_ctrl_gain",
    MODEL_TENSOR.LOOP_Q_CTRL_FORGET:        "blk.{bid}.loop_q_ctrl_forget",
    MODEL_TENSOR.LOOP_K_CTRL_DOWN:          "blk.{bid}.loop_k_ctrl_down",
    MODEL_TENSOR.LOOP_K_CTRL_GAIN:          "blk.{bid}.loop_k_ctrl_gain",
    MODEL_TENSOR.LOOP_K_CTRL_FORGET:        "blk.{bid}.loop_k_ctrl_forget",
    MODEL_TENSOR.LOOP_V_CTRL_DOWN:          "blk.{bid}.loop_v_ctrl_down",
    MODEL_TENSOR.LOOP_V_CTRL_GAIN:          "blk.{bid}.loop_v_ctrl_gain",
    MODEL_TENSOR.LOOP_V_CTRL_FORGET:        "blk.{bid}.loop_v_ctrl_forget",
    MODEL_TENSOR.LOOP_FFN_CTRL_DOWN:        "blk.{bid}.loop_ffn_ctrl_down",
    MODEL_TENSOR.LOOP_FFN_CTRL_GAIN:        "blk.{bid}.loop_ffn_ctrl_gain",
    MODEL_TENSOR.LOOP_FFN_CTRL_FORGET:      "blk.{bid}.loop_ffn_ctrl_forget",
    MODEL_TENSOR.LAYER_OUT_SCALE:           "blk.{bid}.layer_output_scale",
    MODEL_TENSOR.ENC_OUTPUT_NORM:           "enc.output_norm",
    MODEL_TENSOR.SSM_CONV1D:                "blk.{bid}.ssm_conv1d",
    MODEL_TENSOR.SSM_DT:                    "blk.{bid}.ssm_dt",
    MODEL_TENSOR.SSM_A:                     "blk.{bid}.ssm_a",
    MODEL_TENSOR.SSM_NORM:                  "blk.{bid}.ssm_norm",
    MODEL_TENSOR.SSM_OUT:                   "blk.{bid}.ssm_out",
    MODEL_TENSOR.SSM_ALPHA:                 "blk.{bid}.ssm_alpha",
    MODEL_TENSOR.SSM_BETA:                  "blk.{bid}.ssm_beta",
    MODEL_TENSOR.ATTN_QKV_GATE:             "blk.{bid}.attn_qkv_gate",
    MODEL_TENSOR.SSM_BETA_ALPHA_CAT:        "blk.{bid}.ssm_beta_alpha",
    # vision
    MODEL_TENSOR.V_MMPROJ:                  "mm.{bid}",
    MODEL_TENSOR.V_ENC_EMBD_CLS:            "v.class_embd",
    MODEL_TENSOR.V_ENC_EMBD_PATCH:          "v.patch_embd",
    MODEL_TENSOR.V_ENC_EMBD_CLS_POS:        "v.class_position_embd",
    MODEL_TENSOR.V_ENC_EMBD_POS:            "v.position_embd",
    MODEL_TENSOR.V_ENC_ATTN_Q:              "v.blk.{bid}.attn_q",
    MODEL_TENSOR.V_ENC_ATTN_Q_NORM:         "v.blk.{bid}.attn_q_norm",
    MODEL_TENSOR.V_ENC_ATTN_K:              "v.blk.{bid}.attn_k",
    MODEL_TENSOR.V_ENC_ATTN_K_NORM:         "v.blk.{bid}.attn_k_norm",
    MODEL_TENSOR.V_ENC_ATTN_V:              "v.blk.{bid}.attn_v",
    MODEL_TENSOR.V_ENC_ATTN_QKV:            "v.blk.{bid}.attn_qkv",
    MODEL_TENSOR.V_ENC_INPUT_NORM:          "v.blk.{bid}.ln1",
    MODEL_TENSOR.V_ENC_ATTN_O:              "v.blk.{bid}.attn_out",
    MODEL_TENSOR.V_ENC_POST_ATTN_NORM:      "v.blk.{bid}.ln2",
    MODEL_TENSOR.V_ENC_FFN_UP:              "v.blk.{bid}.ffn_up",
    MODEL_TENSOR.V_ENC_FFN_GATE:            "v.blk.{bid}.ffn_gate",
    MODEL_TENSOR.V_ENC_FFN_DOWN:            "v.blk.{bid}.ffn_down",
    MODEL_TENSOR.V_LAYER_SCALE_1:           "v.blk.{bid}.ls1",
    MODEL_TENSOR.V_LAYER_SCALE_2:           "v.blk.{bid}.ls2",
    MODEL_TENSOR.V_PRE_NORM:                "v.pre_ln",
    MODEL_TENSOR.V_POST_NORM:               "v.post_ln",
    MODEL_TENSOR.V_ENC_CONV_STEM:           "v.conv_stem.conv",         # gemma3n
    MODEL_TENSOR.V_ENC_CONV_STEM_NORM:      "v.conv_stem.bn",           # gemma3n
    MODEL_TENSOR.V_ENC_MSFA_EXP:            "v.msfa.ffn.pw_exp.conv",   # gemma3n
    MODEL_TENSOR.V_ENC_MSFA_EXP_NORM:       "v.msfa.ffn.pw_exp.bn",     # gemma3n
    MODEL_TENSOR.V_ENC_MSFA_PROJ:           "v.msfa.ffn.pw_proj.conv",  # gemma3n
    MODEL_TENSOR.V_ENC_MSFA_PROJ_NORM:      "v.msfa.ffn.pw_proj.bn",    # gemma3n
    MODEL_TENSOR.V_ENC_MSFA_NORM:           "v.msfa.norm",              # gemma3n
    MODEL_TENSOR.V_DS_NORM:                 "v.deepstack.{bid}.norm",
    MODEL_TENSOR.V_DS_FC1:                  "v.deepstack.{bid}.fc1",
    MODEL_TENSOR.V_DS_FC2:                  "v.deepstack.{bid}.fc2",
    MODEL_TENSOR.V_MERGER_LN1:              "v.vit_merger.ln1",
    MODEL_TENSOR.V_MERGER_ATTN_Q:           "v.vit_merger.attn_q",
    MODEL_TENSOR.V_MERGER_ATTN_K:           "v.vit_merger.attn_k",
    MODEL_TENSOR.V_MERGER_ATTN_V:           "v.vit_merger.attn_v",
    MODEL_TENSOR.V_MERGER_ATTN_O:           "v.vit_merger.attn_out",
    MODEL_TENSOR.V_MERGER_DS_LN:            "v.vit_merger.ds_ln",
    MODEL_TENSOR.V_MERGER_DS_UP:            "v.vit_merger.ds_ffn_up",
    MODEL_TENSOR.V_MERGER_DS_DOWN:          "v.vit_merger.ds_ffn_down",
    MODEL_TENSOR.V_STD_BIAS:                "v.std_bias", # gemma4
    MODEL_TENSOR.V_STD_SCALE:               "v.std_scale", # gemma4

    # dflash / dspark
    MODEL_TENSOR.FC:                      "fc",
    MODEL_TENSOR.DSPARK_MARKOV_W1:          "markov_w1",
    MODEL_TENSOR.DSPARK_MARKOV_W2:          "markov_w2",
    MODEL_TENSOR.DSPARK_CONF_PROJ:          "conf_proj",
    MODEL_TENSOR.DSPARK_JOINT_PROJ:         "joint_proj",
    MODEL_TENSOR.D2T:                       "d2t",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.MMPROJ: [
        MODEL_TENSOR.V_MMPROJ,
        MODEL_TENSOR.V_ENC_EMBD_CLS,
        MODEL_TENSOR.V_ENC_EMBD_PATCH,
        MODEL_TENSOR.V_ENC_EMBD_CLS_POS,
        MODEL_TENSOR.V_ENC_EMBD_POS,
        MODEL_TENSOR.V_ENC_INPUT_NORM,
        MODEL_TENSOR.V_ENC_ATTN_Q,
        MODEL_TENSOR.V_ENC_ATTN_Q_NORM,
        MODEL_TENSOR.V_ENC_ATTN_K,
        MODEL_TENSOR.V_ENC_ATTN_K_NORM,
        MODEL_TENSOR.V_ENC_ATTN_V,
        MODEL_TENSOR.V_ENC_ATTN_QKV,
        MODEL_TENSOR.V_ENC_ATTN_O,
        MODEL_TENSOR.V_ENC_POST_ATTN_NORM,
        MODEL_TENSOR.V_ENC_FFN_UP,
        MODEL_TENSOR.V_ENC_FFN_GATE,
        MODEL_TENSOR.V_ENC_FFN_DOWN,
        MODEL_TENSOR.V_LAYER_SCALE_1,
        MODEL_TENSOR.V_LAYER_SCALE_2,
        MODEL_TENSOR.V_PRE_NORM,
        MODEL_TENSOR.V_POST_NORM,
        MODEL_TENSOR.V_ENC_CONV_STEM,
        MODEL_TENSOR.V_ENC_CONV_STEM_NORM,
        MODEL_TENSOR.V_ENC_MSFA_EXP,
        MODEL_TENSOR.V_ENC_MSFA_EXP_NORM,
        MODEL_TENSOR.V_ENC_MSFA_PROJ,
        MODEL_TENSOR.V_ENC_MSFA_PROJ_NORM,
        MODEL_TENSOR.V_ENC_MSFA_NORM,
        MODEL_TENSOR.V_DS_NORM,
        MODEL_TENSOR.V_DS_FC1,
        MODEL_TENSOR.V_DS_FC2,
        MODEL_TENSOR.V_MERGER_LN1,
        MODEL_TENSOR.V_MERGER_ATTN_Q,
        MODEL_TENSOR.V_MERGER_ATTN_K,
        MODEL_TENSOR.V_MERGER_ATTN_V,
        MODEL_TENSOR.V_MERGER_ATTN_O,
        MODEL_TENSOR.V_MERGER_DS_LN,
        MODEL_TENSOR.V_MERGER_DS_UP,
        MODEL_TENSOR.V_MERGER_DS_DOWN,
        MODEL_TENSOR.V_STD_BIAS,
        MODEL_TENSOR.V_STD_SCALE,
    ],

    MODEL_ARCH.DFLASH: [
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_POST_NORM,
        MODEL_TENSOR.LAYER_OUT_SCALE,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_SINKS,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FC,
        MODEL_TENSOR.ENC_OUTPUT_NORM,
        MODEL_TENSOR.D2T,
        # optional DSpark heads
        MODEL_TENSOR.DSPARK_MARKOV_W1,
        MODEL_TENSOR.DSPARK_MARKOV_W2,
        MODEL_TENSOR.DSPARK_CONF_PROJ,
        MODEL_TENSOR.DSPARK_JOINT_PROJ,
    ],
    MODEL_ARCH.INKLING2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT_RES_QUERY,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_GATE,
        MODEL_TENSOR.ATTN_REL,
        MODEL_TENSOR.ATTN_REL_BIAS,
        MODEL_TENSOR.ATTN_Q_CLIP,
        MODEL_TENSOR.ATTN_REL_CLIP,
        MODEL_TENSOR.ATTN_SINK_PROJ,
        MODEL_TENSOR.ATTN_Q_INP_NORM,
        MODEL_TENSOR.ATTN_K_INP_NORM,
        MODEL_TENSOR.ATTN_V_INP_NORM,
        MODEL_TENSOR.ATTN_K_CONV,
        MODEL_TENSOR.ATTN_V_CONV,
        MODEL_TENSOR.ATTN_K_CONV_PROJ,
        MODEL_TENSOR.ATTN_V_CONV_PROJ,
        MODEL_TENSOR.ATTN_POST_CONV,
        MODEL_TENSOR.ATTN_RES_QUERY,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_UP_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_POST_CONV,
        MODEL_TENSOR.FFN_RES_QUERY,
    ],
    MODEL_ARCH.INKLING2_LOOP: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT_RES_QUERY,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_GATE,
        MODEL_TENSOR.ATTN_REL,
        MODEL_TENSOR.ATTN_REL_BIAS,
        MODEL_TENSOR.ATTN_Q_CLIP,
        MODEL_TENSOR.ATTN_REL_CLIP,
        MODEL_TENSOR.ATTN_SINK_PROJ,
        MODEL_TENSOR.ATTN_Q_INP_NORM,
        MODEL_TENSOR.ATTN_K_INP_NORM,
        MODEL_TENSOR.ATTN_V_INP_NORM,
        MODEL_TENSOR.ATTN_K_CONV,
        MODEL_TENSOR.ATTN_V_CONV,
        MODEL_TENSOR.ATTN_K_CONV_PROJ,
        MODEL_TENSOR.ATTN_V_CONV_PROJ,
        MODEL_TENSOR.ATTN_POST_CONV,
        MODEL_TENSOR.ATTN_RES_QUERY,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_UP_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_POST_CONV,
        MODEL_TENSOR.FFN_RES_QUERY,
        MODEL_TENSOR.LOOP_MSG_QUERY,
        MODEL_TENSOR.LOOP_Q_CTRL_DOWN,
        MODEL_TENSOR.LOOP_Q_CTRL_GAIN,
        MODEL_TENSOR.LOOP_Q_CTRL_FORGET,
        MODEL_TENSOR.LOOP_K_CTRL_DOWN,
        MODEL_TENSOR.LOOP_K_CTRL_GAIN,
        MODEL_TENSOR.LOOP_K_CTRL_FORGET,
        MODEL_TENSOR.LOOP_V_CTRL_DOWN,
        MODEL_TENSOR.LOOP_V_CTRL_GAIN,
        MODEL_TENSOR.LOOP_V_CTRL_FORGET,
        MODEL_TENSOR.LOOP_FFN_CTRL_DOWN,
        MODEL_TENSOR.LOOP_FFN_CTRL_GAIN,
        MODEL_TENSOR.LOOP_FFN_CTRL_FORGET,
    ],
    MODEL_ARCH.QWEN35: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.ATTN_GATE,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_QKV_GATE,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_NORM,
        MODEL_TENSOR.SSM_BETA,
        MODEL_TENSOR.SSM_ALPHA,
        MODEL_TENSOR.SSM_BETA_ALPHA_CAT,
        MODEL_TENSOR.SSM_OUT,
    ],
    MODEL_ARCH.QWEN35MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.ATTN_GATE,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_INP_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_GATE_UP_EXP,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_NORM,
        MODEL_TENSOR.SSM_BETA,
        MODEL_TENSOR.SSM_ALPHA,
        MODEL_TENSOR.SSM_OUT,
    ],
}


#
# types
#


class TokenType(IntEnum):
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6


class RopeScalingType(Enum):
    NONE     = 'none'
    LINEAR   = 'linear'
    YARN     = 'yarn'
    LONGROPE = 'longrope'


# M-RoPE: input position slot (t, y, x, z) that feeds each RoPE section, in section order
class RopeSectionOrder(Enum):
    TYXZ = 'tyxz' # default
    ZYXT = 'zyxt'


class PoolingType(IntEnum):
    NONE = 0
    MEAN = 1
    CLS  = 2
    LAST = 3
    RANK = 4


class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30
    TQ1_0   = 34
    TQ2_0   = 35
    MXFP4   = 39
    NVFP4   = 40
    Q1_0    = 41
    Q2_0    = 42
    F8_E4M3 = 43


class ExpertGatingFuncType(IntEnum):
    SOFTMAX       = 1
    SIGMOID       = 2
    SQRTSOFTPLUS  = 4


# TODO: add GGMLFileType from ggml_ftype in ggml.h


# from llama_ftype in llama.h
# ALL VALUES SHOULD BE THE SAME HERE AS THEY ARE OVER THERE.
class LlamaFileType(IntEnum):
    ALL_F32              = 0
    MOSTLY_F16           = 1   # except 1d tensors
    MOSTLY_Q4_0          = 2   # except 1d tensors
    MOSTLY_Q4_1          = 3   # except 1d tensors
    # MOSTLY_Q4_1_SOME_F16 = 4   # tok_embeddings.weight and output.weight are F16
    # MOSTLY_Q4_2        = 5   # support has been removed
    # MOSTLY_Q4_3        = 6   # support has been removed
    MOSTLY_Q8_0          = 7   # except 1d tensors
    MOSTLY_Q5_0          = 8   # except 1d tensors
    MOSTLY_Q5_1          = 9   # except 1d tensors
    MOSTLY_Q2_K          = 10  # except 1d tensors
    MOSTLY_Q3_K_S        = 11  # except 1d tensors
    MOSTLY_Q3_K_M        = 12  # except 1d tensors
    MOSTLY_Q3_K_L        = 13  # except 1d tensors
    MOSTLY_Q4_K_S        = 14  # except 1d tensors
    MOSTLY_Q4_K_M        = 15  # except 1d tensors
    MOSTLY_Q5_K_S        = 16  # except 1d tensors
    MOSTLY_Q5_K_M        = 17  # except 1d tensors
    MOSTLY_Q6_K          = 18  # except 1d tensors
    MOSTLY_IQ2_XXS       = 19  # except 1d tensors
    MOSTLY_IQ2_XS        = 20  # except 1d tensors
    MOSTLY_Q2_K_S        = 21  # except 1d tensors
    MOSTLY_IQ3_XS        = 22  # except 1d tensors
    MOSTLY_IQ3_XXS       = 23  # except 1d tensors
    MOSTLY_IQ1_S         = 24  # except 1d tensors
    MOSTLY_IQ4_NL        = 25  # except 1d tensors
    MOSTLY_IQ3_S         = 26  # except 1d tensors
    MOSTLY_IQ3_M         = 27  # except 1d tensors
    MOSTLY_IQ2_S         = 28  # except 1d tensors
    MOSTLY_IQ2_M         = 29  # except 1d tensors
    MOSTLY_IQ4_XS        = 30  # except 1d tensors
    MOSTLY_IQ1_M         = 31  # except 1d tensors
    MOSTLY_BF16          = 32  # except 1d tensors
    # MOSTLY_Q4_0_4_4      = 33  # removed from gguf files, use Q4_0 and runtime repack
    # MOSTLY_Q4_0_4_8      = 34  # removed from gguf files, use Q4_0 and runtime repack
    # MOSTLY_Q4_0_8_8      = 35  # removed from gguf files, use Q4_0 and runtime repack
    MOSTLY_TQ1_0         = 36  # except 1d tensors
    MOSTLY_TQ2_0         = 37  # except 1d tensors
    MOSTLY_MXFP4_MOE     = 38  # except 1d tensors
    MOSTLY_NVFP4         = 39  # except 1d tensors
    MOSTLY_Q1_0          = 40  # except 1d tensors
    MOSTLY_Q2_0          = 41  # except 1d tensors
    MOSTLY_F8_E4M3       = 42  # except 1d tensors

    GUESSED              = 1024  # not specified in the model file


class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            raise ValueError(f"Unknown type: {type(val)}")


class VisionProjectorType:
    INKLING2V      = "inkling2v"
    QWEN3VL        = "qwen3vl_merger"


# Items here are (block size, type size)
QK_K = 256
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 2 + 16),
    GGMLQuantizationType.Q4_1:    (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0:    (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1:    (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0:    (32, 2 + 32),
    GGMLQuantizationType.Q8_1:    (32, 2 + 2 + 32),
    GGMLQuantizationType.Q2_K:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K:    (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K:    (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL:  (32, 2 + 16),
    GGMLQuantizationType.IQ3_S:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, QK_K // 8 + QK_K // 16  + QK_K // 32),
    GGMLQuantizationType.BF16:    (1, 2),
    GGMLQuantizationType.TQ1_0:   (256, 2 + 4 * 13),
    GGMLQuantizationType.TQ2_0:   (256, 2 + 64),
    GGMLQuantizationType.MXFP4:   (32, 1 + 16),
    GGMLQuantizationType.NVFP4:   (64, 4 + 32),
    GGMLQuantizationType.Q1_0:    (128, 2 + 16),
    GGMLQuantizationType.Q2_0:    (64, 2 + 16),
    GGMLQuantizationType.F8_E4M3: (1, 1),
}


# Aliases for backward compatibility.

# general
KEY_GENERAL_ARCHITECTURE         = Keys.General.ARCHITECTURE
KEY_GENERAL_QUANTIZATION_VERSION = Keys.General.QUANTIZATION_VERSION
KEY_GENERAL_ALIGNMENT            = Keys.General.ALIGNMENT
KEY_GENERAL_NAME                 = Keys.General.NAME
KEY_GENERAL_AUTHOR               = Keys.General.AUTHOR
KEY_GENERAL_URL                  = Keys.General.URL
KEY_GENERAL_DESCRIPTION          = Keys.General.DESCRIPTION
KEY_GENERAL_LICENSE              = Keys.General.LICENSE
KEY_GENERAL_SOURCE_URL           = Keys.General.SOURCE_URL
KEY_GENERAL_FILE_TYPE            = Keys.General.FILE_TYPE

# LLM
KEY_VOCAB_SIZE            = Keys.LLM.VOCAB_SIZE
KEY_CONTEXT_LENGTH        = Keys.LLM.CONTEXT_LENGTH
KEY_EMBEDDING_LENGTH      = Keys.LLM.EMBEDDING_LENGTH
KEY_BLOCK_COUNT           = Keys.LLM.BLOCK_COUNT
KEY_FEED_FORWARD_LENGTH   = Keys.LLM.FEED_FORWARD_LENGTH
KEY_TENSOR_DATA_LAYOUT    = Keys.LLM.TENSOR_DATA_LAYOUT

# attention
KEY_ATTENTION_HEAD_COUNT        = Keys.Attention.HEAD_COUNT
KEY_ATTENTION_HEAD_COUNT_KV     = Keys.Attention.HEAD_COUNT_KV
KEY_ATTENTION_MAX_ALIBI_BIAS    = Keys.Attention.MAX_ALIBI_BIAS
KEY_ATTENTION_CLAMP_KQV         = Keys.Attention.CLAMP_KQV
KEY_ATTENTION_LAYERNORM_EPS     = Keys.Attention.LAYERNORM_EPS
KEY_ATTENTION_LAYERNORM_RMS_EPS = Keys.Attention.LAYERNORM_RMS_EPS

# RoPE
KEY_ROPE_DIMENSION_COUNT           = Keys.Rope.DIMENSION_COUNT
KEY_ROPE_FREQ_BASE                 = Keys.Rope.FREQ_BASE
KEY_ROPE_SCALING_TYPE              = Keys.Rope.SCALING_TYPE
KEY_ROPE_SCALING_FACTOR            = Keys.Rope.SCALING_FACTOR
KEY_ROPE_SCALING_ORIG_CTX_LEN      = Keys.Rope.SCALING_ORIG_CTX_LEN
KEY_ROPE_SCALING_FINETUNED         = Keys.Rope.SCALING_FINETUNED

# SSM
KEY_SSM_CONV_KERNEL    = Keys.SSM.CONV_KERNEL
KEY_SSM_INNER_SIZE     = Keys.SSM.INNER_SIZE
KEY_SSM_STATE_SIZE     = Keys.SSM.STATE_SIZE
KEY_SSM_TIME_STEP_RANK = Keys.SSM.TIME_STEP_RANK
KEY_SSM_GROUP_COUNT    = Keys.SSM.GROUP_COUNT

# KDA
KEY_KDA_HEAD_DIM         = Keys.KDA.HEAD_DIM


# tokenization
KEY_TOKENIZER_MODEL      = Keys.Tokenizer.MODEL
KEY_TOKENIZER_PRE        = Keys.Tokenizer.PRE
KEY_TOKENIZER_LIST       = Keys.Tokenizer.LIST
KEY_TOKENIZER_TOKEN_TYPE = Keys.Tokenizer.TOKEN_TYPE
KEY_TOKENIZER_SCORES     = Keys.Tokenizer.SCORES
KEY_TOKENIZER_MERGES     = Keys.Tokenizer.MERGES
KEY_TOKENIZER_BOS_ID     = Keys.Tokenizer.BOS_ID
KEY_TOKENIZER_EOS_ID     = Keys.Tokenizer.EOS_ID
KEY_TOKENIZER_EOT_ID     = Keys.Tokenizer.EOT_ID
KEY_TOKENIZER_EOM_ID     = Keys.Tokenizer.EOM_ID
KEY_TOKENIZER_UNK_ID     = Keys.Tokenizer.UNK_ID
KEY_TOKENIZER_SEP_ID     = Keys.Tokenizer.SEP_ID
KEY_TOKENIZER_PAD_ID     = Keys.Tokenizer.PAD_ID
KEY_TOKENIZER_MASK_ID    = Keys.Tokenizer.MASK_ID
KEY_TOKENIZER_HF_JSON    = Keys.Tokenizer.HF_JSON
KEY_TOKENIZER_RWKV       = Keys.Tokenizer.RWKV

KEY_TOKENIZER_FIM_PRE_ID = Keys.Tokenizer.FIM_PRE_ID
KEY_TOKENIZER_FIM_SUF_ID = Keys.Tokenizer.FIM_SUF_ID
KEY_TOKENIZER_FIM_MID_ID = Keys.Tokenizer.FIM_MID_ID
KEY_TOKENIZER_FIM_PAD_ID = Keys.Tokenizer.FIM_PAD_ID
KEY_TOKENIZER_FIM_REP_ID = Keys.Tokenizer.FIM_REP_ID
KEY_TOKENIZER_FIM_SEP_ID = Keys.Tokenizer.FIM_SEP_ID

# deprecated
KEY_TOKENIZER_PREFIX_ID  = Keys.Tokenizer.PREFIX_ID
KEY_TOKENIZER_SUFFIX_ID  = Keys.Tokenizer.SUFFIX_ID
KEY_TOKENIZER_MIDDLE_ID  = Keys.Tokenizer.MIDDLE_ID
