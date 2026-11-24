# libmtmd dev guide

## History

Please refer to [multimodal.md](../../docs/multimodal.md) for a broader context.

In short:
- `libmtmd` started as a wrapper around `libllava` / `clip.cpp`
- Various components that used to be in `clip.cpp` are moved progressively to mtmd. For example, preprocessor is now part of mtmd
- `clip.cpp` has since been split into `mtmd-model` (hparams and weights, one `mtmd_model` per modality), `mtmd-subcontext` (backends and compute, one `mtmd_subcontext` per modality) and `mtmd-graph` (graph builders, see `models/`). Only the GGUF format keeps the `clip.` name (metadata keys and the `clip` architecture), for compatibility with existing mmproj files

## Terminologies

- mtmd: **M**ul**T**i**M**o**D**al
- bitmap: representing a raw input data, for example: RGB image, PCM audio
- tiles / slices: for llava-uhd-style models, the preprocessor breaks a large input into smaller square images called tiles or slices
- chunk: a mtmd_input_chunk represents a preprocessed input that can then be passed through `mtmd_encode()`

## Structure

Overview of the `libmtmd` structure:

- `mtmd_context` (`mtmd.cpp`): the public handle
    - `mtmd_models`: one mmproj file, loaded by `mtmd_model_load()`
        - `mtmd_model` `v` / `a` / `gen_a` (`mtmd-model.h`): one per modality, holds hparams, weights (`ctx_data` + `buf`) and requirements
    - `mtmd_backends` (`mtmd-subcontext.h`): GPU/CPU backend instances, CPU threadpool and one scheduler, shared by the subcontexts
    - `compute_mutex`: held by every public function that runs a compute pass, so only one pass at a time runs on the backends
    - `mtmd_subcontext` `ctx_v` / `ctx_a` / `ctx_gen_a` (`mtmd-subcontext.h`): one per model, holds a const ref to it and a ref to the backends
        - `buf_compute_meta` and the resolved flash attn of its graphs
        - `mtmd_graph` (`mtmd-graph.h`, `models/`): built for each `encode()` call, temporary
    - `mtmd_image_preprocessor` / `mtmd_audio_preprocessor` (`mtmd-image.h`, `mtmd-audio.h`): read the hparams of their model
- `mtmd_gen_state` / `mtmd_audio_stream`: public, one per sequence / stream
    - `mtmd_subctx_gen_state` / `mtmd_subctx_enc_state`: allocated on the backend of a subcontext, see `gen_state_init()` / `enc_state_init()`

Rules of thumb:
- `mtmd_model` is read-only once loaded. Anything that depends on the backend (flash attn resolution, warmup) belongs to `mtmd_subcontext`
- Like the pp/tg graphs of `llama_context`, the subcontexts share one scheduler: each warmup reserves its worst-case graph on it, so the compute buffers fit the largest one, and switching modality re-plans inside them
- A model declares its runtime needs (e.g. `max_nodes`, `flash_attn_auto`) in `load_hparams`, the subcontext applies them
- Models and backends must outlive the subcontexts, `mtmd_context` declares them first for that reason
- Models having both audio and vision encoders must have different arch names (projector types), postfixed by `a` and `v` (the vision encoder of Inkling 2 is `inkling2v`)
- Locking happens only at the public API boundary (`mtmd.cpp`): a new public function that runs a graph must take `compute_mutex`. The internals (subcontexts, preprocessors, `mtmd_audio_log_mel()`) assume it is held and never lock
- No magic numbers for a model's fixed shapes (layer counts, window lengths...): define them once as `static constexpr` in the graph struct in `models/models.h`, and refer to them from the loader or the subcontext

## Pipeline

A typical pipeline of the core libmtmd is as follows:
- A bitmap (RGB image or PCM audio) is created
- Bitmap and the text prompt is provided to `mtmd_tokenize()` that breaks the input into chunks
    - Alternatively, `mtmd_tokenize_from_parts()` takes a list of pre-split text/media parts instead of a marker-based prompt
    - The tokenizer function first expands a "lazy" bitmap if it finds one. Typically, this is used by video, so that one media token corresponds to one input bitmap
    - For models that support "fused" temporal frames like Qwen-VL, the tokenizer tries to merge pair of consecutive frames into one batch. Only bitmaps marked by `mtmd_bitmap_set_mergeable()` are merged
    - The preprocessor will then be called, which produces a list of chunks
    - Depending on the model itself, special tokens will be injected to separate image chunks (i.e. llava-uhd-style models)
- Multiple bitmaps may be batched together to form a larger `mtmd_batch()`
- Single image or batch is encoded, via `mtmd_encode()` or `mtmd_batch_encode()`
- Get the output embeddings

## Helper

We provide a set of helper functions via `mtmd_helper` to make using libmtmd easier. The helper provides:
- Image, audio and video file decoding (for example, decode raw JPEG into RGB bitmap)
- Manage `llama_batch` and calls to `llama_decode`

## Encoders in this fork

This fork keeps one encoder, `inkling2v` (`models/inkling2v.cpp`): a ViT with a CLS token, learned position embeddings (resized to the input) and 2D RoPE, followed by a patch merge and a 2-layer projector. It is built from the shared `mtmd_graph` helpers (`build_vit`, `build_rope_2d`, `build_patch_merge_permute`, `build_ffn`). Audio input and audio generation (TTS) models are removed.
