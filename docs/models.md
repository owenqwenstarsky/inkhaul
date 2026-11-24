# Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts [thousands of models](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)

You can use any `llama.cpp`-compatible model from [Hugging Face](https://huggingface.co/) using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama cli -hf ggml-org/gemma-3-1b-it-GGUF
```

You can use the same CLI invocation to download from other sites, by pointing the `MODEL_ENDPOINT` environment variable to an endpoint compatible with the Hugging Face API.
`llama.cpp` can also run models you have downloaded locally to your filesystem.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.
To learn more about model quantization, [read this documentation](../tools/quantize/README.md)

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

## Loading extra shards

The converter in this fork has no per-layer embedding (PLE) export (`--no-ple` and `--ple-only` are gone). The loader still accepts extra shards, as described below.

`--ple` is an alias for `--shard`. Extra shards can be comma separated or supplied with repeated options, for example `-m backbone.gguf --shard /fast/ple.gguf,/other/token-embd.gguf`. Paths may be on different drives. Tensor names must be unique across all files, except for the PLE case below.

A numbered backbone is auto-discovered and checked as usual; explicitly supplied paths outside that numbered set are appended as extra shards. Their split numbering is ignored, and the loader uses the actual combined tensor count. Paths already in the discovered set are loaded once. If the main filename does not follow the numbered naming scheme, an explicit backbone list must still match its declared split count and order.

If an extra shard holds only `per_layer_token_embd.weight`, it replaces the PLE table of the model. If the model already has the table, the extra shard overrides it. A backbone split that holds only that tensor is skipped, and it can be absent from disk. This lets one PLE file be shared across quants and releases, for example `-m model-Q4_K_M-00001-of-00003.gguf --shard /fast/ple-Q8_0.gguf`.
