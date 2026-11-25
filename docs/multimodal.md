# Multimodal

llama.cpp supports multimodal input via `libmtmd`. These tools support this feature:
- [llama-cli](../tools/cli/README.md)
- [llama-server](../tools/server/README.md) via OpenAI-compatible `/chat/completions` API
- [llama-mtmd-cli](../tools/mtmd/README.md), for testing and development

This fork keeps two encoders: `inkling2v`, the vision encoder of Inkling 2, and `qwen3vl_merger`, the vision encoder of Qwen 3.5 / 3.6. Both take **image** input. They do not take audio input, and Qwen video input is not supported.

Ling-3.2-air runs as a text model only: its vision encoder (`ling3vl`) is not in this fork, and `--mmproj` files for it do not load.

To enable it, you can use one of the 2 methods below:

- Use `-hf` option with a repo that ships an `mmproj` file
    - To load a model using `-hf` while disabling multimodal, use `--no-mmproj`
    - To load a model using `-hf` while using a custom mmproj file, use `--mmproj local_file.gguf`
- Use `-m model.gguf` option with `--mmproj file.gguf` to specify text and multimodal projector respectively

By default, multimodal projector will be offloaded to GPU. To disable this, add `--no-mmproj-offload`

Replace `(tool_name)` with the name of binary you want to use, for example `llama-mtmd-cli` or `llama-server`:

```sh
# download the model and its mmproj file
(tool_name) -hf AtomicChat/Inkling-2-Mini-GGUF

# using local files
(tool_name) -m Inkling-2-Mini-AD-Q4_K.gguf --mmproj mmproj-Inkling-2-Mini.gguf

# no GPU offload of the projector
(tool_name) -hf AtomicChat/Inkling-2-Mini-GGUF --no-mmproj-offload

# Qwen 3.6 35B-A3B and its mmproj file
(tool_name) -hf ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M
```

The file names above are examples. Use the names of the files you have.

Inkling 2 and Qwen 3.5 / 3.6 use Jinja chat templates. `llama-mtmd-cli` does not enable Jinja by default, so add `--jinja` there.

## Converting the mmproj file

`convert_hf_to_gguf.py` with the `--mmproj` flag writes the projector of an Inkling 2 checkpoint (`Inkling2ForConditionalGeneration` or `Inkling2LoopForConditionalGeneration`, `inkling2v`) or a Qwen 3.5 / 3.6 checkpoint (`Qwen3_5ForConditionalGeneration` or `Qwen3_5MoeForConditionalGeneration`, `qwen3vl_merger`) to its own GGUF file.
