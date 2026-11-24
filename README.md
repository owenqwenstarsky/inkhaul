# llama.cpp (Inkling 2 Mini + LongHaul fork)

> [!NOTE]
> This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) trimmed to run [Inkling 2 Mini](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF) (arch `inkling2_loop`) on Apple Silicon Macs with LongHaul.
> LongHaul keeps the MoE experts on disk and holds the recently used ones in a fixed-size RAM cache, so the model runs on Macs with less RAM than the model file needs.
> Set the cache size with `-lhc, --longhaul-cache` (default: `512M`). See [docs/longhaul.md](docs/longhaul.md) for how it works and how to size the cache.
>
> Other model architectures, the backends other than Metal, CPU and BLAS (Accelerate), and the platforms other than macOS on Apple Silicon are removed.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

`llama.cpp` runs LLM (and MLLM) inference in C/C++ with minimal setup. It is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

Build from source on an Apple Silicon Mac (Metal is enabled by default):

```bash
cmake -B build
cmake --build build --config Release -j
```

The binaries are in `build/bin`. Then:

```bash
# Download Inkling-2-Mini-AD-Q4_K.gguf from Hugging Face and chat with it, with a 1 GiB expert cache
./build/bin/llama-cli -hf AtomicChat/Inkling-2-Mini-GGUF:Q4_K -lhc 1G

# Or use a local GGUF file
./build/bin/llama-cli -m Inkling-2-Mini-AD-Q4_K.gguf -lhc 1G

# Launch an OpenAI-compatible API server
./build/bin/llama-server -hf AtomicChat/Inkling-2-Mini-GGUF:Q4_K -lhc 1G
```

A larger cache gives faster generation and prompt processing. See [docs/build.md](docs/build.md) for the build options.

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [quantize](tools/quantize/README.md)
- [perplexity](tools/perplexity/README.md)
- [llama-bench](tools/llama-bench/README.md)
- [gguf-py](gguf-py/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [LongHaul](docs/longhaul.md)
- [How to build](docs/build.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Changes that are not specific to this fork belong in [upstream llama.cpp](https://github.com/ggml-org/llama.cpp)

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
