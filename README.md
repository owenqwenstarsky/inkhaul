# llama.cpp (Inkling 2 Mini, Qwen 3.5 / 3.6, Ling-3.2-air, Kolibri 1.1 and Hy4-flash + LongHaul fork)

> [!NOTE]
> This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) trimmed to run [Inkling 2 Mini](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF) (arch `inkling2_loop`), Qwen 3.5 / 3.6 (archs `qwen35` and `qwen35moe`, text and images), [Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air) (arch `bailingmoe3`, text only), [Kolibri 1.1](https://huggingface.co/Aleph-Alpha/Kolibri-1.1) (arch `kolibri1`, also Kolibri 1, text only) and [Hy4-flash](https://huggingface.co/AtomicChat/Hy4-flash-GGUF) (arch `hy_v4_flash`, text and images) with LongHaul on Apple Silicon Macs (Metal) and on NVIDIA GPUs (CUDA, Linux and Windows on x86_64).
> LongHaul keeps the MoE experts on disk and holds the recently used ones in a fixed-size GPU memory cache, so the model runs with less memory than the model file needs.
> Set the cache size with `-lhc, --longhaul-cache` (default: `512M`). See [docs/longhaul.md](docs/longhaul.md) for how it works and how to size the cache.
>
> Other model architectures and the backends other than Metal, CUDA, CPU, BLAS (Accelerate) and RPC are removed. LongHaul can split a model over several GPUs, or over several Macs connected with Thunderbolt cables (see [docs/longhaul.md](docs/longhaul.md#several-laptops)). With `-es 60` the remote Mac computes about 60% of the experts of every layer, at the same time as the main Mac (see [Expert split](docs/longhaul.md#expert-split)).

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

# Qwen 3.6 35B-A3B, with its experts on disk and the vision projector (mmproj) downloaded with it
./build/bin/llama-server -hf ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M -lhc 1G

# Ling-3.2-air (124B, 512 experts per layer) with a 4 GiB expert cache
./build/bin/llama-cli -hf bartowski/Ling-3.2-air-GGUF:Q4_K_M -lhc 4G

# Kolibri 1.1 (78B, 384 experts per layer, sparse attention) with a 2 GiB expert cache
./build/bin/llama-cli -hf Aleph-Alpha/Kolibri-1.1-GGUF:Q4_K_M -lhc 2G

# Hy4-flash (219B, 512 experts per layer, sparse attention, images) with a 2 GiB expert cache
./build/bin/llama-server -hf AtomicChat/Hy4-flash-GGUF:IQ2_XXS -lhc 2G
```

The Qwen MTP (multi-token prediction) block is not run, and `mtp-*.gguf` files are not downloaded. Files with the block in the main GGUF load and skip it.

Ling-3.2-air reads about 1 GB from disk per generated token when no expert is cached, so a fast SSD and a larger cache help more than for Inkling. See [docs/longhaul.md](docs/longhaul.md#sizing-for-ling-32-air) for the cache sizes.

Kolibri 1.1 reads about 650 MiB per generated token when no expert is cached. Its IQ1 / IQ2 files have many expert layouts, which makes prompt processing slow with LongHaul, so prefer the Q4_K_M file. With several parallel sequences, pass `--kv-unified`. See [docs/longhaul.md](docs/longhaul.md#sizing-for-kolibri-11) for the cache sizes.

Hy4-flash reads about 1.2 GiB per generated token when no expert is cached. Its Engram table (33 GiB) stays in the file and is read on demand through the OS page cache. For images, pass `-b` and `-ub` of at least the image tokens + 512. To turn thinking off, use `--reasoning-effort no_think`. See [docs/longhaul.md](docs/longhaul.md#sizing-for-hy4-flash) for the cache sizes.

On Linux or Windows with an NVIDIA GPU, build with CUDA (needs the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)) and run the same commands:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

A larger cache gives faster generation and prompt processing; with enough VRAM, `-lhc 16G` holds all of Mini's experts. See [docs/build.md](docs/build.md) for the build options.

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [quantize](tools/quantize/README.md)
- [perplexity](tools/perplexity/README.md)
- [llama-bench](tools/llama-bench/README.md)
- [rpc](tools/rpc/README.md)
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
