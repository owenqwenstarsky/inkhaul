# CI

`ci/run.sh` runs the heavy-duty CI (debug and release builds, `ctest`, `test-backend-ops` and the `test-llama-archs` tensor-split check) on an Apple Silicon Mac or an x86_64 Linux machine with an NVIDIA GPU. It builds the CPU backend by default and can enable Metal, CUDA, BLAS (Accelerate) and KleidiAI.

It is a good practice, before publishing changes, to execute the full CI locally on your machine. For example:

```bash
mkdir tmp

# CPU-only build
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with Metal support
GG_BUILD_METAL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with CUDA support (the GPU architecture is read from nvidia-smi)
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with BLAS support (Accelerate)
GG_BUILD_BLAS=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with KleidiAI support
GG_BUILD_KLEIDIAI=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

The upstream model runs (Qwen3, BGE, reranker) are removed: those archs cannot be loaded by this fork. Set `GG_BUILD_LOW_PERF=1` to skip `test-opt` in the release ctest run.
