# Build llama.cpp locally

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

This fork builds only for macOS on Apple Silicon (arm64). The backends are Metal (GPU), CPU and BLAS (Accelerate).

**To get the Code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

The following sections describe how to build with the available backends and options.

* [CPU Build](#cpu-build)
* [BLAS Build](#blas-build)
* [Metal Build](#metal-build)
* [Arm® KleidiAI™](#arm-kleidiai)
* [Notes about GPU-accelerated backends](#notes-about-gpu-accelerated-backends)

## CPU Build

Build llama.cpp using `CMake`:

```bash
cmake -B build
cmake --build build --config Release
```

On macOS this build already includes the Metal and BLAS (Accelerate) backends, see below.

**Notes**:

- For faster compilation, add the `-j` argument to run multiple jobs in parallel, or use a generator that does this automatically such as Ninja. For example, `cmake --build build --config Release -j 8` will run 8 jobs in parallel.
- For faster repeated compilation, install [ccache](https://ccache.dev/)
- For debug builds, there are two cases:

    1. Single-config generators (e.g. default = `Unix Makefiles`; note that they just ignore the `--config` flag):

       ```bash
       cmake -B build -DCMAKE_BUILD_TYPE=Debug
       cmake --build build
       ```

    2. Multi-config generators (`-G` param set to Xcode...):

       ```bash
       cmake -B build -G "Xcode"
       cmake --build build --config Debug
       ```

    For more details and a list of supported generators, see the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).
- For static builds, add `-DBUILD_SHARED_LIBS=OFF`:
  ```
  cmake -B build -DBUILD_SHARED_LIBS=OFF
  cmake --build build --config Release
  ```

- If you want HTTPS/TLS features, you may install OpenSSL development libraries. If not installed, the project will build and run without SSL support.

## BLAS Build

Building the program with BLAS support may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). Using BLAS doesn't affect the generation performance.

### Accelerate Framework

This is only available on Mac PCs and it's enabled by default. You can just build using the normal instructions.

To disable it, use the `-DGGML_BLAS=OFF` cmake option.

## Metal Build

On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU.
To disable the Metal build at compile time use the `-DGGML_METAL=OFF` cmake option.

When built with Metal support, you can explicitly disable GPU inference with the `--n-gpu-layers 0` command-line argument.

## Arm® KleidiAI™
KleidiAI provides optimized Arm CPU microkernels used by the ggml CPU backend. Enabling it at build time makes those kernels available; it does not force every operation to use KleidiAI. At runtime, llama.cpp selects the best compatible CPU kernel from the detected CPU features, tensor type, operation shape, and active backend priority.

On Apple arm64, runtime CPU feature detection is automatic. Non-streaming SVE vector length is treated as unavailable.

### Native arm64 build

From the llama.cpp source directory:

```bash
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release
```

### Verifying the build

Run an installed or in-tree binary:

```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "What is a car?"
```

If KleidiAI is enabled, the output contains a line similar to:

```
load_tensors: CPU_KLEIDIAI model buffer size =  3474.00 MiB
```

This confirms that the model has tensors allocated through the KleidiAI CPU buffer. It does not prove that every operation, or any specific SME-family operation, used a KleidiAI microkernel. Runtime CPU features, tensor type, operation shape, and backend priority still control dispatch.

Metal has higher priority than the CPU backend. To force CPU execution for a run, disable Metal at build time with `-DGGML_METAL=OFF`, or use the runtime device option `--device none`.

### Runtime dispatch

KleidiAI microkernels use Arm CPU features such as dotprod, i8mm, SVE, and SME/SME2. Build-time configuration makes the kernels available. Runtime dispatch selects a compatible kernel for the detected CPU and operation. Older or lower-feature CPUs fall back automatically to compatible kernels.

KleidiAI accelerates selected `GGML_OP_MUL_MAT` paths for F32 and common quantized formats. Exact coverage depends on the bundled KleidiAI version and the llama.cpp runtime selector, so unsupported tensor types, unsupported operation shapes, or higher priority backends may bypass KleidiAI even when the CPU supports the required Arm feature. This is also why a model may not use SME-family kernels on SME-capable hardware.

The current llama.cpp KleidiAI SVE selector only enables SVE kernels when the runtime SVE vector length is known to be QK8_0 bytes, currently 32 bytes. Apple reports SVE capability separately from userspace non-streaming SVE availability, so llama.cpp treats the SVE vector length as unknown there.

The set of available SME-family kernels depends on the bundled KleidiAI version and the detected CPU capabilities. Production configuration does not require any KleidiAI runtime environment variables.

### Diagnostics and debug overrides

KleidiAI runtime environment variables are diagnostics/debug overrides, not production configuration. Leave them unset for normal use.

`GGML_KLEIDIAI_SME` controls SME-family kernel selection and overrides the maximum number of threads assigned to selected quantized SME-family kernels:

- Not set: use automatic runtime detection.
- `0`: disable SME-family kernels.
- `<n> > 0`: enable compatible SME-family kernels and allow up to `<n>` threads for quantized SME-family kernels.

If the CPU does not support the required SME-family capability for a bundled kernel, that kernel is disabled regardless of the environment variable.

---
## Notes about GPU-accelerated backends

The GPU may still be used to accelerate some parts of the computation even when using the `-ngl 0` option. You can fully disable GPU acceleration by using `--device none`.

At runtime, you can specify which backend devices to use with the `--device` option. To see a list of available devices, use the `--list-devices` option.

Backends can be built as dynamic libraries that can be loaded dynamically at runtime. To enable this feature, use the `GGML_BACKEND_DL` option when building.
