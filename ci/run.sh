#!/usr/bin/env bash
#
# sample usage:
#
# mkdir tmp
#
# # CPU-only build
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with Metal support
# GG_BUILD_METAL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with CUDA support
# GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with KLEIDIAI support
# GG_BUILD_KLEIDIAI=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with BLAS support (Accelerate)
# GG_BUILD_BLAS=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -f $OUT/*.log

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

CMAKE_EXTRA="-DLLAMA_FATAL_WARNINGS=${LLAMA_FATAL_WARNINGS:-ON} -DLLAMA_OPENSSL=OFF -DGGML_SCHED_NO_REALLOC=ON"
CTEST_EXTRA=""

# Default to use make unless specified for compatibility
CMAKE_GENERATOR="Unix Makefiles"

if [ ! -z "${GG_BUILD_NINJA}" ]; then
    CMAKE_GENERATOR="Ninja"
fi

if [ ! -z ${GG_BUILD_METAL} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=ON"
else
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=OFF"
fi

if [ ! -z ${GG_BUILD_CUDA} ]; then
    # TODO: Drop GGML_CUDA_CCCL_VERSION when CUDA CI uses CTK >= 13.5, which bundles CCCL >= 3.5.
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_CUDA=ON -DGGML_CUDA_CCCL_VERSION=v3.4.3"

    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')
        if [[ -n "$CUDA_ARCH" && "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
        else
            echo "Warning: Using fallback CUDA architectures"
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89"
        fi
    else
        echo "Error: nvidia-smi not found, cannot build with CUDA"
        exit 1
    fi
fi

if [ ! -z ${GG_BUILD_NO_SVE} ]; then
    # arm 9 and newer enables sve by default, adjust these flags depending on the cpu used
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm"
fi

if [ -n "${GG_BUILD_KLEIDIAI}" ]; then
    echo ">>===== Enabling KleidiAI support"
    CMAKE_EXTRA="${CMAKE_EXTRA:+$CMAKE_EXTRA } -DGGML_CPU_KLEIDIAI=ON"
fi

if [ ! -z ${GG_BUILD_BLAS} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=${GG_BUILD_BLAS_VENDOR:-Apple}"
else
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_BLAS=OFF"
fi

## helpers

function gg_run {
    ci=$1

    set -o pipefail
    set -x

    gg_run_$ci | tee $OUT/$ci.log
    cur=$?

    set +x
    set +o pipefail

    ret=$((ret | cur))
}

## ci

# ctest_debug

function gg_run_ctest_debug {
    cd ${SRC}

    rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug

    set -e

    # Check required binaries are installed
    gg_check_build_requirements

    (cmake -G "${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time cmake --build . --config Debug -j$(nproc)) 2>&1 | tee -a $OUT/${ci}-make.log

    (time ctest -C Debug --output-on-failure -L main -E "test-opt|test-llama-archs" ${CTEST_EXTRA}) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
}

# ctest_release

function gg_run_ctest_release {
    cd ${SRC}

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    # Check required binaries are installed
    gg_check_build_requirements

    (cmake -G "${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time cmake --build . --config Release -j$(nproc)) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ -z ${GG_BUILD_LOW_PERF} ]; then
        (time ctest -C Release --output-on-failure -L main ${CTEST_EXTRA}) 2>&1 | tee -a $OUT/${ci}-ctest.log
    else
        (time ctest -C Release --output-on-failure -L main -E test-opt ${CTEST_EXTRA}) 2>&1 | tee -a $OUT/${ci}-ctest.log
    fi

    set +e
}

# test_llama_archs_tensor_split

function gg_run_test_llama_archs_tensor_split {
    cd ${SRC}

    set -e

    if [ ! -z ${GG_BUILD_CUDA} ]; then
        GGML_CUDA_DEVICES=1 ./build-ci-release/bin/test-llama-archs -s 1 2>&1
        GGML_CUDA_DEVICES=2 ./build-ci-release/bin/test-llama-archs -s 1 2>&1
        GGML_CUDA_DEVICES=3 ./build-ci-release/bin/test-llama-archs -s 1 2>&1
        GGML_CUDA_DEVICES=4 ./build-ci-release/bin/test-llama-archs -s 1 2>&1
    fi

    if [ ! -z ${GG_BUILD_METAL} ]; then
        GGML_METAL_DEVICES=1 ./build-ci-release/bin/test-llama-archs -s 1 2>&1
        GGML_METAL_DEVICES=2 ./build-ci-release/bin/test-llama-archs -s 1 2>&1
        GGML_METAL_DEVICES=3 ./build-ci-release/bin/test-llama-archs -s 1 2>&1
        GGML_METAL_DEVICES=4 ./build-ci-release/bin/test-llama-archs -s 1 2>&1
    fi

    set +e
}

function gg_check_build_requirements {
    if ! command -v git &> /dev/null; then
        echo 'git not found, please install'
        exit 1
    fi

    if ! command -v git-lfs &> /dev/null; then
        echo 'git-lfs not found, please install'
        exit 1
    fi

    if ! git config --get filter.lfs.clean &> /dev/null; then
        echo 'git-lfs not initialized, please run `git lfs install`'
        exit 1
    fi

    if ! command -v wget &> /dev/null; then
        echo 'wget not found, please install'
        exit 1
    fi

    if ! command -v python3 &> /dev/null; then
        echo 'python3 not found, please install'
        exit 1
    fi

    if ! command -v pip3 &> /dev/null; then
        echo 'pip3 not found, please install'
        exit 1
    fi

    if ! python3 -m ensurepip --help &> /dev/null; then
        echo 'ensurepip not found, please install python3-venv package'
        exit 1
    fi

    if ! command -v cmake &> /dev/null; then
        echo 'cmake not found, please install'
        exit 1
    fi

    if ! command -v ccache &> /dev/null; then
        echo 'ccache not found, please consider installing for faster builds'
    fi

    if ! command -v ctest &> /dev/null; then
        echo 'ctest not found, please install'
        exit 1
    fi

    if ! command -v unzip &> /dev/null; then
        echo 'unzip not found, please install'
        exit 1
    fi
}

function gg_run_test_backend_ops {
    cd ${SRC}

    cd build-ci-release

    set -e

    local n_jobs=$(nproc)
    if [ "${n_jobs}" -gt 2 ]; then
        n_jobs=2
    fi
    local args_extra="-j ${n_jobs}"

    # TODO: reduce the test-backend-ops timeout to 1800s
    if [ ! -z ${GG_BUILD_HIGH_PERF} ]; then
        (time timeout 3600 ./bin/test-backend-ops ${args_extra} -b CPU) 2>&1 | tee -a $OUT/${ci}-test-backend-ops.log
    else
        (time timeout 3600 ./bin/test-backend-ops ${args_extra}       ) 2>&1 | tee -a $OUT/${ci}-test-backend-ops.log
    fi

    set +e
}

## main

export LLAMA_ARG_LOG_PREFIX=1
export LLAMA_ARG_LOG_TIMESTAMPS=1

ret=0

test $ret -eq 0 && gg_run ctest_debug
test $ret -eq 0 && gg_run ctest_release

test $ret -eq 0 && gg_run test_backend_ops

test $ret -eq 0 && gg_run test_llama_archs_tensor_split

exit $ret
