#!/usr/bin/env bash
#
# usage: ./build-xcframework.sh
# builds a macOS (Apple Silicon, arm64) llama.xcframework in build-apple/
#
# Options
MACOS_MIN_OS_VERSION=13.3

BUILD_SHARED_LIBS=OFF
LLAMA_BUILD_APP=OFF
LLAMA_BUILD_COMMON=OFF
LLAMA_BUILD_EXAMPLES=OFF
LLAMA_BUILD_TOOLS=OFF
LLAMA_BUILD_TESTS=OFF
LLAMA_BUILD_SERVER=OFF
LLAMA_BUILD_MTMD=ON
GGML_METAL=ON
GGML_METAL_EMBED_LIBRARY=${GGML_METAL_EMBED_LIBRARY:-ON}
GGML_BLAS_DEFAULT=ON
GGML_OPENMP=OFF

if [[ $# -gt 0 ]]; then
    echo "Error: this script takes no arguments, it only builds the macOS slice" >&2
    exit 1
fi

JOBS=$(sysctl -n hw.logicalcpu)
if [[ "$JOBS" -lt 1 ]]; then
    JOBS=1
fi

BUILD_DIR=build-macos
RELEASE_DIR=Release

COMMON_C_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"
COMMON_CXX_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"

COMMON_CMAKE_ARGS=(
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=""
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
    -DCMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT="dwarf-with-dsym"
    -DCMAKE_XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS=YES
    -DCMAKE_XCODE_ATTRIBUTE_COPY_PHASE_STRIP=NO
    -DCMAKE_XCODE_ATTRIBUTE_STRIP_INSTALLED_PRODUCT=NO
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=ggml
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DLLAMA_BUILD_APP=${LLAMA_BUILD_APP}
    -DLLAMA_BUILD_COMMON=${LLAMA_BUILD_COMMON}
    -DLLAMA_BUILD_EXAMPLES=${LLAMA_BUILD_EXAMPLES}
    -DLLAMA_BUILD_TOOLS=${LLAMA_BUILD_TOOLS}
    -DLLAMA_BUILD_TESTS=${LLAMA_BUILD_TESTS}
    -DLLAMA_BUILD_SERVER=${LLAMA_BUILD_SERVER}
    -DLLAMA_BUILD_MTMD=${LLAMA_BUILD_MTMD}
    -DGGML_METAL_EMBED_LIBRARY=${GGML_METAL_EMBED_LIBRARY}
    -DGGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT}
    -DGGML_METAL=${GGML_METAL}
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=${GGML_OPENMP}
)

check_required_tool() {
    local tool=$1
    local install_message=$2

    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is required but not found."
        echo "$install_message"
        exit 1
    fi
}
echo "Checking for required tools..."
check_required_tool "cmake" "Please install CMake 3.28.0 or later (brew install cmake)"
check_required_tool "xcrun" "Please install Xcode and Xcode Command Line Tools (xcode-select --install)"

XCODE_VERSION=$(xcrun xcodebuild -version 2>/dev/null | head -n1 | awk '{ print $2 }')
echo "Detected Xcode version: $XCODE_VERSION"

set -e

## Clean up previous builds
rm -rf build-apple
rm -rf ${BUILD_DIR}

# Setup the macOS framework directory structure
setup_framework_structure() {
    local build_dir=$1
    local min_os_version=$2
    local framework_name="llama"
    local fw_dir=${build_dir}/framework/${framework_name}.framework

    echo "Creating macOS framework structure for ${build_dir}"

    # macOS frameworks use versioned directories
    mkdir -p ${fw_dir}/Versions/A/Headers
    mkdir -p ${fw_dir}/Versions/A/Modules
    mkdir -p ${fw_dir}/Versions/A/Resources

    ln -sf A ${fw_dir}/Versions/Current
    ln -sf Versions/Current/Headers ${fw_dir}/Headers
    ln -sf Versions/Current/Modules ${fw_dir}/Modules
    ln -sf Versions/Current/Resources ${fw_dir}/Resources
    ln -sf Versions/Current/${framework_name} ${fw_dir}/${framework_name}

    local header_path=${fw_dir}/Versions/A/Headers/
    local module_path=${fw_dir}/Versions/A/Modules/

    cp include/llama.h             ${header_path}
    cp ggml/include/ggml.h         ${header_path}
    cp ggml/include/ggml-opt.h     ${header_path}
    cp ggml/include/ggml-alloc.h   ${header_path}
    cp ggml/include/ggml-backend.h ${header_path}
    cp ggml/include/ggml-metal.h   ${header_path}
    cp ggml/include/ggml-cpu.h     ${header_path}
    cp ggml/include/ggml-blas.h    ${header_path}
    cp ggml/include/gguf.h         ${header_path}
    cp tools/mtmd/mtmd.h           ${header_path}
    cp tools/mtmd/mtmd-helper.h    ${header_path}

    if [[ "$GGML_METAL_EMBED_LIBRARY" == "OFF" ]]; then
        cp ${build_dir}/bin/*.metallib ${fw_dir}/Versions/A/Resources/
    fi

    cat > ${module_path}module.modulemap << EOF
framework module llama {
    umbrella "Headers"

    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF

    cat > ${fw_dir}/Versions/A/Resources/Info.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>llama</string>
    <key>CFBundleIdentifier</key>
    <string>org.ggml.llama</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>llama</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${min_os_version}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
    <key>DTPlatformName</key>
    <string>macosx</string>
    <key>DTSDKName</key>
    <string>macosx${min_os_version}</string>
</dict>
</plist>
EOF
}

# Create a dynamic library from the static libraries
combine_static_libraries() {
    local build_dir="$1"
    local release_dir="$2"
    local base_dir="$(pwd)"
    local framework_name="llama"
    local output_lib="${build_dir}/framework/${framework_name}.framework/Versions/A/${framework_name}"

    local libs=(
        "${base_dir}/${build_dir}/src/${release_dir}/libllama.a"
        "${base_dir}/${build_dir}/ggml/src/${release_dir}/libggml.a"
        "${base_dir}/${build_dir}/ggml/src/${release_dir}/libggml-base.a"
        "${base_dir}/${build_dir}/ggml/src/${release_dir}/libggml-cpu.a"
        "${base_dir}/${build_dir}/ggml/src/ggml-metal/${release_dir}/libggml-metal.a"
        "${base_dir}/${build_dir}/ggml/src/ggml-blas/${release_dir}/libggml-blas.a"
        "${base_dir}/${build_dir}/tools/mtmd/${release_dir}/libmtmd.a"
        "${base_dir}/${build_dir}/vendor/hash/${release_dir}/libvendor-hash.a"
    )

    local temp_dir="${base_dir}/${build_dir}/temp"
    mkdir -p "${temp_dir}"

    xcrun libtool -static -o "${temp_dir}/combined.a" "${libs[@]}" 2> /dev/null

    echo "Creating dynamic library for macOS."
    xcrun -sdk macosx clang++ -dynamiclib \
        -isysroot $(xcrun --sdk macosx --show-sdk-path) \
        -arch arm64 \
        -mmacosx-version-min=${MACOS_MIN_OS_VERSION} \
        -Wl,-force_load,"${temp_dir}/combined.a" \
        -framework Foundation -framework Metal -framework Accelerate \
        -install_name "@rpath/llama.framework/Versions/Current/llama" \
        -o "${base_dir}/${output_lib}"

    echo "Creating properly formatted dSYM..."
    mkdir -p "${base_dir}/${build_dir}/dSYMs"

    # strip debug info to a separate file, then make the dSYM from the unstripped binary
    xcrun strip -S "${base_dir}/${output_lib}" -o "${temp_dir}/stripped_lib"
    xcrun dsymutil "${base_dir}/${output_lib}" -o "${base_dir}/${build_dir}/dSYMs/llama.dSYM"
    mv "${temp_dir}/stripped_lib" "${base_dir}/${output_lib}"

    # a dSYM inside the framework fails bundle validation
    if [ -d "${base_dir}/${output_lib}.dSYM" ]; then
        echo "Removing generated dSYM file in framework structure: ${base_dir}/${output_lib}.dSYM"
        rm -rf "${base_dir}/${output_lib}.dSYM"
    fi

    rm -rf "${temp_dir}"
}

echo "Building for macOS (arm64)..."
cmake -B ${BUILD_DIR} -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_MIN_OS_VERSION} \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DLLAMA_OPENSSL=OFF \
    -S .
cmake --build ${BUILD_DIR} --config Release -j "${JOBS}" -- -quiet

echo "Setting up framework structure..."
setup_framework_structure "${BUILD_DIR}" "${MACOS_MIN_OS_VERSION}"

echo "Creating dynamic library from static libraries..."
combine_static_libraries "${BUILD_DIR}" "${RELEASE_DIR}"

echo "Creating XCFramework..."
xcrun xcodebuild -create-xcframework \
    -framework "$(pwd)/${BUILD_DIR}/framework/llama.framework" \
    -debug-symbols "$(pwd)/${BUILD_DIR}/dSYMs/llama.dSYM" \
    -output "$(pwd)/build-apple/llama.xcframework"
