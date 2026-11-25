function(ggml_add_cpu_backend_features cpu_name arch)
    # The feature detection code is compiled as a separate target so that
    # it can be built without the architecture flags
    # Since multiple variants of the CPU backend may be included in the same
    # build, using set_source_files_properties() to set the arch flags is not possible
    set(GGML_CPU_FEATS_NAME ${cpu_name}-feats)
    add_library(${GGML_CPU_FEATS_NAME} OBJECT ggml-cpu/arch/${arch}/cpu-feats.cpp)
    target_include_directories(${GGML_CPU_FEATS_NAME} PRIVATE . ../include)
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE ${ARGN})
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    set_target_properties(${GGML_CPU_FEATS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Disable LTO for the feature detection code to prevent cross-module optimization
    # from inlining architecture-specific instructions into the score function.
    # Without this, LTO can cause SIGILL when loading backends on older CPUs
    # (e.g., loading power10 backend on power9 crashes before feature check runs).
    target_compile_options(${GGML_CPU_FEATS_NAME} PRIVATE -fno-lto)
    target_link_libraries(${cpu_name} PRIVATE ${GGML_CPU_FEATS_NAME})
endfunction()

function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        set(GGML_CPU_NAME ggml-cpu-${tag_name})
    else()
        set(GGML_CPU_NAME ggml-cpu)
    endif()

    ggml_add_backend_library(${GGML_CPU_NAME})

    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/hbm.cpp
        ggml-cpu/hbm.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/amx/amx.cpp
        ggml-cpu/amx/amx.h
        ggml-cpu/amx/mmq.cpp
        ggml-cpu/amx/mmq.h
        ggml-cpu/tiled/tiled.cpp
        ggml-cpu/tiled/tiled.h
        ggml-cpu/tiled/tiled-kernel.cpp
        ggml-cpu/tiled/tiled-kernel.h
        ggml-cpu/ggml-cpu-impl.h
        ggml-cpu/common.h
        ggml-cpu/binary-ops.h
        ggml-cpu/binary-ops.cpp
        ggml-cpu/unary-ops.h
        ggml-cpu/unary-ops.cpp
        ggml-cpu/simd-mappings.h
        ggml-cpu/vec.h
        ggml-cpu/vec.cpp
        ggml-cpu/ops.h
        ggml-cpu/ops.cpp
        )

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if (ACCELERATE_FRAMEWORK)
            message(STATUS "Accelerate framework found")

            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_ACCELERATE)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_NEW_LAPACK)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_LAPACK_ILP64)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
        else()
            message(WARNING "Accelerate framework not found")
        endif()
    endif()

    if (GGML_OPENMP_ENABLED)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)
        target_link_libraries(${GGML_CPU_NAME} PRIVATE ${GGML_OPENMP_TARGET_C} ${GGML_OPENMP_TARGET_CXX})
    endif()

    if (GGML_LLAMAFILE)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_LLAMAFILE)

        list(APPEND GGML_CPU_SOURCES
                    ggml-cpu/llamafile/sgemm.cpp
                    ggml-cpu/llamafile/sgemm.h)
    endif()

    if (GGML_CPU_HBM)
        find_library(memkind memkind REQUIRED)

        message(STATUS "Using memkind for CPU HBM")

        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_HBM)

        target_link_libraries(${GGML_CPU_NAME} PUBLIC memkind)
    endif()

    if (GGML_SYSTEM_ARCH STREQUAL "ARM")
        message(STATUS "ARM detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/arm/quants.c
            ggml-cpu/arch/arm/repack.cpp
            )

        # MSVC proper (not clang-cl) has no -mcpu/-march for ARM: features can only be
        # selected by probing the host at configure time and forwarding -D__ARM_FEATURE_*
        if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
            set(GGML_ARM_MSVC ON)
        else()
            set(GGML_ARM_MSVC OFF)
        endif()

        if (GGML_ARM_MSVC AND NOT GGML_NATIVE)
            message(FATAL_ERROR "MSVC on ARM requires GGML_NATIVE=ON, use clang for GGML_CPU_ARM_ARCH / GGML_CPU_ALL_VARIANTS")
        else()
            if (NOT GGML_ARM_MSVC)
                check_cxx_compiler_flag(-mfp16-format=ieee GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E)
                if (NOT "${GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
                    list(APPEND ARCH_FLAGS -mfp16-format=ieee)
                endif()
            endif()

            if (GGML_NATIVE)
                # -mcpu=native does not always enable all the features in some compilers,
                # so we check for them manually and enable them if available

                # cl.exe cannot be queried this way: no /dev/null, and it rejects the flags.
                # ARM_NATIVE_FLAG stays empty for MSVC and is never appended below.
                if (NOT GGML_ARM_MSVC)
                    execute_process(
                        COMMAND ${CMAKE_C_COMPILER} -mcpu=native -E -v -
                        INPUT_FILE "/dev/null"
                        OUTPUT_QUIET
                        ERROR_VARIABLE ARM_MCPU
                        RESULT_VARIABLE ARM_MCPU_RESULT
                    )
                    if (NOT ARM_MCPU_RESULT)
                        string(REGEX MATCH "-mcpu=[^ ']+" ARM_MCPU_FLAG "${ARM_MCPU}")
                        string(REGEX MATCH "-march=[^ ']+" ARM_MARCH_FLAG "${ARM_MCPU}")

                        # on some old GCC we need to read -march=
                        if (ARM_MARCH_FLAG AND NOT "${ARM_MARCH_FLAG}" STREQUAL "-march=native")
                            set(ARM_NATIVE_FLAG "${ARM_MARCH_FLAG}")
                        elseif(ARM_MCPU_FLAG AND NOT "${ARM_MCPU_FLAG}" STREQUAL "-mcpu=native")
                            set(ARM_NATIVE_FLAG "${ARM_MCPU_FLAG}")
                        endif()
                    endif()

                    if ("${ARM_NATIVE_FLAG}" STREQUAL "")
                        set(ARM_NATIVE_FLAG -mcpu=native)
                        message(WARNING "ARM -march/-mcpu not found, -mcpu=native will be used")
                    else()
                        message(STATUS "ARM detected flags: ${ARM_NATIVE_FLAG}")
                    endif()
                endif()

                include(CheckCXXSourceRuns)

                macro(check_arm_feature tag feature code)
                    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
                    if (GGML_ARM_MSVC)
                        set(ARM_PROBE_FLAG   "")
                        set(ARM_PROBE_NOFLAG "")
                    else()
                        set(ARM_PROBE_FLAG   "${ARM_NATIVE_FLAG}+${tag}")
                        set(ARM_PROBE_NOFLAG "${ARM_NATIVE_FLAG}+no${tag}")
                    endif()
                    set(CMAKE_REQUIRED_FLAGS "${ARM_PROBE_FLAG}")
                    check_cxx_source_runs("${code}" GGML_MACHINE_SUPPORTS_${tag})
                    if (GGML_MACHINE_SUPPORTS_${tag})
                        set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+${tag}")
                        if (GGML_ARM_MSVC)
                            # forward runtime probing decisions for MSVC
                            list(APPEND ARCH_FLAGS "-D__ARM_FEATURE_${feature}=1")
                        endif()
                    else()
                        set(CMAKE_REQUIRED_FLAGS "${ARM_PROBE_NOFLAG}")
                        check_cxx_source_compiles("int main() { return 0; }" GGML_MACHINE_SUPPORTS_no${tag})
                        if (GGML_MACHINE_SUPPORTS_no${tag})
                            set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+no${tag}")
                            list(APPEND ARCH_FLAGS -U__ARM_FEATURE_${feature})
                        endif()
                    endif()
                    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
                endmacro()

                if (GGML_ARM_MSVC)
                    # drop volatile
                    check_arm_feature(dotprod DOTPROD     "#include <arm_neon.h>\nint main() { int8x16_t _a = vdupq_n_s8(0), _b = vdupq_n_s8(0); int32x4_t _s = vdupq_n_s32(0); _s = vdotq_s32(_s, _a, _b); return 0; }")
                    check_arm_feature(i8mm    MATMUL_INT8 "#include <arm_neon.h>\nint main() { int8x16_t _a = vdupq_n_s8(0), _b = vdupq_n_s8(0); int32x4_t _s = vdupq_n_s32(0); _s = vmmlaq_s32(_s, _a, _b); return 0; }")
                    check_arm_feature(sve     SVE         "#include <arm_sve.h>\nint main() { const svbool_t pg = svptrue_b8(); const svint8_t a = svdup_n_s8(0), b = svdup_n_s8(0); svint32_t s = svdup_n_s32(0); s = svdot_s32(s, a, b); (void) pg; (void) s; return 0; }")
                    # FMA is mandatory on ARMv8-A but MSVC never defines __ARM_FEATURE_FMA,
                    # so probe it like the others and forward the define
                    check_arm_feature(fma     FMA         "#include <arm_neon.h>\nint main() { float32x4_t _a = vdupq_n_f32(0), _b = vdupq_n_f32(0), _c = vdupq_n_f32(0); _a = vfmaq_f32(_a, _b, _c); return 0; }")
                else()
                    check_arm_feature(dotprod DOTPROD     "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }")
                    check_arm_feature(i8mm    MATMUL_INT8 "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vmmlaq_s32(_s, _a, _b); return 0; }")
                    check_arm_feature(sve     SVE         "#include <arm_sve.h>\nint main()  { svfloat32_t _a, _b; volatile svfloat32_t _c = svadd_f32_z(svptrue_b8(), _a, _b); return 0; }")
                endif()
                check_arm_feature(sme     SME         "#include <arm_sme.h>\n__arm_locally_streaming int main() { __asm__ volatile(\"smstart; smstop;\"); return 0; }")

                if (NOT GGML_ARM_MSVC)
                    list(APPEND ARCH_FLAGS "${ARM_NATIVE_FLAG}${ARM_NATIVE_FLAG_FIX}")
                endif()
            else()
                if (GGML_CPU_ARM_ARCH)
                    list(APPEND ARCH_FLAGS -march=${GGML_CPU_ARM_ARCH})
                elseif(GGML_CPU_ALL_VARIANTS)
                    # Begin with the lowest baseline
                    set(ARM_MCPU "armv8-a")
                    set(ARCH_TAGS "")
                    set(ARCH_DEFINITIONS "")

                    # When a feature is selected, bump the MCPU to the first
                    # version that supported it
                    if (GGML_INTERNAL_DOTPROD)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+dotprod")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_DOTPROD)
                    endif()
                    if (GGML_INTERNAL_FP16_VECTOR_ARITHMETIC)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+fp16")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_FP16_VECTOR_ARITHMETIC)
                    endif()
                    if (GGML_INTERNAL_SVE)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE)
                    endif()
                    if (GGML_INTERNAL_MATMUL_INT8)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+i8mm")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_MATMUL_INT8)
                    endif()
                    if (GGML_INTERNAL_SVE2)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve2")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE2)
                    endif()
                    if (GGML_INTERNAL_NOSVE)
                        set(ARCH_TAGS "${ARCH_TAGS}+nosve")
                    endif()
                    if (GGML_INTERNAL_SME)
                        set(ARM_MCPU "armv9.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sme")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SME)
                    endif()
                    list(APPEND ARCH_FLAGS "-march=${ARM_MCPU}${ARCH_TAGS}")
                    ggml_add_cpu_backend_features(${GGML_CPU_NAME} arm ${ARCH_DEFINITIONS})
                endif()
            endif()

            message(STATUS "Checking for ARM features using flags:")
            foreach(flag IN LISTS ARCH_FLAGS)
                message(STATUS "  ${flag}")
            endforeach()

            include(CheckCXXSourceCompiles)
            set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
            string(REPLACE ";" " " ARCH_FLAGS_STR "${ARCH_FLAGS}")
            set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS_STR}")
            foreach(feature DOTPROD SVE MATMUL_INT8 FMA FP16_VECTOR_ARITHMETIC SME)
                set(ARM_FEATURE "HAVE_${feature}")
                check_cxx_source_compiles(
                    "
                    #if !defined(__ARM_FEATURE_${feature})
                    #  error \"Feature ${feature} is not defined\"
                    #endif
                    int main() { return 0; }
                    "
                    ${ARM_FEATURE}
                )
            endforeach()
            set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/x86/quants.c
            ggml-cpu/arch/x86/repack.cpp
            )

        if (MSVC)
            # instruction set detection for MSVC only
            if (GGML_NATIVE)
                include(ggml-cpu/cmake/FindSIMD.cmake)
            endif ()
            if (GGML_AVX512)
                list(APPEND ARCH_FLAGS /arch:AVX512)
                # /arch:AVX512 includes: __AVX512F__, __AVX512CD__, __AVX512BW__, __AVX512DQ__, and __AVX512VL__
                # MSVC has no compile-time flags enabling specific
                # AVX512 extensions, neither it defines the
                # macros corresponding to the extensions.
                # Do it manually.
                list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VBMI__)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vbmi)
                    endif()
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VNNI__ GGML_AVX512_VNNI)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vnni)
                    endif()
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_DEFINITIONS __AVX512BF16__ GGML_AVX512_BF16)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512bf16)
                    endif()
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_DEFINITIONS __AMX_TILE__ GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_DEFINITIONS __AMX_INT8__ GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_DEFINITIONS __AMX_BF16__ GGML_AMX_BF16)
                endif()
            elseif (GGML_AVX2)
                list(APPEND ARCH_FLAGS /arch:AVX2)
                list(APPEND ARCH_DEFINITIONS GGML_AVX2 GGML_FMA GGML_F16C)
            elseif (GGML_AVX)
                list(APPEND ARCH_FLAGS /arch:AVX)
                list(APPEND ARCH_DEFINITIONS GGML_AVX)
            elseif (GGML_SSE42)
                list(APPEND ARCH_FLAGS /arch:SSE4.2)
                list(APPEND ARCH_DEFINITIONS GGML_SSE42)
            endif()
            if (GGML_AVX_VNNI)
                list(APPEND ARCH_DEFINITIONS __AVXVNNI__ GGML_AVX_VNNI)
            endif()
            if (GGML_BMI2)
                # MSVC does not define macro __BMI2__
                list(APPEND ARCH_DEFINITIONS __BMI2__ GGML_BMI2)
            endif()
        else ()
            if (GGML_NATIVE)
                list(APPEND ARCH_FLAGS -march=native)
            else ()
                if (GGML_SSE42)
                    list(APPEND ARCH_FLAGS -msse4.2)
                    list(APPEND ARCH_DEFINITIONS GGML_SSE42)
                endif()
                if (GGML_F16C)
                    list(APPEND ARCH_FLAGS -mf16c)
                    list(APPEND ARCH_DEFINITIONS GGML_F16C)
                endif()
                if (GGML_FMA)
                    list(APPEND ARCH_FLAGS -mfma)
                    list(APPEND ARCH_DEFINITIONS GGML_FMA)
                endif()
                if (GGML_BMI2)
                    list(APPEND ARCH_FLAGS -mbmi2)
                    list(APPEND ARCH_DEFINITIONS GGML_BMI2)
                endif()
                if (GGML_AVX)
                    list(APPEND ARCH_FLAGS -mavx)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX)
                endif()
                if (GGML_AVX2)
                    list(APPEND ARCH_FLAGS -mavx2)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX2)
                endif()
                if (GGML_AVX_VNNI)
                    list(APPEND ARCH_FLAGS -mavxvnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX_VNNI)
                endif()
                if (GGML_AVX512)
                    list(APPEND ARCH_FLAGS -mavx512f)
                    list(APPEND ARCH_FLAGS -mavx512cd)
                    list(APPEND ARCH_FLAGS -mavx512vl)
                    list(APPEND ARCH_FLAGS -mavx512dq)
                    list(APPEND ARCH_FLAGS -mavx512bw)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                endif()
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_FLAGS -mavx512vbmi)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VBMI)
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_FLAGS -mavx512vnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VNNI)
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_FLAGS -mavx512bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_BF16)
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_FLAGS -mamx-tile)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_FLAGS -mamx-int8)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_FLAGS -mamx-bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_BF16)
                endif()
            endif()
        endif()

        if (GGML_BACKEND_DL)
            if (GGML_NATIVE)
                # the feature check relies on ARCH_DEFINITIONS, but it is not set with GGML_NATIVE
                message(FATAL_ERROR "GGML_NATIVE is not compatible with GGML_BACKEND_DL, consider using GGML_CPU_ALL_VARIANTS")
            endif()
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} x86 ${ARCH_DEFINITIONS})
        endif()
    else()
        message(WARNING "Unknown CPU architecture. Falling back to generic implementations.")
        list(APPEND ARCH_FLAGS -DGGML_CPU_GENERIC)
    endif()

    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()

    if (GGML_CPU_KLEIDIAI)
        # upstream repo requires at least cmake 3.16
        if (CMAKE_VERSION VERSION_LESS 3.16)
            message(FATAL_ERROR "GGML_CPU_KLEIDIAI requires CMake >= 3.16")
        endif()

        set(GGML_CPU_KLEIDIAI_AARCH64 OFF)
        if (GGML_SYSTEM_ARCH STREQUAL "ARM" AND
            (APPLE OR WIN32 OR CMAKE_SYSTEM_NAME MATCHES "^(Linux|Android|OHOS)$") AND
            (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|arm64-v8a)$" OR
             CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR
             CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
             CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a"))
            set(GGML_CPU_KLEIDIAI_AARCH64 ON)
        endif()
        if (NOT GGML_CPU_KLEIDIAI_AARCH64)
            message(FATAL_ERROR "GGML_CPU_KLEIDIAI requires a Linux, Android, Apple, or Windows AArch64/arm64 target")
        endif()

        message(STATUS "Using KleidiAI optimized kernels if applicable")

        # Fetch KleidiAI sources:
        include(FetchContent)
        set(KLEIDIAI_COMMIT_TAG "v1.32.0")
        set(KLEIDIAI_DOWNLOAD_URL "https://github.com/ARM-software/kleidiai/releases/download/${KLEIDIAI_COMMIT_TAG}/kleidiai-${KLEIDIAI_COMMIT_TAG}-src.tar.gz")
        set(KLEIDIAI_RELEASE_ARCHIVE_MD5  "05f1953428ffe8e8c7f585dc6d508887")

        set(KLEIDIAI_FETCH_ARGS
            URL ${KLEIDIAI_DOWNLOAD_URL}
            URL_HASH MD5=${KLEIDIAI_RELEASE_ARCHIVE_MD5}
        )
        if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            list(APPEND KLEIDIAI_FETCH_ARGS DOWNLOAD_EXTRACT_TIMESTAMP NEW)
        endif()

        FetchContent_Declare(kleidiai
            ${KLEIDIAI_FETCH_ARGS}
        )

        # Disable tests and benchmark building
        set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

        # Use the Populate/add_subdirectory flow for compatibility with CMake 3.16.
        FetchContent_GetProperties(kleidiai
            SOURCE_DIR KLEIDIAI_SRC
            BINARY_DIR KLEIDIAI_BIN
            POPULATED KLEIDIAI_POPULATED
        )
        if (NOT KLEIDIAI_POPULATED)
            FetchContent_Populate(kleidiai)
            FetchContent_GetProperties(kleidiai
                SOURCE_DIR KLEIDIAI_SRC
                BINARY_DIR KLEIDIAI_BIN
            )
        endif()

        if (NOT TARGET kleidiai)
            add_subdirectory(
                "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu/kleidiai"
                "${CMAKE_CURRENT_BINARY_DIR}/kleidiai-wrapper"
                EXCLUDE_FROM_ALL
            )
            if (NOT CMAKE_SKIP_INSTALL_RULES AND
                (NOT DEFINED BUILD_SHARED_LIBS OR NOT BUILD_SHARED_LIBS))
                install(TARGETS kleidiai ARCHIVE)
            endif()
        endif()

        if (NOT TARGET kleidiai)
            message(FATAL_ERROR "KleidiAI target was not created")
        endif()

        set_target_properties(kleidiai PROPERTIES POSITION_INDEPENDENT_CODE ON)

        target_link_libraries(${GGML_CPU_NAME} PRIVATE kleidiai)

        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_KLEIDIAI)

        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/kleidiai/kleidiai.cpp
            ggml-cpu/kleidiai/kernels.cpp
            ggml-cpu/kleidiai/kleidiai.h
            ggml-cpu/kleidiai/kernels.h
            )
    endif()

    message(STATUS "Adding CPU backend variant ${GGML_CPU_NAME}: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})
endfunction()
