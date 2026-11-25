set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS AND UNIX AND NOT GGML_BACKEND_DL)
    add_executable(test-rpc-multi-server ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.cpp)
    target_link_libraries(test-rpc-multi-server PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-multi-server PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(
        NAME test-rpc-multi-server
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-multi-server>)
    set_property(TEST test-rpc-multi-server PROPERTY LABELS rpc)
    add_executable(test-rpc-unreachable ${PROJECT_SOURCE_DIR}/tests/test-rpc-unreachable.cpp)
    target_link_libraries(test-rpc-unreachable PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-unreachable PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(NAME test-rpc-unreachable COMMAND test-rpc-unreachable)
    set_property(TEST test-rpc-unreachable PROPERTY LABELS main)
    add_executable(test-rpc-fingerprint ${PROJECT_SOURCE_DIR}/tests/test-rpc-fingerprint.cpp)
    target_link_libraries(test-rpc-fingerprint PRIVATE ggml ggml-rpc)
    target_compile_features(test-rpc-fingerprint PRIVATE cxx_std_17)
    add_test(NAME test-rpc-fingerprint COMMAND test-rpc-fingerprint)
    set_property(TEST test-rpc-fingerprint PROPERTY LABELS main)
    # starts an RPC server on a thread, listening on 127.0.0.1
    add_executable(test-rpc-loopback ${PROJECT_SOURCE_DIR}/tests/test-rpc-loopback.cpp)
    target_link_libraries(test-rpc-loopback PRIVATE ggml ggml-rpc)
    target_compile_features(test-rpc-loopback PRIVATE cxx_std_17)
    add_test(NAME test-rpc-loopback COMMAND test-rpc-loopback)
    set_property(TEST test-rpc-loopback PROPERTY LABELS rpc)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
