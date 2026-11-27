set(TARGET llama-tp-probe)
add_executable(${TARGET} tp-probe.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS)
    # the network check listens on 127.0.0.1 only, on a thread of the test
    add_test(NAME test-tp-probe COMMAND ${TARGET} self-test)
    set_property(TEST test-tp-probe PROPERTY LABELS rpc)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
