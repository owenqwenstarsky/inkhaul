set(TARGET llama-eval-callback)
add_executable(${TARGET} eval-callback.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_BUILD_TESTS)
    # uses the tiny Inkling 2 model from the tests/ generate-models fixture
    # LongHaul needs a GPU device for the expert cache, so the layers are offloaded
    set(MODEL_DEST "${CMAKE_BINARY_DIR}/tests/test-models/inkling2-moe.gguf")
    set(TEST_TARGET test-eval-callback)
    add_test(NAME ${TEST_TARGET} COMMAND llama-eval-callback -m "${MODEL_DEST}" --prompt hello --seed 42 -ngl 99)
    set_tests_properties(${TEST_TARGET} PROPERTIES FIXTURES_REQUIRED generate-models)
endif()
