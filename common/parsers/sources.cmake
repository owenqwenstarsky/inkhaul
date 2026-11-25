# Specialized chat template parsers, listed explicitly so that adding or removing one re-runs CMake instead of leaving an incremental build stale.

set(LLAMA_CHAT_PARSERS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/parsers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/parsers.h
    ${CMAKE_CURRENT_LIST_DIR}/inkling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ling3.cpp
)
