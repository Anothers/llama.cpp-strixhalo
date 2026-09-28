include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# MSVC emits a PCH bookkeeping symbol that WINDOWS_EXPORT_ALL_SYMBOLS exports as an ambiguous "__"

set(LLAMA_SERVER_PCH ON)

if (BUILD_SHARED_LIBS AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(LLAMA_SERVER_PCH OFF)
endif()

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
    server-anthropic-content.cpp
    server-anthropic-content.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# Anthropic PDF document blocks: pages rendered with poppler-cpp and encoded with libjpeg (optional;
# without them PDF documents are rejected with a 400)
option(LLAMA_SERVER_PDF "llama-server: render PDF document blocks (needs poppler-cpp and libjpeg)" ON)
if (LLAMA_SERVER_PDF)
    find_package(PkgConfig QUIET)
    find_package(JPEG QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(POPPLER_CPP QUIET IMPORTED_TARGET poppler-cpp)
    endif()
    if (POPPLER_CPP_FOUND AND JPEG_FOUND)
        target_sources(${TARGET} PRIVATE server-pdf.cpp)
        target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVER_PDF)
        target_link_libraries(${TARGET} PRIVATE PkgConfig::POPPLER_CPP JPEG::JPEG)
        message(STATUS "llama-server: PDF document support enabled (poppler-cpp ${POPPLER_CPP_VERSION})")
    else()
        message(STATUS "llama-server: PDF document support disabled (poppler-cpp or libjpeg not found)")
    endif()
endif()

if (LLAMA_SERVER_PCH)
    target_precompile_headers(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/common/common.h)
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_SERVER_PCH)
    target_precompile_headers(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/common/common.h)
endif()

add_dependencies(${TARGET} llama-ui-assets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
