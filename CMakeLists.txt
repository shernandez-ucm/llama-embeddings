cmake_minimum_required(VERSION 3.12)
project(llama-embeddings)

# Add llama.cpp subdirectory
set(LLAMA_BUILD_COMMON ON CACHE INTERNAL "")
add_subdirectory(llama.cpp)

# Set include directory for llama.cpp
set(LLAMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
set(LLAMA_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common")
llama_add_compile_flags()
find_package(Threads REQUIRED)

set(TARGET llama-embedding)
add_executable(${TARGET} embedding.cpp)
target_include_directories(${TARGET} PRIVATE ${LLAMA_INCLUDE_DIR} ${LLAMA_COMMON_DIR})
target_link_libraries(${TARGET} PRIVATE llama llama-common Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)

set(TARGET llama-embedding-md)
add_executable(${TARGET} embedding_md.cpp)
target_include_directories(${TARGET} PRIVATE ${LLAMA_INCLUDE_DIR} ${LLAMA_COMMON_DIR})
target_link_libraries(${TARGET} PRIVATE llama llama-common Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)

# New latam-embeddings target
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
find_package(Eigen3 REQUIRED)

set(TARGET latam-embeddings)
add_executable(${TARGET} latam_embeddings.cpp)
target_include_directories(${TARGET} PRIVATE 
    ${LLAMA_INCLUDE_DIR} 
    ${LLAMA_COMMON_DIR} 
    ${JSONCPP_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(${TARGET} PRIVATE 
    llama 
    llama-common 
    Threads::Threads 
    ${JSONCPP_LIBRARIES}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)
