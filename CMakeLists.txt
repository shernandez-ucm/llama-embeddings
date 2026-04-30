cmake_minimum_required(VERSION 3.12)
project(llama-embeddings)

# Add llama.cpp subdirectory
include(FetchContent)

# 1. Set the acceleration variables BEFORE MakeAvailable
# Use CACHE INTERNAL to override the default 'option()' in llama.cpp
set(GGML_METAL ON CACHE INTERNAL "Enable Metal for Apple Silicon")
set(GGML_CUDA OFF CACHE INTERNAL "Disable CUDA if not needed")

# 2. Add other specific llama.cpp build options
set(LLAMA_BUILD_EXAMPLES OFF CACHE INTERNAL "Don't build llama.cpp example binaries")
set(LLAMA_BUILD_SERVER OFF CACHE INTERNAL "Don't build the server binary")
set(LLAMA_BUILD_COMMON ON CACHE INTERNAL "Build common utilities library")

# 2. Declare the remote repository details
FetchContent_Declare(
  llama
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
  GIT_TAG        master # Or a specific tag/commit hash for stability
  #SOURCE_DIR     "${CMAKE_SOURCE_DIR}/external/llama.cpp" # Downloads here
  SOURCE_DIR     "/Users/sergio/code/llama.cpp" # Downloads here
)
# Set include directory for llama.cpp
#set(LLAMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp")
#set(LLAMA_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/common")
set(LLAMA_INCLUDE_DIR "/Users/sergio/code/llama.cpp")
set(LLAMA_COMMON_DIR "/Users/sergio/code/llama.cpp/common")


# 3. Make it available (downloads and adds to your build tree)
FetchContent_MakeAvailable(llama)
llama_add_compile_flags()
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
include(FindPkgConfig)

# Find jsoncpp using pkg-config
pkg_check_modules(JSONCPP jsoncpp)
if(NOT JSONCPP_FOUND)
    message(FATAL_ERROR "jsoncpp not found. Install it with: sudo apt install libjsoncpp-dev")
endif()


set(TARGET llama-embedding)
add_executable(${TARGET} embedding.cpp)
target_include_directories(${TARGET} PRIVATE ${LLAMA_INCLUDE_DIR} ${LLAMA_COMMON_DIR})
target_link_libraries(${TARGET} PRIVATE llama llama-common Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)

add_executable(json_embeddings json_embeddings.cpp)
# Link libraries
target_include_directories(json_embeddings PRIVATE ${JSONCPP_INCLUDE_DIRS})
target_link_directories(json_embeddings PRIVATE ${JSONCPP_LIBRARY_DIRS})
target_link_libraries(json_embeddings 
    PRIVATE
        llama 
        llama-common 
        Threads::Threads 
        ${JSONCPP_LIBRARIES}
        Eigen3::Eigen
)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "jsoncpp include: ${JSONCPP_INCLUDE_DIRS}")
message(STATUS "jsoncpp libraries: ${JSONCPP_LIBRARIES}")