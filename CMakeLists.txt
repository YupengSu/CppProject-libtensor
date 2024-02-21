cmake_minimum_required(VERSION 3.8.0)
project(libtensor VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/cuda")
link_directories("/usr/local/cuda/lib64")

include(CTest)
enable_testing()
find_package(CUDAToolkit REQUIRED)

include_directories("include/")
FILE(GLOB SRC "src/*.cpp" "src/*.cu")

add_executable(libtensor ${SRC} main.cpp)
target_link_libraries(libtensor ${CUDAToolkit_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)