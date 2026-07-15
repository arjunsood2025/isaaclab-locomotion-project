cmake_minimum_required(VERSION 3.18)
project(policy_loop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# point CMake at libtorch:  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

add_executable(policy_loop main.cpp)
target_link_libraries(policy_loop "${TORCH_LIBRARIES}")
target_compile_options(policy_loop PRIVATE ${TORCH_CXX_FLAGS})
