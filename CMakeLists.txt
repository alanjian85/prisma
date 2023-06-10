cmake_minimum_required(VERSION 3.26)
project(prism LANGUAGES CXX CUDA)

add_executable(prism
    src/main.cu
)

target_include_directories(prism PRIVATE src)

add_subdirectory(ext)
target_link_libraries(prism PRIVATE ext)
