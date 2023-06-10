cmake_minimum_required(VERSION 3.26)
project(prism LANGUAGES CXX CUDA)

add_executable(prism
    src/main.cu
)
target_include_directories(prism PRIVATE src)
target_compile_features(prism PUBLIC cxx_std_11)

add_subdirectory(ext)
target_link_libraries(prism PRIVATE ext)
