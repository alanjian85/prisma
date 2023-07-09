cmake_minimum_required(VERSION 3.26)
project(prism LANGUAGES CXX CUDA)

add_executable(prism
    src/main.cu
    src/camera/camera.cu
    src/core/bounds.cu
    src/scene/bvh.cu
    src/scene/triangle.cu
)
target_include_directories(prism PRIVATE src)
target_compile_features(prism PUBLIC cxx_std_11)
set_target_properties(prism PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(ext)
target_link_libraries(prism PRIVATE ext)
