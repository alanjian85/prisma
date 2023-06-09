project(ext)

add_library(ext stb.c)
target_include_directories(ext PUBLIC stb)
