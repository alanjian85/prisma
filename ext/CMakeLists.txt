project(ext)

add_subdirectory(sol2)

add_library(ext stb.c)
target_include_directories(ext PUBLIC stb)
target_link_libraries(ext PUBLIC sol2)
