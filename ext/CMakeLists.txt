project(ext)

add_subdirectory(tinyobjloader)

find_package(Lua REQUIRED)

add_library(ext stb.c)
target_include_directories(ext PUBLIC stb ${LUA_INCLUDE_DIR})
target_link_libraries(ext PUBLIC tinyobjloader ${LUA_LIBRARIES})
