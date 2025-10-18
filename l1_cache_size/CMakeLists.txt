cmake_minimum_required(VERSION 3.1)
project(l1_cache_size)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-O2")

add_executable(l1_cache_size main.cpp)
target_compile_options(l1_cache_size PRIVATE -Wall -Wextra -Wpedantic -Werror)
