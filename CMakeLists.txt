cmake_minimum_required(VERSION 3.13)
project(succ)

add_definitions(-w)

set(CMAKE_CXX_STANDARD 20)

configure_file(
  lib/ds2i/ds2i_config.hpp.in
  lib/ds2i/ds2i_config.hpp
  ESCAPE_QUOTES
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -pedantic")

find_package(Boost REQUIRED)

include_directories(include lib/ds2i lib/sux)
add_subdirectory(lib/ds2i/succinct)
add_subdirectory(lib/sdsl-lite)

add_executable(test test.cpp)
target_link_libraries(test sdsl succinct ${Boost_LIBRARIES})
