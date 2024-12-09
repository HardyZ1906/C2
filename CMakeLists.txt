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
target_include_directories(test PRIVATE ${Boost_INCLUDE_DIR})

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark sdsl succinct ${Boost_LIBRARIES})
target_include_directories(benchmark PRIVATE ${Boost_INCLUDE_DIR} baseline_coco baseline_fst)

add_executable(profile benchmark.cpp)
target_include_directories(profile PRIVATE ${Boost_INCLUDE_DIR} baseline_coco baseline_fst)
target_link_libraries(profile sdsl succinct ${Boost_LIBRARIES})
target_compile_options(profile PRIVATE "-ggdb3")
