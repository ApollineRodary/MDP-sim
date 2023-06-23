cmake_minimum_required(VERSION 3.22.1)
project(mdp-sim VERSION 1.0)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(src)

set(SOURCES
    src/mdp.cpp
    src/algorithms.cpp
)

add_executable(riverswim.exe tests/riverswim.cpp ${SOURCES})
add_executable(coprime_steps.exe tests/coprime_steps.cpp ${SOURCES})