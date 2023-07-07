cmake_minimum_required(VERSION 3.22.1)
project(mdp-sim VERSION 1.0)

# find_package(Python3 REQUIRED)
set(Python3_INCLUDE_DIRS "/usr/include/python3.10")
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(src)

find_package(Python3 REQUIRED COMPONENTS Development)

set(SOURCES
    src/mdp.cpp
    src/algorithms.cpp
)

add_executable(riverswim.exe tests/riverswim.cpp ${SOURCES})
target_link_libraries(riverswim.exe PRIVATE Python3::Python)
add_executable(coprime_steps.exe tests/coprime_steps.cpp ${SOURCES})