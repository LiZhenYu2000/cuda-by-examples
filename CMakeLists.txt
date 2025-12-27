cmake_minimum_required(VERSION 3.29)

project(LearnCUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "75")

message("-- CMake Source Path: ${CMAKE_SOURCE_DIR}")
message("-- Project Source Path: ${PROJECT_SOURCE_DIR}")
message("-- Project Binary Path: ${PROJECT_BINARY_DIR}")

add_executable(Main main.cu)

add_library(GLUT STATIC IMPORTED)
set_target_properties(GLUT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/common"
    IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/lib/glut64.lib"
)

target_include_directories(Main PUBLIC ${CMAKE_SOURCE_DIR}/common)

target_link_libraries(Main PRIVATE GLUT)