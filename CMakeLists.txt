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
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/glut64.lib"
)

add_library(CPP_LIBS STATIC "${CMAKE_SOURCE_DIR}/lib/book_cpp.cu")

add_library(LABS STATIC 
    "${CMAKE_SOURCE_DIR}/labs/ch_3.cu" 
    "${CMAKE_SOURCE_DIR}/labs/ch_4.cu" )

target_include_directories(CPP_LIBS PUBLIC ${CMAKE_SOURCE_DIR}/common)

target_include_directories(LABS PUBLIC ${CMAKE_SOURCE_DIR}/common)

target_include_directories(Main PUBLIC ${CMAKE_SOURCE_DIR}/common)

target_link_libraries(Main PRIVATE GLUT LABS CPP_LIBS)