cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nogo)

#set(CUDNN_INCLUDE_DIR "/usr/lib/cuda")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3")

add_executable(nogo nogo.cpp)
target_link_libraries(nogo "${TORCH_LIBRARIES}")
set_property(TARGET nogo PROPERTY CXX_STANDARD 17)
