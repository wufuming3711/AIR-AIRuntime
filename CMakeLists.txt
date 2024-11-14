cmake_minimum_required(VERSION 3.1)
project(runtime LANGUAGES CXX CUDA)

# 设置CUDA架构，避免硬编码
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")
message(STATUS "CUDA Lib Dir: \n${CUDA_LIB_DIR}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

#============================
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/three_party/grpc/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/three_party/grpc/lib")

set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/three_party/grpc/lib/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/three_party/grpc/lib/cmake/grpc")

# absl
set(absl_DIR "${CMAKE_CURRENT_SOURCE_DIR}/three_party/grpc/lib/cmake/absl")
message(STATUS ">>>>>>>>>>>>>>>: \n${absl_DIR}\n")
find_package(absl REQUIRED)
message(STATUS "absl Libs: \n${absl_LIBS}\n")
message(STATUS "absl Libraries: \n${absl_LIBRARIES}\n")
message(STATUS "absl Headers: \n${absl_INCLUDE_DIRS}\n")

# utf8_range
set(utf8_range_DIR "${CMAKE_CURRENT_SOURCE_DIR}/three_party/grpc/lib/cmake/utf8_range")
find_package(utf8_range REQUIRED)
message(STATUS "utf8_range Libs: \n${absl_LIBS}\n")
message(STATUS "utf8_range Libraries: \n${absl_LIBRARIES}\n")
message(STATUS "utf8_range Headers: \n${absl_INCLUDE_DIRS}\n")
message(STATUS "utf8_range_INCLUDE_DIRS: ${utf8_range_INCLUDE_DIRS}\n")

set(protobuf_DIR "${CMAKE_CURRENT_SOURCE_DIR}/three_party/grpc/lib/cmake/protobuf")
find_package(protobuf REQUIRED)
message(STATUS "protobuf_INCLUDE_DIRS: ${protobuf_INCLUDE_DIRS}\n")

set(gRPC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/three_party/grpc/lib/cmake/grpc")
find_package(gRPC REQUIRED)
message(STATUS "gRPC_INCLUDE_DIRS: ${gRPC_INCLUDE_DIRS}\n")
#============================

# TensorRT
set(TensorRT_INCLUDE_DIRS 
    /usr/include/x86_64-linux-gnu  
    /usr/local/TensorRT-8.6.1.6/include
)

# 将库名称添加到TensorRT_LIBRARIES列表中
list(APPEND TensorRT_LIBRARIES nvinfer nvinfer_plugin)
set(TensorRT_LIB_DIR 
    /usr/lib/x86_64-linux-gnu  
    /usr/local/TensorRT-8.6.1.6/lib
)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Lib Dir: \n${TensorRT_LIB_DIR}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")



list(APPEND ALL_LIBS
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${TensorRT_LIBRARIES}
)
message(STATUS "ALL_LIBS: \n${ALL_LIBS}\n")

list(APPEND ALL_LIB_DIRS
    ${CUDA_LIB_DIR}
    ${TensorRT_LIB_DIR}
)
message(STATUS "ALL_LIB_DIRS: \n${ALL_LIB_DIRS}\n")

# 添加源文件
file(GLOB_RECURSE 
    SOURCES
    "*.cc" "*.cpp"
    "src/*.cc" "src/*.cpp"
    "src/utils/*.cpp"
    "src/models/*.cc" "src/models/*.cpp"
    "src/process/*.cc" "src/process/*.cpp"
    "proto/*.cc"
)

file(GLOB_RECURSE 
    HEADERS  
    "config/*.h"
    "include/*.h" 
    "include/*.inl"
    "include/models/*.h"
    "include/process/*.h"
    "include/structSet/*.h"
    "include/workflow/*.h"
    "include/utils/*.h"
    "include/common/*.h"
    "three_party/include/*.h" "three_party/include/*.hpp"
    "proto/*.h"
)

list(FILTER SOURCES EXCLUDE REGEX "CMakeFiles")
list(FILTER SOURCES EXCLUDE REGEX "CMakeCXXCompilerId")
list(FILTER SOURCES EXCLUDE REGEX "CMakeCUDACompilerId")



list(APPEND ALL_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${absl_INCLUDE_DIRS}
    ${utf8_range_INCLUDE_DIRS}
    ${protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
    three_party/include
    ./${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/include/process
    ${CMAKE_CURRENT_SOURCE_DIR}/include/models
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structSet
    ${CMAKE_CURRENT_SOURCE_DIR}/include/workflow
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
    # ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# include_directories(${ALL_INCLUDE_DIRS} )
# 添加动态库
add_library(runtime SHARED  ${SOURCES}  ${HEADERS} )

# 设置输出库的名称
set_target_properties(runtime PROPERTIES OUTPUT_NAME "runtime")


# 配置目标依赖
target_include_directories(runtime PUBLIC ${ALL_INCLUDE_DIRS} )

target_link_directories(runtime PUBLIC ${ALL_LIB_DIRS}  three_party/lib)


target_link_libraries(runtime PRIVATE ${ALL_LIBS} 
    cjson
    absl::flags
    absl::flags_parse
    absl::strings
    gRPC::grpc++
    protobuf::libprotobuf
)