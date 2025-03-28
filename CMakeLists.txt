if(NOT DEFINED PROJECT_NAME)
    set(PROJECT_NAME "analysis_service")  # 默认项目名称
endif()

cmake_minimum_required(VERSION 3.1)
project(${PROJECT_NAME} C CXX CUDA)

# ----------------手动打包指定头文件与源文件----------------
list(APPEND PROTOBUF_SHARE_SOURCE_LIST
    src/model/task_exchange.grpc.pb.cc
    src/model/task_exchange.pb.cc
)
list(APPEND PROTOBUF_SHARE_INCLUDE_LIST
    src/model/task_exchange.grpc.pb.cc
    src/model/task_exchange.pb.cc
)
list(APPEND RUNTIME_SOURCES_LIST
    src/workflow/workflowBase.cc
    src/common/gpu_resource_manager.cc
    src/common/logger.cc
    src/models/algorithmBase.cc
    src/models/modelCls.cc
    src/models/modelDet.cc
    src/preprocess/resizeNormalize.cc
    src/preprocess/letterbox.cc
    src/utils/loadJsonConfig.cc
    src/workflow/runtime.cc
)

list(APPEND RUNTIME_INCLUDES_LIST
    src/preprocess/resizeNormalize.h
    src/preprocess/letterbox.h
    src/structSet/workflow.h
    src/structSet/networkSpace.h
    src/utils/loadJsonConfig.h
    src/workflow/workflowBase.h
    src/workflow/runtime.h
    src/common/gpu_resource_manager.h
    src/common/common.h
    src/common/logger.h
    src/models/modelCls.h
    src/models/modelDet.h
    src/models/algorithmBase.h
    src/models/opt.h
    src/model/task_exchange.grpc.pb.h
    src/model/task_exchange.pb.h
    3rd/include/cJSON.h
)
list(APPEND GRPC_SOURCES_LIST
    src/conf/Config.cpp
    src/model/task_exchange.grpc.pb.cc
    src/service/Service.cpp
    src/tools/File.cpp
    src/tools/TrackableStatus.cpp
    src/service/StreamHandle.cpp
    src/service/ai/Manager.cpp
    src/service/ffmpeg/ffmpeg_linker.cpp
    src/service/ffmpeg/ipc.c
    src/service/ffmpeg/decoder_wrapper.c
    src/service/FFMPEGCallbackHelper.cpp
)
list(APPEND GRPC_INCLUDES_LIST
    src/conf/Config.h
    src/service/Service.h
    src/tools/File.h
    src/tools/TrackableStatus.h
    src/service/StreamHandle.h
    src/service/ai/Manager.h
    src/service/ffmpeg/ffmpeg_linker.h
    src/service/ffmpeg/ipc.h
    src/service/ffmpeg/decoder_wrapper.h
    src/service/FFMPEGCallbackHelper.h
)
# ^^^^^^^^^^^^^^^^手动打包指定头文件与源文件^^^^^^^^^^^^^^^^

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g ${ASAN_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g ${ASAN_FLAGS}")
else()
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG ${ASAN_FLAGS}" CACHE STRING "C compiler flags" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG ${ASAN_FLAGS}" CACHE STRING "C++ compiler flags" FORCE)
endif()

# 设置CUDA架构，避免硬编码
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CMAKE_CXX_STANDARD 20)

# CUDA
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)

# NVML
find_library(NVML_LIBRARY
             NAMES nvidia-ml libnvidia-ml.so
             PATHS /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs
             NO_DEFAULT_PATH)

find_path(NVML_INCLUDE_DIR
          NAMES nvml.h
          PATHS /usr/include /usr/local/cuda-11.8/targets/x86_64-linux/include
          NO_DEFAULT_PATH)

if (NOT NVML_LIBRARY OR NOT NVML_INCLUDE_DIR)
    message(FATAL_ERROR "NVML library or include directory not found")
endif()

# TensorRT
set(TensorRT_INCLUDE_DIRS 
    /usr/include/x86_64-linux-gnu  
    /usr/local/TensorRT-8.6.1.6/include
)
list(APPEND TensorRT_LIBRARIES
    nvinfer nvinfer_plugin)
set(TensorRT_LIB_DIR 
    /usr/lib/x86_64-linux-gnu  
    /usr/local/TensorRT-8.6.1.6/lib
)

# OpenCV
find_package(OpenCV REQUIRED)

# gRPC and related libraries
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/lib")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rd/ffmpeg/lib")
set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/lib/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/lib/cmake/grpc")
set(absl_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/lib/cmake/absl")
find_package(absl REQUIRED)
set(utf8_range_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/lib/cmake/utf8_range")
find_package(utf8_range REQUIRED)
set(protobuf_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/lib/cmake/protobuf")
find_package(protobuf REQUIRED)
set(gRPC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/lib/cmake/grpc")
find_package(gRPC REQUIRED)

# 添加三方头文件
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${absl_INCLUDE_DIRS})
include_directories(${utf8_range_INCLUDE_DIRS})
include_directories(${protobuf_INCLUDE_DIRS})
include_directories(${gRPC_INCLUDE_DIRS})
include_directories(${NVML_INCLUDE_DIR})

# 添加工程三方头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/toml)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/ffmpeg/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/opencv/include/opencv4)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/grpc/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/toml)

# 添加开发头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/ai)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/models)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/structSet)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/workflow)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/config)

# 添加全局依赖库 动态库
link_directories(
    ${CUDA_LIB_DIR}
    ${TensorRT_LIB_DIR}
    /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs
)

add_executable(saida_decoder
    src/service/ffmpeg/main.c
    src/service/ffmpeg/ffmpeg_decode.c
    src/service/ffmpeg/ipc.c
)
target_compile_options(saida_decoder 
    PRIVATE
        -O3
        -march=native
        -DNDEBUG
        -Wall
        ${ASAN_FLAGS}
        ${FRAME_POINTER_FLAG}
)
target_link_libraries(saida_decoder
    swscale
    avformat
    avcodec
    avutil
    m
    z
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    # pb协议 源文件
    ${PROTOBUF_SHARE_SOURCE_LIST}
    ${PROTOBUF_SHARE_INCLUDE_LIST}
    # RUNTIME 源文件
    ${RUNTIME_SOURCES_LIST}
    ${RUNTIME_INCLUDES_LIST}
    # grpc 源文件
    ${GRPC_SOURCES_LIST}
    ${GRPC_INCLUDES_LIST}
)
target_include_directories(${PROJECT_NAME} 
    PRIVATE
)
target_link_libraries(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/lib/libcjson.a
    absl::flags
    absl::flags_parse
    absl::strings
    gRPC::grpc++
    protobuf::libprotobuf
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NVML_LIBRARY}
)

# 编译调试程序
add_executable(RUNTIME
    src/runtime_main.cc
    # pb协议 源文件
    ${PROTOBUF_SHARE_SOURCE_LIST}
    ${PROTOBUF_SHARE_INCLUDE_LIST}
    # RUNTIME 源文件
    ${RUNTIME_SOURCES_LIST}
    ${RUNTIME_INCLUDES_LIST}
)
target_link_libraries(RUNTIME
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/lib/libcjson.a
    absl::flags
    absl::flags_parse
    absl::strings
    gRPC::grpc++
    protobuf::libprotobuf
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NVML_LIBRARY}
)