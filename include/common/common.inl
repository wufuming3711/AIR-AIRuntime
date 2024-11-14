#ifndef INCLUDE_COMMON_COMMON_INL
#define INCLUDE_COMMON_COMMON_INL

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring> 

#include "workflow.h"

#define CHECK(call)                                                                \
    do {                                                                           \
        const cudaError_t error_code = call;                                       \
        if (error_code != cudaSuccess) {                                           \
            printf("CUDA Error:\n");                                               \
            printf("    File:       %s\n", __FILE__);                              \
            printf("    Line:       %d\n", __LINE__);                              \
            printf("    Error code: %d\n", error_code);                            \
            printf("    Error text: %s\n", cudaGetErrorString(error_code))<< '\n'; \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

inline void trtVersion(){
    std::cout << "TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;
}

inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string& path)
{
    if (access(path.c_str(), 0) == F_OK) {
        return true;
    }
    return false;
}

inline bool IsFile(const std::string& path)
{
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string& path)
{
    if (!IsPathExist(path)) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

inline void concatenate(const char* first, const char* second, std::string& result) {
    /* 用法示例
    std::string msg = "";
    concatenate(errtype_c, modelName_c, msg);
    */
    result = first; // 将第一个字符串赋值给结果
    result += second; // 追加第二个字符串
}

#endif  // INCLUDE_COMMON_COMMON_INL