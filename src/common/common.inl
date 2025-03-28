#ifndef INCLUDE_COMMON_COMMON_INL
#define INCLUDE_COMMON_COMMON_INL

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <cstdarg>  // 用于 va_list, va_start, va_end, vsnprintf 
#include <memory>
#include <cstddef>
#include <cstdio>
#include <string>
#include <sstream>

#include "workflow.h"

constexpr size_t INVALID_GPU_ID = static_cast<size_t>(-1);

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

#define RUNTIME_LOG(logger, severity, msg)                                         \
    do {                                                                           \
        if ((logger)) {                                                            \
            (logger)->logWithLocation((severity), (msg), __FILE__, __LINE__);      \
        }                                                                          \
    } while (0)


#define RUNTIME_CHECK(logger, call)                                                \
    do {                                                                           \
        const cudaError_t error_code = (call);                                     \
        if (error_code != cudaSuccess) {                                           \
            std::string errorText = cudaGetErrorString(error_code);                \
            std::string errorMessage = "CUDA Error: " + errorText + " in " #call;  \
            RUNTIME_LOG(                                                           \
                logger,                                                            \
                nvinfer1::ILogger::Severity::kERROR,                               \
                errorMessage.c_str());                                             \
        }                                                                          \
    } while (0)

inline const char* trtVersion() {
    static std::string versionStr;
    if (versionStr.empty()) {
        std::ostringstream oss;
        oss << "TensorRT version: " << NV_TENSORRT_MAJOR 
            << "." << NV_TENSORRT_MINOR 
            << "." << NV_TENSORRT_PATCH;
        versionStr = oss.str();
    }
    return versionStr.c_str();
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
    result = first;
    result += second;
}

inline std::string format_to_string(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);

    int len = vsnprintf(nullptr, 0, fmt, args);
    va_end(args);

    if (len < 0) {
        va_end(args_copy);
        return "";
    }

    std::string buffer(len + 1, '\0');
    vsnprintf(&buffer[0], buffer.size(), fmt, args_copy);
    va_end(args_copy);

    return buffer;
}

#endif  // INCLUDE_COMMON_COMMON_INL