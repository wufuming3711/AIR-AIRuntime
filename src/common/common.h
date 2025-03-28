#ifndef INCLUDE_COMMON_COMMON_H
#define INCLUDE_COMMON_COMMON_H

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"


class Logger: public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO):
        reportableSeverity(severity)
    {}

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};

#endif  // INCLUDE_COMMON_COMMON_H