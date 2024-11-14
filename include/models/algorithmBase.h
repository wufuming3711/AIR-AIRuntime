#ifndef INCLUDE_MODELS_ALGORITHMBASE_H
#define INCLUDE_MODELS_ALGORITHMBASE_H

#include "NvInferPlugin.h"
#include <fstream>
#include <cmath>      // 包含exp函数的头文件
#include <algorithm>  // 包含max_element函数

#include "common.h"
#include "algorithmBase.h"
#include "networkSpace.h"


class ALGORITHM_BASE
{

public:
    ALGORITHM_BASE(const std::string& engineFilePath);
    virtual ~ALGORITHM_BASE();

    virtual bool commitImages(
        const std::vector<cv::Mat>& images
    ) = 0;
    virtual void postprocess(
    ) = 0;
    virtual void draw_boxes(
    ) = 0;

    // virtual void singleImageCrop(
    //     const cv::Mat& oriImage, 
    //     const std::vector<networkSpace::Object>& boxes_vec,
    //     std::vector<cv::Mat>& bufferImages_vec,
    //     int padding = 10
    // ) {};
    
    virtual void singleImageCrop(
        const cv::Mat& oriImage, 
        networkSpace::Object& box,
        int padding = 10
    ) {};

    virtual bool loadEngine(const std::string& engineFilePath);
    virtual bool createContext();
    virtual bool setCurtContext(int batchSize, int channel, int imgh, int imgw);
    virtual bool warmup(int warmupNum = 10);
    virtual void inferCore();

    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
    networkSpace::BaseAlgoParser baseAlgoParser;    
};

#endif  // INCLUDE_MODELS_ALGORITHMBASE_H