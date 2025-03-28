#ifndef INCLUDE_MODELS_ALGORITHMBASE_H
#define INCLUDE_MODELS_ALGORITHMBASE_H

#include <atomic>
#include "NvInferPlugin.h"
#include <fstream>
#include <cmath>    
#include <algorithm>

#include "common.h"
#include "algorithmBase.h"
#include "networkSpace.h"
#include "logger.h"


class AlgorithmBase
{

public:
    AlgorithmBase(
        const std::string& nvptrEngine_FilePath,
        std::shared_ptr<logger::CustomLogger>& logger
    );
    virtual ~AlgorithmBase();

    virtual void postprocess(
    ) = 0;
    virtual void draw_boxes(size_t save_img_max_num) = 0;
        
    virtual void singleImageCrop(
        const cv::Mat& oriImage, 
        network_space::Object& box,
        int padding = 10
    ) {};

    virtual bool loadEngine(
        const std::string& nvptrEngine_FilePath
    );
    virtual bool createContext();
    virtual bool setCurtContext(
        int batchSize, 
        int channel, 
        int imgh, 
        int imgw
    );
    virtual bool inferCore();

    virtual bool commitImages(
        const std::vector<cv::Mat>& images,
        const char* preprocess
    );

    nvinfer1::ICudaEngine*       nvptrEngine_  = nullptr;
    nvinfer1::IRuntime*          nvptrRuntime_ = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
    std::shared_ptr<logger::CustomLogger>& sptrLogger_;

    network_space::BaseAlgoParser baseAlgoParser;
    std::string                  modelName;
    std::atomic<int> entry_counter = 0;
    bool bIsInitial_ = false;
};

#endif  // INCLUDE_MODELS_ALGORITHMBASE_H