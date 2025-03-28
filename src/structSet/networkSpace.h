#ifndef INCLUDE_STRUCTSET_NETWORKSPACE_H
#define INCLUDE_STRUCTSET_NETWORKSPACE_H

#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "opencv2/opencv.hpp"


namespace network_space {

// 记录常见算法模块类型
enum AlgoType {
    Detection,
    Classification,
    ImageCrop,
    ChangeDetect,
};

struct Binding {
    size_t size_i  = 0;
    size_t dsize_i = 0;
    const char* name_c = nullptr;
    nvinfer1::Dims dims;
};

struct EngineParser{
    size_t               iNumBindings_ = 0;

    std::vector<std::vector<int>> vviInputSizeHW_;   
    size_t               iNumInputs_   = 0; 
    size_t               iNumOutputs_  = 0;
    std::vector<Binding> vbindInputBindings_;
    std::vector<Binding> vbinOutputBindings_;
    size_t               iMaxBatch_  = 32;
    size_t               iBestBatch_ = 16;
    size_t               iCurtBatch_ = 1;
    size_t               iOutClsNum_ = 0;
    std::vector<void*> vvoidptrDeviceIns_;
    std::vector<void*> vvoidptrDeviceOuts_;
    std::vector<void*> vvoidptrHostOuts_;

    void reset_EngineParser_vvoidptrX() {
        for (auto& ptr : vvoidptrDeviceIns_) {
            if (ptr != nullptr) {
                cudaFree(ptr);
                ptr = nullptr;
            }
        }
        vvoidptrDeviceIns_.clear();
        for (auto& ptr : vvoidptrDeviceOuts_) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
        }
        vvoidptrDeviceOuts_.clear();
        for (auto& ptr : vvoidptrHostOuts_) {
            if (ptr != nullptr) {
                cudaFreeHost(ptr);
            }
        }
        vvoidptrHostOuts_.clear();
    }
};

struct PreprocessParser{
    cv::Size size;
    size_t iOriImgHeight_ = 0;
    size_t iOriImgWidth_  = 0;
    float ratio_f         = 1.0f;
    float padw_f          = 0.0f;
    float padh_f          = 0.0f;
};

struct InputData{
    InputData(PreprocessParser& preParser)
        : preParser(preParser), inputImage(), oriImage() {}

    PreprocessParser preParser;
    cv::Mat oriImage;   
    cv::Mat inputImage; 
};

struct Object {
    cv::Rect_<float> rect;
    cv::Mat cvmatCropImage_;
    size_t label_i = 0;
    float prob_f  = 0.0;
    void reset(){
        this->label_i = 0;
        this->prob_f  = 0.0;
    }
};

struct InOutPutData {
    std::vector<InputData> input;
    std::vector<std::vector<Object>> output;
};

struct BaseAlgoParser{
    EngineParser nvptrEngine_Parser;
    InOutPutData inOutPutData;
};

}  // namespace networkSpace
#endif  // INCLUDE_STRUCTSET_NETWORKSPACE_H