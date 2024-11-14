#ifndef INCLUDE_MODELS_MODELCLS_H
#define INCLUDE_MODELS_MODELCLS_H

#include "NvInferPlugin.h"
#include <fstream>
#include <cmath>      // 包含exp函数的头文件
#include <algorithm>  // 包含max_element函数

#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "resizeNormalize.h"


class CLASSIFIER : public ALGORITHM_BASE
{
public:
    explicit CLASSIFIER(const std::string& engineFilePath) : ALGORITHM_BASE(engineFilePath) {}
    ~CLASSIFIER() override {
        std::cout << "CLASSIFIER destructor called." << std::endl;
    }

    bool commitImages(const std::vector<cv::Mat>& images);
    void postprocess();
    void draw_boxes();
};


#endif  // INCLUDE_MODELS_MODELCLS_H