#ifndef INCLUDE_MODELS_MODELDER_H
#define INCLUDE_MODELS_MODELDER_H

#include "NvInferPlugin.h"
#include "fstream"
#include <cmath>      // 包含exp函数的头文件
#include <algorithm>  // 包含max_element函数

#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "letterbox.h"


class DETECTOR : public ALGORITHM_BASE
{
public:
    explicit DETECTOR(const std::string& engineFilePath) : ALGORITHM_BASE(engineFilePath) {}
    ~DETECTOR() override {
        std::cout << "DETECTOR destructor called." << std::endl;
    }

    bool commitImages(const std::vector<cv::Mat>& images);
    void postprocess();
    void draw_boxes();

    inline float clamp(float value, float min, float max) {
        return std::max(min, std::min(value, max));
    }
};


#endif  // INCLUDE_MODELS_MODELDER_H