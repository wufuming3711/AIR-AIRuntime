#ifndef INCLUDE_COMMON_UTILS_RUNTIME_H
#define INCLUDE_COMMON_UTILS_RUNTIME_H

#include <vector>
#include <opencv2/opencv.hpp>


#include "task_exchange.pb.h"

class Interface {
public:
    Interface(pb::DetectionAlgorithm ALGONAME, size_t gpuId);

    ~Interface();

    // static int getGPUUsagePercentageMinusOneForAlgorithm(pb::DetectionAlgorithm algorithm) {
    //     return 99;
    // };

    bool analysisSingle(cv::Mat& oriImage, pb::OnAIResultGotReply::ResultWrapper& resultWrapper);
};


#endif  // INCLUDE_COMMON_UTILS_RUNTIME_H