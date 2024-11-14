#ifndef INCLUDE_PROCESS_RESIZENORMALIZE_H
#define INCLUDE_PROCESS_RESIZENORMALIZE_H

#include <iostream>
#include <opencv2/opencv.hpp>

#include "algorithmBase.h"
#include "networkSpace.h"

void resizeNormalize(
    const cv::Mat& oriImage, 
    cv::Mat& inputImage, 
    cv::Size& size, 
    networkSpace::PreprocessParser& preParser
);

#endif // INCLUDE_PROCESS_RESIZENORMALIZE_H