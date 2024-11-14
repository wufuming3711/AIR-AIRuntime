#ifndef INCLUDE_PROCESS_LETTERBOX_H
#define INCLUDE_PROCESS_LETTERBOX_H

#include <iostream>
#include <opencv2/opencv.hpp>

#include "algorithmBase.h"
#include "networkSpace.h"

void letterbox(
    const cv::Mat& oriImage, 
    cv::Mat& inputImage, 
    cv::Size& size, 
    networkSpace::PreprocessParser& preParser
);

#endif // INCLUDE_PROCESS_LETTERBOX_H