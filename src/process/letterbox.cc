#include <opencv2/opencv.hpp>
#include <cmath>

#include "letterbox.h"
#include "networkSpace.h"


void letterbox(
    const cv::Mat& oriImage, 
    cv::Mat& inputImage, 
    cv::Size& size, 
    networkSpace::PreprocessParser& preParser
){
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = oriImage.rows;
    float       width  = oriImage.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(oriImage, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = oriImage.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    inputImage.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)inputImage.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)inputImage.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)inputImage.data + (int)inp_h * (int)inp_w * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    preParser.ratio_f  = 1 / r;
    preParser.padw_f   = dw;
    preParser.padh_f   = dh;
    preParser.oriImgHeight_i = height;
    preParser.oriImgWidth_i  = width;

    #ifdef DEBUG
    std::cout << "[INFO] letterbox-> preParser.ratio_f = " << preParser.ratio_f << std::endl;
    std::cout << "[INFO] letterbox-> preParser.padw_f = " << preParser.padw_f << std::endl;
    std::cout << "[INFO] letterbox-> preParser.padh_f = " << preParser.padh_f << std::endl;
    std::cout << "[INFO] letterbox-> preParser.oriImgHeight_i = " << preParser.oriImgHeight_i << std::endl;
    std::cout << "[INFO] letterbox-> preParser.oriImgWidth_i = " << preParser.oriImgWidth_i << std::endl;
    #endif
}