#include <opencv2/opencv.hpp>
#include <cmath>

#include "letterbox.h"
#include "resizeNormalize.h"


void resizeNormalize(
    const cv::Mat& oriImage, 
    cv::Mat& inputImage, 
    cv::Size& size, 
    networkSpace::PreprocessParser& preParser
) {
    /*
    仅对图像resize+255归一化操作
    */
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = oriImage.rows;
    float       width  = oriImage.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);
    
    cv::Mat tmp;
    cv::resize(oriImage, tmp, size);

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = 0;
    int bottom = 0;
    int left   = 0;
    int right  = 0;

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    inputImage.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)inputImage.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)inputImage.data + (int)inp_h * (int)inp_w);
    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)inputImage.data + (int)inp_h * (int)inp_w * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);  // R->B
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);  // G->G
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);  // B->G

    preParser.ratio_f  = 1 / r;
    preParser.padw_f   = dw;
    preParser.padh_f   = dh;
    preParser.oriImgHeight_i = height;
    preParser.oriImgWidth_i  = width;
}