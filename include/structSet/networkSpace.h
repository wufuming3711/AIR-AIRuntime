#ifndef INCLUDE_STRUCTSET_NETWORKSPACE_H
#define INCLUDE_STRUCTSET_NETWORKSPACE_H

#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "opencv2/opencv.hpp"

namespace networkSpace {

struct Binding {
    size_t size_i  = 0;
    size_t dsize_i = 0;
    const char* name_c = nullptr;
    nvinfer1::Dims dims;
};

struct EngineParser{
    // 申请一组资源记录满batch下的所有信息
    size_t               numBindings_i = 0;  // 模型总的输入与输出数量

    std::vector<std::vector<int>> inputSizeHW_vec;  // 模型输入尺寸 支持模型多输入
    // 示例数据如下
    // inputSizeHW_vec.push_back({768, 1024}); // 第一个元素，表示高度1024和宽度768
    
    size_t               numInputs_i   = 0;  // 网络输入数量
    size_t               numOutputs_i  = 0;  // 网络输出数量
    // 单batch下
    std::vector<Binding> inputBindings_vec;  // 记录模型输入节点的维度信息 支持多输入节点
    std::vector<Binding> outputBindings_vec; // 记录模型输出节点的维度信息 支持多输出节点
    // 多batch参数
    size_t               maxBatch_i  = 32;
    size_t               bestBatch_i = 16;
    size_t               curtBatch_i = 1;
    // 网络预测的类别数量
    size_t               outClsNum_i = 0;
    std::vector<void*> deviceInPtrs_vec;
    std::vector<void*> deviceOutPtrs_vec;
    std::vector<void*> hostOutPtrs_vec;
};

struct PreprocessParser{
    cv::Size size;
    size_t oriImgHeight_i = 0;
    size_t oriImgWidth_i  = 0;
    float ratio_f         = 1.0f;  // resize缩放比例
    float padw_f          = 0.0f;  // letterbox padding总宽
    float padh_f          = 0.0f;  // letterbox padding总高
};

struct InputData{
    InputData(PreprocessParser& preParser)
        : preParser(preParser), inputImage(), oriImage() {}

    PreprocessParser preParser;
    cv::Mat oriImage;   // 原始图片 这里使用引用方式
    cv::Mat inputImage;  // resize之后的图片 创建一个新的变量对象
};

struct Object {
    cv::Rect_<float> rect;
    cv::Mat cropImage;  // 记录检测结果的抠图部分
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
    // AlgoInfo algoInfo;
    EngineParser engineParser;
    InOutPutData inOutPutData;
};

}  // namespace networkSpace
#endif  // INCLUDE_STRUCTSET_NETWORKSPACE_H