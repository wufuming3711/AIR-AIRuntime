#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "cJSON.h"
#include "loadJsonConfig.h"
#include "common.inl"
#include "config.inl"
#include "modelDet.h"
#include "modelCls.h"
#include "workflow.h"
#include "opt.h"
#include "workflowBase.h"
// #include "runtime.h"

// protobuffer
#include "task_exchange.pb.h"

class Interface {
public:
    Interface(pb::DetectionAlgorithm ALGONAME, size_t gpuId);

    ~Interface();

    bool analysisSingle(cv::Mat& oriImage, pb::OnAIResultGotReply::ResultWrapper& resultWrapper);

    bool __write2Wrapper__(pb::OnAIResultGotReply::ResultWrapper& resultWrapper);
private:
    std::string* algoName_s = nullptr;
    std::shared_ptr<WorkflowInfer> instance = nullptr;
    std::shared_ptr<RESULTS::analysisResult> result = nullptr;
};


Interface::Interface(pb::DetectionAlgorithm ALGONAME, size_t gpuId) {
    std::string s_algoName = pb::DetectionAlgorithm_Name(ALGONAME);
    this->algoName_s = new std::string(s_algoName);  // 动态分配字符串
    this->instance = std::make_shared<WorkflowInfer>(this->algoName_s->c_str(), gpuId);
    if (this->instance == nullptr) {
        printf("[ERROR] %s 模型初始化失败\n", this->algoName_s->c_str());
        exit(-1);
    }
    printf("[INFO] %s 实例化成功\n", this->algoName_s->c_str());
}

Interface::~Interface() {
    delete this->algoName_s;  // 释放动态分配的字符串
}

bool Interface::analysisSingle(cv::Mat& oriImage, pb::OnAIResultGotReply::ResultWrapper& resultWrapper) {
    std::vector<cv::Mat> images = {oriImage};
    this->result = nullptr;
    this->result = this->instance->analysisImages(images);
    if (this->result == nullptr) {
        printf("[INFO] %s this->instance->analysisImages分析失败\n", this->algoName_s->c_str());
        return false;
    }
    this->__write2Wrapper__(resultWrapper);
    return true;
}

bool Interface::__write2Wrapper__(pb::OnAIResultGotReply::ResultWrapper& resultWrapper) {
    size_t imagesNum = this->result->imagesNum;
    for (size_t i = 0; i < imagesNum; ++i) {
        std::shared_ptr<RESULTS::singleAnalysisResult> singResPtr = this->result->singResPtr_vec[i];
        std::vector<networkSpace::Object>& boxes_vec = singResPtr->boxes_vec;
        size_t resNum = singResPtr->resNum;  // 获取结果数量
        for (size_t j = 0; j < resNum; ++j) {
            networkSpace::Object& object = boxes_vec[j];
            
            pb::OnAIResultGotReply::Result* result = resultWrapper.add_rs();
            result->set_prob(object.prob_f);
            result->set_label(object.label_i);
            
            pb::OnAIResultGotReply::Result::Rect* rect = result->mutable_rect();
            rect->set_minx(object.rect.x);
            rect->set_maxx(object.rect.x + object.rect.width);
            rect->set_miny(object.rect.y);
            rect->set_maxy(object.rect.y + object.rect.height);
        }
    }
    return true;
}