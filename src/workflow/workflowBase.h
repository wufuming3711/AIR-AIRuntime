#ifndef INCLUDE_WORKFLOW_WORKFLOWBASE_H
#define INCLUDE_WORKFLOW_WORKFLOWBASE_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <memory>


#include "workflow.h"

#include "cJSON.h"
#include "loadJsonConfig.h"
#include "common.inl"
#include "config.inl"
#include "modelDet.h"
#include "modelCls.h"
#include "networkSpace.h"
#include "cJSON.h"
#include "logger.h"

using namespace std;

namespace workflow_results{

struct SingleAnalysisResult {
    size_t szOriImageId_ = 0;
    size_t resNum = 0;
    std::vector<network_space::Object> boxes_vec;
};

struct AnalysisResult {
    AnalysisResult(
        std::vector<cv::Mat>& images
    ) : 
        images_vec(images)
        , imagesNum(images.size()
    ) {
        imagesNum = images.size();
    }
    
    std::vector<cv::Mat>& images_vec;
    size_t imagesNum = 0;
    std::vector<std::shared_ptr<SingleAnalysisResult>> singResPtr_vec;
};
}  // workflow_results


class WorkflowInfer{
public:
    bool bIsInitialized_ = false; 
    std::string sModelName_ = "WorkflowInferNone"; 
    std::shared_ptr<WorkflowConfg> sptrWorkflowConfig_ = nullptr;
    std::shared_ptr<logger::CustomLogger> sptrLogger_ = nullptr; 
    const char* ccWorkflowConfig_ = nullptr;
    std::string sModelZoo_;
    // 这里创建一个鸟类专用的YOLO检测器
    std::shared_ptr<Detector> sptrBirdDetector_ = nullptr;
    int bird_crop_padding_ = 0;

public:
    WorkflowInfer(
        const std::string& modelName,
        size_t deviceId,
        const string configWorkflow,
        const string modelZoo,
        std::shared_ptr<logger::CustomLogger> logger
    );
    
    bool loadConfigFromJson();

    bool parseJsonToMap(
        cJSON *item
    );

    bool createSingNodeModel(
        std::shared_ptr<SingleAlgoNodeConfig>& sptrSigAlgNodeCfg,
        std::string sTmpModelName
    );

    bool buildWorkflow();

    std::shared_ptr<workflow_results::AnalysisResult> analysisImages(
        std::shared_ptr<workflow_results::AnalysisResult>& sptrAnalysisResult,
        size_t batch
    );

    void outputIdFilter(
        std::vector<std::vector<network_space::Object>>& output_vec,  
        const std::vector<int>& viOutputId_
    );

    void synchronizeAlgoRes(
        std::shared_ptr<workflow_results::AnalysisResult> sptrAnalysisResult,
        std::vector<std::vector<network_space::Object>>& output_vec
    );

    void resetMembers() {
        sptrWorkflowConfig_ = nullptr;
        bIsInitialized_ = false;      
        sptrLogger_ = nullptr;        
    }
};

#endif  // #ifndef INCLUDE_WORKFLOW_WORKFLOWBASE_H