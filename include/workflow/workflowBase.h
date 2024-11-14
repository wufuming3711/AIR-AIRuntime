#ifndef INCLUDE_WORKFLOW_WORKFLOWBASE_H
#define INCLUDE_WORKFLOW_WORKFLOWBASE_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "workflow.h"

#include "cJSON.h"
#include "loadJsonConfig.h"
#include "common.inl"
#include "config.inl"
#include "modelDet.h"
#include "modelCls.h"
// #include "infer.h"
#include "networkSpace.h"
// #include "opt.h"


namespace RESULTS{

struct singleAnalysisResult {
    size_t oriImageId_i = 0;
    size_t resNum = 0;
    std::vector<networkSpace::Object> boxes_vec;
};

struct analysisResult {

    analysisResult(std::vector<cv::Mat>& images) : images_vec(images), imagesNum(images.size()) {
        imagesNum = images.size();
    }
    
    std::vector<cv::Mat>& images_vec;
    size_t imagesNum = 0;
    std::vector<std::shared_ptr<singleAnalysisResult>> singResPtr_vec;
};


}  // RESULTS


class WorkflowInfer{
public:
    WorkflowInfer(std::string modelName_s);
    WorkflowInfer(std::string modelName_s, size_t deviceId);
    
    bool createNodeModel(
        std::shared_ptr<SingleAlgoNodeConfig>& nodePtr, 
        string modelName_s
    );

    bool buildWorkflow();
    std::shared_ptr<RESULTS::analysisResult> analysisImages(std::vector<cv::Mat> images);

    void outputIdFilter(
        std::vector<std::vector<networkSpace::Object>>& output_vec,  
        const std::vector<int>& outputId_vec
    );

    void synchronizeAlgoRes(
        std::shared_ptr<RESULTS::analysisResult> analysisResult_ptr,
        std::vector<std::vector<networkSpace::Object>>& output_vec
    );

    std::string modelName_s;
    std::shared_ptr<WorkflowConfg> workflowConfgPtr = std::make_shared<WorkflowConfg>();
};

#endif  // #ifndef INCLUDE_WORKFLOW_WORKFLOWBASE_H