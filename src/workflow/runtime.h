#ifndef INCLUDE_COMMON_UTILS_RUNTIME_H
#define INCLUDE_COMMON_UTILS_RUNTIME_H

#include <vector>
#include <opencv2/opencv.hpp>


#include "task_exchange.pb.h"
#include "workflow.h"
#include "opt.h"
#include "workflowBase.h"
#include "logger.h"
#include "gpu_resource_manager.h"


class Interface {
private:
    std::mutex mutex_; 
    std::string sAlgSolutionName_ = "";
    std::string sErrAlgSolutionName_ = "OutOfBoundsAlgorithmName";
    const size_t cntszMinAlgNameLength_ = 1;
    const size_t cntszMaxAlgNameLength_ = 64;
    std::shared_ptr<WorkflowInfer> sptrInstance_ = nullptr;
    std::shared_ptr<workflow_results::AnalysisResult> sptrResult_;
    std::shared_ptr<logger::CustomLogger> sptrLogger_ = nullptr;
public:
    pb::DetectionAlgorithm algorithm;
    std::shared_ptr<GpuResourceManager> sptrGpuResourceManager_;
public:

    Interface(
        pb::DetectionAlgorithm ALGONAME, 
        size_t szGpuId,
        const string logDir,
        const string configWorkflow,
        const string modelZoo
    );

    ~Interface() {};

    static int getGPUUsagePercentageMinusOneForAlgorithm(
        pb::DetectionAlgorithm algorithm
    ) {
        return 99;
    };

    bool analysisSingle(
        cv::Mat& cvmatOriImage, 
        pb::OnAIResultGotReply::ResultWrapper& pbResultWrapper
    );

    bool __write2Wrapper__(
        pb::OnAIResultGotReply::ResultWrapper& pbResultWrapper
    );

    void setAlgName(
        const std::string& cntsAlgName
    );

};

#endif  // INCLUDE_COMMON_UTILS_RUNTIME_H