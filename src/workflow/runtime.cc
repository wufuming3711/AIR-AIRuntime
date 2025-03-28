#include <mutex>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <random>
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
#include "runtime.h"
#include "logger.h"
#include "gpu_resource_manager.h"

#include "task_exchange.grpc.pb.h"


Interface::Interface(
    pb::DetectionAlgorithm pbALGONAME, 
    size_t szGpuId,
    const string logDir,
    const string configWorkflow,
    const string modelZoo
) {
    sptrResult_ = nullptr;
    algorithm = pbALGONAME;
    sptrLogger_ = std::make_shared<logger::CustomLogger>(logDir);
    std::string sAlgName = pb::DetectionAlgorithm_Name(pbALGONAME);
    setAlgName(sAlgName);
    sptrLogger_->initialize(sAlgSolutionName_);

    size_t optimal_gpu_id = 0;
    sptrGpuResourceManager_ = std::make_shared<GpuResourceManager>(
        sptrLogger_,
        80.0f,  // gpu使用率
        80.0f   // cpu使用率
    );
    if (!sptrGpuResourceManager_->bIsInitial) {
        RUNTIME_LOG(sptrLogger_,
            nvinfer1::ILogger::Severity::kERROR, 
            "GPU自动调度器初始化失败, 尝试将算法加载到gpu0上 ......"
        );
    }
    if (!sptrGpuResourceManager_->initialize()) {
        RUNTIME_LOG(
            sptrLogger_,
            nvinfer1::ILogger::Severity::kERROR, 
            "尝试获取gpu基础信息失败"
        );
        sptrInstance_ = nullptr;
        return;
    }

    optimal_gpu_id = sptrGpuResourceManager_->get_optimal_gpu_id();
    if (optimal_gpu_id == INVALID_GPU_ID) {
        RUNTIME_LOG(sptrLogger_,
            nvinfer1::ILogger::Severity::kERROR, 
            "无法找到合适的GPU设备"
        );
        sptrInstance_ = nullptr;
        return;
    } else {
        RUNTIME_LOG(sptrLogger_,
            nvinfer1::ILogger::Severity::kINFO, 
            format_to_string("选择了GPU设备ID: %zu", optimal_gpu_id).c_str()
        );
    }

    sptrInstance_ = std::make_shared<WorkflowInfer>(
        sAlgSolutionName_.c_str(), 
        optimal_gpu_id,
        configWorkflow,
        modelZoo,
        sptrLogger_
    );

    if (sptrInstance_->bIsInitialized_ == false) {
        RUNTIME_LOG(sptrLogger_,
            nvinfer1::ILogger::Severity::kERROR, 
            "模型创建失败"
        );
        sptrInstance_ = nullptr;
    }
}

bool Interface::analysisSingle(
    cv::Mat& cvmatOriImage, 
    pb::OnAIResultGotReply::ResultWrapper& pbResultWrapper
) {
    // // TODO 这个位置随机模拟50~100ms耗时，然后return true;
    // // 模拟50~100ms的随机耗时
    // std::random_device rd;  // 随机数种子
    // std::mt19937 gen(rd()); // 使用Mersenne Twister算法的随机数生成器
    // std::uniform_int_distribution<> dis(44, 45); // 生成50到100之间的随机数

    // int random_delay = dis(gen); // 获取随机延迟时间
    // std::this_thread::sleep_for(std::chrono::milliseconds(random_delay)); // 让线程休眠
    // return true;


    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<cv::Mat> vcvmatOriImage = {cvmatOriImage};
    // 重置引用
    sptrResult_ = nullptr;
    if (sptrInstance_ != nullptr) {
        // 外部创建共享指针 传递到内部进行编辑
        std::shared_ptr<workflow_results::AnalysisResult> sptrAnalysisResult 
            = std::make_shared<workflow_results::AnalysisResult>(vcvmatOriImage);
        size_t batch = vcvmatOriImage.size();
        // 初始化sptrAnalysisResult
        for (int i = 0; i < batch; ++i) {
            std::shared_ptr<workflow_results::SingleAnalysisResult> singResPtr 
                = std::make_shared<workflow_results::SingleAnalysisResult>();
            singResPtr->szOriImageId_ = i;
            sptrAnalysisResult->singResPtr_vec.push_back(singResPtr);
        }
        RUNTIME_LOG(sptrLogger_, 
            nvinfer1::ILogger::Severity::kINFO, 
            "``->正在同步数据->sptrInstance_->analysisImages(sptrAnalysisResult, batch);"
        );

        sptrResult_ = sptrInstance_->analysisImages(sptrAnalysisResult, batch);
        if (sptrResult_ == nullptr) {
            RUNTIME_LOG(sptrLogger_, 
                nvinfer1::ILogger::Severity::kERROR, 
                format_to_string("[%s] Interface::analysisSingle->sptrInstance_->analysisImages分析失败", sAlgSolutionName_).c_str());
            return false;
        }
        if (!__write2Wrapper__(pbResultWrapper)) {
            RUNTIME_LOG(sptrLogger_, 
                nvinfer1::ILogger::Severity::kERROR, 
                format_to_string("[%s]Interface::__write2Wrapper__结果同步失败", sAlgSolutionName_).c_str());
            return false;
        }
    }
    else {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR, 
            "Interface没有初始化");
    }

    if (sptrLogger_->vsInterErr_.size() > 0) {
        pbResultWrapper.set_desc(sptrLogger_->getInterErrAsString().c_str());
        pbResultWrapper.set_shouldupdate(true);
    }
    pbResultWrapper.set_algo(algorithm);
    sptrResult_ = nullptr;
    return true;
}


bool Interface::__write2Wrapper__(
    pb::OnAIResultGotReply::ResultWrapper& pbResultWrapper
) {
    // return true;  // @wfm
    if (!sptrResult_) {
        RUNTIME_LOG(sptrLogger_, 
            nvinfer1::ILogger::Severity::kERROR, 
            "Interface::__write2Wrapper__ 尝试同步推理结果到pb结构体，发现sptrResult_指针为空"
        );
        return false;
    }

    size_t imagesNum = sptrResult_->imagesNum;
    for (size_t i = 0; i < imagesNum; ++i) {
        // 使用引用避免创建额外的 std::shared_ptr
        const std::shared_ptr<workflow_results::SingleAnalysisResult>& singResPtr = sptrResult_->singResPtr_vec[i];
        const std::vector<network_space::Object>& boxes_vec = singResPtr->boxes_vec;
        size_t resNum = singResPtr->resNum;

        for (size_t j = 0; j < resNum; ++j) {
            const network_space::Object& object = boxes_vec[j];
            
            pb::OnAIResultGotReply::Result* result = pbResultWrapper.add_rs();
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

void Interface::setAlgName(const std::string& cntsAlgName) {
    if (cntsAlgName.length() < cntszMinAlgNameLength_ 
        || cntsAlgName.length() > cntszMaxAlgNameLength_
    ) {
        sAlgSolutionName_ = sErrAlgSolutionName_ + "-[" + cntsAlgName + "]";
        sptrLogger_->vsInterErr_.push_back(
            format_to_string("算法名称非法: [%s]，名称重定向为 [%s]", cntsAlgName.c_str(), sAlgSolutionName_.c_str()).c_str()
        );
    }
    sAlgSolutionName_ = cntsAlgName;
}