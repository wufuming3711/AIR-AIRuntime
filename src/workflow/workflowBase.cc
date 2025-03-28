#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cstdlib>

#include "cJSON.h"
#include "loadJsonConfig.h"
#include "common.inl"
#include "config.inl"
#include "modelDet.h"
#include "modelCls.h"
#include "workflow.h"
#include "opt.h"
#include "workflowBase.h"

WorkflowInfer::WorkflowInfer(
    const std::string &modelName,
    size_t deviceId,
    const string configWorkflow,
    const string modelZoo,
    std::shared_ptr<logger::CustomLogger> logger) : sptrLogger_(logger), sModelName_(modelName)
{
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, trtVersion());
    this->ccWorkflowConfig_ = (configWorkflow != "") ? configWorkflow.c_str() : WORKFLOW_CONFIG_CC;
    this->sModelZoo_ = (modelZoo != "") ? modelZoo : "";
    sptrWorkflowConfig_ = std::make_shared<WorkflowConfg>();
    if (!sptrWorkflowConfig_)
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    "`sptrWorkflowConfig_ = std::make_shared<WorkflowConfg>();`初始化智能指针失败");
        resetMembers();
        return;
    }
    else
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                    format_to_string("服务配置文件内容成功").c_str());
        if (!loadConfigFromJson())
        {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                        ("无法从配置文件中读取到所需的算法配置参数, workflow名称: " + modelName, ", 配置文件: " + std::string(this->ccWorkflowConfig_)).c_str());
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                        (" " + modelName, " 配置文件: " + std::string(this->ccWorkflowConfig_)).c_str());
            resetMembers();
            return;
        }
        else
        {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                        format_to_string("成功从配置文件内容中解析出参数").c_str());
            if (!sptrWorkflowConfig_->bIsloadedConfig_)
            {
                RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                            ("无法从配置文件中读取到所需的算法配置参数, workflow名称: " + modelName, ", 配置文件: " + std::string(this->ccWorkflowConfig_)).c_str());
                resetMembers();
                return;
            }
            else
            {
                RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                            format_to_string("二次校验配置文件是否成功初始化: 是").c_str());
                RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                            ("加载配置文件完成, workflow名称: " + modelName).c_str());
                if (deviceId != INVALID_GPU_ID)
                    sptrWorkflowConfig_->szWorkflowDevice_ = deviceId;
                if (!this->buildWorkflow())
                {
                    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                                ("Workflow工作流加载失败, workflow名称: " + modelName).c_str());
                    resetMembers();
                    return;
                }
                bIsInitialized_ = true;
                RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                            ("workflow构建成功, workflow名称: " + modelName).c_str());
            }
        }
    }

    // ***************初始化鸟类专用的YOLO检测器--YOLOv8-COCO***************
    RUNTIME_CHECK(sptrLogger_, cudaSetDevice(sptrWorkflowConfig_->szWorkflowDevice_));
    sptrBirdDetector_ = std::make_shared<Detector>("./yolov8x_coco.engine", sptrLogger_);
    std::vector<int> inputSz = {640, 640};
    sptrBirdDetector_->baseAlgoParser.nvptrEngine_Parser.vviInputSizeHW_.push_back(inputSz);
    sptrBirdDetector_->baseAlgoParser.nvptrEngine_Parser.iOutClsNum_ = 80;
    // *******************************************************************
}

bool WorkflowInfer::parseJsonToMap(
    cJSON *item)
{
    if (!cJSON_IsObject(item))
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string("config配置文件中未找到`%s`算法的配置参数", sModelName_.c_str()).c_str());
        return false;
    }

    cJSON *model = cJSON_GetObjectItem(item, sModelName_.c_str());
    if (!model || !cJSON_IsObject(model))
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string("模型配置参数`%s`为空", sModelName_.c_str()).c_str());
        return false;
    }
    sptrWorkflowConfig_->cntcptrWorkflowSolution_ = sModelName_.c_str();
    cJSON *model_workflow = cJSON_GetObjectItem(model, "workflow");
    if (!model_workflow)
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string("模型`%s`工作流为空，请检查对应配置参数", sModelName_.c_str()).c_str());
        return false;
    }
    sptrWorkflowConfig_->szWorkflowDevice_ = 0;
    if (cJSON_HasObjectItem(model, "device"))
    {
        cJSON *model_device = cJSON_GetObjectItem(model, "device");
        if (!model_device)
        {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kWARNING,
                        format_to_string(
                            "算法`%s`缺少默认配置参数`device`, 模型默认加载到第`%d`块显卡上", sModelName_.c_str(), sptrWorkflowConfig_->szWorkflowDevice_)
                            .c_str());
        }
        else
        {
            sptrWorkflowConfig_->szWorkflowDevice_ = model_device->valueint;
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                        format_to_string("算法`%s`加载到第`%d`块显卡上", sModelName_.c_str(), sptrWorkflowConfig_->szWorkflowDevice_).c_str());
        }
    }
    cJSON *nodeSet = cJSON_GetObjectItem(model_workflow, "node_set");
    if (!nodeSet || !cJSON_IsArray(nodeSet))
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR, "node_set is not an array");
        return false;
    }
    for (int i = 0; i < cJSON_GetArraySize(nodeSet); ++i)
    {
        cJSON *local_cntcNodeName_ = cJSON_GetArrayItem(nodeSet, i);
        if (local_cntcNodeName_ && local_cntcNodeName_->valuestring)
        {
            std::shared_ptr<SingleAlgoNodeConfig> local_nodePtr = std::make_shared<SingleAlgoNodeConfig>();
            std::string local_nodeName_s = local_cntcNodeName_->valuestring;
            sptrWorkflowConfig_->mssptrSingleAlgoNodeConfig_[local_nodeName_s] = local_nodePtr;
        }
        else
        {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                        format_to_string("Node at index `%d` is null or has no valuestring", i).c_str());
            return false;
        }
    }
    if (nodeSet && nodeSet->type == cJSON_Array)
    {
        cJSON *nodeObj;
        cJSON_ArrayForEach(nodeObj, nodeSet)
        {
            if (!nodeObj || nodeObj->type != cJSON_String)
                continue;
            std::string nodeName_s(nodeObj->valuestring);
            const char *cntcNodeName_ = nodeName_s.c_str();
            cJSON *nodeData = cJSON_GetObjectItem(model_workflow, cntcNodeName_);
            if (!nodeData || nodeData->type != cJSON_Object)
                continue;

            std::shared_ptr<SingleAlgoNodeConfig> nodePtr = nullptr;
            auto it = sptrWorkflowConfig_->mssptrSingleAlgoNodeConfig_.find(nodeName_s);
            if (it != sptrWorkflowConfig_->mssptrSingleAlgoNodeConfig_.end())
            {
                nodePtr = it->second;
            }
            nodePtr->cntcNodeName_ = cntcNodeName_;
            cJSON *modelType = cJSON_GetObjectItemCaseSensitive(nodeData, "model_type");
            if (modelType && cJSON_IsString(modelType))
            {
                if (modelType->valuestring != nullptr)
                    nodePtr->cptrNodeModelType_ = strdup(modelType->valuestring);
                else
                {
                    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                                format_to_string("参数配置错误, `%s` 未配置正确的 `model_type`", cntcNodeName_).c_str());
                    return false;
                }
            }
            if (cJSON_HasObjectItem(nodeData, "model_file") == false)
            {
                RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                            format_to_string("`%s`, 模型文件参数`model_file`没有配置", cntcNodeName_).c_str());
                return false;
            }
            cJSON *modelFile = cJSON_GetObjectItemCaseSensitive(nodeData, "model_file");
            if (modelFile && cJSON_IsString(modelFile))
            {
                if (modelFile->valuestring != nullptr)
                {
                    std::string modelFilePath = modelFile->valuestring;
                    // if (modelFilePath != "image_crop") {
                    // 这里不再以 `image_crop` 作为判定关键词，而是以 `operation` 为判定关键词
                    if (std::string(modelType->valuestring) != "operation")
                    { // 跳过操作符节点
                        if (modelFilePath[0] != '/' && modelFilePath.find(":") == std::string::npos)
                        {
                            modelFilePath = this->sModelZoo_ + "/" + modelFilePath;
                        }
                        if (access(modelFilePath.c_str(), F_OK) == -1)
                        {
                            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                                        format_to_string("文件 `%s` 不存在, modelType->valuestring=%s", modelFilePath.c_str(), modelType->valuestring).c_str());
                            return false;
                        }
                    }
                    nodePtr->cptrNodeModelFile_ = strdup(modelFilePath.c_str());
                }
            }
            if (access(nodePtr->cptrNodeModelFile_, F_OK) == -1 && (std::string(modelType->valuestring) != "operation"))
            {
                RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                            format_to_string(
                                "`%s`, 模型文件不存在`model_file`: %s", cntcNodeName_, nodePtr->cptrNodeModelFile_)
                                .c_str());
                return false;
            }
            cJSON *outputIdArray = cJSON_GetObjectItemCaseSensitive(nodeData, "output_id");
            if (outputIdArray && outputIdArray->type == cJSON_Array)
            {
                cJSON *outputIdItem;
                cJSON_ArrayForEach(outputIdItem, outputIdArray)
                {
                    if (outputIdItem && outputIdItem->type == cJSON_Number)
                    {
                        nodePtr->viOutputId_.push_back(outputIdItem->valueint);
                    }
                }
            }
            // 可选项 存储图片功能
            cJSON *save_img_max_num = cJSON_GetObjectItemCaseSensitive(nodeData, "save_img_max_num");
            if (save_img_max_num && cJSON_IsNumber(save_img_max_num))
                nodePtr->save_img_max_num = save_img_max_num->valueint;
            // 可选项 "classifier_action 配置说明": "labal_replace: 标签替换,将用分类器判定的label结果替换检测器label; box_delete: 删除负样本box,分类器对检测器结果判定,如果为负样本,就将这个检测结果删除",
            cJSON *classifier_action = cJSON_GetObjectItemCaseSensitive(nodeData, "classifier_action");
            if (classifier_action && cJSON_IsString(classifier_action))
                nodePtr->classifier_action = strdup(classifier_action->valuestring);
            // 可选项 后处理操作
            cJSON *preprocess = cJSON_GetObjectItemCaseSensitive(nodeData, "preprocess");
            if (preprocess && cJSON_IsString(preprocess))
            {
                if (preprocess->valuestring != nullptr)
                    nodePtr->cptrNodePreprocess_ = strdup(preprocess->valuestring);
                else
                {
                    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kWARNING,
                                format_to_string("`%s` 配置文件 `preprocess` 未指定前处理方法，默认使用 `letterbox`", cntcNodeName_).c_str());
                    nodePtr->cptrNodePreprocess_ = strdup("letterbox");
                }
            }
            cJSON *height = cJSON_GetObjectItemCaseSensitive(nodeData, "height");
            if (height && cJSON_IsNumber(height))
                nodePtr->iInputImgHeight_ = height->valueint;
            cJSON *width = cJSON_GetObjectItemCaseSensitive(nodeData, "width");
            if (width && cJSON_IsNumber(width))
                nodePtr->iInputImgWidth_ = width->valueint;
            cJSON *outNum = cJSON_GetObjectItemCaseSensitive(nodeData, "out_num");
            if (outNum && cJSON_IsNumber(outNum))
                nodePtr->iOutNum_ = outNum->valueint;
            cJSON *max_batch = cJSON_GetObjectItemCaseSensitive(nodeData, "max_batch");
            if (max_batch && cJSON_IsNumber(max_batch))
                nodePtr->iMaxBatch_ = max_batch->valueint;
            cJSON *best_batch = cJSON_GetObjectItemCaseSensitive(nodeData, "best_batch");
            if (best_batch && cJSON_IsNumber(best_batch))
                nodePtr->iBestBatch_ = best_batch->valueint;
            if (strcmp(nodePtr->cptrNodeModelType_, "classification") == 0)
            {
                cJSON *numClasses = cJSON_GetObjectItemCaseSensitive(nodeData, "num_classes");
                if (numClasses && cJSON_IsNumber(numClasses))
                    nodePtr->iOutNum_ = numClasses->valueint;
            }
            else if (strcmp(nodePtr->cptrNodeModelType_, "operation") == 0)
            {
                cJSON *node_max_len = cJSON_GetObjectItemCaseSensitive(nodeData, "max_len");
                if (node_max_len && cJSON_IsNumber(node_max_len))
                    nodePtr->iImageCropMaxLen_ = node_max_len->valueint;
            }
        }
    }
    sptrWorkflowConfig_->bIsloadedConfig_ = true;
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string("`%s` 配置文件中参数加载完成", sModelName_.c_str()).c_str());
    return true;
}

bool WorkflowInfer::loadConfigFromJson()
{
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string("Json file:%s", this->ccWorkflowConfig_).c_str());
    FILE *fp = fopen(this->ccWorkflowConfig_, "rb");
    if (!fp)
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string("Failed to open file: %s", this->ccWorkflowConfig_).c_str());
        return false;
    }

    fseek(fp, 0L, SEEK_END);
    size_t file_size = ftell(fp);
    rewind(fp);

    char *json_str = (char *)malloc(file_size + 1);
    if (!json_str)
    {
        fclose(fp);
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    "Failed to allocate memory");
        return false;
    }

    size_t bytes_read = fread(json_str, 1, file_size, fp);
    json_str[bytes_read] = '\0';
    fclose(fp);

    if (bytes_read != file_size)
    {
        free(json_str);
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    "Failed to read file content");
        return false;
    }

    cJSON *root = cJSON_Parse(json_str);
    if (!root)
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string("Before parsing %s", cJSON_GetErrorPtr()).c_str());
        free(json_str);
        return false;
    }
    cJSON *model_list = cJSON_GetObjectItem(root, "model_list");
    if (!model_list || !cJSON_IsArray(model_list))
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    "`model_list` is not an array");
        cJSON_Delete(root);
        return false;
    }
    bool status = parseJsonToMap(root);
    cJSON_Delete(root);
    free(json_str);
    if (!status)
        return false;
    return true;
}

void WorkflowInfer::outputIdFilter(
    std::vector<std::vector<network_space::Object>> &output_vec,
    const std::vector<int> &viOutputId_)
{
    if (viOutputId_.size() == 0)
        return;
    std::vector<std::vector<network_space::Object>> newOutput_vec;
    int batchSize = output_vec.size();
    for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        std::vector<network_space::Object> &obj_vec = output_vec[batchIdx];
        // printf("[INFO] [batch %d], obj_vec.size()=%d\n", batchIdx, obj_vec.size());
        std::vector<network_space::Object> tmp;
        for (auto &obj : obj_vec)
        {
            if (std::find(
                    viOutputId_.begin(), viOutputId_.end(), obj.label_i) != viOutputId_.end())
            {
                // printf("[INFO] [batch %d] outputIdFilter-> 匹配到结果 %d\n", batchIdx, obj.label_i);
                tmp.push_back(obj);
            }
            else
            {
                printf("[INFO] [batch %d] outputIdFilter-> 没有匹配到结果 %d\n", batchIdx, obj.label_i);
            }
        }
        newOutput_vec.push_back(tmp);
    }
    output_vec.clear();
    output_vec = newOutput_vec;
}

void WorkflowInfer::synchronizeAlgoRes(
    std::shared_ptr<workflow_results::AnalysisResult> sptrAnalysisResult,
    std::vector<std::vector<network_space::Object>> &output_vec)
{
    size_t batchSize = output_vec.size();
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("src/workflow/workflowBase.cc -> WorkflowInfer::synchronizeAlgoRes->batchSize=%d\n", batchSize).c_str());
    for (int batch = 0; batch < batchSize; ++batch)
    {
        std::vector<network_space::Object> &res_vec = output_vec[batch];
        std::shared_ptr<workflow_results::SingleAnalysisResult> singResPtr = sptrAnalysisResult->singResPtr_vec[batch];

        int resNum = res_vec.size();
        if (resNum > 0)
        {
            singResPtr->resNum = resNum;
            for (int j = 0; j < resNum; ++j)
            {
                singResPtr->boxes_vec.push_back(res_vec[j]);
            }
        }
    }
    RUNTIME_LOG(sptrLogger_,
                nvinfer1::ILogger::Severity::kINFO,
                "src/workflow/workflowBase.cc -> WorkflowInfer::synchronizeAlgoRes->同步完成");
}

bool WorkflowInfer::createSingNodeModel(
    std::shared_ptr<SingleAlgoNodeConfig> &sptrSigAlgNodeCfg,
    std::string sTmpModelName)
{
    sptrSigAlgNodeCfg->cntcNodeName_ = sTmpModelName.c_str();
    if (sptrSigAlgNodeCfg->sptrNodeModel_ != nullptr || sptrSigAlgNodeCfg->bIsInitial_ == true)
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kINFO,
                    format_to_string(
                        "%s 已存在，无需重复创建", sptrSigAlgNodeCfg->cntcNodeName_)
                        .c_str());
        return true;
    }
    if (strcmp("classification", sptrSigAlgNodeCfg->cptrNodeModelType_) == 0)
    {
        sptrSigAlgNodeCfg->sptrNodeModel_ = std::make_shared<Classifier>(
            sptrSigAlgNodeCfg->cptrNodeModelFile_,
            sptrLogger_);
        if (!sptrSigAlgNodeCfg->sptrNodeModel_->bIsInitial_)
        {
            RUNTIME_LOG(
                sptrLogger_,
                nvinfer1::ILogger::Severity::kERROR,
                "分类算法 %s 初始化失败");
            return false;
        }
        std::vector<int> inputSz = {sptrSigAlgNodeCfg->iInputImgHeight_, sptrSigAlgNodeCfg->iInputImgWidth_};
        sptrSigAlgNodeCfg->sptrNodeModel_->baseAlgoParser.nvptrEngine_Parser.vviInputSizeHW_.push_back(inputSz);
        sptrSigAlgNodeCfg->sptrNodeModel_->baseAlgoParser.nvptrEngine_Parser.iOutClsNum_ = sptrSigAlgNodeCfg->iOutNum_;
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kINFO,
                    format_to_string(
                        "创建分类算法 %s 成功", sptrSigAlgNodeCfg->cptrNodeModelFile_)
                        .c_str());
    }
    else if (strcmp("detection", sptrSigAlgNodeCfg->cptrNodeModelType_) == 0)
    {
        sptrSigAlgNodeCfg->sptrNodeModel_ = std::make_shared<Detector>(
            sptrSigAlgNodeCfg->cptrNodeModelFile_,
            sptrLogger_);
        std::vector<int> inputSz = {
            sptrSigAlgNodeCfg->iInputImgHeight_, sptrSigAlgNodeCfg->iInputImgWidth_};
        sptrSigAlgNodeCfg->sptrNodeModel_
            ->baseAlgoParser.nvptrEngine_Parser.vviInputSizeHW_
            .push_back(inputSz);
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kINFO,
                    format_to_string(
                        "创建检测算法 %s 成功", sptrSigAlgNodeCfg->cptrNodeModelFile_)
                        .c_str());
    }
    else if (strcmp("operation", sptrSigAlgNodeCfg->cptrNodeModelType_) == 0)
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kINFO,
                    format_to_string(
                        "创建操作节点 %s", sptrSigAlgNodeCfg->cptrNodeModelFile_)
                        .c_str());
        sptrSigAlgNodeCfg->sptrNodeModel_ = std::make_shared<Operation>(
            sptrSigAlgNodeCfg->cptrNodeModelFile_,
            sptrLogger_);
        return true;
    }
    else
        return false;
    sptrSigAlgNodeCfg->sptrNodeModel_->modelName = sModelName_;
    sptrSigAlgNodeCfg->bIsInitial_ = true;
    return true;
}

std::shared_ptr<workflow_results::AnalysisResult> WorkflowInfer::analysisImages(
    std::shared_ptr<workflow_results::AnalysisResult> &sptrAnalysisResult,
    size_t batch)
{
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "WorkflowInfer::analysisImages获取 batch = %d", batch)
                    .c_str());
    RUNTIME_CHECK(sptrLogger_, cudaSetDevice(sptrWorkflowConfig_->szWorkflowDevice_));
    size_t nodeNum = sptrWorkflowConfig_->mssptrSingleAlgoNodeConfig_.size();

    // std::vector<std::vector<cv::Mat>> bufferImages_vec;

    std::shared_ptr<SingleAlgoNodeConfig> preNodePtr = nullptr;
    for (auto &[nodeName_s, nodePtr] : sptrWorkflowConfig_->mssptrSingleAlgoNodeConfig_)
    {
        nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output.clear();
        nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.input.clear();
        const char *preprocess = nodePtr->cptrNodePreprocess_;
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                    format_to_string("analysisImages-> %s", nodePtr->cptrNodeModelType_).c_str());
        if (strcmp(nodePtr->cptrNodeModelType_, "detection") == 0)
        {
            // 推理
            nodePtr->sptrNodeModel_->commitImages(sptrAnalysisResult->images_vec, preprocess);
            nodePtr->sptrNodeModel_->postprocess();
            // 是否存储图片
            if (nodePtr->save_img_max_num > 0)
            {
                std::cout << "[DEBUG] 保存图片 nodePtr->save_img_max_num = " << nodePtr->save_img_max_num << std::endl;
                nodePtr->sptrNodeModel_->draw_boxes(nodePtr->save_img_max_num);
            }
            std::vector<std::vector<network_space::Object>> &output_vec = nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
            const std::vector<int> &viOutputId_ = nodePtr->viOutputId_;
            this->outputIdFilter(
                output_vec, viOutputId_);

            int batchSize = output_vec.size();
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] 检测算法推理完成\n", nodePtr->sptrNodeModel_->modelName.c_str()).c_str());

            if (1 == sptrWorkflowConfig_->mssptrSingleAlgoNodeConfig_.size())
            {
                this->synchronizeAlgoRes(sptrAnalysisResult, output_vec);
            }
        }
        else if (strcmp(nodePtr->cptrNodeModelType_, "operation") == 0)
        {
            assert(preNodePtr != nullptr);
            std::vector<network_space::InputData> &oriImages_vec = preNodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.input;
            std::vector<std::vector<network_space::Object>> &output_vec = preNodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
            for (int batchIdx = 0; batchIdx < batch; ++batchIdx)
            {
                cv::Mat &oriImage = oriImages_vec[batchIdx].oriImage;
                std::vector<network_space::Object> &sigImgBoxes_vec = output_vec[batchIdx];
                int boxNum = sigImgBoxes_vec.size();
                if (boxNum < 1)
                    continue;

                for (int i = 0; i < boxNum; ++i)
                {
                    network_space::Object &box = sigImgBoxes_vec[i];
                    std::cout << "[DEBUG] boxNum = " << boxNum << std::endl;
                    if (boxNum <= 4)
                    {
                        // int width = static_cast<int>(box.rect.width);
                        // int height = static_cast<int>(box.rect.height);
                        // int maxValue = width > height? width:height;
                        // this->bird_crop_padding_ = maxValue;
                        this->bird_crop_padding_ = 100; // 20250319 如果只有一个box检测结果 就设置padding=100
                    }
                    nodePtr->sptrNodeModel_->singleImageCrop(
                        oriImage, box, this->bird_crop_padding_ // 20250319 这个参数设置抠图拓展的像素值
                    );
                }
            }
            std::vector<std::vector<network_space::Object>> &preOutput_vec = preNodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
            std::vector<std::vector<network_space::Object>> &curtOutput_vec = nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
            curtOutput_vec = preOutput_vec;
            this->bird_crop_padding_ = 0; // 像素值恢复为0
        }
        else if (strcmp(nodePtr->cptrNodeModelType_, "classification") == 0)
        {
            if (strcmp(nodeName_s.c_str(), "node1") == 0)
            {
                nodePtr->sptrNodeModel_->commitImages(
                    sptrAnalysisResult->images_vec,
                    preprocess);
                nodePtr->sptrNodeModel_->postprocess();
                std::vector<std::vector<network_space::Object>> &output_vec = nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
                const std::vector<int> &viOutputId_ = nodePtr->viOutputId_;
                if (viOutputId_.size() > 0)
                {
                    this->outputIdFilter(output_vec, viOutputId_);
                }
                {
                    std::vector<std::vector<network_space::Object>> &output_vec = nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
                    this->synchronizeAlgoRes(sptrAnalysisResult, output_vec);
                }
            }
            else
            {
                std::vector<network_space::InputData> &preOriImages_vec = preNodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.input;
                std::vector<std::vector<network_space::Object>> &preOutput_vec = preNodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
                for (int batchIdx = 0; batchIdx < batch; ++batchIdx)
                {
                    std::vector<network_space::Object> &sigImgBoxes_vec = preOutput_vec[batchIdx];
                    int boxNum = sigImgBoxes_vec.size();
                    if (boxNum < 1)
                    {
                        continue;
                    }
                    // ********************分类器算法分支********************
                    // 进入这个分支，意味着分类器作为方案的一个环节 方案一定是：det+cls
                    // 检查`分类器行为策略`参数classifier_action
                    if (nodePtr->classifier_action == "labal_replace")
                    {
                        for (int i = 0; i < boxNum; ++i)
                        {
                            std::vector<std::vector<network_space::Object>> &curtOutput_vec = nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
                            curtOutput_vec.clear();
                            nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.input.clear();
                            std::vector<cv::Mat> buffer;
                            buffer.push_back(sigImgBoxes_vec[i].cvmatCropImage_);
                            bool status = nodePtr->sptrNodeModel_->commitImages(
                                buffer, preprocess);
                            nodePtr->sptrNodeModel_->postprocess();
                            std::vector<network_space::Object> &obj_vec = curtOutput_vec[0];
                            network_space::Object &obj = obj_vec[0];
                            sigImgBoxes_vec[i].label_i = obj.label_i; // 替换标签
                        }
                    }
                    else if (nodePtr->classifier_action == "box_delete")
                    {
                        // ****************** 判断boxNum==1 决定是否使用鸟类检测器 ******************
                        std::vector<network_space::Object> filtered_boxes;
                        if (boxNum <= 4)
                        {
                            network_space::Object &box = sigImgBoxes_vec[0];
                            // 使用鸟类检测器 如果鸟类检测器输出结果id==14 就保留这个box并添加到filtered_boxes
                            std::vector<cv::Mat> buffer;
                            buffer.push_back(box.cvmatCropImage_);
                            sptrBirdDetector_->baseAlgoParser.inOutPutData.input.clear();
                            sptrBirdDetector_->commitImages(buffer, preprocess);
                            sptrBirdDetector_->postprocess();
                            auto &obj_vec = sptrBirdDetector_->baseAlgoParser.inOutPutData.output[0];
                            if (obj_vec.size() != 0)
                            {
                                network_space::Object &obj = obj_vec[0];
                                if (obj.label_i == 14)
                                { // 假设 label_i == 14 表示鸟类
                                    filtered_boxes.push_back(sigImgBoxes_vec[0]);
                                }
                            }
                            obj_vec.clear();
                        }
                        else
                        {
                            for (int i = 0; i < boxNum; ++i)
                            {
                                std::vector<std::vector<network_space::Object>> &curtOutput_vec = nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
                                curtOutput_vec.clear();
                                nodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.input.clear();
                                std::vector<cv::Mat> buffer;
                                buffer.push_back(sigImgBoxes_vec[i].cvmatCropImage_);
                                bool status = nodePtr->sptrNodeModel_->commitImages(
                                    buffer, preprocess);
                                nodePtr->sptrNodeModel_->postprocess();
                                std::vector<network_space::Object> &obj_vec = curtOutput_vec[0];
                                network_space::Object &obj = obj_vec[0];

                                if (obj.label_i != 1)
                                { // 假设 label_i == 1 表示负样本
                                    filtered_boxes.push_back(sigImgBoxes_vec[i]);
                                }
                            }
                        }
                        preOutput_vec[batchIdx] = filtered_boxes; // 更新前节点的输出
                    }
                    else
                    { //  (nodePtr->classifier_action == "None")
                        //
                        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                                    "异常逻辑，直接抛出系统性错误,检测器+分类器的组合方案中必须配置`classifier_action`参数，参数目前仅支持`labal_replace` or `box_delete`");
                        std::exit(EXIT_FAILURE); // 终止程序并返回失败状态码
                    }
                }
                {
                    std::vector<std::vector<network_space::Object>> &preOutput_vec = preNodePtr->sptrNodeModel_->baseAlgoParser.inOutPutData.output;
                    this->synchronizeAlgoRes(sptrAnalysisResult, preOutput_vec);
                }
            }
        }
        if (strcmp(nodeName_s.c_str(), "node3") != 0)
        {
            preNodePtr = nodePtr;
        }
    }
    return sptrAnalysisResult;
}

bool WorkflowInfer::buildWorkflow()
{
    RUNTIME_CHECK(sptrLogger_,
                  cudaSetDevice(sptrWorkflowConfig_->szWorkflowDevice_));
    for (auto &[sNodeName, sptrSigAlgNodeCfg] : sptrWorkflowConfig_->mssptrSingleAlgoNodeConfig_)
    {
        if (false == sptrSigAlgNodeCfg->bIsInitial_)
        {
            std::string sTmpModelName;
            sTmpModelName = std::string(
                                sptrWorkflowConfig_->cntcptrWorkflowSolution_) +
                            "-node";
            if (
                !createSingNodeModel(
                    sptrSigAlgNodeCfg, sTmpModelName))
            {
                RUNTIME_LOG(sptrLogger_,
                            nvinfer1::ILogger::Severity::kERROR,
                            format_to_string(
                                "工作流`%s`创建失败，对应的节点名称是 [%s]", sptrWorkflowConfig_->cntcptrWorkflowSolution_, sTmpModelName.c_str())
                                .c_str());
                return false;
            }
        }
        else
        {
            RUNTIME_LOG(sptrLogger_,
                        nvinfer1::ILogger::Severity::kWARNING,
                        format_to_string(
                            "工作流`%s`算法节点`%s`已经存在，不需要重复创建（疑问？为什么会进入这个分支）", sModelName_.c_str(), sNodeName.c_str())
                            .c_str());
        }
    }
    return true;
}