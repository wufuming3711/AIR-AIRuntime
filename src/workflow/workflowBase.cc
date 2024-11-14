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
// #include "infer.h"
#include "workflow.h"
#include "opt.h"
#include "workflowBase.h"


void WorkflowInfer::outputIdFilter(
    std::vector<std::vector<networkSpace::Object>>& output_vec,  
    const std::vector<int>& outputId_vec
) {
    if (outputId_vec.size() == 0) return;
    std::vector<std::vector<networkSpace::Object>> newOutput_vec;
    int batchSize = output_vec.size();
    for (int batchIdx=0; batchIdx < batchSize; ++batchIdx) {
        std::vector<networkSpace::Object>& obj_vec = output_vec[batchIdx];
        printf("[INFO] [batch %d], obj_vec.size()=%d\n", batchIdx, obj_vec.size());
        std::vector<networkSpace::Object> tmp;
        for (auto& obj : obj_vec) {
            // 判断是否包含
            if (std::find(
                outputId_vec.begin(), outputId_vec.end(), obj.label_i
            ) != outputId_vec.end()) {
                printf("[INFO] [batch %d] outputIdFilter-> 匹配到结果 %d\n", batchIdx, obj.label_i);
                tmp.push_back(obj);
            }
            else {
                printf("[INFO] [batch %d] outputIdFilter-> 没有匹配到结果 %d\n", batchIdx, obj.label_i);
            } 
        }
        newOutput_vec.push_back(tmp);
    }
    output_vec.clear();
    output_vec = newOutput_vec;
}

void WorkflowInfer::synchronizeAlgoRes(
    std::shared_ptr<RESULTS::analysisResult> analysisResult_ptr,
    std::vector<std::vector<networkSpace::Object>>& output_vec
) {
        size_t batchSize = output_vec.size();
        for (int batch = 0; batch < batchSize; ++batch) {

            std::vector<networkSpace::Object>& res_vec = output_vec[batch];
            std::shared_ptr<RESULTS::singleAnalysisResult> singResPtr = analysisResult_ptr->singResPtr_vec[batch];

            int resNum = res_vec.size();
            if (resNum > 0) {
                singResPtr->resNum = resNum;
                for (int j = 0; j < resNum; ++j) {
                    singResPtr->boxes_vec.push_back(res_vec[j]);
                }
            }
        }
    }

WorkflowInfer::WorkflowInfer(
    std::string modelName_s
) : modelName_s(modelName_s) {
    trtVersion();
    if (!loadConfigFromJson(WORKFLOW_CONFIG_CC, modelName_s.c_str(), workflowConfgPtr)) {
        printf("[ERROR] 配置文件加载失败\n");
        exit(-1);
    }
    printf("[INFO] 加载配置文件完成\n");
    if (!this->buildWorkflow()) {
        printf("[ERROR] Workflow工作流加载失败 \n");
        exit(-1);
    }
    printf("[INFO] ---------------------workflow构建完成-----------------------\n");
}

WorkflowInfer::WorkflowInfer(
    std::string modelName_s,
    size_t deviceId
) {
    this->modelName_s = modelName_s;
    trtVersion();
    if (!loadConfigFromJson(WORKFLOW_CONFIG_CC, modelName_s.c_str(), workflowConfgPtr)) {
        printf("[ERROR] 配置文件加载失败\n");
        exit(-1);
    }
    this->workflowConfgPtr->workflowDevice_i = deviceId;
    printf("[INFO] 加载配置文件完成\n");
    if (!this->buildWorkflow()) {
        printf("[ERROR] Workflow工作流加载失败 \n");
        exit(-1);
    }
    printf("[INFO] ---------------------workflow构建完成-----------------------\n");
}

bool WorkflowInfer::createNodeModel(
    std::shared_ptr<SingleAlgoNodeConfig>& nodePtr, 
    string modelName_s
){
    nodePtr->nodeName_c = modelName_s.c_str();
    if(nodePtr->nodeModel != nullptr
        || nodePtr->nodeStatus_b == true
    ) {
        printf("[INFO] %s 已存在，无需重复创建\n", nodePtr->nodeName_c);
        return true;
    }
    if(strcmp("classification" , nodePtr->nodeModelType_c) == 0){
        printf("[INFO] 创建分类算法 %s \n", nodePtr->nodeModelFile_c);
        nodePtr->nodeModel = std::make_shared<CLASSIFIER>(
            nodePtr->nodeModelFile_c);
        std::vector<int> inputSz = {nodePtr->inputImgHeight_i, nodePtr->inputImgWidth_i};
        nodePtr->nodeModel->baseAlgoParser.engineParser.inputSizeHW_vec.push_back(inputSz);
        nodePtr->nodeModel->baseAlgoParser.engineParser.outClsNum_i = nodePtr->outNum_i;
    }
    else if(strcmp("detection", nodePtr->nodeModelType_c) == 0 ){
        printf("[INFO] 创建检测算法 %s \n", nodePtr->nodeModelFile_c);
        nodePtr->nodeModel = std::make_shared<DETECTOR>(
            nodePtr->nodeModelFile_c);
        std::vector<int> inputSz = {nodePtr->inputImgHeight_i, nodePtr->inputImgWidth_i};        
        nodePtr->nodeModel->baseAlgoParser.engineParser.inputSizeHW_vec.push_back(inputSz);
    }
    else if(strcmp("operation", nodePtr->nodeModelType_c) == 0 ){
        printf("[INFO] 创建操作节点 %s \n", nodePtr->nodeModelFile_c);
        nodePtr->nodeModel = std::make_shared<Operation>(nodePtr->nodeModelFile_c);
        return true;
    }
    else return false;
    nodePtr->nodeStatus_b = true;
    // printf("[INFO] `%s`初始化完成，模型文件：%s\n", modelName_s.c_str(), nodePtr->nodeModelFile_c);
    return true;
}

std::shared_ptr<RESULTS::analysisResult> WorkflowInfer::analysisImages(
    std::vector<cv::Mat> images
){
    cudaSetDevice(workflowConfgPtr->workflowDevice_i);
    size_t nodeNum = workflowConfgPtr->singleAlgoNodeConfig_map.size();
    size_t batch = images.size();

    // 将传入的图片存储在当前的结构体中 创建一个智能指针
    std::shared_ptr<
        RESULTS::analysisResult
    > analysisResult_ptr = std::make_shared<
            RESULTS::analysisResult
        >(images);
    // 初始化智能指针参数
    // analysisResult_ptr->images_vec = images;
    // analysisResult_ptr->imagesNum = batch;
    for (int i = 0; i < batch; ++i) {
        // 创建一个智能指针变量
        std::shared_ptr<RESULTS::singleAnalysisResult> singResPtr = std::make_shared<RESULTS::singleAnalysisResult>();
        // 初始化这个指针singResPtr
        singResPtr->oriImageId_i = i;
        // 将智能指针添加到队列中
        analysisResult_ptr->singResPtr_vec.push_back(singResPtr);
    }

    // 存储临时图片结果
    std::vector<std::vector<cv::Mat>> bufferImages_vec;  
    
    // 记录前一个节点
    std::shared_ptr<SingleAlgoNodeConfig> preNodePtr = nullptr;

    // 根据预设的node顺序执行算法节点
    for(auto & [nodeName_s, nodePtr] : workflowConfgPtr->singleAlgoNodeConfig_map){
        // 当前节点名前处理名称 不同算法有不同的前处理名称 默认是空指针
        // TODO 前处理写成函数参数 这样就可以解决抽象类的功能合并 需要调试
        const char* preprocess = nodePtr->nodePreprocess_c;
        // 打印节点名称
        std::cout << "[INFO] analysisImages-> " << nodeName_s << "->" << nodePtr->nodeModelType_c << std::endl; 
        // 进入节点判断分支
        if(strcmp(nodePtr->nodeModelType_c, "detection") == 0){
            nodePtr->nodeModel->warmup(0); 
            nodePtr->nodeModel->commitImages(analysisResult_ptr->images_vec); 
            nodePtr->nodeModel->postprocess();
            
            // 对检测结果过滤
            std::vector<std::vector<networkSpace::Object>>& output_vec = nodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
            const std::vector<int>& outputId_vec = nodePtr->outputId_vec;
            this->outputIdFilter(
                output_vec, outputId_vec
            );

            nodePtr->nodeModel->draw_boxes();  // 调试使用
            int batchSize = output_vec.size();
            printf("[INFO] 检测算法推理完成\n");

            // ---------------------调试代码块 打印检测信息---------------------
            for (int batchIdx=0; batchIdx < batchSize; ++batchIdx) {
                std::vector<networkSpace::Object>& obj_vec = output_vec[batchIdx];
                printf("[INFO] >> [batch %d], obj_vec.size()=%d\n", batchIdx, obj_vec.size());
            }

                if (1 == workflowConfgPtr->singleAlgoNodeConfig_map.size()) {  // 将检测结果同步到analysisResult_ptr
                    // 取出当前模型的推理结果
                    // std::vector<std::vector<networkSpace::Object>>& output_vec = nodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
                    // 将当前节点的推理结果同步到`analysisResult_ptr`指针中
                    this->synchronizeAlgoRes(analysisResult_ptr, output_vec);
                }
        }
        else if(strcmp(nodePtr->nodeModelType_c, "operation") == 0) {
            assert(preNodePtr != nullptr);
            // 直接操作前一个检测算法节点对象
            std::vector<networkSpace::InputData>& oriImages_vec 
                = preNodePtr->nodeModel->baseAlgoParser.inOutPutData.input;
            std::vector<std::vector<networkSpace::Object>>& output_vec 
                = preNodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
            for (int batchIdx = 0; batchIdx < batch; ++batchIdx) {
                cv::Mat& oriImage = oriImages_vec[batchIdx].oriImage;
                std::vector<networkSpace::Object>& sigImgBoxes_vec = output_vec[batchIdx];
                int boxNum = sigImgBoxes_vec.size();
                if (boxNum < 1) continue;
                for (int i = 0; i < boxNum; ++i) {
                    networkSpace::Object& box = sigImgBoxes_vec[i];
                    // 根据坐标在原始图片上抠图 抠图结果存储在sigBoxes.cropImage
                    nodePtr->nodeModel->singleImageCrop(
                        oriImage, box,
                        10  // padding =10 这里手动写死
                    );
                }
            }
            
            
            // --------------直接继承前一个模型节点检测结果--------------
            // 1. 取出前一个模型检测结果
            std::vector<std::vector<networkSpace::Object>>& preOutput_vec = preNodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
            // 2. 取出当前节点的输出结果
            std::vector<std::vector<networkSpace::Object>>& curtOutput_vec = nodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
            // 3. 用前一个节点中的数据覆盖当前节点数据 当前节点是一个操作模块 没有生成多余数据
            curtOutput_vec = preOutput_vec;  // 深拷贝
            // 4. 现在完成了全部box的抠图操作
            printf("[INFO] 抠图操作完成\n");
        }
        else if(strcmp(nodePtr->nodeModelType_c, "classification") == 0){
            if (strcmp(nodeName_s.c_str(), "node1") == 0) {
                // 第一个节点上是分类模型 直接进行单张图片的预测
                printf("[INFO] 单分类算法任务\n");
                // printf("[INFO] analysisImages-> warmup\n");
                nodePtr->nodeModel->warmup(0); 
                // printf("[INFO] analysisImages-> commitImages\n");
                nodePtr->nodeModel->commitImages(analysisResult_ptr->images_vec); 
                // printf("[INFO] analysisImages-> postprocess\n");
                nodePtr->nodeModel->postprocess();

                // 对检测结果过滤
                std::vector<std::vector<networkSpace::Object>>& output_vec = nodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
                const std::vector<int>& outputId_vec = nodePtr->outputId_vec;
                if (outputId_vec.size() > 0) {
                    this->outputIdFilter(output_vec, outputId_vec);
                }
                
                printf("[INFO] analysisImages-> draw_boxes\n");
                nodePtr->nodeModel->draw_boxes();  // 调试使用
                    {  // 将检测结果同步到analysisResult_ptr
                    // 取出当前模型的推理结果
                    std::vector<std::vector<networkSpace::Object>>& output_vec = nodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
                    // 取出前一个模型的推理结果
                    // 将当前节点的推理结果同步到`analysisResult_ptr`指针中
                    this->synchronizeAlgoRes(analysisResult_ptr, output_vec);
                    }
            }
            else {
                // for循环遍历每一张图片 将单张图片中的检测框集合送入分类器推理
                printf("[INFO] 当前batch=%d \n", batch);

                // 直接操作前一个检测算法节点对象
                std::vector<networkSpace::InputData>& preOriImages_vec 
                    = preNodePtr->nodeModel->baseAlgoParser.inOutPutData.input;
                std::vector<std::vector<networkSpace::Object>>& preOutput_vec 
                    = preNodePtr->nodeModel->baseAlgoParser.inOutPutData.output;

                for (int batchIdx = 0; batchIdx < batch; ++batchIdx) {
                    // 获取单张图片中的全部box结果
                    std::vector<networkSpace::Object>& sigImgBoxes_vec = preOutput_vec[batchIdx];
                    int boxNum = sigImgBoxes_vec.size();
                    // 打印状态
                    printf("[INFO] analysisImages->  第 %d 个batch, 进入分类模型分支, boxNum=%d\n"
                        , batchIdx, boxNum);

                    // 如果当前图片没有检测出任何结果 就不使用分类模型处理
                    if (boxNum < 1) {
                        printf("[INFO] [batch %d] 当前图片未检出任何结果\n", batchIdx);
                        continue;
                    }
                    // 循环取出所有图片 然后将图片按照顺序组合送入分类模型
                    std::vector<cv::Mat> buffer;
                    for (auto & box : sigImgBoxes_vec) {
                        buffer.push_back(box.cropImage);
                    }
                    // 将box.cropImage传递给分类模型推理
                    nodePtr->nodeModel->commitImages(buffer); 
                    nodePtr->nodeModel->postprocess();
                    nodePtr->nodeModel->draw_boxes();  // 调试使用
                    // TODO 分类结果同步给检测结果
                    // 分类模型的推理结果队列
                    std::vector<std::vector<networkSpace::Object>>& curtOutput_vec 
                        = nodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
                    for (int i = 0; i < boxNum; ++i) {
                        std::vector<networkSpace::Object>& obj_vec = curtOutput_vec[i];
                        networkSpace::Object& obj = obj_vec[0];
                        // 取出obj中类别信息复制给前一个节点
                        sigImgBoxes_vec[i].label_i = obj.label_i;
                    }
                    // TODO 这部分暂时不做 分类模型过滤标签 
                }

                // TODO 分类for循环处理完毕 检测结果同步
                {  // 将检测结果同步到analysisResult_ptr
                // 取出当前模型的推理结果
                std::vector<std::vector<networkSpace::Object>>& preOutput_vec = preNodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
                // 将当前节点的推理结果同步到`analysisResult_ptr`指针中
                this->synchronizeAlgoRes(analysisResult_ptr, preOutput_vec);
                }
            }

        }
        if (strcmp(nodeName_s.c_str(), "node3") != 0) {
            preNodePtr = nodePtr;  // 更新指针
        }
        
    }
    printf("[INFO] analysisImages-> algorithmBatchInfer done.\n");
    // 打印出最终检测结果
    // std::vector<std::vector<networkSpace::Object>>& printPreOutput_vec 
    //     = preNodePtr->nodeModel->baseAlgoParser.inOutPutData.output;
    // for (int bi = 0; bi < batch; ++bi) {
    //     std::vector<networkSpace::Object> bi_Output_vec = printPreOutput_vec[bi];
    //     int num = bi_Output_vec.size();
    //     if (num == 0) continue;
    //     for (int boxid = 0; boxid < num; ++boxid) {
    //         networkSpace::Object obj = bi_Output_vec[boxid];
    //         printf("[>>] ===========[batch %d] [boxid %d] [num %d] ===========\n", batch, boxid, num );
    //         printf("[>>] [obj.rect.x %f]\n", obj.rect.x);
    //         printf("[>>] [obj.rect.y %f]\n", obj.rect.y);
    //         printf("[>>] [obj.rect.width %f]\n", obj.rect.width);
    //         printf("[>>] [obj.rect.height %f]\n", obj.rect.height);
    //         printf("[>>] [obj.prob_f %f]\n", obj.prob_f);
    //         printf("[>>] [obj.label_i %d]\n", obj.label_i);
    //     }
    // }
    return analysisResult_ptr;
}

bool WorkflowInfer::buildWorkflow(){
    cudaSetDevice(this->workflowConfgPtr->workflowDevice_i);
    for(auto & [_, nodePtr] : this->workflowConfgPtr->singleAlgoNodeConfig_map){
        if(false == nodePtr->nodeStatus_b){
            char* __tmp_c = "-node";
            std::string modelName_s;
            concatenate(this->workflowConfgPtr->workflowSolution_c, __tmp_c, modelName_s);
            if (!this->createNodeModel(nodePtr, modelName_s)){
                printf("[ERROR] workflow-%s 节点 [%s] 创建失败\n", 
                    this->workflowConfgPtr->workflowSolution_c, modelName_s.c_str());
                return false;
            }
            // printf("[INFO] workflow-%s 节点 [%s] 创建完成\n", 
            //     this->workflowConfgPtr->workflowSolution_c, modelName_s.c_str());
            // printf("[INFO] 模型文件 %s \n", nodePtr->nodeModelFile_c);
            nodePtr->nodeStatus_b = true;
        }
        else{
            // printf("[INFO] workflow-%s 节点 [%s] 已经存在\n", 
            //     this->workflowConfgPtr->workflowSolution_c, modelName_s.c_str());
            return true;
        }
    }
    return true;
}