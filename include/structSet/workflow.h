#ifndef INCLUDE_STRUCTSET_WORKFLOW_H
#define INCLUDE_STRUCTSET_WORKFLOW_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>
#include <iostream>
#include <string>

#include "networkSpace.h"
#include "algorithmBase.h"

struct SingleAlgoNodeConfig {
    bool nodeStatus_b                = false ;      // 标记当前模型是否已经初始化
    // node参数全部记录在这里 与配置文件一一对应
    int maxBatch_i                   = 32;          // 动态batch
    int bestBatch_i                  = 16;          // 动态batch
    int inputImgHeight_i             = 0;           // 模型输入的高度
    int inputImgWidth_i              = 0;           // 模型输入的宽度
    int outNum_i                     = 0;           // 分类模型输出的类别数量
    int imageCropMaxLen_i            = 100;         // 抠图操作支持的最大数量
    const char* nodeName_c           = nullptr;     // 当前算法node在workflow中的名称
    char* nodePreprocess_c           = nullptr;     // 前处理操作
    char* nodeModelType_c            = nullptr;     // workflow节点算法的类型：detection classification
    char* nodeModelFile_c            = nullptr;     // workflow节点算法模型文件
    std::vector<int> outputId_vec;                       // 对模型输出的类别进行过滤
    // 记录对应算法参数
    std::shared_ptr<ALGORITHM_BASE> nodeModel        
                                     = nullptr;     // 模型指针
};


struct WorkflowConfg {
    bool loadedConfig = false;                        // 标记当前config参数是否已经从配置文件中加载过
    size_t workflowDevice_i = 0;                      // GPU编号
    const char* workflowSolution_c = nullptr;               // 算法方案名称
    char* workflowResult_c = nullptr;                 // 当前算法推理结果
    std::map<
        std::string, std::shared_ptr<SingleAlgoNodeConfig>
    > singleAlgoNodeConfig_map;                       // 复杂的算法方案由多个子算法组成，每个子算法作为一个节点被记录到model_vec中
};


// typedef std::map<std::string, std::shared_ptr<WORKFLOW>> WORKFLOW_MAP;
// extern WORKFLOW_MAP GLOBAL_WORKFLOW_MAP;

#endif  // INCLUDE_STRUCTSET_WORKFLOW_H