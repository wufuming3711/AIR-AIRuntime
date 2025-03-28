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
    bool bIsInitial_                = false ;      
    int iMaxBatch_                   = 32;         
    int iBestBatch_                  = 16;         
    int iInputImgHeight_             = 0;          
    int iInputImgWidth_              = 0;          
    int iOutNum_                     = 0;          
    int iImageCropMaxLen_            = 100;        
    const char* cntcNodeName_           = nullptr;
    char* cptrNodePreprocess_           = nullptr;
    char* cptrNodeModelType_            = nullptr;
    char* cptrNodeModelFile_            = nullptr;
    std::vector<int> viOutputId_; 
    std::shared_ptr<AlgorithmBase> sptrNodeModel_        
                                     = nullptr;
    // 可选项 储存图片的参数
    size_t save_img_max_num          = 0;  // 默认关闭存图片功能
    // 可选项 分类器行为策略
    // 可选项 "classifier_action_配置说明": "labal_replace: 标签替换,将用分类器判定的label结果替换检测器label; box_delete: 删除负样本box,分类器对检测器结果判定,如果为负样本,就将这个检测结果删除",
    std::string classifier_action    = "None";
};


struct WorkflowConfg {
    bool bIsloadedConfig_ = false;
    size_t szWorkflowDevice_ = 0;
    const char* cntcptrWorkflowSolution_ = nullptr;
    std::map<
        std::string, std::shared_ptr<SingleAlgoNodeConfig>
    > mssptrSingleAlgoNodeConfig_;
};

#endif  // INCLUDE_STRUCTSET_WORKFLOW_H