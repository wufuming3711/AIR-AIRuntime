#include "loadJsonConfig.h"
#include "common.inl"
#include "workflow.h"


bool parseJsonToMap(
    cJSON *item, 
    const char* key, 
    std::shared_ptr<WorkflowConfg>& workflowConfgPtr
){
    if (!cJSON_IsObject(item)) {
        printf("[ERROR] config配置文件中未找到 %s 算法的配置参数\n", key);
        return false;
    }

    cJSON *model = cJSON_GetObjectItem(item, key);
    if (!model || !cJSON_IsObject(model)){
        std::cout << "key: " << key << std::endl;
        printf("[ERROR] 模型配置参数 %s 为空\n", key);
        return false;
    }
    workflowConfgPtr->workflowSolution_c = key;  // 使用 strdup 复制字符串
    cJSON *model_workflow = cJSON_GetObjectItem(model, "workflow");
    if(!model_workflow) {
        printf("[ERROR] 模型 %s 工作流为空，请检查对应配置参数\n", key);
        return false;
    }
    // 获取"device" 必须参数 如果不配置，默认值则为0
    workflowConfgPtr->workflowDevice_i = 0;  // 如果没有配置这个device参数，就默认算法加载到0显卡上
    if (cJSON_HasObjectItem(model, "device")){
        cJSON *model_device = cJSON_GetObjectItem(model, "device");
        if(!model_device) {
            printf("[WARNING] config没有为算法`%s`配置`device`参数，模型默认加载到第`%d`块显卡上\n", key, workflowConfgPtr->workflowDevice_i);
        }
        else{
            workflowConfgPtr->workflowDevice_i = model_device->valueint;  // 将json文件中int类型直接赋值
        }
    }

    cJSON *nodeSet = cJSON_GetObjectItem(model_workflow, "node_set");
    if (!nodeSet || !cJSON_IsArray(nodeSet)){
        printf("[ERROR] node_set is not an array\n");
        return false;
    }
    for (int i = 0; i < cJSON_GetArraySize(nodeSet); ++i) {
        cJSON* local_nodeName_c = cJSON_GetArrayItem(nodeSet, i);
        if (local_nodeName_c && local_nodeName_c->valuestring) {
            std::shared_ptr<SingleAlgoNodeConfig> local_nodePtr = std::make_shared<SingleAlgoNodeConfig>();
            std::string local_nodeName_s = local_nodeName_c->valuestring;
            workflowConfgPtr->singleAlgoNodeConfig_map[local_nodeName_s] = local_nodePtr; // std::move();
        } else {
            printf("[ERROR] Node at index %d is null or has no valuestring\n", i);
            continue;
        }
    }
    if (nodeSet && nodeSet->type == cJSON_Array){
         cJSON *nodeObj;
         cJSON_ArrayForEach(nodeObj, nodeSet){
            if (!nodeObj || nodeObj->type != cJSON_String) continue;
            std::string nodeName_s(nodeObj->valuestring);
            const char *nodeName_c = nodeName_s.c_str();
            cJSON *nodeData = cJSON_GetObjectItem(model_workflow, nodeName_c);
            if (!nodeData || nodeData->type != cJSON_Object) continue;

            std::shared_ptr<SingleAlgoNodeConfig> nodePtr = nullptr;
            auto it = workflowConfgPtr->singleAlgoNodeConfig_map.find(nodeName_s);
            if (it != workflowConfgPtr->singleAlgoNodeConfig_map.end()) {
                nodePtr = it->second; // std::move();
            }
            
            nodePtr->nodeName_c = nodeName_c;  // strdup();
            cJSON *modelType = cJSON_GetObjectItemCaseSensitive(nodeData, "model_type");
            if (modelType && cJSON_IsString(modelType)){
                if(modelType->valuestring != nullptr)
                    nodePtr->nodeModelType_c = strdup(modelType->valuestring);  // 使用 strdup 复制字符串
                else{
                    printf("[ERROR] 参数配置错误, `%s` 未配置正确的 `model_type`\n", nodeName_c);
                    return false;
                }  
            }
            if (cJSON_HasObjectItem(nodeData, "model_file") == false){
                printf("[ERROR] %s, 模型文件不存在`model_file`\n", nodeName_c);
                exit(0);
            }
            
            cJSON *modelFile = cJSON_GetObjectItemCaseSensitive(nodeData, "model_file");
            if (modelFile && cJSON_IsString(modelFile)){
                if(modelFile->valuestring != nullptr)
                    nodePtr->nodeModelFile_c = strdup(modelFile->valuestring);  // 使用 strdup 复制字符串
            }
            
            cJSON *outputIdArray = cJSON_GetObjectItemCaseSensitive(nodeData, "output_id");
            if (outputIdArray && outputIdArray->type == cJSON_Array){
                cJSON *outputIdItem;
                cJSON_ArrayForEach(outputIdItem, outputIdArray){
                    if (outputIdItem && outputIdItem->type == cJSON_Number){
                        nodePtr->outputId_vec.push_back(outputIdItem->valueint);
                    }
                }
            }
            cJSON *preprocess = cJSON_GetObjectItemCaseSensitive(nodeData, "preprocess");
            if (preprocess && cJSON_IsString(preprocess)){
                if(preprocess->valuestring != nullptr)
                    nodePtr->nodePreprocess_c = strdup(preprocess->valuestring);  // 使用 strdup 复制字符串
                else{
                    printf("[WARNING] `%s` 配置文件 `preprocess` 未指定前处理方法，默认使用 `letterbox`\n", nodeName_c);
                    nodePtr->nodePreprocess_c = strdup("letterbox");  // 使用 strdup 复制字符串
                }
            }
            
            cJSON *height = cJSON_GetObjectItemCaseSensitive(nodeData, "height");
            if (height && cJSON_IsNumber(height)) nodePtr->inputImgHeight_i = height->valueint;
            // else{printf("[ERROR] 配置文件没有指定模型输入分辨率尺寸\n"); return false;}

            cJSON *width = cJSON_GetObjectItemCaseSensitive(nodeData, "width");
            if (width && cJSON_IsNumber(width)) nodePtr->inputImgWidth_i = width->valueint;
            // else{printf("[ERROR] 配置文件没有指定模型输入分辨率尺寸\n"); return false;}

            cJSON *outNum = cJSON_GetObjectItemCaseSensitive(nodeData, "out_num");
            if (outNum && cJSON_IsNumber(outNum)) nodePtr->outNum_i = outNum->valueint;
            // else{printf("[ERROR] 配置文件没有指定模型输出数量\n"); return false;}
            
            cJSON *max_batch = cJSON_GetObjectItemCaseSensitive(nodeData, "max_batch");
            if (max_batch && cJSON_IsNumber(max_batch)) nodePtr->maxBatch_i = max_batch->valueint;
            // else{printf("[WARNING] 使用默认动态max_batch=32\n"); }
            
            cJSON *best_batch = cJSON_GetObjectItemCaseSensitive(nodeData, "best_batch");
            if (best_batch && cJSON_IsNumber(best_batch)) nodePtr->bestBatch_i = best_batch->valueint;
            // else{printf("[WARNING] 使用默认动态best_batch=16\n"); }

            if(strcmp(nodePtr->nodeModelType_c, "classification") == 0){
                cJSON *numClasses = cJSON_GetObjectItemCaseSensitive(nodeData, "num_classes");
                if (numClasses && cJSON_IsNumber(numClasses)) nodePtr->outNum_i = numClasses->valueint;
            }
            else if (strcmp(nodePtr->nodeModelType_c, "operation") == 0){
                cJSON *node_max_len = cJSON_GetObjectItemCaseSensitive(nodeData, "max_len");
                if (node_max_len && cJSON_IsNumber(node_max_len)) nodePtr->imageCropMaxLen_i = node_max_len->valueint;
            }
         }
    }
    workflowConfgPtr->loadedConfig = true;
    printf("[INFO] `%s` 参数加载完成\n", key);
    return true;
}


// 读取json配置文件
bool loadConfigFromJson(
    const char* WORKFLOW_CONFIG_CC, 
    const char* modelName_c,
    std::shared_ptr<WorkflowConfg>& workflowConfgPtr
){
    printf("[INFO] json file: %s \n", WORKFLOW_CONFIG_CC);
    FILE *fp = fopen(WORKFLOW_CONFIG_CC, "rb");
    if (!fp){
        printf("[ERROR] Failed to open file %s\n" , WORKFLOW_CONFIG_CC);
        return false;
    }

    fseek(fp, 0L, SEEK_END);
    size_t file_size = ftell(fp);
    rewind(fp);

    char *json_str = (char*)malloc(file_size + 1);
    if (!json_str){
        fclose(fp);
        printf("[ERROR] Failed to allocate memory\n");
        return false;
    }

    size_t bytes_read = fread(json_str, 1, file_size, fp);
    json_str[bytes_read] = '\0';
    fclose(fp);

    if (bytes_read != file_size){
        free(json_str);
        printf("[ERROR] Failed to read file content\n");
        return false;
    }

    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        printf("[ERROR] before parsing: %s\n", cJSON_GetErrorPtr());
        free(json_str);
        return false;
    }
    cJSON *model_list = cJSON_GetObjectItem(root, "model_list");
    if (!model_list || !cJSON_IsArray(model_list)) {
        printf("[ERROR] model_list is not an array\n");
        cJSON_Delete(root);
        return false;
    }
    bool status = parseJsonToMap(root, modelName_c, workflowConfgPtr);
    cJSON_Delete(root);
    free(json_str);
    if (!status) return false;
    return true;
}