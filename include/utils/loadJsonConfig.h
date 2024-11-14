#ifndef INCLUDE_UTILS_LOADJSONCONFIG_H
#define INCLUDE_UTILS_LOADJSONCONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <memory>

#include "cJSON.h"
#include "workflow.h"


using namespace std;

bool loadConfigFromJson(
    const char* WORKFLOW_CONFIG_CC, 
    const char* modelName_c,
    std::shared_ptr<WorkflowConfg>& workflowConfgPtr
);

bool parseJsonToMap(
    cJSON *item, 
    const char* key, 
    std::shared_ptr<WorkflowConfg>& workflowConfgPtr
);



#endif  // INCLUDE_UTILS_LOADJSONCONFIG_H