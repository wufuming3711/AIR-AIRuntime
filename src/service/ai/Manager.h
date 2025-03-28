//
// Created by Oboi on 2024/11/13.
//

#ifndef ANALYSIS_SERVICE_MANAGER_H
#define ANALYSIS_SERVICE_MANAGER_H

#include <mutex>
#include <condition_variable>
#include "../../workflow/runtime.h"

namespace ai
{
    using namespace pb;
    using namespace std;

    class Manager
    {
    public:
        static Manager& getInstance();

        Manager(const Manager&) = delete;

        Manager& operator=(const Manager&) = delete;

        ~Manager() = delete;

        void initHandlers(
            const int gpuCount,
            const string logDir,
            const string configWorkflow,
            const string modelZoo
        );

        bool useAlgorithm(DetectionAlgorithm algorithm,
                          cv::Mat& oriImage,
                          OnAIResultGotReply::ResultWrapper& resultWrapper);

    private:
        Manager();

        map<int, map<DetectionAlgorithm, Interface*>*> algorithmHandlerDic;
        map<DetectionAlgorithm, int> algorithmGPUUsagePercentageMinusOneDic;

        struct GPUModel
        {
            int gpuIdx;
            int gpuIdlePercentageMinusOne;
        };

        array<std::deque<GPUModel*>*, 100> gpuQueueArray{};
        mutex lock;
        condition_variable conditionVar;
    };
} // ai

#endif //ANALYSIS_SERVICE_MANAGER_H
