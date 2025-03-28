//
// Created by Oboi on 2024/11/13.
//

#include "Manager.h"
#include <mutex>
#include "../../conf/Config.h"

namespace ai
{
    using namespace std;
    using namespace pb;

    Manager::Manager() = default;

    static Manager* instance = nullptr;
    static once_flag onceFlag;

    Manager& Manager::getInstance()
    {
        call_once(onceFlag, []()
        {
            instance = new Manager;
        });
        return *instance;
    }

    void Manager::initHandlers(
        const int gpuCount,
        const string logDir,
        const string configWorkflow,
        const string modelZoo
    ) {
        lock.lock();
        for (int i = 0; i < 100; ++i)
        {
            gpuQueueArray[i] = new deque<GPUModel*>();
        }
        const auto gpuQueue = gpuQueueArray[99];
        for (int i = 0; i < gpuCount; ++i)
        {
            gpuQueue->push_front(new GPUModel{i, 99});
            const auto algorithms = new map<DetectionAlgorithm, Interface*>;
            algorithmHandlerDic[i] = algorithms;
            for (auto algorithm =static_cast<DetectionAlgorithm>(conf::Config::getInstance().ai->detectionAlgorithmStart);
                 algorithm <= static_cast<DetectionAlgorithm>(conf::Config::getInstance().ai->detectionAlgorithmEnd);
                 algorithm = static_cast<DetectionAlgorithm>(algorithm + 1))
            {
                (*algorithms)[algorithm] = new Interface(
                    algorithm, 
                    static_cast<size_t>(i),
                    logDir,
                    configWorkflow,
                    modelZoo
                );
                cout << "准备在GPU:" << i << "上创建算法" << DetectionAlgorithm_Name(algorithm) << endl;
            }
        }
        for (auto algorithm =
                 static_cast<DetectionAlgorithm>(conf::Config::getInstance().ai->detectionAlgorithmStart);
             algorithm <= static_cast<DetectionAlgorithm>(conf::Config::getInstance().ai->detectionAlgorithmEnd);
             algorithm = static_cast<DetectionAlgorithm>(algorithm + 1))
        {
            algorithmGPUUsagePercentageMinusOneDic[algorithm] =
                Interface::getGPUUsagePercentageMinusOneForAlgorithm(algorithm);
        }
        lock.unlock();
    }

    bool Manager::useAlgorithm(const DetectionAlgorithm algorithm, cv::Mat& oriImage,
                               OnAIResultGotReply::ResultWrapper& resultWrapper)
    {
        int wantedIdle = algorithmGPUUsagePercentageMinusOneDic[algorithm];
        if (wantedIdle == 0)
        {
            return false;
        }
        std::unique_lock lk(lock);
        GPUModel* gpuModel;
        conditionVar.wait(lk, [wantedIdle, &gpuModel, this]()
        {
            for (int i = wantedIdle; i < 100; ++i)
            {
                if (!gpuQueueArray[i]->empty())
                {
                    gpuModel = gpuQueueArray[i]->front();
                    gpuQueueArray[i]->pop_front();
                    gpuModel->gpuIdlePercentageMinusOne -= wantedIdle;
                    gpuQueueArray[gpuModel->gpuIdlePercentageMinusOne]->push_front(gpuModel);
                    return true;
                }
            }
            return false;
        });
        bool rs = false;
        lk.unlock();
        conditionVar.notify_one();
        const auto algorithms = algorithmHandlerDic[gpuModel->gpuIdx];
        if (const auto handler = (*algorithms)[algorithm]; handler != nullptr)
        {
            auto st = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).
                count();
            rs = handler->analysisSingle(oriImage, resultWrapper);
            auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).
                count() - st;
            if (duration > 40)
            {
                resultWrapper.set_shouldupdate(true);
                stringstream ss;
                ss << "\nai处理异常缓慢,用时" << duration << " ms" << endl;
                resultWrapper.mutable_desc()->append(ss.str());
            }
        }
        else
        {
            cout << "[Error] 无效的算法调用: " << DetectionAlgorithm_Name(algorithm) << endl;
        }
        lk.lock();
        for (auto it = gpuQueueArray[gpuModel->gpuIdlePercentageMinusOne]->begin();
             it != gpuQueueArray[gpuModel->gpuIdlePercentageMinusOne]->end();
        )
        {
            if (*it == gpuModel)
            {
                gpuQueueArray[gpuModel->gpuIdlePercentageMinusOne]->erase(it);
                break;
            }
            ++it;
        }
        gpuModel->gpuIdlePercentageMinusOne += wantedIdle;
        gpuQueueArray[gpuModel->gpuIdlePercentageMinusOne]->push_front(gpuModel);
        lk.unlock();
        conditionVar.notify_one();
        return rs;
    }
} // ai
