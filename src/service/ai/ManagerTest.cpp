//
// Created by Oboi on 2024/11/13.
//
// ManagerTest.cpp
#include "Manager.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace ai;
using namespace pb;

class ManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化 Manager 实例
        manager = &Manager::getInstance();
        manager->initHandlers(1);  // 假设有 2 个 GPU
    }

    Manager *manager;
};

TEST_F(ManagerTest, UseAlgorithmSingleThread) {
    DetectionAlgorithm algorithm = DetectionAlgorithm_MIN;
    cv::Mat dummyImage = cv::Mat::zeros(100, 100, CV_8UC3);  // 创建一个空的黑色图像
    OnAIResultGotReply::ResultWrapper resultWrapper;

    // 调用 useAlgorithm，检查返回结果是否为 true
    bool result = manager->useAlgorithm(algorithm, dummyImage, resultWrapper);
    EXPECT_TRUE(result);  // 假设 handler->analysisSingle 应该返回 true
}

TEST_F(ManagerTest, UseAlgorithmMultiThread) {
    for (int k = 0; k < 10; ++k) {
        cv::Mat dummyImage = cv::Mat::zeros(100, 100, CV_8UC3);

        // 多线程测试
        std::vector<std::thread> threads;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 500; ++i) {
        OnAIResultGotReply::ResultWrapper resultWrapper;
            threads.emplace_back([this,i , &dummyImage, &resultWrapper]() {
            auto algorithm = static_cast<DetectionAlgorithm>(3333);
                cout << DetectionAlgorithm_Name(algorithm) << endl;
                bool result = manager->useAlgorithm(algorithm, dummyImage, resultWrapper);
                EXPECT_TRUE(result);
            });
        }
        for (auto &thread: threads) {
            thread.join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    每个程序运行20->300ms;500个任务
// 160ms * 500 = 80s
        std::cout << "程序运行时长: " << duration.count() << " 毫秒" << std::endl;
        // flush
        OnAIResultGotReply::ResultWrapper resultWrapper;
        manager->useAlgorithm(DETECTION_ALGORITHM_UNKNOWN, dummyImage, resultWrapper);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
