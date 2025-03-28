// changeDet.h
#ifndef CHANGEDETECT_H
#define CHANGEDETECT_H

#include <string>
#include <list>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "algorithmBase.h"

struct ChangeRegion {
    int x1, y1, x2, y2;
};

class ChangeDetec : public AlgorithmBase {
public:
    WarehouseDetect(const std::string& backgroundImagePath = "../background_images/",
                   int intervalBackgroundPolicy = 1,
                   bool enableLog = true);

    // DetectStatusChange function
    std::list<ChangeRegion> DetectStatusChange(const std::string& cameraNo,
                                              const cv::Mat& inputImage,
                                              int statusThreshold = 900); // 30*30

private:
    // Variables
    std::string backgroundImagePath;
    int intervalBackgroundPolicy;
    bool isEnableLog;

    // For policy=1
    std::list<std::pair<std::string, cv::Mat>> listBackgroundImage;
    std::mutex mutexPolicy1;

    // For policy=2
    std::list<std::pair<std::string, std::string>> listBackgroundImagePath;
    std::mutex mutexPolicy2;

    // Logging
    std::mutex mutexLog;
    void log(const std::string& message);

    // Helper functions
    cv::Mat loadImageFromPath(const std::string& path);
    void saveImageToPath(const std::string& path, const cv::Mat& image);
    std::pair<std::string, cv::Mat> findBackgroundImage(const std::string& cameraNo);
    std::pair<std::string, std::string> findBackgroundImagePath(const std::string& cameraNo);
};

#endif // CHANGEDETECT_H