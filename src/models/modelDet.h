#ifndef INCLUDE_MODELS_MODELDER_H
#define INCLUDE_MODELS_MODELDER_H

#include "NvInferPlugin.h"
#include "fstream"
#include <cmath>
#include <algorithm>

#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "letterbox.h"


class Detector : public AlgorithmBase
{
public:
    explicit Detector(
        const std::string& nvptrEngine_FilePath,
        std::shared_ptr<logger::CustomLogger>& logger
    ) : AlgorithmBase(nvptrEngine_FilePath, logger) {
    }
    
    ~Detector() override {
        std::cout << "Detector destructor called." << std::endl;
    }

    void postprocess();
    void draw_boxes(size_t save_img_max_num);

    inline float clamp(
        float value, 
        float min, 
        float max
    ) {
        return std::max(min, std::min(value, max));
    }

private:  // 新增附属函数
    void createDateFolder(const std::string& imagesFolderPath, std::string& currentDateFolder);
    void cleanupOldFolders(const std::string& imagesFolderPath);
    void generateJsonFile(const std::string& imagePath, const cv::Mat& image, const std::vector<network_space::Object>& objects);
    // string description = "Detector";
};


#endif  // INCLUDE_MODELS_MODELDER_H