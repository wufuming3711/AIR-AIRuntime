#ifndef INCLUDE_MODELS_OPERATION_OPT_H
#define INCLUDE_MODELS_OPERATION_OPT_H

#include "NvInferPlugin.h"
#include "fstream"
#include <cmath> 
#include <algorithm> 
#include <ctime>
#include <random>
#include <string>
#include <iostream>

#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "letterbox.h"

class Operation : public AlgorithmBase{
private:
    std::string m_optName;
public:
    Operation() = delete;
    explicit Operation(
        const std::string& operationName,
        std::shared_ptr<logger::CustomLogger>& logger
    ) : AlgorithmBase(operationName, logger) {
    }
    ~Operation() {
        std::cout << "Detector destructor called." << std::endl;
    }

    virtual bool commitImages(
        const std::vector<cv::Mat>& images,
        const char* preprocess
    ) {};
    virtual void postprocess(
    ) {};
    virtual void draw_boxes(size_t save_img_max_num) {};

    // void singleImageCrop(
    //     const cv::Mat& oriImage, 
    //     network_space::Object& box,
    //     int padding = 10
    // ) {
    //     try {
    //         int x = static_cast<int>(box.rect.x);
    //         int y = static_cast<int>(box.rect.y);
    //         int width = static_cast<int>(box.rect.width);
    //         int height = static_cast<int>(box.rect.height);

    //         int x1 = std::max(0, x - padding);
    //         int y1 = std::max(0, y - padding);
    //         int x2 = std::min(oriImage.cols - 1, x + width + padding);
    //         int y2 = std::min(oriImage.rows - 1, y + height + padding);

    //         int newWidth = x2 - x1;
    //         int newHeight = y2 - y1;

    //         if (newWidth <= 0 || newHeight <= 0) {
    //             throw std::runtime_error("Invalid bounding box dimensions after padding.");
    //         }

    //         cv::Mat croppedImage = oriImage(cv::Rect(x1, y1, newWidth, newHeight));
    //         box.cvmatCropImage_ = croppedImage.clone();

    //          // 获取当前时间作为文件名
    //         std::time_t now = std::time(nullptr);
    //         std::tm* timeinfo = std::localtime(&now);
    //         char buffer[80];
    //         std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", timeinfo);
            
    //         // 生成随机数
    //         std::random_device rd;
    //         std::mt19937 gen(rd());
    //         std::uniform_int_distribution<> distrib(1000, 9999); // 生成1000-9999之间的随机数
    //         int random_num = distrib(gen);
            
    //         // 构建保存路径和文件名
    //         std::string savePath = "/data/01_Project/Saida-runtime-ubuntu22.04-sanhe/build/1736818481845_mp4-1/";
    //         std::string filename = savePath + "crop_" + std::string(buffer) + "_" + std::to_string(random_num) + ".jpg";
            
    //         // 保存裁剪后的图片
    //         cv::imwrite(filename, box.cvmatCropImage_);
    //         std::cout << "[INFO] Cropped image saved to: " << filename << std::endl;

    //     } catch (const std::exception& e) {
    //         std::cerr << "[ERROR] " << e.what() << std::endl;
    //     }
    // }

    void singleImageCrop(
        const cv::Mat& oriImage, 
        network_space::Object& box,
        int padding = 10
    ) {
        try {
            int x = static_cast<int>(box.rect.x);
            int y = static_cast<int>(box.rect.y);
            int width = static_cast<int>(box.rect.width);
            int height = static_cast<int>(box.rect.height);

            int x1 = std::max(0, x - padding);
            int y1 = std::max(0, y - padding);
            int x2 = std::min(oriImage.cols - 1, x + width + padding);
            int y2 = std::min(oriImage.rows - 1, y + height + padding);

            int newWidth = x2 - x1;
            int newHeight = y2 - y1;

            if (newWidth <= 0 || newHeight <= 0) {
                throw std::runtime_error("Invalid bounding box dimensions after padding.");
            }

            cv::Mat croppedImage = oriImage(cv::Rect(x1, y1, newWidth, newHeight));
            box.cvmatCropImage_ = croppedImage.clone();
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] " << e.what() << std::endl;
        }
    }


};


#endif  // INCLUDE_MODELS_OPERATION_OPT_H