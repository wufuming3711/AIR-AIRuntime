#ifndef INCLUDE_MODELS_OPERATION_OPT_H
#define INCLUDE_MODELS_OPERATION_OPT_H

#include "NvInferPlugin.h"
#include "fstream"
#include <cmath> 
#include <algorithm> 

#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "letterbox.h"

class Operation : public ALGORITHM_BASE{
private:
    std::string m_optName;
public:
    Operation() = delete;
    explicit Operation(const std::string& operationName) : ALGORITHM_BASE(operationName) {
        // m_optName = operationName;
    }
    ~Operation() {
        std::cout << "DETECTOR destructor called." << std::endl;
    }

    virtual bool commitImages(
        const std::vector<cv::Mat>& images
    ) {};
    virtual void postprocess(
    ) {};
    virtual void draw_boxes(
    ) {};

    void singleImageCrop(
        const cv::Mat& oriImage, 
        networkSpace::Object& box,
        int padding = 10
    ) {
        try {
            // 提取sigBox中记录的坐标信息
            int x = static_cast<int>(box.rect.x);
            int y = static_cast<int>(box.rect.y);
            int width = static_cast<int>(box.rect.width);
            int height = static_cast<int>(box.rect.height);

            // 计算扩展后的边界框坐标
            int x1 = std::max(0, x - padding);
            int y1 = std::max(0, y - padding);
            int x2 = std::min(oriImage.cols - 1, x + width + padding);
            int y2 = std::min(oriImage.rows - 1, y + height + padding);

            // 计算扩展后的宽度和高度
            int newWidth = x2 - x1;
            int newHeight = y2 - y1;

            // 检查边界框是否有效
            if (newWidth <= 0 || newHeight <= 0) {
                throw std::runtime_error("Invalid bounding box dimensions after padding.");
            }

            // 抠图并存储到 bufferImages_vec
            cv::Mat croppedImage = oriImage(cv::Rect(x1, y1, newWidth, newHeight));
            box.cropImage = croppedImage.clone();  // 深拷贝

            // 打印成功信息
            std::cout << "[INFO] Crop operation successful for object at (" << x << ", " << y << ") with dimensions (" << width << ", " << height << ")" << std::endl;

            // 保存抠图图片到文件
            std::string filename = "cropped_image_" + std::to_string(x) + "_" + std::to_string(y) + ".png";
            cv::imwrite(filename, box.cropImage);
            std::cout << "[INFO] Cropped image saved to " << filename << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[ERROR] " << e.what() << std::endl;
        }
    }

    /*
    void singleImageCrop(
        const cv::Mat& oriImage, 
        // const std::vector<networkSpace::Object>& boxes_vec,
        // std::vector<cv::Mat>& bufferImages_vec,
        int padding = 10
    ) {
        // 清空 bufferImages_vec
        bufferImages_vec.clear();

        // 遍历每个边界框
        for (const auto& box : boxes_vec) {
            // 获取边界框的坐标
            int x = static_cast<int>(box.rect.x);
            int y = static_cast<int>(box.rect.y);
            int width = static_cast<int>(box.rect.width);
            int height = static_cast<int>(box.rect.height);

            // 计算扩展后的边界框坐标
            int x1 = std::max(0, x - padding);
            int y1 = std::max(0, y - padding);
            int x2 = std::min(oriImage.cols - 1, x + width + padding);
            int y2 = std::min(oriImage.rows - 1, y + height + padding);

            // 计算扩展后的宽度和高度
            int newWidth = x2 - x1;
            int newHeight = y2 - y1;

            // 抠图并存储到 bufferImages_vec
            cv::Mat croppedImage = oriImage(cv::Rect(x1, y1, newWidth, newHeight));
            bufferImages_vec.push_back(croppedImage);
        }
    }
    */

};


#endif  // INCLUDE_MODELS_OPERATION_OPT_H