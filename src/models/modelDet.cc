#include <nlohmann/json.hpp>
#include <unistd.h> 
#include <atomic>
#include "NvInferPlugin.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cmath>      
#include <algorithm>  
#include <sys/stat.h>
#include <dirent.h>
#include <sstream>
#include <vector>
#include <map>
#include <filesystem> // C++17 文件系统库
#include <ctime>

#include "letterbox.h"
#include "modelDet.h"
#include "networkSpace.h"
#include "common.inl"


void Detector::postprocess() {
    std::vector<std::vector<network_space::Object>>& output_vec = this->baseAlgoParser.inOutPutData.output;
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] postprocess-> output_vec.size() = %zu", this->modelName.c_str(), output_vec.size()).c_str());
    output_vec.clear();
    auto& input_vec = this->baseAlgoParser.inOutPutData.input;
    int batch = input_vec.size();
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] postprocess-> 后处理 batch = %d", this->modelName.c_str(), batch).c_str());
    auto& vvoidptrHostOuts_ = this->baseAlgoParser.nvptrEngine_Parser.vvoidptrHostOuts_;

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] postprocess-> vvoidptrHostOuts_.size() = %zu", this->modelName.c_str(), vvoidptrHostOuts_.size()).c_str());
    if (vvoidptrHostOuts_.size() != this->baseAlgoParser.nvptrEngine_Parser.iNumOutputs_) {
        RUNTIME_LOG(sptrLogger_,nvinfer1::ILogger::Severity::kINFO,format_to_string("postprocess-> Error: vvoidptrHostOuts_ size mismatch. Expected 4, got %d ", vvoidptrHostOuts_.size()).c_str());
        std::cerr 
        << "postprocess-> Error: "
        << this->modelName.c_str()
        << " vvoidptrHostOuts_ size mismatch. Expected 4, got " << vvoidptrHostOuts_.size() << std::endl;
        return;
    }
    RUNTIME_LOG(
        sptrLogger_,
        nvinfer1::ILogger::Severity::kINFO,
        format_to_string(
            "[%s] postprocess-> 当前batch = %d",
            this->modelName.c_str(), batch
        ).c_str()
    );
    
    RUNTIME_LOG(
        sptrLogger_,
        nvinfer1::ILogger::Severity::kINFO,
        format_to_string(
            "[%s] postprocess-> 打印 vvoidptrHostOuts_[1] 中的所有检测框坐标",
            this->modelName.c_str()
        ).c_str()
    );
    int total_boxes = 0;
    for (int idx = 0; idx < batch; ++idx) {
        int* num_dets = static_cast<int*>(vvoidptrHostOuts_[0]) + idx;
        total_boxes += *num_dets;
    }
    float* boxes = static_cast<float*>(vvoidptrHostOuts_[1]);
    float* scores = static_cast<float*>(vvoidptrHostOuts_[2]);
    int* labels = static_cast<int*>(vvoidptrHostOuts_[3]);

    for (int idx = 0; idx < batch; ++idx) {
        std::vector<network_space::Object> subOutput_vec;
        int* num_dets = static_cast<int*>(vvoidptrHostOuts_[0]) + idx;
        float* boxes = static_cast<float*>(vvoidptrHostOuts_[1]) + 100 * 4 * idx;
        float* scores = static_cast<float*>(vvoidptrHostOuts_[2]) + 100 * idx;
        int* labels = static_cast<int*>(vvoidptrHostOuts_[3]) + 100 * idx;

        size_t& iOriImgHeight_ = input_vec[idx].preParser.iOriImgHeight_;
        size_t& iOriImgWidth_ = input_vec[idx].preParser.iOriImgWidth_;
        auto& ratio_f = input_vec[idx].preParser.ratio_f;
        float& padw_f = input_vec[idx].preParser.padw_f;
        float& padh_f = input_vec[idx].preParser.padh_f;

        for (int i = 0; i < *num_dets; ++i) {
            network_space::Object obj;
            float* ptr = boxes + i * 4;
            
            float x0   = *ptr++ - padw_f;
            float y0   = *ptr++ - padh_f;
            float x1   = *ptr++ - padw_f;
            float y1   = *ptr - padh_f;

            x0         = this->clamp(x0 * ratio_f, 0.f, static_cast<float>(iOriImgWidth_));
            y0         = this->clamp(y0 * ratio_f, 0.f, static_cast<float>(iOriImgHeight_));
            x1         = this->clamp(x1 * ratio_f, 0.f, static_cast<float>(iOriImgWidth_));
            y1         = this->clamp(y1 * ratio_f, 0.f, static_cast<float>(iOriImgHeight_));

            obj.rect.x      = x0;
            obj.rect.y      = y0;
            obj.rect.width  = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.prob_f      = scores[i];
            obj.label_i     = labels[i];
            
            // 补充逻辑 清除小框
            if (obj.rect.width < 32 && obj.rect.height < 32) {
                continue;   
            }
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] postprocess-> obj.rect.x = %f\n", this->modelName.c_str(), obj.rect.x).c_str());
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] postprocess-> obj.rect.y = %f\n", this->modelName.c_str(), obj.rect.y).c_str());
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] postprocess-> obj.rect.width = %f\n", this->modelName.c_str(), obj.rect.width).c_str());
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] postprocess-> obj.rect.height = %f\n", this->modelName.c_str(), obj.rect.height).c_str());
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("postprocess-> obj.prob_f = %f\n", this->modelName.c_str(), obj.prob_f).c_str());
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, format_to_string("[%s] postprocess-> obj.label_i = %d\n", this->modelName.c_str(), obj.label_i).c_str());
            subOutput_vec.push_back(obj);
        }
        output_vec.push_back(subOutput_vec);
    }
    this->baseAlgoParser.nvptrEngine_Parser.reset_EngineParser_vvoidptrX();
}


// Helper function to count existing _src.jpg files in the images folder
size_t countExistingSrcImages(const std::string& imagesFolderPath) {
    size_t count = 0;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(imagesFolderPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string fileName(ent->d_name);
            if (fileName.size() > 8 && fileName.substr(fileName.size() - 8) == "_src.jpg") {
                count++;
            }
        }
        closedir(dir);
    }
    return count;
}


// // 仅保存检测结果
// void Detector::draw_boxes(size_t save_img_max_num) {
//     std::string logDirPath = this->sptrLogger_->logDirPath;
//     std::string imagesFolderPath = logDirPath + "/images";
//     // 创建主 images 文件夹
//     if (!std::filesystem::exists(imagesFolderPath)) {
//         std::filesystem::create_directory(imagesFolderPath);
//     }
//     // 创建按日期命名的子文件夹
//     std::string currentDateFolder;
//     createDateFolder(imagesFolderPath, currentDateFolder);
//     // 清理超过 5 天的文件夹
//     cleanupOldFolders(imagesFolderPath);
//     // 统计当前日期文件夹中已有的 _src.jpg 文件数量
//     size_t startIndex = 0;
//     for (const auto& entry : std::filesystem::directory_iterator(currentDateFolder)) {
//         if (entry.path().extension() == ".jpg" && entry.path().filename().string().find("_src.jpg") != std::string::npos) {
//             startIndex++;
//         }
//     }
//     std::vector<std::vector<network_space::Object>> output_vec = this->baseAlgoParser.inOutPutData.output;
//     std::vector<network_space::InputData> input_vec = this->baseAlgoParser.inOutPutData.input;
//     if (input_vec.size() != output_vec.size()) {
//         RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
//                     "Error: The number of input images and the number of object batches do not match.");
//         return;
//     }
//     size_t savedCount = 0; // 当前日期文件夹中保存的图片计数
//     for (size_t i = 0; i < input_vec.size() && savedCount < save_img_max_num; ++i) {
//         // 检查是否有检测结果
//         if (output_vec[i].empty()) {
//             continue; // 如果没有检测结果，跳过该图片
//         }
//         cv::Mat originalImage = input_vec[i].oriImage.clone();
//         cv::Mat detectedImage = input_vec[i].oriImage.clone();
//         if (originalImage.empty() || detectedImage.empty()) {
//             RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
//                         format_to_string("Error: Failed to clone image at index %zu", i).c_str());
//             continue;
//         }
//         // 保存原始图像
//         std::ostringstream oss;
//         oss << currentDateFolder << "/" << startIndex + savedCount << "_src.jpg";
//         std::string srcImagePath = oss.str();
//         if (!cv::imwrite(srcImagePath, originalImage)) {
//             RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
//                         format_to_string("Error: Failed to save image to %s", srcImagePath.c_str()).c_str());
//             continue; // 如果保存失败，跳过后续操作
//         }
//         // 绘制检测框
//         for (const auto& obj : output_vec[i]) {
//             cv::Scalar color = cv::Scalar(0, 0, 255); // 红色
//             cv::rectangle(detectedImage, obj.rect, color, 2);
//             char text[256];
//             sprintf(text, "%d %.1f%%", obj.label_i, obj.prob_f * 100);
//             int baseLine = 0;
//             cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
//             int x = static_cast<int>(obj.rect.x);
//             int y = static_cast<int>(obj.rect.y) + 1;
//             if (y > detectedImage.rows) {
//                 y = detectedImage.rows;
//             }
//             cv::rectangle(detectedImage, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
//             cv::putText(detectedImage, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
//         }
//         // 保存带检测框的图像
//         oss.str(""); // 清空字符串流
//         oss << currentDateFolder << "/" << startIndex + savedCount << "_det.jpg";
//         std::string detImagePath = oss.str();
//         if (!cv::imwrite(detImagePath, detectedImage)) {
//             RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
//                         format_to_string("Error: Failed to save image to %s", detImagePath.c_str()).c_str());
//             continue; // 如果保存失败，跳过后续操作
//         }
//         // 生成 JSON 文件
//         generateJsonFile(srcImagePath, originalImage, output_vec[i]);
//         // 更新保存计数
//         savedCount++;
//     }
// }


// 保存全部图片
void Detector::draw_boxes(size_t save_img_max_num) {
    std::string logDirPath = this->sptrLogger_->logDirPath;
    std::string imagesFolderPath = logDirPath + "/images";
    // 创建主 images 文件夹
    if (!std::filesystem::exists(imagesFolderPath)) {
        std::filesystem::create_directory(imagesFolderPath);
    }
    // 创建按日期命名的子文件夹
    std::string currentDateFolder;
    createDateFolder(imagesFolderPath, currentDateFolder);
    // 清理超过 5 天的文件夹
    cleanupOldFolders(imagesFolderPath);
    // 统计当前日期文件夹中已有的 _src.jpg 文件数量
    size_t startIndex = 0;
    for (const auto& entry : std::filesystem::directory_iterator(currentDateFolder)) {
        if (entry.path().extension() == ".jpg" && entry.path().filename().string().find("_src.jpg") != std::string::npos) {
            startIndex++;
        }
    }
    std::vector<std::vector<network_space::Object>> output_vec = this->baseAlgoParser.inOutPutData.output;
    std::vector<network_space::InputData> input_vec = this->baseAlgoParser.inOutPutData.input;
    if (input_vec.size() != output_vec.size()) {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    "Error: The number of input images and the number of object batches do not match.");
        return;
    }
    size_t savedCount = 0; // 当前日期文件夹中保存的图片计数
    for (size_t i = 0; i < input_vec.size() && savedCount < save_img_max_num; ++i) {
        cv::Mat originalImage = input_vec[i].oriImage.clone();
        cv::Mat detectedImage = input_vec[i].oriImage.clone();
        if (originalImage.empty() || detectedImage.empty()) {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                        format_to_string("Error: Failed to clone image at index %zu", i).c_str());
            continue;
        }
        // 保存原始图像
        std::ostringstream oss;
        oss << currentDateFolder << "/" << startIndex + savedCount << "_src.jpg";
        std::string srcImagePath = oss.str();
        if (!cv::imwrite(srcImagePath, originalImage)) {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                        format_to_string("Error: Failed to save image to %s", srcImagePath.c_str()).c_str());
            continue; // 如果保存失败，跳过后续操作
        }
        // 绘制检测框（如果有检测结果）
        if (!output_vec[i].empty()) {
            for (const auto& obj : output_vec[i]) {
                cv::Scalar color = cv::Scalar(0, 0, 255); // 红色
                cv::rectangle(detectedImage, obj.rect, color, 2);
                char text[256];
                sprintf(text, "%d %.1f%%", obj.label_i, obj.prob_f * 100);
                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
                int x = static_cast<int>(obj.rect.x);
                int y = static_cast<int>(obj.rect.y) + 1;
                if (y > detectedImage.rows) {
                    y = detectedImage.rows;
                }
                cv::rectangle(detectedImage, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
                cv::putText(detectedImage, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
            }
        }
        // 保存带检测框的图像
        oss.str(""); // 清空字符串流
        oss << currentDateFolder << "/" << startIndex + savedCount << "_det.jpg";
        std::string detImagePath = oss.str();
        if (!cv::imwrite(detImagePath, detectedImage)) {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                        format_to_string("Error: Failed to save image to %s", detImagePath.c_str()).c_str());
            continue; // 如果保存失败，跳过后续操作
        }
        // 生成 JSON 文件（如果有检测结果）
        if (!output_vec[i].empty()) {
            generateJsonFile(srcImagePath, originalImage, output_vec[i]);
        }
        // 更新保存计数
        savedCount++;
    }
}


void Detector::createDateFolder(const std::string& imagesFolderPath, std::string& currentDateFolder) {
    // 获取当前日期
    time_t now = time(nullptr);
    tm* localTime = localtime(&now);
    char dateBuffer[9];
    strftime(dateBuffer, sizeof(dateBuffer), "%Y%m%d", localTime);
    std::string dateStr(dateBuffer);

    // 创建日期文件夹
    currentDateFolder = imagesFolderPath + "/" + dateStr + "_images";
    if (!std::filesystem::exists(currentDateFolder)) {
        std::filesystem::create_directory(currentDateFolder);
    }
}

void Detector::cleanupOldFolders(const std::string& imagesFolderPath) {
    std::vector<std::pair<std::string, std::string>> folders; // 存储文件夹路径和日期

    // 遍历所有文件夹，提取日期
    for (const auto& entry : std::filesystem::directory_iterator(imagesFolderPath)) {
        if (std::filesystem::is_directory(entry)) {
            std::string folderName = entry.path().filename().string();
            if (folderName.find("_images") != std::string::npos) {
                std::string dateStr = folderName.substr(0, 8); // 提取 YYYYMMDD
                folders.emplace_back(dateStr, entry.path().string());
            }
        }
    }

    // 按日期排序（从旧到新）
    std::sort(folders.begin(), folders.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // 删除超过 5 天的文件夹
    while (folders.size() > 5) {
        std::filesystem::remove_all(folders.front().second);
        folders.erase(folders.begin());
    }
}

void Detector::generateJsonFile(const std::string& imagePath, const cv::Mat& image, const std::vector<network_space::Object>& objects) {
    // 构造 JSON 文件路径
    std::string jsonPath = imagePath.substr(0, imagePath.rfind('.')) + ".json";

    // 构造 JSON 数据
    nlohmann::json jsonData; // 使用 nlohmann/json 库
    jsonData["version"] = "2.4.4";
    jsonData["flags"] = {};
    jsonData["imagePath"] = std::filesystem::path(imagePath).filename().string();
    jsonData["imageData"] = nullptr;
    jsonData["imageHeight"] = image.rows;
    jsonData["imageWidth"] = image.cols;
    jsonData["text"] = "";
    jsonData["description"] = "";

    // 填充 shapes 数组
    jsonData["shapes"] = nlohmann::json::array();
    for (const auto& obj : objects) {
        nlohmann::json shape;
        shape["label"] = std::to_string(obj.label_i); // 转换为字符串
        shape["score"] = obj.prob_f;
        shape["points"] = {
            {obj.rect.x, obj.rect.y},
            {obj.rect.x + obj.rect.width, obj.rect.y},
            {obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height},
            {obj.rect.x, obj.rect.y + obj.rect.height}
        };
        shape["group_id"] = nullptr;
        shape["description"] = "";
        shape["difficult"] = false;
        shape["shape_type"] = "rectangle";
        shape["flags"] = {};
        shape["attributes"] = {};
        shape["kie_linking"] = {};
        jsonData["shapes"].push_back(shape);
    }

    // 写入 JSON 文件
    std::ofstream jsonFile(jsonPath);
    if (jsonFile.is_open()) {
        jsonFile << jsonData.dump(4); // 格式化输出，缩进 4 个空格
        jsonFile.close();
    } else {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string("Error: Failed to write JSON file to %s", jsonPath.c_str()).c_str());
    }
}