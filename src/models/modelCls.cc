#include "NvInferPlugin.h"
#include <fstream>
#include <cmath>      
#include <algorithm> 

#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "resizeNormalize.h"
#include "modelCls.h"



void Classifier::postprocess() {
    std::vector<std::vector<network_space::Object>>& output_vec = this->baseAlgoParser.inOutPutData.output;
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, 
        format_to_string(
            "[%s] postprocess-> output_vec.size() = %zu\n"
            , this->modelName.c_str()
            , output_vec.size()
        ).c_str()
    );
    output_vec.clear();
    auto& input_vec = this->baseAlgoParser.inOutPutData.input;
    int batch = input_vec.size();
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, 
        format_to_string(
            "[%s] postprocess-> 后处理 batch = %d\n", this->modelName.c_str(), batch
        ).c_str()
    );
    auto& vvoidptrHostOuts_ = this->baseAlgoParser.nvptrEngine_Parser.vvoidptrHostOuts_;

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, 
        format_to_string(
            "[%s] postprocess-> vvoidptrHostOuts_.size() = %zu\n"
            , this->modelName.c_str()
            , vvoidptrHostOuts_.size()
        ).c_str()
    );

    if (vvoidptrHostOuts_.size() != this->baseAlgoParser.nvptrEngine_Parser.iNumOutputs_) {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR, 
            format_to_string(
                "postprocess-> Error: vvoidptrHostOuts_ size mismatch. Expected 1, got %d"
                , vvoidptrHostOuts_.size()
            ).c_str()
        );
        return;
    }
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, 
        format_to_string(
            "[%s] postprocess-> 当前batch = %d\n"
            , this->modelName.c_str()
            , batch
        ).c_str()
    );
    float* probabilities = static_cast<float*>(vvoidptrHostOuts_[0]);
    int num_classes = this->baseAlgoParser.nvptrEngine_Parser.iOutClsNum_;

    for (int idx = 0; idx < batch; ++idx) {
        std::vector<network_space::Object> subOutput_vec;

        float* probs = probabilities + idx * num_classes;

        int max_idx = std::distance(probs, std::max_element(probs, probs + num_classes));
        float max_prob = probs[max_idx];

        network_space::Object obj;
        obj.prob_f = max_prob;
        obj.label_i = max_idx;

        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, 
            format_to_string(
                "[%s] postprocess-> obj.prob_f = %f"
                , this->modelName.c_str()
                , obj.prob_f
            ).c_str()
        );
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO, 
            format_to_string(
                "[%s] postprocess-> obj.label_i = %d\n"
                , this->modelName.c_str()
                , obj.label_i
            ).c_str()
        );
        subOutput_vec.push_back(obj);

        output_vec.push_back(subOutput_vec);
    }
    this->baseAlgoParser.nvptrEngine_Parser.reset_EngineParser_vvoidptrX();
}

void Classifier::draw_boxes(size_t save_img_max_num) {
    std::vector<std::vector<network_space::Object>>& output_vec = this->baseAlgoParser.inOutPutData.output;
    std::vector<network_space::InputData>& input_vec = this->baseAlgoParser.inOutPutData.input;

    if (input_vec.size() != output_vec.size()) {
        std::cerr << "Error: The number of input images and the number of object batches do not match." << std::endl;
        return;
    }
    

    for (size_t i = 0; i < input_vec.size(); ++i) {
        cv::Mat res = input_vec[i].oriImage.clone();

        if (res.empty()) {
            std::cerr << "Error: Failed to clone image at index " << i  << std::endl;
            continue;
        }

        for (const auto& obj : output_vec[i]) {
            cv::Scalar color = cv::Scalar({0, 0, 255});
            cv::rectangle(res, obj.rect, color, 2);

            char text[256];
            sprintf(text, "%zu %.1f%%", obj.label_i, obj.prob_f * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            int x = static_cast<int>(obj.rect.x);
            int y = static_cast<int>(obj.rect.y) + 1;

            if (y > res.rows) {
                y = res.rows;
            }

            cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

            cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
        }

        std::string output_path = "/data/01_Project/algoLibraryBatch/res/cls_result_" + std::to_string(i) + ".jpg";
        if (!cv::imwrite(output_path, res)) {
            std::cerr << "Error: Failed to save image to " << output_path << std::endl;
        } else {
        }
    }
}
