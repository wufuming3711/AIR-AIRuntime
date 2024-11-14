#include "NvInferPlugin.h"
#include <fstream>
#include <cmath>      // 包含exp函数的头文件
#include <algorithm>  // 包含max_element函数

#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "resizeNormalize.h"
#include "modelCls.h"


// TODO 这里的前处理函数不一样 剩下的都一样 统一后可以合并
bool CLASSIFIER::commitImages(const std::vector<cv::Mat>& images){
    networkSpace::EngineParser& engineParser = this->baseAlgoParser.engineParser;
    int batchSize = images.size();
    
    engineParser.curtBatch_i = batchSize;
    // printf("[INFO] commitImages-> engineParser.curtBatch_i = %d\n", batchSize);
    
    int inputH_i = engineParser.inputSizeHW_vec[0][0];
    int inputW_i = engineParser.inputSizeHW_vec[0][1];
    
    engineParser.deviceInPtrs_vec.clear();
    engineParser.deviceOutPtrs_vec.clear();
    engineParser.hostOutPtrs_vec.clear();

    // printf("[INFO] commitImages-> 设置context\n");
    if (!this->setCurtContext(batchSize, 3, inputH_i, inputW_i)) return false;

    // printf("[INFO] commitImages-> 分配输入空间 CUDA\n");
    for (auto& inBindings : engineParser.inputBindings_vec) {
        size_t insize = inBindings.size_i * inBindings.dsize_i * batchSize;
        // printf("[INFO] inBindings.size_i: %d, inBindings.dsize_i: %d, batchSize: %d\n", 
        //     inBindings.size_i, inBindings.dsize_i, batchSize);
        void* in_d_ptr;
        CHECK(cudaMallocAsync(&in_d_ptr, insize, this->stream));
        engineParser.deviceInPtrs_vec.push_back(in_d_ptr);
    }
    // printf("[INFO] commitImages-> engineParser.deviceInPtrs_vec.size() = %d\n", engineParser.deviceInPtrs_vec.size());

    // printf("[INFO] commitImages-> 分配输出空间 CUDA与CPU \n");
    for (auto& outBindings : engineParser.outputBindings_vec) {
        size_t outsize = outBindings.size_i * outBindings.dsize_i * batchSize;
        // printf("[INFO] outBindings.size_i: %d, outBindings.dsize_i: %d, batchSize: %d\n", 
        //     outBindings.size_i, outBindings.dsize_i, batchSize);
        void * out_d_ptr;
        CHECK(cudaMallocAsync(&out_d_ptr, outsize, this->stream));
        engineParser.deviceOutPtrs_vec.push_back(out_d_ptr);

        void * out_h_ptr;
        CHECK(cudaHostAlloc(&out_h_ptr, outsize, 0));
        engineParser.hostOutPtrs_vec.push_back(out_h_ptr);
    }

    // printf("[INFO] commitImages-> 分配输入空间 默认只有一个输入\n");
    auto& inBinding = engineParser.inputBindings_vec[0];
    size_t insize = inBinding.size_i * inBinding.dsize_i * batchSize;
    // printf("[INFO] inBinding.size_i: %d, inBinding.dsize_i: %d, batchSize: %d\n", 
    //     inBinding.size_i, inBinding.dsize_i, batchSize);
    void* in_h_ptr = malloc(insize);
    if (in_h_ptr == nullptr) {
        printf("[INFO] commitImages-> 分配内存失败 include/models/modelDet.hpp # `void* in_h_ptr = malloc(insize);`\n");
        return false;
    }
    memset(in_h_ptr, 0, insize);

    
    // printf("[INFO] commitImages-> 数据前处理 宽+高\n");
    cv::Size size{inputW_i, inputH_i};
    for (size_t imgIdx = 0; imgIdx < batchSize; ++imgIdx) {
        cv::Mat oriImage = images[imgIdx].clone();
        networkSpace::PreprocessParser preParser;
        preParser.size = size;
        networkSpace::InputData input(preParser);  // 引用
        auto& inputImage = input.inputImage;
        input.oriImage = oriImage.clone();
        auto& inputOriImage = input.oriImage;
        resizeNormalize(inputOriImage, inputImage, size, input.preParser);
        this->baseAlgoParser.inOutPutData.input.push_back(input);  // 深拷贝
        memcpy(
            static_cast<char*>(in_h_ptr) // 起始地址
                + imgIdx 
                * inBinding.size_i // 单个维度所需空间
                * inBinding.dsize_i, // 总的维度数量
            inputImage.data,   // 当前图片数据
            inputImage.total() * inputImage.elemSize()  // 当前图片所需空间
        );
        // printf("[INFO] commitImages-> 起始地址: %d\n", static_cast<char*>(in_h_ptr) + imgIdx * inBinding.size_i * inBinding.dsize_i);
        // printf("[INFO] commitImages-> 偏移量: %d\n", imgIdx * inBinding.size_i * inBinding.dsize_i);
        // printf("[INFO] commitImages-> 申请内存总量: %d\n", insize);
        // printf("[INFO] imgIdx: %d, inBinding.size_i: %d, inBinding.dsize_i: %d, inputImage.data: %d, inputImage.total(): %d, inputImage.elemSize(): %d\n"
        //     , imgIdx, inBinding.size_i, inBinding.dsize_i
        //     , inputImage.data
        //     , inputImage.total(), inputImage.elemSize());
        // std::cout << "[INFO] commitImages-> Image dimensions: " << inputOriImage.cols << "x" << inputOriImage.rows << std::endl;
    }
    // printf("[INFO] commitImages-> 数据同步到GPU\n");
    CHECK(cudaMemcpyAsync(engineParser.deviceInPtrs_vec[0],  // 单输入模型
        in_h_ptr, insize, cudaMemcpyHostToDevice, this->stream));
    free(in_h_ptr);

    // printf("[INFO] commitImages-> 开始推理\n");
    this->inferCore();
    engineParser.curtBatch_i = 0;
    // printf("[INFO] commitImages-> commitImages done.\n");
    return true;
}

void CLASSIFIER::postprocess() {
    std::vector<std::vector<networkSpace::Object>>& output_vec = this->baseAlgoParser.inOutPutData.output;
    // printf("[INFO] postprocess-> output_vec.size() = %zu\n", output_vec.size());
    output_vec.clear();
    auto& input_vec = this->baseAlgoParser.inOutPutData.input;
    int batch = input_vec.size();
    // printf("[INFO] postprocess-> 后处理 batch = %d\n", batch);

    auto& hostOutPtrs_vec = this->baseAlgoParser.engineParser.hostOutPtrs_vec;

    // 确保 hostOutPtrs_vec 的大小为 1
    // printf("[INFO] postprocess-> hostOutPtrs_vec.size() = %zu\n", hostOutPtrs_vec.size());

    if (hostOutPtrs_vec.size() != this->baseAlgoParser.engineParser.numOutputs_i) {
        std::cerr << "postprocess-> Error: hostOutPtrs_vec size mismatch. Expected 1, got " << hostOutPtrs_vec.size() << std::endl;
        return;
    }

    // printf("[INFO] postprocess-> 当前batch = %d\n", batch);

    // 获取模型输出的概率值
    float* probabilities = static_cast<float*>(hostOutPtrs_vec[0]);
    int num_classes = this->baseAlgoParser.engineParser.outClsNum_i;

    for (int idx = 0; idx < batch; ++idx) {
        // printf("-------------------------%d-------------------------\n", idx);
        std::vector<networkSpace::Object> subOutput_vec;

        // 获取当前 batch 的输出指针
        float* probs = probabilities + idx * num_classes;

        // 找到最大概率的类别
        int max_idx = std::distance(probs, std::max_element(probs, probs + num_classes));
        float max_prob = probs[max_idx];

        // 创建 Object 结构体并填充数据
        networkSpace::Object obj;
        obj.prob_f = max_prob;
        obj.label_i = max_idx;

        // 调试输出：检查最终的 obj 值
        // printf("[INFO] postprocess-> obj.prob_f = %f\n", obj.prob_f);
        // printf("[INFO] postprocess-> obj.label_i = %d\n", obj.label_i);

        subOutput_vec.push_back(obj);

        // 将当前 batch 的结果添加到 output_vec 中
        output_vec.push_back(subOutput_vec);
    }
}

void CLASSIFIER::draw_boxes() {
    std::vector<std::vector<networkSpace::Object>>& output_vec = this->baseAlgoParser.inOutPutData.output;
    std::vector<networkSpace::InputData>& input_vec = this->baseAlgoParser.inOutPutData.input;

    if (input_vec.size() != output_vec.size()) {
        std::cerr << "Error: The number of input images and the number of object batches do not match." << std::endl;
        return;
    }
    

    for (size_t i = 0; i < input_vec.size(); ++i) {
        cv::Mat res = input_vec[i].oriImage.clone();
        // cv::Mat res = input_vec[i].oriImage.clone();  // 克隆原始图像用于绘制

        if (res.empty()) {
            std::cerr << "Error: Failed to clone image at index " << i  << std::endl;
            continue;
        }
        // std::cout << "Image dimensions: " << res.cols << "x" << res.rows << std::endl;

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

        // 将绘制结果保存或显示
        std::string output_path = "/data/01_Project/algoLibraryBatch/res/cls_result_" + std::to_string(i) + ".jpg";
        if (!cv::imwrite(output_path, res)) {
            std::cerr << "Error: Failed to save image to " << output_path << std::endl;
        } else {
            // std::cout << "Saved image to " << output_path << std::endl;
        }
    }
}
