#include "NvInferPlugin.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cmath>      // 包含exp函数的头文件
#include <algorithm>  // 包含max_element函数

#include "letterbox.h"
#include "modelDet.h"
#include "networkSpace.h"
#include "common.inl"


// TODO 这里的前处理函数不一样 剩下的都一样 统一后可以合并
bool DETECTOR::commitImages(const std::vector<cv::Mat>& images){
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

    // 
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
        letterbox(inputOriImage, inputImage, size, input.preParser);
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

void DETECTOR::postprocess() {
    std::vector<std::vector<networkSpace::Object>>& output_vec = this->baseAlgoParser.inOutPutData.output;
    // printf("[INFO] postprocess-> output_vec.size() = %zu\n", output_vec.size());
    output_vec.clear();
    auto& input_vec = this->baseAlgoParser.inOutPutData.input;
    int batch = input_vec.size();
    // printf("[INFO] postprocess-> 后处理 batch = %d\n", batch);

    auto& hostOutPtrs_vec = this->baseAlgoParser.engineParser.hostOutPtrs_vec;

    // 确保 hostOutPtrs_vec 的大小为 4
    // printf("[INFO] postprocess-> hostOutPtrs_vec.size() = %zu\n", hostOutPtrs_vec.size());
    if (hostOutPtrs_vec.size() != this->baseAlgoParser.engineParser.numOutputs_i) {
        std::cerr << "postprocess-> Error: hostOutPtrs_vec size mismatch. Expected 4, got " << hostOutPtrs_vec.size() << std::endl;
        return;
    }

    // printf("[INFO] postprocess-> 当前batch = %d\n", batch);

    // 打印 hostOutPtrs_vec[1] 中的所有检测框坐标
    // printf("[INFO] postprocess-> 打印 hostOutPtrs_vec[1] 中的所有检测框坐标\n");
    int total_boxes = 0;
    for (int idx = 0; idx < batch; ++idx) {
        int* num_dets = static_cast<int*>(hostOutPtrs_vec[0]) + idx;
        total_boxes += *num_dets;
    }
    float* boxes = static_cast<float*>(hostOutPtrs_vec[1]);
    float* scores = static_cast<float*>(hostOutPtrs_vec[2]);
    int* labels = static_cast<int*>(hostOutPtrs_vec[3]);

    for (int i = 0; i < total_boxes; ++i) {
        float* box_ptr = boxes + i * 4;
        // printf("[DEBUG] boxes[%d] = (%f, %f, %f, %f)\n", i, box_ptr[0], box_ptr[1], box_ptr[2], box_ptr[3]);
        // printf("[DEBUG] scores[%d] = %f\n", i, scores[i]);
        // printf("[DEBUG] labels[%d] = %d\n", i, labels[i]);
    }

    for (int idx = 0; idx < batch; ++idx) {
        // printf("-------------------------%d-------------------------\n", idx);
        std::vector<networkSpace::Object> subOutput_vec;
        // 获取当前 batch 的输出指针
        int* num_dets = static_cast<int*>(hostOutPtrs_vec[0]) + idx;
        float* boxes = static_cast<float*>(hostOutPtrs_vec[1]) + 100 * 4 * idx;
        float* scores = static_cast<float*>(hostOutPtrs_vec[2]) + 100 * idx;
        int* labels = static_cast<int*>(hostOutPtrs_vec[3]) + 100 * idx;

        size_t& oriImgHeight_i = input_vec[idx].preParser.oriImgHeight_i;
        size_t& oriImgWidth_i = input_vec[idx].preParser.oriImgWidth_i;
        auto& ratio_f = input_vec[idx].preParser.ratio_f;
        float& padw_f = input_vec[idx].preParser.padw_f;
        float& padh_f = input_vec[idx].preParser.padh_f;

        // std::cout << "[INFO] postprocess-> oriImgHeight_i = " << oriImgHeight_i << std::endl;
        // std::cout << "[INFO] postprocess-> oriImgWidth_i = " << oriImgWidth_i << std::endl;
        // std::cout << "[INFO] postprocess-> ratio_f = " << ratio_f << std::endl;
        // std::cout << "[INFO] postprocess-> padw_f = " << padw_f << std::endl;
        // std::cout << "[INFO] postprocess-> padh_f = " << padh_f << std::endl;
        // printf("[INFO] postprocess-> *num_dets = %d\n", *num_dets);

        // 调试输出：检查 boxes, scores, labels 的初始值
        for (int i = 0; i < *num_dets; ++i) {
            float* box_ptr = boxes + i * 4;
            // printf("[DEBUG] boxes[%d] = (%f, %f, %f, %f)\n", i, box_ptr[0], box_ptr[1], box_ptr[2], box_ptr[3]);
            // printf("[DEBUG] scores[%d] = %f\n", i, scores[i]);
            // printf("[DEBUG] labels[%d] = %d\n", i, labels[i]);
        }

        for (int i = 0; i < *num_dets; ++i) {
            networkSpace::Object obj;
            float* ptr = boxes + i * 4;
            float x0   = *ptr++ - padw_f;
            float y0   = *ptr++ - padh_f;
            float x1   = *ptr++ - padw_f;
            float y1   = *ptr - padh_f;

            // 打印中间结果
            // printf("[DEBUG] pre-clamp x0 = %f, y0 = %f, x1 = %f, y1 = %f\n", x0, y0, x1, y1);

            x0         = this->clamp(x0 * ratio_f, 0.f, static_cast<float>(oriImgWidth_i));
            y0         = this->clamp(y0 * ratio_f, 0.f, static_cast<float>(oriImgHeight_i));
            x1         = this->clamp(x1 * ratio_f, 0.f, static_cast<float>(oriImgWidth_i));
            y1         = this->clamp(y1 * ratio_f, 0.f, static_cast<float>(oriImgHeight_i));

            // 打印最终结果
            // printf("[DEBUG] post-clamp x0 = %f, y0 = %f, x1 = %f, y1 = %f\n", x0, y0, x1, y1);

            obj.rect.x      = x0;
            obj.rect.y      = y0;
            obj.rect.width  = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.prob_f      = scores[i];
            obj.label_i     = labels[i];

            // 调试输出：检查最终的 obj 值
            // printf("[INFO] postprocess-> obj.rect.x = %f\n", obj.rect.x);
            // printf("[INFO] postprocess-> obj.rect.y = %f\n", obj.rect.y);
            // printf("[INFO] postprocess-> obj.rect.width = %f\n", obj.rect.width);
            // printf("[INFO] postprocess-> obj.rect.height = %f\n", obj.rect.height);
            // printf("[INFO] postprocess-> obj.prob_f = %f\n", obj.prob_f);
            // printf("[INFO] postprocess-> obj.label_i = %d\n", obj.label_i);
            subOutput_vec.push_back(obj);
        }

        // 将当前 batch 的结果添加到 output_vec 中
        output_vec.push_back(subOutput_vec);
    }
}

void DETECTOR::draw_boxes() {
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
        std::string output_path = "/data/01_Project/algoLibraryBatch/res/det_result_" + std::to_string(i) + ".jpg";
        if (!cv::imwrite(output_path, res)) {
            std::cerr << "Error: Failed to save image to " << output_path << std::endl;
        } else {
            // std::cout << "Saved image to " << output_path << std::endl;
        }
    }
}