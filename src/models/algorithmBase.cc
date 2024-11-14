#include "algorithmBase.h"
#include "common.h"
#include "common.inl"
#include "networkSpace.h"


ALGORITHM_BASE::ALGORITHM_BASE(const std::string& engineFilePath){
    // printf("[INFO] ALGORITHM_BASE-> 准备加载engine\n");
    if (this->loadEngine(engineFilePath)) {
        // printf("[INFO] ALGORITHM_BASE-> 加载engine完成\n");
        networkSpace::EngineParser& engineParser = this->baseAlgoParser.engineParser;
        nvinfer1::Dims minDims = this->engine->getProfileDimensions(
            0, 0, nvinfer1::OptProfileSelector::kMIN);  // 1
        nvinfer1::Dims optDims = this->engine->getProfileDimensions(
            0, 0, nvinfer1::OptProfileSelector::kOPT);  // 16
        nvinfer1::Dims maxDims = this->engine->getProfileDimensions(
            0, 0, nvinfer1::OptProfileSelector::kMAX);  // 32
        // printf("[INFO] ALGORITHM_BASE-> 从网络中获取batch信息\n");
        engineParser.maxBatch_i = maxDims.d[0];
        engineParser.bestBatch_i = optDims.d[0];
        engineParser.numBindings_i = this->engine->getNbBindings();
        // printf("[INFO] ALGORITHM_BASE-> maxBatch_i = %d, bestBatch_i = %d, numBindings_i = %d\n", 
        //     engineParser.maxBatch_i, engineParser.bestBatch_i, engineParser.numBindings_i);
        // 遍历模型输入输出节点 获取模型基础信息
        for (int i = 0; i < engineParser.numBindings_i; ++i) {
            networkSpace::Binding binding;
            nvinfer1::Dims dims;
            nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
            std::string name_s = this->engine->getBindingName(i);
            binding.name_c = name_s.c_str();
            binding.dsize_i = type_to_size(dtype);

            bool IsInput = engine->bindingIsInput(i);
            if (IsInput) {
                engineParser.numInputs_i += 1;
                dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);
                dims.d[0] = 1;
                binding.size_i = get_size_by_dims(dims);
                binding.dims = dims;
                engineParser.inputBindings_vec.push_back(binding);
                engineParser.inputSizeHW_vec.push_back({dims.d[2], dims.d[3]});
            }
            else {
                if (!context) {
                    std::cerr << "Context is not initialized." << std::endl;
                    return;
                }

                // 设置输入维度
                for (int j = 0; j < engineParser.numInputs_i; ++j) {
                    this->context->setBindingDimensions(j, engineParser.inputBindings_vec[j].dims);
                }

                // 获取输出绑定维度
                dims = context->getBindingDimensions(i);
                binding.size_i = get_size_by_dims(dims);
                binding.dims = dims;
                engineParser.outputBindings_vec.push_back(binding);
                engineParser.numOutputs_i += 1;
            }
        }
        // printf("[INFO] ALGORITHM_BASE-> 模型构造完成\n");
    }
}

ALGORITHM_BASE::~ALGORITHM_BASE() {
    printf("[INFO] ~ALGORITHM_BASE-> 调用析构函数释放内存\n");
    // 1. 销毁 IExecutionContext 对象
    if (this->context) {
        this->context->destroy();
        this->context = nullptr;
        printf("[INFO] ~ALGORITHM_BASE-> this->context->destroy();\n");
    }
    // 2. 销毁 Engine 对象
    if (this->engine) {
        this->engine->destroy();
        this->engine = nullptr;
        printf("[INFO] ~ALGORITHM_BASE-> this->engine->destroy();\n");
    }
    // 3. 销毁 Runtime 对象
    if (this->runtime) {
        this->runtime->destroy();
        this->runtime = nullptr;
        printf("[INFO] ~ALGORITHM_BASE-> this->runtime->destroy();\n");
    }
    // 4. 销毁 CUDA 流对象
    if (this->stream) {
        cudaStreamDestroy(this->stream);
        this->stream = nullptr;
        printf("[INFO] ~ALGORITHM_BASE-> cudaStreamDestroy(this->stream);\n");
    }
    // 5. 释放设备和主机内存
    networkSpace::EngineParser& engineParser = this->baseAlgoParser.engineParser;
    for (auto& ptr : engineParser.deviceInPtrs_vec) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (auto& ptr : engineParser.deviceOutPtrs_vec) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (auto& ptr : engineParser.hostOutPtrs_vec) {
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
    }
}

bool ALGORITHM_BASE::loadEngine(const std::string& engineFilePath) {
    std::ifstream file(engineFilePath, std::ios::binary);
    if(!file.good()){
        printf("[ERROR] loadEngine-> can not open file %s\n", engineFilePath.c_str());  
        return false;
    }
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    char* trtModelStream = new char[size];
    if(trtModelStream == nullptr){
        printf("[ERROR] loadEngine-> can not new char[size]\n");
        return false;
    }

    file.read(trtModelStream, size);

    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    if(this->runtime == nullptr){
        printf("[ERROR] loadEngine-> can not createInferRuntime\n");
        delete[] trtModelStream;
        return false;
    }

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    if(this->engine == nullptr){
        printf("[ERROR] loadEngine-> can not deserializeCudaEngine\n");
        delete[] trtModelStream;
        return false;
    }

    cudaStreamCreate(&this->stream);
    if (!this->createContext()) return false;
    delete[] trtModelStream;
    return true;
}

bool ALGORITHM_BASE::createContext() {
    this->context = this->engine->createExecutionContext();
    if (!this->context){
        printf("[ERROR] createContext-> 初始化context失败");
        return false;
    }
    return true;
}

bool ALGORITHM_BASE::setCurtContext(int batchSize, int channel, int imgh, int imgw){
    if (batchSize <= 0 && batchSize > this->engine->getMaxBatchSize()){
        printf("[INFO] setCurtContext-> 当前batch不在推理引擎执行范围, 0 < batchSize=%d < %d\n", batchSize, this->engine->getMaxBatchSize());
        return false;
    }
    this->context->setBindingDimensions(0, nvinfer1::Dims4(batchSize, channel, imgh, imgw));
    if (!this->context){
        printf("[ERROR] setCurtContext-> 根据动态batch设置context上下文失败 (batchSize:%d, channel:%d, imgh:%d, imgw:%d)\n",
            batchSize, channel, imgh, imgw);
        return false;
    }
    return true;
}

bool ALGORITHM_BASE::warmup(int warmupNum) {
    if (warmupNum <= 0) return true;
    bool warmup = true;
    int inputH_i = this->baseAlgoParser.engineParser.inputSizeHW_vec[0][0];
    int inputW_i = this->baseAlgoParser.engineParser.inputSizeHW_vec[0][1];
    for (size_t batch = 1; batch <= this->baseAlgoParser.engineParser.maxBatch_i; batch += 1) {
        std::vector<cv::Mat> images;
        // 生成测试数据
        std::srand(std::time(0));
        for (int i = 0; i < batch; ++i) {
            int type = CV_8UC3;
            cv::Mat randomImage(inputH_i, inputW_i, type);
            
            cv::randu(randomImage, cv::Scalar::all(0), cv::Scalar::all(255));
            images.push_back(randomImage);
        }
        printf("[INFO] warmup-> 开始测试\n");
        for (int i = 0; i < warmupNum; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            this->commitImages(images);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("[INFO] warmup-> [batch %d] [warmup %d] warmup耗时: %d 毫秒\n", batch, i, duration);
        }
    }
    printf("[INFO] warmup-> model warmup %d times\n", warmupNum);
    return true;
}

void ALGORITHM_BASE::inferCore() {
    auto& engineParser = this->baseAlgoParser.engineParser;
    std::vector<networkSpace::Binding>& outputBindings_vec = engineParser.outputBindings_vec;
    // printf("[INFO] inferCore-> engineParser.deviceInPtrs_vec.size() = %d\n", engineParser.deviceInPtrs_vec.size());
    std::vector<void*>& deviceInPtrs_vec = engineParser.deviceInPtrs_vec;
    std::vector<void*>& deviceOutPtrs_vec = engineParser.deviceOutPtrs_vec;

    std::vector<void*> bindings(engineParser.deviceInPtrs_vec.size() + deviceOutPtrs_vec.size());
    std::copy(deviceInPtrs_vec.begin(), deviceInPtrs_vec.end(), bindings.begin());
    std::copy(deviceOutPtrs_vec.begin(), deviceOutPtrs_vec.end(), bindings.begin() + deviceInPtrs_vec.size());

    // printf("[INFO] inferCore-> 确保 bindings 的长度正确\n");
    // printf("[INFO] inferCore-> 输入输出总数 = %d\n", deviceInPtrs_vec.size() + deviceOutPtrs_vec.size());
    // printf("[INFO] inferCore-> bindings.size() = %d\n", bindings.size());
    if (engineParser.numBindings_i != deviceInPtrs_vec.size() + deviceOutPtrs_vec.size()) {
        std::cerr << "inferCore-> Error: numBindings_i mismatch. Expected " << deviceInPtrs_vec.size() + deviceOutPtrs_vec.size() << ", got " << engineParser.numBindings_i << std::endl;
        exit(-1);
    }

    // printf("[INFO] inferCore-> 调用 enqueueV2 方法\n");
    bool success = this->context->enqueueV2(bindings.data(), this->stream, nullptr);

    if (!success) {
        std::cerr << "inferCore-> Failed to enqueue the execution context" << std::endl;
        exit(-1);
    }

    // printf("[INFO] inferCore-> context->enqueueV2 done.\n");
    // printf("[INFO] inferCore-> engineParser.numInputs_i = %d\n", engineParser.numInputs_i);

    for (size_t i = 0; i < engineParser.numOutputs_i; ++i) {
        // printf("[INFO] inferCore-> [%zu / %d] 推理结果同步到\n", i, engineParser.numOutputs_i);
        // printf("[INFO] outputBindings_vec[%zu].size_i=%d,  outputBindings_vec[%zu].dsize_i=%d, engineParser.curtBatch_i=%d, \n", 
        //        i, outputBindings_vec[i].size_i, i, outputBindings_vec[i].dsize_i, engineParser.curtBatch_i);
        size_t osize = outputBindings_vec[i].size_i * outputBindings_vec[i].dsize_i * engineParser.curtBatch_i;
        // printf("[INFO] inferCore-> [%zu / %d] 输出数据所需空间： %zu\n", i, engineParser.numOutputs_i, osize);

        // printf("[INFO] inferCore-> hostOutPtrs_vec[%zu] 地址: %p, 大小: %zu\n", i, engineParser.hostOutPtrs_vec[i], osize);

        // printf("[INFO] inferCore-> 推理结果同步到host \n");
        CHECK(cudaMemcpyAsync(
            engineParser.hostOutPtrs_vec[i],
            deviceOutPtrs_vec[i],
            osize, 
            cudaMemcpyDeviceToHost, 
            this->stream
        ));

        CHECK(cudaStreamSynchronize(this->stream));

        #ifdef DEBUG
        printf("[INFO] inferCore-> 打印 hostOutPtrs_vec[%zu] 中的数据\n", i);
        switch (i) {
            case 0: {  // 假设第0个输出是检测框数量
                int* num_dets = static_cast<int*>(engineParser.hostOutPtrs_vec[i]);
                for (int j = 0; j < engineParser.curtBatch_i; ++j) {
                    printf("[DEBUG] num_dets[%d] = %d\n", j, num_dets[j]);
                }
                break;
            }
            case 1: {  // 假设第1个输出是检测框坐标
                float* boxes = static_cast<float*>(engineParser.hostOutPtrs_vec[i]);
                for (int j = 0; j < osize / sizeof(float); j += 4) {
                    printf("[DEBUG] boxes[%d] = (%f, %f, %f, %f)\n", j / 4, boxes[j], boxes[j + 1], boxes[j + 2], boxes[j + 3]);
                }
                break;
            }
            case 2: {  // 假设第2个输出是检测框得分
                float* scores = static_cast<float*>(engineParser.hostOutPtrs_vec[i]);
                for (int j = 0; j < osize / sizeof(float); ++j) {
                    printf("[DEBUG] scores[%d] = %f\n", j, scores[j]);
                }
                break;
            }
            case 3: {  // 假设第3个输出是检测框类别
                int* labels = static_cast<int*>(engineParser.hostOutPtrs_vec[i]);
                for (int j = 0; j < osize / sizeof(int); ++j) {
                    printf("[DEBUG] labels[%d] = %d\n", j, labels[j]);
                }
                break;
            }
            default:
                printf("[WARNING] inferCore-> 未知的输出索引: %zu\n", i);
        }
        #endif  // DEBUG
    }

    CHECK(cudaStreamSynchronize(this->stream));
    // printf("[INFO] inferCore-> infer done.\n");
}