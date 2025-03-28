#include "algorithmBase.h"
#include "common.h"
#include "common.inl"
#include "networkSpace.h"
#include "letterbox.h"

auto getDimsToStr = [](const nvinfer1::Dims &dims, const char *type) -> std::string
{
    std::ostringstream oss;
    oss << type << " Dimensions (nbDims=" << dims.nbDims << "): ";

    for (int i = 0; i < dims.nbDims; ++i)
    {
        oss << dims.d[i];
        if (i < dims.nbDims - 1)
        {
            oss << " x ";
        }
    }
    return oss.str();
};

AlgorithmBase::AlgorithmBase(
    const std::string &nvptrEngine_FilePath,
    std::shared_ptr<logger::CustomLogger> &logger) : sptrLogger_(logger)
{
    if (this->loadEngine(nvptrEngine_FilePath))
    {
        network_space::EngineParser &nvptrEngine_Parser = this->baseAlgoParser.nvptrEngine_Parser;
        int numProfiles = this->nvptrEngine_->getNbOptimizationProfiles();
        if (numProfiles > 0)
        {
            RUNTIME_LOG(sptrLogger_,
                        nvinfer1::ILogger::Severity::kINFO,
                        "当前nvptrEngine_支持动态batch");
            nvinfer1::Dims minDims = this->nvptrEngine_->getProfileDimensions(
                0, 0, nvinfer1::OptProfileSelector::kMIN);
            nvinfer1::Dims optDims = this->nvptrEngine_->getProfileDimensions(
                0, 0, nvinfer1::OptProfileSelector::kOPT);
            nvinfer1::Dims maxDims = this->nvptrEngine_->getProfileDimensions(
                0, 0, nvinfer1::OptProfileSelector::kMAX);

            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                        getDimsToStr(minDims, "Min").c_str());
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                        getDimsToStr(optDims, "Opt").c_str());
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                        getDimsToStr(maxDims, "Max").c_str());
            nvptrEngine_Parser.iMaxBatch_ = maxDims.d[0];
            nvptrEngine_Parser.iBestBatch_ = optDims.d[0];
        }
        else
        {
            RUNTIME_LOG(sptrLogger_,
                        nvinfer1::ILogger::Severity::kINFO,
                        "当前nvptrEngine_不支持动态batch 采用batch=1策略推理");
            nvptrEngine_Parser.iMaxBatch_ = 1;
            nvptrEngine_Parser.iBestBatch_ = 1;
        }
        nvptrEngine_Parser.iNumBindings_ = this->nvptrEngine_->getNbBindings();
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kINFO,
                    format_to_string(
                        "当前模型输入输出节点总数=%d", nvptrEngine_Parser.iNumBindings_)
                        .c_str());

        for (int i = 0; i < nvptrEngine_Parser.iNumBindings_; ++i)
        {
            network_space::Binding binding;
            nvinfer1::Dims dims;
            nvinfer1::DataType dtype = this->nvptrEngine_->getBindingDataType(i);
            std::string name_s = this->nvptrEngine_->getBindingName(i);
            binding.name_c = name_s.c_str();
            binding.dsize_i = type_to_size(dtype);

            bool IsInput = nvptrEngine_->bindingIsInput(i);
            if (IsInput)
            {
                nvptrEngine_Parser.iNumInputs_ += 1;
                dims = this->nvptrEngine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);
                dims.d[0] = 1;
                binding.size_i = get_size_by_dims(dims);
                binding.dims = dims;
                nvptrEngine_Parser.vbindInputBindings_.push_back(binding);
                nvptrEngine_Parser.vviInputSizeHW_.push_back({dims.d[2], dims.d[3]});
            }
            else
            {
                if (!context)
                {
                    RUNTIME_LOG(sptrLogger_,
                                nvinfer1::ILogger::Severity::kERROR,
                                "Context is not initialized.");
                    return;
                }
                for (int j = 0; j < nvptrEngine_Parser.iNumInputs_; ++j)
                {
                    if (!this->context->setBindingDimensions(j, nvptrEngine_Parser.vbindInputBindings_[j].dims))
                    {
                        RUNTIME_LOG(
                            sptrLogger_,
                            nvinfer1::ILogger::Severity::kERROR,
                            format_to_string(
                                "Failed to set binding dimensions for binding index %d (跳过iNumInputs_节点数据)", j)
                                .c_str());
                        return;
                    }
                }

                dims = context->getBindingDimensions(i);
                binding.size_i = get_size_by_dims(dims);
                binding.dims = dims;
                nvptrEngine_Parser.vbinOutputBindings_.push_back(binding);
                nvptrEngine_Parser.iNumOutputs_ += 1;
            }
        }
        RUNTIME_LOG(
            sptrLogger_,
            nvinfer1::ILogger::Severity::kINFO,
            "模型构造完成.");
    }
    else
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    "this->loadEngine(nvptrEngine_FilePath) 失败");
    }
    bIsInitial_ = true;
}

AlgorithmBase::~AlgorithmBase()
{
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                "~AlgorithmBase-> 调用析构函数释放内存");
    if (this->context)
    {
        this->context->destroy();
        this->context = nullptr;
    }
    if (this->nvptrEngine_)
    {
        this->nvptrEngine_->destroy();
        this->nvptrEngine_ = nullptr;
    }
    if (this->nvptrRuntime_)
    {
        this->nvptrRuntime_->destroy();
        this->nvptrRuntime_ = nullptr;
    }
    if (this->stream)
    {
        cudaStreamDestroy(this->stream);
        this->stream = nullptr;
    }
    network_space::EngineParser &nvptrEngine_Parser = this->baseAlgoParser.nvptrEngine_Parser;
    for (auto &ptr : nvptrEngine_Parser.vvoidptrDeviceIns_)
    {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (auto &ptr : nvptrEngine_Parser.vvoidptrDeviceOuts_)
    {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (auto &ptr : nvptrEngine_Parser.vvoidptrHostOuts_)
    {
        if (ptr)
        {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
    }
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                "~AlgorithmBase析构函数结束 资源释放");
}

bool AlgorithmBase::loadEngine(
    const std::string &nvptrEngine_FilePath)
{
    std::ifstream file(nvptrEngine_FilePath, std::ios::binary);
    if (!file.good())
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kERROR,
                    format_to_string(
                        "loadEngine-> can not open file %s", nvptrEngine_FilePath.c_str())
                        .c_str());
        return false;
    }
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char *trtModelStream = new char[size];
    if (trtModelStream == nullptr)
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kERROR,
                    "loadEngine-> can not new char[size] 罕见问题 加载nvptrEngine_文件之前尝试申请缓存内存 申请失败");
        return false;
    }

    file.read(trtModelStream, size);

    file.close();
    initLibNvInferPlugins(&(gLogger), "");
    this->nvptrRuntime_ = nvinfer1::createInferRuntime(gLogger);
    if (this->nvptrRuntime_ == nullptr)
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kERROR,
                    "loadEngine-> can not createInferRuntime");
        delete[] trtModelStream;
        return false;
    }
    this->nvptrEngine_ = this->nvptrRuntime_->deserializeCudaEngine(trtModelStream, size);
    if (this->nvptrEngine_ == nullptr)
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kERROR,
                    "loadEngine-> can not deserializeCudaEngine");
        delete[] trtModelStream;
        return false;
    }
    RUNTIME_CHECK(sptrLogger_, cudaStreamCreate(&this->stream));
    if (!this->createContext())
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kERROR,
                    "创建上下文失败");
        delete[] trtModelStream;
        return false;
    };
    delete[] trtModelStream;
    return true;
}

bool AlgorithmBase::createContext()
{
    this->context = this->nvptrEngine_->createExecutionContext();
    if (!this->context)
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kERROR,
                    "createContext-> 初始化context失败");
        return false;
    }
    return true;
}

bool AlgorithmBase::setCurtContext(
    int batchSize,
    int channel,
    int imgh,
    int imgw)
{
    if (batchSize <= 0 && batchSize > this->nvptrEngine_->getMaxBatchSize())
    {
        RUNTIME_LOG(sptrLogger_,
                    nvinfer1::ILogger::Severity::kERROR,
                    format_to_string(
                        "setCurtContext-> 当前batch不在推理引擎执行范围, 0 < batchSize=%d < %d", batchSize, this->nvptrEngine_->getMaxBatchSize())
                        .c_str());
        return false;
    }
    bool successful = this->context->setBindingDimensions(
        0,
        nvinfer1::Dims4(batchSize, channel, imgh, imgw));
    if (!successful)
    {
        RUNTIME_LOG(
            sptrLogger_,
            nvinfer1::ILogger::Severity::kERROR,
            format_to_string(
                "`context->setBindingDimensions` 执行失败,(batchSize=%d, channel=%d, imgh=%d, imgw=%d)", batchSize, channel, imgh, imgw)
                .c_str());
        return false;
    }
    if (!this->context)
    {
        RUNTIME_LOG(
            sptrLogger_,
            nvinfer1::ILogger::Severity::kERROR,
            format_to_string(
                "`context->setBindingDimensions` 执行失败, 根据动态batch设置context上下文失败 (batchSize:%d, channel:%d, imgh:%d, imgw:%d)", batchSize, channel, imgh, imgw)
                .c_str());
        return false;
    }
    return true;
}

bool AlgorithmBase::inferCore()
{
    auto &nvptrEngine_Parser = this->baseAlgoParser.nvptrEngine_Parser;
    std::vector<network_space::Binding> &vbinOutputBindings_ = nvptrEngine_Parser.vbinOutputBindings_;
    std::vector<void *> &vvoidptrDeviceIns_ = nvptrEngine_Parser.vvoidptrDeviceIns_;
    std::vector<void *> &vvoidptrDeviceOuts_ = nvptrEngine_Parser.vvoidptrDeviceOuts_;

    std::vector<void *> bindings(nvptrEngine_Parser.vvoidptrDeviceIns_.size() + vvoidptrDeviceOuts_.size());
    std::copy(vvoidptrDeviceIns_.begin(), vvoidptrDeviceIns_.end(), bindings.begin());
    std::copy(vvoidptrDeviceOuts_.begin(), vvoidptrDeviceOuts_.end(), bindings.begin() + vvoidptrDeviceIns_.size());

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "inferCore-> 确保 bindings 的长度正确")
                    .c_str());
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "inferCore-> vvoidptrDeviceIns_.size() = %d",
                    vvoidptrDeviceIns_.size())
                    .c_str());
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "inferCore-> vvoidptrDeviceOuts_.size() = %d",
                    vvoidptrDeviceOuts_.size())
                    .c_str());
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "inferCore-> nvptrEngine_Parser.iNumBindings_ = %d",
                    nvptrEngine_Parser.iNumBindings_)
                    .c_str());

    if (nvptrEngine_Parser.iNumBindings_ != vvoidptrDeviceIns_.size() + vvoidptrDeviceOuts_.size())
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string(
                        "inferCore-> Error: iNumBindings_ mismatch. Expected %d, got %d", nvptrEngine_Parser.iNumBindings_, vvoidptrDeviceIns_.size() + vvoidptrDeviceOuts_.size())
                        .c_str());
        return false;
    }

    bool success = this->context->enqueueV2(bindings.data(), this->stream, nullptr);

    if (!success)
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string(
                        "inferCore-> Failed to enqueue the execution context")
                        .c_str());
        return false;
    }
    for (size_t i = 0; i < nvptrEngine_Parser.iNumOutputs_; ++i)
    {
        size_t osize = vbinOutputBindings_[i].size_i * vbinOutputBindings_[i].dsize_i * nvptrEngine_Parser.iCurtBatch_;
        RUNTIME_CHECK(
            sptrLogger_,
            cudaMemcpyAsync(
                nvptrEngine_Parser.vvoidptrHostOuts_[i],
                vvoidptrDeviceOuts_[i],
                osize,
                cudaMemcpyDeviceToHost,
                this->stream));
        RUNTIME_CHECK(
            sptrLogger_,
            cudaStreamSynchronize(this->stream));
        RUNTIME_CHECK(
            sptrLogger_,
            cudaFree(vvoidptrDeviceOuts_[i]));
        vvoidptrDeviceOuts_[i] = nullptr;
    }
    CHECK(cudaStreamSynchronize(this->stream));
    for (size_t i = 0; i < nvptrEngine_Parser.iNumInputs_; ++i)
    {
        cudaFree(vvoidptrDeviceIns_[i]);
        vvoidptrDeviceIns_[i] = nullptr;
    }
}

bool AlgorithmBase::commitImages(
    const std::vector<cv::Mat> &images,
    const char *preprocess)
{
    network_space::EngineParser &nvptrEngine_Parser = this->baseAlgoParser.nvptrEngine_Parser;
    int batchSize = images.size();

    nvptrEngine_Parser.iCurtBatch_ = batchSize;

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[%s] commitImages-> nvptrEngine_Parser.iCurtBatch_ = %d", this->modelName.c_str(), batchSize)
                    .c_str());

    int inputH_i = nvptrEngine_Parser.vviInputSizeHW_[0][0];
    int inputW_i = nvptrEngine_Parser.vviInputSizeHW_[0][1];

    nvptrEngine_Parser.vvoidptrDeviceIns_.clear();
    nvptrEngine_Parser.vvoidptrDeviceOuts_.clear();
    nvptrEngine_Parser.vvoidptrHostOuts_.clear();

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[%s] commitImages-> 设置context, batchSize=%d, inputH_i=%d, inputW_i=%d", this->modelName.c_str(), batchSize, inputH_i, inputW_i)
                    .c_str());

    if (!this->setCurtContext(batchSize, 3, inputH_i, inputW_i))
    {
        return false;
    }

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[%s] commitImages-> 分配输入空间 CUDA, nvptrEngine_Parser.vbindInputBindings_.size()=%d", this->modelName.c_str(), nvptrEngine_Parser.vbindInputBindings_.size())
                    .c_str());
    int tmpcount = 0;
    for (auto &inBindings : nvptrEngine_Parser.vbindInputBindings_)
    {
        size_t insize = inBindings.size_i * inBindings.dsize_i * batchSize;
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                    format_to_string(
                        "[%s] inBindings.size_i: %d, inBindings.dsize_i: %d, batchSize: %d\n", this->modelName.c_str(), inBindings.size_i, inBindings.dsize_i, batchSize)
                        .c_str());
        void *in_d_ptr = nullptr;
        CHECK(cudaMallocAsync(&in_d_ptr, insize, this->stream));
        nvptrEngine_Parser.vvoidptrDeviceIns_.push_back(in_d_ptr);
    }

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[%s] commitImages-> nvptrEngine_Parser.vvoidptrDeviceIns_.size() = %d\n", this->modelName.c_str(), nvptrEngine_Parser.vvoidptrDeviceIns_.size())
                    .c_str());

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[INFO] [%s] nvptrEngine_Parser.vbinOutputBindings_.size() = %d\n", this->modelName.c_str(), nvptrEngine_Parser.vbinOutputBindings_.size())
                    .c_str());
    for (auto &outBindings : nvptrEngine_Parser.vbinOutputBindings_)
    {
        size_t outsize = outBindings.size_i * outBindings.dsize_i * batchSize;
        void *out_d_ptr;
        CHECK(cudaMallocAsync(&out_d_ptr, outsize, this->stream));
        nvptrEngine_Parser.vvoidptrDeviceOuts_.push_back(out_d_ptr);

        void *out_h_ptr;
        CHECK(cudaHostAlloc(&out_h_ptr, outsize, 0));
        nvptrEngine_Parser.vvoidptrHostOuts_.push_back(out_h_ptr);
    }
    // @wfm 没有出现内存泄漏
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[INFO] [%s] commitImages-> 分配输入空间 默认只有一个输入\n", this->modelName.c_str())
                    .c_str());
    auto &inBinding = nvptrEngine_Parser.vbindInputBindings_[0];
    size_t insize = inBinding.size_i * inBinding.dsize_i * batchSize;
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[INFO] [%s] inBinding.size_i: %d, inBinding.dsize_i: %d, batchSize: %d\n",
                    this->modelName.c_str(), inBinding.size_i, inBinding.dsize_i, batchSize)
                    .c_str());
    void *in_h_ptr = malloc(insize);
    if (in_h_ptr == nullptr)
    {
        RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                    format_to_string(
                        "[%s] commitImages-> 分配内存失败 include/models/modelDet.hpp # `void* in_h_ptr = malloc(insize);`\n", this->modelName.c_str())
                        .c_str());
        return false;
    }
    memset(in_h_ptr, 0, insize);

    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[%s] commitImages-> batchSize = %d\n", this->modelName.c_str(), batchSize)
                    .c_str());
    RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kINFO,
                format_to_string(
                    "[INFO] [%s] commitImages-> inputW_i = %d, inputH_i = %d\n", this->modelName.c_str(), inputW_i, inputH_i)
                    .c_str());
    cv::Size size{inputW_i, inputH_i};
    for (size_t imgIdx = 0; imgIdx < batchSize; ++imgIdx)
    {
        cv::Mat oriImage = images[imgIdx].clone();
        network_space::PreprocessParser preParser;
        preParser.size = size;
        network_space::InputData input(preParser);
        auto &inputImage = input.inputImage;
        input.oriImage = oriImage.clone();
        auto &inputOriImage = input.oriImage;
        if (
            (std::string(preprocess) == "letterbox") || (std::string(preprocess) == "rgb_letterbox"))
        {
            rgb_letterbox(inputOriImage, inputImage, size, input.preParser);
        }
        else if (std::string(preprocess) == "bgr_letterbox")
        {
            bgr_letterbox(inputOriImage, inputImage, size, input.preParser);
        }
        else
        {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                        format_to_string(
                            "[%s] 参数配置非法 preprocess = %s\n", this->modelName.c_str(), preprocess)
                            .c_str());
            exit(0);
        }

        this->baseAlgoParser.inOutPutData.input.push_back(input);
        memcpy(
            static_cast<char *>(in_h_ptr) + imgIdx * inBinding.size_i * inBinding.dsize_i,
            inputImage.data,
            inputImage.total() * inputImage.elemSize());
    }

    RUNTIME_CHECK(
        sptrLogger_,
        cudaMemcpyAsync(
            nvptrEngine_Parser.vvoidptrDeviceIns_[0],
            in_h_ptr,
            insize,
            cudaMemcpyHostToDevice,
            this->stream));
    free(in_h_ptr);
    this->inferCore();
    nvptrEngine_Parser.iCurtBatch_ = 0;

    return true;
}