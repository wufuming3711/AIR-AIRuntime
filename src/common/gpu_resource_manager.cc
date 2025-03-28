#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <nvml.h>

#include "gpu_resource_manager.h"
#include "common.inl"
#include "NvInfer.h"


GpuResourceManager::GpuResourceManager(
    std::shared_ptr<logger::CustomLogger>& logger,
    float max_mem_usage, 
    float max_cpu_usage
) : sptrLogger_(logger)
  , max_memory_usage_(max_mem_usage)
  , max_cpu_usage_(max_cpu_usage) {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        RUNTIME_LOG(
            sptrLogger_,
            nvinfer1::ILogger::Severity::kERROR,
            (
                "Failed to initialize NVML: "
                + std::string(nvmlErrorString(result))
                + "\n"
                + "如果这行代码被执行，意味着在尝试初始化NVML时发生了错误。具体的原因可能包括但不限于以下几种情况\n："
                + "1. NVML库未正确安装或配置：如之前讨论的，确保NVML库已正确安装，并且编译和链接过程中能够找到它。\n"
                + "2. NVIDIA驱动程序问题：NVML依赖于NVIDIA驱动程序，如果驱动程序不兼容或有问题，可能会导致初始化失败。\n"
                + "3. 权限问题：某些操作可能需要管理员权限，特别是在访问硬件资源时。\n"
                + "4. GPU设备不可用：系统中没有可用的NVIDIA GPU，或者所有GPU都被其他进程占用。\n"
                + "5. 库版本不匹配：如果您使用的是较新的NVML API（例如 nvmlInit_v2），而您的库版本较旧，则可能导致不兼容的问题。\n"
            ).c_str()
        );
        return;
    }
    bIsInitial = true;
}

GpuResourceManager::~GpuResourceManager() {
    nvmlShutdown();
}

bool GpuResourceManager::initialize() {
    return update_gpu_status();
}

size_t GpuResourceManager::get_optimal_gpu_id() {
    if (!update_gpu_status()) {
        RUNTIME_LOG(
            sptrLogger_,
            nvinfer1::ILogger::Severity::kERROR,
            "获取最优gpu失败，原因：无法获取gpu基础信息"
        );
        return -1;
    };

    size_t optimal_gpu_id = INVALID_GPU_ID;
    float min_memory_usage = 100.0f;
    float min_cpu_usage = 100.0f;

    for (const auto& [id, status] : gpus_) {
        if (is_optimal(status) 
            && status.memory_usage 
                < min_memory_usage 
                  && status.cpu_usage 
                    < min_cpu_usage
        ) {
            optimal_gpu_id = id;
            min_memory_usage = status.memory_usage;
            min_cpu_usage = status.cpu_usage;
        }
    }

    return optimal_gpu_id;
}

std::vector<size_t> GpuResourceManager::get_available_gpu_ids() const {
    std::vector<size_t> available_gpus;
    for (const auto& [id, status] : gpus_) {
        if (status.is_available) {
            available_gpus.push_back(id);
        }
    }
    return available_gpus;
}

bool GpuResourceManager::is_optimal(const GpuStatus& status) {
    return status.is_available 
        && status.memory_usage < max_memory_usage_ 
        && status.cpu_usage < max_cpu_usage_;
}

bool GpuResourceManager::update_gpu_status() {
    unsigned int device_count = 0;
    nvmlReturn_t result = nvmlDeviceGetCount_v2(&device_count);
    if (result != NVML_SUCCESS) {
        RUNTIME_LOG(
            sptrLogger_, 
            nvinfer1::ILogger::Severity::kERROR,
            (
                "Failed to get GPU count: " 
                + std::string(nvmlErrorString(result))
            ).c_str()
        );
        return false;
    }

    RUNTIME_LOG(
        sptrLogger_, 
        nvinfer1::ILogger::Severity::kINFO,
        format_to_string("可获取的GPU数量为: %d", device_count).c_str()
    );

    std::unordered_map<size_t, GpuStatus> new_statuses;

    for (unsigned int i = 0; i < device_count; ++i) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex_v2(i, &device);
        if (result != NVML_SUCCESS) {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                format_to_string(
                    "Failed to get handle for GPU %d : %s",
                    i,
                    nvmlErrorString(result)
                ).c_str()
            );
            continue;
        }

        nvmlMemory_t memory_info;
        result = nvmlDeviceGetMemoryInfo(device, &memory_info);
        if (result != NVML_SUCCESS) {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                format_to_string(
                    "Failed to get memory info for GPU %d : %s",
                    i,
                    nvmlErrorString(result)
                ).c_str()
            );
            continue;
        }

        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result != NVML_SUCCESS) {
            RUNTIME_LOG(sptrLogger_, nvinfer1::ILogger::Severity::kERROR,
                format_to_string(
                    "Failed to get utilization rates for GPU %d : %s",
                    i,
                    nvmlErrorString(result)
                ).c_str()
            );
            continue;
        }

        float mem_usage = static_cast<float>(memory_info.used) / static_cast<float>(memory_info.total) * 100.0f;
        float gpu_utilization = static_cast<float>(utilization.gpu);

        bool available = mem_usage < max_memory_usage_ && gpu_utilization < max_cpu_usage_;

        new_statuses[i] = {mem_usage, gpu_utilization, available};

        RUNTIME_LOG(
            sptrLogger_,
            nvinfer1::ILogger::Severity::kINFO,
            format_to_string(
                "GPU %d : Memory Usage %.2f%%, GPU Utilization %.2f%%",
                i,
                mem_usage,
                gpu_utilization
            ).c_str()
        );
    }

    gpus_ = new_statuses;
    return true;
}