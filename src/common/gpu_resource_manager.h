#ifndef GPU_RESOURCE_MANAGER_H
#define GPU_RESOURCE_MANAGER_H

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "logger.h"

class GpuResourceManager {
public:
    GpuResourceManager(
        std::shared_ptr<logger::CustomLogger>& logger,
        float max_mem_usage = 90.0f, 
        float max_cpu_usage = 90.0f
    );

    ~GpuResourceManager();

    bool initialize();

    size_t get_optimal_gpu_id();

    std::vector<size_t> get_available_gpu_ids() const;

public:
    bool bIsInitial = false;
    std::shared_ptr<logger::CustomLogger>& sptrLogger_;
private:
    struct GpuStatus {
        float memory_usage;
        float cpu_usage;
        bool is_available;
    };

    bool update_gpu_status();

    bool is_optimal(const GpuStatus& status);

    std::unordered_map<size_t, GpuStatus> gpus_;
    float max_memory_usage_;
    float max_cpu_usage_;
};

#endif // GPU_RESOURCE_MANAGER_H