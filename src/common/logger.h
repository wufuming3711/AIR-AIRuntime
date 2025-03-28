#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <thread>
#include <mutex>
#include "NvInfer.h"  // 引入 TensorRT 的 ILogger 头文件
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <vector>

namespace fs = std::filesystem;

namespace logger {

class CustomLogger : public nvinfer1::ILogger {
private:
    std::string sWorkflowName_ = "UninitializedName-CustomLoggerInitName";
    size_t szMaxLogSize_ = 1024 * 1024;
    size_t szMaxFiles_ = 10;
    std::ofstream logFile_;
    size_t szFileSize_ = 0;
    std::mutex mutex_;
    std::mutex mtx_;
    fs::path logDir_;
    int iCurrentLogFileIndex_ = 1; 
    std::string logDir = "./log/";

private:
    void initLogDirectory();
    void rollLogFile();
    void cleanOldLogs();
    std::string getSeverityString(Severity severity);

public:
    std::string logDirPath;
    std::vector<std::string> vsInterErr_;

public:
    CustomLogger() {};
    CustomLogger(std::string logDir) : logDir(logDir.empty() ? "./log" : logDir) {}
    ~CustomLogger();
    
    void initialize(
        const std::string& sWorkflowName
    );

    void log(
        Severity severity, const char* msg
    ) noexcept override;

    void logWithLocation(
        Severity severity, const char* msg,
        const char* file, int line
    ) noexcept;

    std::string getInterErrAsString(
    );
};

}

#endif  //  _LOGGER_H_