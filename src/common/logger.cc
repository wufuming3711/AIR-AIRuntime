#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <thread>
#include <mutex>
#include "NvInfer.h" 
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <cstring>
#include <vector>
#include <algorithm> 
#include <iomanip> 
#include "logger.h"

namespace fs = std::filesystem;

void logger::CustomLogger::initialize(
    const std::string& sWorkflowName
) {
    sWorkflowName_ = sWorkflowName;
    initLogDirectory();
    rollLogFile();
}

void logger::CustomLogger::log(
    Severity severity, const char* msg
) noexcept {    
    if (severity == Severity::kERROR) {
        this->vsInterErr_.push_back(msg);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto pid = getpid();
    auto tid = syscall(SYS_gettid);

    if (!logFile_.is_open() || szFileSize_ >= szMaxLogSize_) {
        rollLogFile();
    }

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");

    logFile_ 
        << "[" 
        << ss.str()
        << "] [" 
        << getSeverityString(severity) 
        << "] [" 
        << pid 
        << ":" 
        << tid 
        << "] [" 
        << this->sWorkflowName_ 
        << "] " 
        << msg 
        << std::endl;

    szFileSize_ += strlen(msg) + ss.str().length() + 1; 
}

void logger::CustomLogger::initLogDirectory() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y%m%d", localtime(&time_t));

    fs::path dateDir = this->logDir + std::string(buffer);
    if (!fs::exists(dateDir)) {
        fs::create_directories(dateDir);
    }

    char pidBuf[16], tidBuf[16];
    snprintf(pidBuf, sizeof(pidBuf), "%d", getpid());
    snprintf(tidBuf, sizeof(tidBuf), "%ld", syscall(SYS_gettid));
    logDir_ = dateDir / (sWorkflowName_ + "_pid-" + std::string(pidBuf) + "_tid-" + std::string(tidBuf));
    if (!fs::exists(logDir_)) {
        fs::create_directories(logDir_);
    }
    this->logDirPath = logDir_;
}

void logger::CustomLogger::rollLogFile(
) {
    if (logFile_.is_open()) {
        logFile_.close();
    }

    cleanOldLogs();

    std::stringstream ss;
    ss << logDir_.string() << "/" << iCurrentLogFileIndex_ << ".log";
    logFile_.open(ss.str(), std::ios_base::app);
    szFileSize_ = 0;

    iCurrentLogFileIndex_ = (iCurrentLogFileIndex_ % szMaxFiles_) + 1;
}

void logger::CustomLogger::cleanOldLogs(
) {
    std::vector<fs::path> logFiles;
    for (const auto& entry : fs::directory_iterator(logDir_)) {
        if (entry.path().extension() == ".log") {
            logFiles.push_back(entry.path());
        }
    }

    std::sort(logFiles.begin(), logFiles.end(), [](const fs::path& a, const fs::path& b) {
        return fs::last_write_time(a) < fs::last_write_time(b);
    });

    while (logFiles.size() > szMaxFiles_) {
        fs::remove(logFiles.front());
        logFiles.erase(logFiles.begin());
    }
}

std::string logger::CustomLogger::getSeverityString(
    Severity severity
) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case Severity::kERROR: return "ERROR";
        case Severity::kWARNING: return "WARNING";
        case Severity::kINFO: return "INFO";
        default: return "UNKNOWN";
    }
}

void logger::CustomLogger::logWithLocation(
    Severity severity, const char* msg,
    const char* file, int line
) noexcept {
    std::ostringstream oss;
    oss << "[" << file << ":" << line << "] " << msg;
    log(severity, oss.str().c_str());
}

std::string logger::CustomLogger::getInterErrAsString() {
    std::ostringstream oss;
    std::lock_guard<std::mutex> lock(this->mtx_);

    for (size_t i = 0; i < vsInterErr_.size(); ++i) {
        oss << "[" << i << "] " << vsInterErr_[i] << ";";
    }
    return oss.str();
}

logger::CustomLogger::~CustomLogger() {
    if (logFile_.is_open()) {
        logFile_.close();
    }

    bool deletionFailed = false;

    try {
        if (!logDir_.empty() && fs::exists(logDir_)) {
            for (const auto& entry : fs::directory_iterator(logDir_)) {
                fs::remove_all(entry.path());
            }
            fs::remove_all(logDir_);
        }
    } catch (const fs::filesystem_error& e) {
        deletionFailed = true;
        std::cerr << "Error while removing log directory: " << e.what() << std::endl;

        if (iCurrentLogFileIndex_ > 0) {
            std::stringstream ss;
            ss << logDir_.string() << "/" << iCurrentLogFileIndex_ << ".log";
            logFile_.open(ss.str(), std::ios_base::app); 
            
            if (logFile_.is_open()) {
                logFile_ 
                    << "[ERROR] [DESTRUCTOR] "
                    << "Error while removing log directory: " << e.what()
                    << std::endl;
                logFile_.close();
            }
        }
    }

    if (deletionFailed) {
        // 可选：这里可以执行其他操作，比如标记日志文件夹为待清理等
    }
}