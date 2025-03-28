//
// Created by Oboi on 2024/11/5.
//

#ifndef ANALYSIS_SERVICE_CONFIG_H
#define ANALYSIS_SERVICE_CONFIG_H

#include <string>
#include <list>

namespace conf
{
    using namespace std;

    class GRPCConfig
    {
    public:
        unsigned short port;
        string listenHost;
    };

    class TLSConfig
    {
    public:
        string caCertPath;
        string keyPath;
        string certPath;
    };

    class DecodeConfig
    {
    public:
        int imageMaxHeight;
        int imageMaxWidth;
        string execPath;
    };

    class AIConfig
    {
    public:
        int gpuCount;
        int detectionAlgorithmStart;
        int detectionAlgorithmEnd;
    };

    class AlgoConfig  // 新增外部参数 算法日志路径、算法配置文件、算法文件
    {
    public:
        string logDir;
        string configWorkflow;
        string modelZoo;
    };

    class Config
    {
    public:
        DecodeConfig* decode;
        TLSConfig* tls;
        GRPCConfig* gRPC;
        AIConfig* ai;
        AlgoConfig* algoConfig;

        static Config& getInstance();
        Config(const Config&) = delete;
        Config& operator=(const Config&) = delete;
        [[nodiscard]] bool loadFormFile(const string& filename) const;

    private:
        Config();
    };
} // conf

#endif //ANALYSIS_SERVICE_CONFIG_H
