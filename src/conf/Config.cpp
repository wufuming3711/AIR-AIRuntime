#include <mutex>
#include "Config.h"
#include <toml.hpp>
#include <iostream>
#include <sstream>  // 引入 stringstream

using namespace std;

namespace conf
{
    Config::Config()
    {
        decode = new DecodeConfig;
        tls = new TLSConfig;
        gRPC = new GRPCConfig;
        ai = new AIConfig;
        algoConfig = new AlgoConfig;
    };

    static Config* instance = nullptr;
    static once_flag onceFlag;

    Config& Config::getInstance()
    {
        call_once(onceFlag, []()
        {
            instance = new Config;
        });
        return *instance;
    }

    bool Config::loadFormFile(const string& filename) const
    {
        try
        {
            auto fileRoot = toml::parse(filename);

            auto tlsData = fileRoot.at("tls");
            tls->caCertPath = toml::find<string>(tlsData, "caCertPath");
            tls->keyPath = toml::find<string>(tlsData, "keyPath");
            tls->certPath = toml::find<string>(tlsData, "certPath");

            auto grpcData = fileRoot.at("gRPC");
            gRPC->port = grpcData.at("listenPort").as_integer();
            gRPC->listenHost = toml::find<string>(grpcData, "listenHost");

            auto decodeData = fileRoot.at("decode");
            decode->imageMaxHeight = static_cast<int>(decodeData.at("imageMaxHeight").as_integer());
            decode->imageMaxWidth = static_cast<int>(decodeData.at("imageMaxWidth").as_integer());
            decode->execPath = toml::find<string>(decodeData, "execPath");

            auto aiData = fileRoot.at("ai");
            ai->gpuCount = static_cast<int>(aiData.at("gpuCount").as_integer());
            ai->detectionAlgorithmStart = static_cast<int>(aiData.at("detectionAlgorithmStart").as_integer());
            ai->detectionAlgorithmEnd = static_cast<int>(aiData.at("detectionAlgorithmEnd").as_integer());

            // printf("000000000000\n");
            // auto algoConfigData = fileRoot.at("algoConfig");
            // printf("1111111111111\n");
            // algoConfig->logDir = toml::find<string>(algoConfigData, "logDir");
            // printf("222222222222\n");
            // algoConfig->configWorkflow = toml::find<string>(algoConfigData, "configWorkflow");
            // printf("333333333333333\n");
            // algoConfig->modelZoo = toml::find<string>(algoConfigData, "modelZoo");
            // printf("444444444444\n");
            // 检查是否存在 algoConfig 部分
            if (fileRoot.contains("algoConfig"))
            {
                auto algoConfigData = fileRoot.at("algoConfig");

                // 检查并读取 logDir
                if (algoConfigData.contains("logDir")) {
                    auto logDirValue = algoConfigData.at("logDir");
                    if (logDirValue.is_string()) {
                        string logDirStr = logDirValue.as_string();
                        if (logDirStr.empty()) {
                            cout << "logDir 配置值为空字符串，使用默认值" << endl;
                            algoConfig->logDir = "";  // 设置为空字符串
                        } else {
                            algoConfig->logDir = logDirStr;
                        }
                    } else {
                        cout << "logDir 配置值类型错误，应该是字符串" << endl;
                        algoConfig->logDir = "";  // 设置为空字符串
                    }
                } else {
                    cout << "algoConfig 中缺少 logDir 配置，使用默认值" << endl;
                    algoConfig->logDir = "";  // 设置为空字符串
                }

                // 检查并读取 configWorkflow
                if (algoConfigData.contains("configWorkflow")) {
                    auto configWorkflowValue = algoConfigData.at("configWorkflow");
                    if (configWorkflowValue.is_string()) {
                        string configWorkflowStr = configWorkflowValue.as_string();
                        if (configWorkflowStr.empty()) {
                            cout << "configWorkflow 配置值为空字符串，使用默认值" << endl;
                            algoConfig->configWorkflow = "";  // 设置为空字符串
                        } else {
                            algoConfig->configWorkflow = configWorkflowStr;
                        }
                    } else {
                        cout << "configWorkflow 配置值类型错误，应该是字符串" << endl;
                        algoConfig->configWorkflow = "";  // 设置为空字符串
                    }
                } else {
                    cout << "algoConfig 中缺少 configWorkflow 配置，使用默认值" << endl;
                    algoConfig->configWorkflow = "";  // 设置为空字符串
                }

                // 检查并读取 modelZoo
                if (algoConfigData.contains("modelZoo")) {
                    auto modelZooValue = algoConfigData.at("modelZoo");
                    if (modelZooValue.is_string()) {
                        string modelZooStr = modelZooValue.as_string();
                        if (modelZooStr.empty()) {
                            cout << "modelZoo 配置值为空字符串，使用默认值" << endl;
                            algoConfig->modelZoo = "";  // 设置为空字符串
                        } else {
                            algoConfig->modelZoo = modelZooStr;
                        }
                    } else {
                        cout << "modelZoo 配置值类型错误，应该是字符串" << endl;
                        algoConfig->modelZoo = "";  // 设置为空字符串
                    }
                } else {
                    cout << "algoConfig 中缺少 modelZoo 配置，使用默认值" << endl;
                    algoConfig->modelZoo = "";  // 设置为空字符串
                }
            }
            else
            {
                cout << "配置中缺少 algoConfig 部分" << endl;
            }
            cout << "配置读取完成" << endl;
        }
        catch (exception& err)
        {
            stringstream errorMsg;
            errorMsg << "读取配置遇到错误: " << err.what();
            cout << errorMsg.str() << endl;
            return false;
        }
        return true;
    }
} // conf
