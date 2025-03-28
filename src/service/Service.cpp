//
// Created by Oboi on 2024/11/5.
//

#include <fstream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include "Service.h"
#include "StreamHandle.h"
#include "ai/Manager.h"

namespace grpcImpl
{
    using namespace std;
    using namespace pb;
    using namespace grpc;
    using namespace conf;
    using namespace tools;

    Service::Service(const string& addr)
    {
        SslServerCredentialsOptions::PemKeyCertPair key_cert_pair = {
            loadStringFromFile(Config::getInstance().tls->keyPath),
            loadStringFromFile(Config::getInstance().tls->certPath)
        };
        SslServerCredentialsOptions ssl_options;
        ssl_options.pem_key_cert_pairs.emplace_back(key_cert_pair);
        builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
        builder.AddListeningPort(addr,
                                 SslServerCredentials(ssl_options));
    }

    // void Service::start()
    // {
    //     builder.RegisterService(this);
    //     builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS,
    //                                1000 * 60 * 2);
    //     builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
    //                                1000 * 1);
    //     builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    //     builder.AddChannelArgument(
    //         GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
    //         1000 * 30);
    //     const std::unique_ptr server(builder.BuildAndStart());
    //     server->Wait();
    // }

    void Service::start()
    {
        builder.RegisterService(this);
        ResourceQuota quota;
        quota.Resize(1024 * 1024 * 1024);
        builder.SetResourceQuota(quota);
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS,
                                   1000 * 60 * 2);
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
                                   1000 * 1);
        builder.AddChannelArgument(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, 64 * 1024 * 1024); // 64MB
        builder.AddChannelArgument(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 64 * 1024 * 1024);
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
        builder.AddChannelArgument(
            GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
            1000 * 30);
        const std::unique_ptr server(builder.BuildAndStart());
        server->Wait();
    }

    ServerWriteReactor<OnAIResultGotReply>*
    Service::RequestForStream(CallbackServerContext* context, const StreamTask* task)
    {
        auto rs = new StreamHandle(context, task, nullptr);
        rs->startFFMPEG();
        return rs;
    }

    ServerWriteReactor<OnAIResultGotReply>*
    Service::RequestForFile(CallbackServerContext* context, const FileTask* task)
    {
        auto rs = new StreamHandle(context, nullptr, task);
        rs->startFFMPEG();
        return rs;
    }

    ServerUnaryReactor* Service::RequestForImage(CallbackServerContext* context, const ImageTask* task,
                                                 OnAIResultGotReply* reply)
    {
        OnAIResultGotReply::Result result;
        auto* reactor = context->DefaultReactor();
        const std::vector<uchar> data(task->img().begin(), task->img().end());
        auto img = cv::imdecode(data, cv::IMREAD_COLOR);
        if (img.empty())
        {
            reactor->Finish(Status(StatusCode::ABORTED, "错误的图片,OPENCV无法解析"));
        }
        bool suc = false;
        for (auto algorithm : task->algorithms())
        {
            OnAIResultGotReply::ResultWrapper resultWrapper;
            if (!ai::Manager::getInstance().useAlgorithm(static_cast<DetectionAlgorithm>(algorithm), img, resultWrapper))
            {
                continue;
            }
            if (resultWrapper.rs().empty())
            {
                continue;
            }
            reply->mutable_result()->Add()->CopyFrom(resultWrapper);
            if (!suc)
            {
                suc = true;
            }
        }
        if (suc)
        {
            reactor->Finish(Status::OK);
        }
        else
        {
            reactor->Finish(Status(StatusCode::ABORTED, "没有检测到任何结果"));
        }
        return reactor;
    }
} // gRPC
