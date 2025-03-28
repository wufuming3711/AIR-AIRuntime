//
// Created by Oboi on 2024/11/5.
//

#ifndef ANALYSIS_SERVICE_SERVICE_H
#define ANALYSIS_SERVICE_SERVICE_H

#include <grpcpp/grpcpp.h>
#include "../model/task_exchange.grpc.pb.h"

namespace grpcImpl
{
    using namespace grpc;
    using namespace pb;

    class Service final : public TaskExchange::CallbackService
    {
    public:
        explicit Service(const string& addr);

        void start();

        ServerWriteReactor<OnAIResultGotReply>*
        RequestForStream(CallbackServerContext* context, const StreamTask* task) override;

        ServerWriteReactor<OnAIResultGotReply>*
        RequestForFile(CallbackServerContext* context, const FileTask* task) override;

        ServerUnaryReactor* RequestForImage(CallbackServerContext* context, const ImageTask* task,
                                            OnAIResultGotReply* reply) override;

    private:
        ServerBuilder builder;
    };
} // gRPC

#endif //ANALYSIS_SERVICE_SERVICE_H
