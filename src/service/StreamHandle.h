//
// Created by Oboi on 2024/11/7.
//

#ifndef ANALYSIS_SERVICE_STREAMHANDLE_H
#define ANALYSIS_SERVICE_STREAMHANDLE_H

#include "../conf/Config.h"
#include "../tools/File.h"
#include "../tools/TrackableStatus.h"
#include "../model/task_exchange.grpc.pb.h"
#include "ffmpeg/decoder_wrapper.h"
#include "FFMPEGCallbackHelper.h"

#include <thread>

class FFMPEGCallbackHelper;

namespace grpcImpl
{
    using namespace tools;
    using namespace pb;
    using namespace grpc;

    class StreamHandle : public ServerWriteReactor<OnAIResultGotReply>
    {
    public:
        StreamHandle(CallbackServerContext* context,
                     const StreamTask* streamTask, const FileTask* fileTask);

        ~StreamHandle() override;

        void pushOnAIResultGotReply(const shared_ptr<OnAIResultGotReply>& reply);

        void startFFMPEG();

        void OnWriteDone(bool b) override;

        void OnDone() override;

        void OnCancel() override;

    private:
        void ffmpegMainloop();

        void cleanup(const char* reason, bool fromFFMPEGThread);

        void onceFinish(const char* reason, bool fromFFMPEGThread);

        atomic<int> onceFinishTag;
        bool writing;
        char* inputName;
        void* saidaStreamContext;
        FFMPEGCallbackHelper* ffmpegCallbackHelper;
        TrackableStatus* pStatus;
        mutex saidaStreamContextLock;
        mutex queueLock;
        thread ffmpegThread;
        once_flag doneOnce;
        deque<shared_ptr<OnAIResultGotReply>> replyQueue;
    };
} // service

#endif //ANALYSIS_SERVICE_STREAMHANDLE_H
