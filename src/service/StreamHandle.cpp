//
// Created by Oboi on 2024/11/7.
//

#include "StreamHandle.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include "ffmpeg/ffmpeg_linker.h"
#include "ffmpeg/decoder_wrapper.h"

namespace grpcImpl
{
    using namespace grpc;
    static atomic<int> count = 0;
    static atomic<unsigned int> gPRCTaskID = 0;

    StreamHandle::StreamHandle(CallbackServerContext* context, const StreamTask* streamTask, const FileTask* fileTask)
    {
        pStatus = new TrackableStatus("gRPC服务", "由于收到了请求,准备开始创建");
        google::protobuf::RepeatedField<int> algorithms;
        if (streamTask != nullptr)
        {
            ffmpegCallbackHelper = new FFMPEGCallbackHelper(
                pStatus,
                streamTask->args(),
                streamTask->algorithms());
            inputName = strdup(streamTask->stream_url().c_str());
        }
        else
        {
            ffmpegCallbackHelper = new FFMPEGCallbackHelper(
                pStatus,
                fileTask->args(),
                fileTask->algorithms());
            inputName = strdup(fileTask->file_path().c_str());
        }
        saidaStreamContext = nullptr;
        writing = false;
        ffmpegCallbackHelper->setHandle(this);
    }

    StreamHandle::~StreamHandle()
    {
        saidaContextFree(saidaStreamContext);
        delete ffmpegCallbackHelper;
        delete pStatus;
        free(inputName);
    }

    void StreamHandle::OnWriteDone(bool b)
    {
        if (!b)
        {
            return;
        }
        queueLock.lock();
        if (replyQueue.empty())
        {
            writing = false;
            queueLock.unlock();
            return;
        }
        auto reply = replyQueue.front();
        replyQueue.pop_front();
        writing = true;
        queueLock.unlock();
        StartWrite(reply.get());
    }

    void StreamHandle::OnDone()
    {
        onceFinish(strdup("最后的清理"), false);
    }

    void StreamHandle::OnCancel()
    {
        pStatus->append("任务取消");
        onceFinish(strdup("对端断开了连接"), false);
    }

    void StreamHandle::pushOnAIResultGotReply(const shared_ptr<OnAIResultGotReply>& reply)
    {
        queueLock.lock();
        if (!writing)
        {
            writing = true;
            StartWrite(reply.get());
            queueLock.unlock();
            return;
        }
        if (replyQueue.size() > 128)
        {
            queueLock.unlock();
            onceFinish(strdup("累积了过多数据"), false);
            return;
        }
        replyQueue.push_back(reply);
        queueLock.unlock();
    }

    void StreamHandle::startFFMPEG()
    {
        saidaStreamContextLock.lock();
        if (saidaStreamContext == nullptr)
        {
            ffmpegThread = thread(&StreamHandle::ffmpegMainloop, this);
        }
    }

    void StreamHandle::ffmpegMainloop()
    {
        unsigned int fGPRCTaskID = gPRCTaskID++;
        stringstream sss;
        sss << "新的gRPC任务:" << fGPRCTaskID << ", 当前FFMPEG视频处理数量" << ++count << endl;
        pStatus->append(sss.str());
        auto execParams = DecoderArgs{
            ffmpegCallbackHelper,
            ffmpegCallbackHelper->videoCommonArgs.key_frame_only(),
            ffmpegCallbackHelper->videoCommonArgs.callback_interval_in_mill_seconds(),
            conf::Config::getInstance().decode->execPath.c_str(),
            inputName,
            ffmpegCallbackHelper->videoCommonArgs.frame_step(),
            ffmpegCallbackHelper->videoCommonArgs.skip_step(),
            conf::Config::getInstance().decode->imageMaxHeight,
            conf::Config::getInstance().decode->imageMaxWidth,
            &onLogLine,
        };
        saidaStreamContext = saidaContextOpenWithSoftwareDecode(&execParams);
        if (saidaStreamContext == nullptr)
        {
            saidaStreamContextLock.unlock();
            cout << "gRPC任务" << fGPRCTaskID << "结束, 当前FFMPEG视频处理数量" << --count << endl;
            onceFinish(strdup("FFMPEG读取源失败"), true);
            return;
        }
        saidaStreamContextLock.unlock();
        saidaAVMainLoop(saidaStreamContext, &onNewRGBData);
        queueLock.lock();
        replyQueue.clear();
        queueLock.unlock();
        stringstream ss;
        ss << "gRPC任务结束:" << fGPRCTaskID << ", 当前FFMPEG视频处理数量" << --count << endl;
        cout << ss.str() << endl;
        const char* reason = strdup(ss.str().c_str());
        onceFinish(reason, true);
    }

    void StreamHandle::cleanup(const char* reason, bool fromFFMPEGThread)
    {
        pStatus->append("任务结束");
        saidaStreamContextLock.lock();
        if (saidaStreamContext != nullptr)
        {
            saidaContextClose(saidaStreamContext);
        }
        saidaStreamContextLock.unlock();
        if (!fromFFMPEGThread)
        {
            ffmpegThread.join();
        }
        pStatus->append(reason);
        Finish(Status(ABORTED, pStatus->getString()));
        free((void*)reason);
        cout << pStatus->getString() << endl;
    }

    void StreamHandle::onceFinish(const char* reason, bool fromFFMPEGThread)
    {
        if (onceFinishTag++ == 0)
        {
            cleanup(reason, fromFFMPEGThread);
        }
        else
        {
            free((void*)reason);
        }
    }
} // service
