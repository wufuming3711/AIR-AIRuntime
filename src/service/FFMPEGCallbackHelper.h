//
// Created by Oboi on 2024/11/28.
//

#ifndef ANALYSIS_SERVICE_FFMPEGCALLBACKHELPER_H
#define ANALYSIS_SERVICE_FFMPEGCALLBACKHELPER_H

#include <opencv2/opencv.hpp>

#include "../tools/TrackableStatus.h"
#include "../model/task_exchange.grpc.pb.h"
#include "StreamHandle.h"

namespace grpcImpl
{
    using namespace tools;
    using namespace pb;
    using namespace grpc;

    class StreamHandle;

    class FFMPEGCallbackHelper
    {
    public:
        bool keyFrameOnly;
        uint32_t callbackInterval;
        uint64_t lastCallbackTime;
        const VideoCommonArgs& videoCommonArgs;
        uint64_t firstRealWorldTimestamp;
        uint64_t firstFrameTimestamp;
        TrackableStatus* pStatus;
        StreamHandle* streamHandle;
        const google::protobuf::RepeatedField<int>& algorithms;

        FFMPEGCallbackHelper(
            TrackableStatus* pStatus,
            const pb::VideoCommonArgs& videoCommonArgs,
            const google::protobuf::RepeatedField<int>& algorithms);

        void setHandle(StreamHandle* streamHandle);

        ~FFMPEGCallbackHelper();
    };
} // gRPC

#endif //ANALYSIS_SERVICE_FFMPEGCALLBACKHELPER_H
