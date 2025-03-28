//
// Created by Oboi on 2024/11/28.
//

#include "FFMPEGCallbackHelper.h"

namespace grpcImpl
{
    FFMPEGCallbackHelper::~FFMPEGCallbackHelper()
    {
        delete pStatus;
    }


    void FFMPEGCallbackHelper::setHandle(StreamHandle* fStreamHandle)
    {
        streamHandle = fStreamHandle;
    }

    FFMPEGCallbackHelper::FFMPEGCallbackHelper(TrackableStatus* fpStatus,
                                               const VideoCommonArgs& videoCommonArgs,
                                               const google::protobuf::RepeatedField<int>& algorithms)
        : keyFrameOnly(videoCommonArgs.key_frame_only()), callbackInterval(videoCommonArgs.callback_interval_in_mill_seconds()),
          lastCallbackTime(0),
          videoCommonArgs(videoCommonArgs),
          pStatus(fpStatus->fork("ffmpeg回调", "对象已经创建")),
          algorithms(algorithms)
    {
        firstRealWorldTimestamp = 0;
        firstFrameTimestamp = 0;
        streamHandle = nullptr;
    }
}
