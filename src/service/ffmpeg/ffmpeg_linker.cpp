//
// Created by Oboi on 2024/11/14.
//

#include "ffmpeg_linker.h"
#include "../ai/Manager.h"
#include "../StreamHandle.h"

using namespace grpcImpl;
using namespace ai;

void onNewRGBData(void* userData, unsigned char* rgbData,
                  int imgHeight, int imgWidth,
                  unsigned long long timestamp)
{
    auto* model = static_cast<grpcImpl::FFMPEGCallbackHelper*>(userData);
    if (model->firstRealWorldTimestamp == 0)
    {
        model->firstRealWorldTimestamp = chrono::duration_cast<chrono::milliseconds>
            (chrono::system_clock::now().time_since_epoch())
            .count();
        model->firstFrameTimestamp = timestamp;
    }
    auto reply = make_shared<OnAIResultGotReply>();
    reply->set_timestamp(model->firstRealWorldTimestamp + timestamp - model->firstFrameTimestamp);
    cv::Mat img(imgHeight, imgWidth, CV_8UC3, rgbData);
    bool suc = false;
    for (auto algorithm : model->algorithms)
    {
        OnAIResultGotReply::ResultWrapper resultWrapper;
        if (!Manager::getInstance().useAlgorithm(static_cast<DetectionAlgorithm>(algorithm), img, resultWrapper))
        {
            continue;
        }
        if (resultWrapper.shouldupdate())
        {
            model->lastCallbackTime = 0;
        }
        if (resultWrapper.rs().empty() && !resultWrapper.shouldupdate())
        {
            continue;
        }
        reply->mutable_result()->Add()->CopyFrom(resultWrapper);
        if (!suc)
        {
            suc = true;
        }
    }
    if (!suc)
    {
        return;
    }
    vector<uchar> encodedImgBuffer;
    if (!model->keyFrameOnly)
    {
        if (cv::imencode(".jpg", img, encodedImgBuffer))
        {
            string jpegData(encodedImgBuffer.begin(), encodedImgBuffer.end());
            reply->set_imagedata(jpegData);
            reply->set_fmt(IMAGE_FORMAT_JPEG);
            model->streamHandle->pushOnAIResultGotReply(reply);
        }
        return;
    }
    auto now = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    if ((now - model->lastCallbackTime) > model->callbackInterval)
    {
        if (cv::imencode(".jpg", img, encodedImgBuffer))
        {
            string jpegData(encodedImgBuffer.begin(), encodedImgBuffer.end());
            reply->set_imagedata(jpegData);
            reply->set_fmt(IMAGE_FORMAT_JPEG);
            model->streamHandle->pushOnAIResultGotReply(reply);
            model->lastCallbackTime = now;
        }
    }
}

static string trim(const string& str)
{
    size_t first = str.find_first_not_of(" \n\r\t\f\v");
    if (first == string::npos)
        return "";
    size_t last = str.find_last_not_of(" \n\r\t\f\v");
    return str.substr(first, (last - first + 1));
}

void onLogLine(void* userData, const char* logLine)
{
    auto* model = static_cast<grpcImpl::FFMPEGCallbackHelper*>(userData);
    model->pStatus->append(trim(logLine));
}
