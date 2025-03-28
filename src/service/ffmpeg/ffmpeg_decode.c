//
// Created by Oboi on 2024/11/4.
//
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include "ffmpeg_decode.h"
#include "ipc.h"
#include <pthread.h>
#include <libavutil/avutil.h>
#include <mqueue.h>

typedef struct SaidaAVContextModel
{
    AVFormatContext* inputContext;
    const AVCodec* videoCodec;
    AVCodecContext* videoCodecContext;
    AVStream* videoFrameStream;
    struct SwsContext* swsContext;
    int streamIndex;
    char* logBuffer;
    unsigned int tag;
    mqd_t finUnique;
    mqd_t msgUnique;
    void* sharedMem;
} SAIDA_AV_CONTEXT_MODEL;

void softwareDecodeLoop(SAIDA_AV_CONTEXT_MODEL* model,
                        unsigned int frameStep, unsigned int skipStep)
{
    fprintf(stderr, "开始解码循环\n");
    fflush(stderr);
    int rs = 0;
    char errBuffer[768];
    int videoFrameIndex = model->streamIndex;
    AVFormatContext* inputContext = model->inputContext;
    AVCodecContext* codecContext = model->videoCodecContext;
    struct SwsContext* swsContext = model->swsContext;
    unsigned long long decodedFrame = 0;
    unsigned long long step = frameStep + skipStep;
    bool keyFrameOnly = step == 0;
    AVPacket* avPacket = av_packet_alloc();
    AVFrame* avFrame = av_frame_alloc();
    AVFrame* rgbFrame = av_frame_alloc();
    unsigned long long handledFrame = 0;
    unsigned long long skippedFrame = 0;
    char buffer[4];
    size_t bufferLen = sizeof(buffer);
    unsigned int msgPriority = 0;
    struct timespec ts;
    ssize_t msgSize;
    while (true)
    {
        rs = av_read_frame(inputContext, avPacket);
        if (rs < 0)
        {
            av_strerror(rs, errBuffer, 1024);
            fprintf(stderr, "av_read_frame 失败, 错误: %s\n", errBuffer);
            fflush(stderr);
            break;
        }
        if (videoFrameIndex != avPacket->stream_index)
        {
            continue;
        }
        if (keyFrameOnly && (avPacket->flags & AV_PKT_FLAG_KEY) == 0)
        {
            skippedFrame++;
            continue;
        }
        rs = avcodec_send_packet(codecContext, avPacket);
        av_packet_unref(avPacket);
        if (rs < 0)
        {
            av_strerror(rs, errBuffer, 1024);
            fprintf(stderr, "avcodec_send_packet 失败, 错误: %s\n", errBuffer);
            fflush(stderr);
            break;
        }
        while (true)
        {
            rs = avcodec_receive_frame(codecContext, avFrame);
            if (rs == AVERROR_EOF || rs == AVERROR(EAGAIN))
            {
                break;
            }
            if (rs < 0)
            {
                av_strerror(rs, errBuffer, 1024);
                fprintf(stderr, "avcodec_receive_frame 失败, 错误: %s\n", errBuffer);
                fflush(stderr);
                goto end;
            }
            if (!keyFrameOnly && decodedFrame % step >= frameStep)
            {
                skippedFrame++;
                decodedFrame++;
                av_frame_unref(avFrame);
                continue;
            }
            handledFrame++;
            decodedFrame++;
            if (avFrame->pts == AV_NOPTS_VALUE)
            {
                continue;
            }
            unsigned long long realWorldTimestamp = (unsigned long long)((double)avFrame->pts *
                av_q2d(model->videoFrameStream->time_base) *
                1000.0);
            rs = sws_scale_frame(swsContext, rgbFrame, avFrame);
            av_frame_unref(avFrame);
            if (rs < 0)
            {
                av_strerror(rs, errBuffer, 1024);
                fprintf(stderr, "sws_scale_frame 失败, 错误: %s\n", errBuffer);
                fflush(stderr);
                goto end;
            }
            memcpy(model->sharedMem, rgbFrame->data[0], rgbFrame->height * rgbFrame->width * 3);
            SAIDA_IPC_MESSAGE msg = {rgbFrame->height, rgbFrame->width, realWorldTimestamp};
            av_frame_unref(rgbFrame);
            if (!saidaPostMessageToUnique(&msg, model->msgUnique))
            {
                fprintf(stderr, "无法通知主进程开始处理\n");
                fflush(stderr);
                goto end;
            }
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 1;
            msgSize = mq_timedreceive(model->finUnique, buffer, bufferLen, &msgPriority, &ts);
            if (msgSize != bufferLen)
            {
                const int err = errno;
                switch (err)
                {
                case 110:
                    fprintf(stderr, "无法及时收到主进程处理完成的信号%d___%zu\n", rs, bufferLen);
                    fflush(stderr);
                    break;
                case 9:
                    break;
                default:
                    fprintf(stderr, "无法收到主进程处理完成的信号%d___%zu\n", rs, bufferLen);
                    fflush(stderr);
                    break;
                }
                goto end;
            }
            if (buffer[0] != 0)
            {
                fprintf(stderr, "主进程要求退出\n");
                fflush(stderr);
                goto end;
            }
        }
    }
end:
    av_packet_free(&avPacket);
    av_frame_free(&avFrame);
    av_frame_free(&rgbFrame);
    fprintf(stderr, "ffmpeg进程已退出\n");
    fflush(stderr);
    fprintf(stderr, "解码帧数: %llu\n", decodedFrame);
    fflush(stderr);
    fprintf(stderr, "跳过帧数: %llu\n", skippedFrame);
    fflush(stderr);
    fprintf(stderr, "处理帧数: %llu\n", handledFrame);
    fflush(stderr);
    saidaCloseFinishedSignalUnique(model->tag, model->finUnique);
    saidaCloseMessageUnique(model->tag, model->msgUnique);
}

int AVMainLoop(void* context,
               int imageMaxHeight, int imageMaxWidth,
               unsigned int frameStep, unsigned int skipStep)
{
    SAIDA_AV_CONTEXT_MODEL* model = context;
    fprintf(stderr, "尝试打开解码器: %s\n", model->videoCodec->name);
    fflush(stderr);
    int rs = avcodec_open2(model->videoCodecContext, model->videoCodec, NULL);
    if (rs != 0)
    {
        char errBuffer[768];
        av_strerror(rs, errBuffer, 1024);
        fprintf(stderr, "解码器打开失败, 错误: %s\n", errBuffer);
        fflush(stderr);
        return -1;
    }
    fprintf(stderr, "解码器打开成功: %s,尝试读取分辨率\n", model->videoCodec->name);
    fflush(stderr);
    int origWidth = model->videoFrameStream->codecpar->width;
    int origHeight = model->videoFrameStream->codecpar->height;
    enum AVPixelFormat origFmt = model->videoFrameStream->codecpar->format;
    if (origWidth <= 0 ||
        origHeight <= 0 ||
        origWidth >= 8192 ||
        origHeight >= 8192 ||
        origFmt == AV_PIX_FMT_NONE)
    {
        fprintf(stderr, "读取分辨率等信息失败,流质量不理想\n");
        fflush(stderr);
        return -1;
    }
    int dstWidth;
    int dstHeight;
    if ((imageMaxWidth < 0 && imageMaxHeight < 0) ||
        // 不能超分辨率
        (imageMaxWidth >= origWidth || imageMaxHeight >= origHeight))
    {
        dstWidth = origWidth;
        dstHeight = origHeight;
    }
    else
    {
        dstWidth = imageMaxWidth;
        dstHeight = (int)((double)dstWidth * (double)origHeight / (double)origWidth);
        if (dstHeight > imageMaxHeight)
        {
            dstHeight = imageMaxHeight;
            dstWidth = (int)((double)dstHeight * (double)origWidth / (double)origHeight);
        }
    }
    dstWidth = FFALIGN(dstWidth, 16);
    dstHeight = FFALIGN(dstHeight, 16);
    model->swsContext = sws_getContext(origWidth, origHeight, origFmt,
                                       dstWidth, dstHeight, AV_PIX_FMT_BGR24,
                                       SWS_FAST_BILINEAR, NULL, NULL, NULL);
    if (model->swsContext == NULL)
    {
        fprintf(stderr, "sws_getContext 失败\n");
        fflush(stderr);
        return -1;
    }
    fprintf(stderr, "分辨率已经读取到:%p\n"
            "model->swsContext origWidth:%d\n"
            "model->swsContext origHeight:%d\n"
            "model->swsContext origFmt:%d\n"
            "model->swsContext dstWidth:%d\n"
            "model->swsContext dstHeight:%d\n",
            model->swsContext, origWidth, origHeight, origFmt, dstWidth, dstHeight);
    fflush(stderr);
    softwareDecodeLoop(model, frameStep, skipStep);
    return 0;
}

SAIDA_AV_CONTEXT_MODEL* ContextInit(const char* inputFileName,
                                    unsigned int memSize,
                                    unsigned int tag)
{
    char errBuffer[768];
    SAIDA_AV_CONTEXT_MODEL* model = malloc(sizeof(SAIDA_AV_CONTEXT_MODEL));
    memset(model, 0, sizeof(SAIDA_AV_CONTEXT_MODEL));
    model->logBuffer = malloc(1024);
    model->tag = tag;
    AVDictionary* options = NULL;
    if (strncmp(inputFileName, "rtsp://", 7) == 0)
    {
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
        char timeoutStr[32];
        snprintf(timeoutStr, sizeof(timeoutStr), "%d", 1000 * 1000 * 5);
        av_dict_set(&options, "timeout", timeoutStr, 0);
    }
    else if (strncmp(inputFileName, "rtmp://", 7) == 0)
    {
        av_dict_set(&options, "rtmp_connect_timeout", "5", 0); // 设置连接超时为 30 秒
        av_dict_set(&options, "rtmp_read_timeout", "5", 0); // 设置读取超时为 60 秒
    }
    fprintf(stderr, "准备打开输入( %s )\n", inputFileName);
    fflush(stderr);
    int rs = avformat_open_input(&model->inputContext, inputFileName, NULL, &options);
    av_dict_free(&options);
    if (rs != 0)
    {
        av_strerror(rs, errBuffer, 1024);
        fprintf(stderr, "无法打开输入( %s ),错误: %s\n", inputFileName, errBuffer);
        fflush(stderr);
        goto err1;
    }
    fprintf(stderr, "输入成功打开,准备获取流信息\n");
    fflush(stderr);
    rs = avformat_find_stream_info(model->inputContext, NULL);
    if (rs < 0)
    {
        av_strerror(rs, errBuffer, 1024);
        fprintf(stderr, "avformat_find_stream_info 失败, 错误: %s\n", errBuffer);
        fflush(stderr);
        goto err2;
    }
    fprintf(stderr, "流信息已取得,准备开始处理\n");
    fflush(stderr);
    model->streamIndex = av_find_best_stream(model->inputContext,
                                             AVMEDIA_TYPE_VIDEO,
                                             -1, -1,
                                             &model->videoCodec, 0);
    if (model->streamIndex < 0)
    {
        av_strerror(model->streamIndex, errBuffer, 1024);
        fprintf(stderr, "av_find_best_stream 失败, 错误: %s\n", errBuffer);
        fflush(stderr);
        goto err2;
    }
    fprintf(stderr, "准备使用解码器( %s )开始解码\n", model->videoCodec->name);
    fflush(stderr);
    model->videoCodecContext = avcodec_alloc_context3(model->videoCodec);
    if (model->videoCodecContext == NULL)
    {
        fprintf(stderr, "avcodec_alloc_context3 失败\n");
        fflush(stderr);
        goto err2;
    }
    model->videoFrameStream = model->inputContext->streams[model->streamIndex];
    rs = avcodec_parameters_to_context(model->videoCodecContext, model->videoFrameStream->codecpar);
    if (rs < 0)
    {
        av_strerror(rs, errBuffer, 1024);
        fprintf(stderr, "avcodec_parameters_to_context 失败, 错误: %s\n", errBuffer);
        fflush(stderr);
        goto err3;
    }
    model->sharedMem = saidaGetSharedMemory(tag, memSize);
    if (model->sharedMem == NULL)
    {
        fprintf(stderr, "无法取得共享内存\n");
        fflush(stderr);
        goto err3;
    }
    model->finUnique = saidaGetFinishedSignalUnique(tag);
    if (model->finUnique == -1)
    {
        fprintf(stderr, "无法取得完成通道\n");
        fflush(stderr);
        goto err3;
    }
    model->msgUnique = saidaGetMessageUnique(tag);
    if (model->msgUnique == -1)
    {
        fprintf(stderr, "无法取得消息通道\n");
        fflush(stderr);
        goto err3;
    }
    return model;
err3:
    avcodec_free_context(&model->videoCodecContext);
err2:
    avformat_close_input(&model->inputContext);
err1:
    free(model->logBuffer);
    free(model);
    return NULL;
}

SAIDA_C_EXT void ContextPrintLibrariesCodecsAndMutexes()
{
    void* opaque = NULL;
    const AVCodec* codec = NULL;
    for (codec = av_codec_iterate(&opaque); codec != NULL; codec = av_codec_iterate(&opaque))
    {
        fprintf(stderr, "Codec: %s\n", codec->name);
        fflush(stderr);
    }

    opaque = NULL;
    const AVOutputFormat* oFormat = NULL;
    for (oFormat = av_muxer_iterate(&opaque); oFormat != NULL; oFormat = av_muxer_iterate(&opaque))
    {
        fprintf(stderr, "Muxer: %s\n", oFormat->name);
        fflush(stderr);
    }

    opaque = NULL;
    const AVInputFormat* iFormat = NULL;
    for (iFormat = av_demuxer_iterate(&opaque); iFormat != NULL; iFormat = av_demuxer_iterate(&opaque))
    {
        fprintf(stderr, "DeMuxer: %s\n", iFormat->name);
        fflush(stderr);
    }

    opaque = NULL;
    const AVCodecParser* parser = NULL;
    for (parser = av_parser_iterate(&opaque); parser != NULL; parser = av_parser_iterate(&opaque))
    {
        for (int i = 0; i < 7; ++i)
        {
            int id = parser->codec_ids[i];
            if (id != AV_CODEC_ID_NONE)
            {
                fprintf(stderr, "Parser: %s\n", avcodec_get_name(id));
                fflush(stderr);
            }
        }
    }
}

SAIDA_C_EXT void* ContextOpenWithSoftwareDecode(const char* inputFileName,
                                                unsigned int memSize,
                                                unsigned int tag)
{
    return ContextInit(inputFileName, memSize, tag);
}
