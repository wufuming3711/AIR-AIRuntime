//
// Created by Oboi on 2024/11/25.
//

#ifndef ANALYSIS_SERVICE_DECODER_WRAPPER_H
#define ANALYSIS_SERVICE_DECODER_WRAPPER_H
#ifdef __cplusplus
#define SAIDA_C_EXT extern "C"
#else
#define SAIDA_C_EXT

#include <stdbool.h>

#endif
#include <stdint.h>

typedef void FOnNewRGBData(void* userData, unsigned char* rgbData,
                           int imgHeight, int imgWidth,
                           unsigned long long timestamp);

typedef void FOnLogLine(void* userData, const char* logLine);

typedef struct DecoderArgs
{
    void* userData;
    bool keyFrameOnly;
    uint32_t callbackInterval;
    const char* execPath;
    const char* inputFileName;
    unsigned int frameStep;
    unsigned int skipStep;
    int imageMaxHeight;
    int imageMaxWidth;
    FOnLogLine* onLogLine;
} DECODER_ARGS;

SAIDA_C_EXT void* saidaContextOpenWithSoftwareDecode(DECODER_ARGS* args);

SAIDA_C_EXT int saidaAVMainLoop(void* context, FOnNewRGBData onDecoded);

SAIDA_C_EXT void saidaContextClose(void* context);

SAIDA_C_EXT void saidaContextFree(void* context);

#endif //ANALYSIS_SERVICE_DECODER_WRAPPER_H
