//
// Created by Oboi on 2024/11/4.
//

#ifndef DECODER_FFMPEG_DECODE_H
#define DECODER_FFMPEG_DECODE_H
#ifdef __cplusplus
#ifndef SAIDA_C_EXT
#define SAIDA_C_EXT extern "C"
#endif
#else
#ifndef SAIDA_C_EXT
#define SAIDA_C_EXT
#endif

#include <stdbool.h>

#endif

SAIDA_C_EXT void ContextPrintLibrariesCodecsAndMutexes();

SAIDA_C_EXT void* ContextOpenWithSoftwareDecode(const char* inputFileName,
                                                unsigned int memSize,
                                                unsigned int tag);

SAIDA_C_EXT int AVMainLoop(void* context,
                           int imageMaxHeight, int imageMaxWidth,
                           unsigned int frameStep, unsigned int skipStep);

#endif //DECODER_FFMPEG_DECODE_H
