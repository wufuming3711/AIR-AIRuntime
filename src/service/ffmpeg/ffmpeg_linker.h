//
// Created by Oboi on 2024/11/14.
//

#ifndef ANALYSIS_SERVICE_FFMPEG_LINKER_H
#define ANALYSIS_SERVICE_FFMPEG_LINKER_H

void onLogLine(void* userData, const char* logLine);

void onNewRGBData(void* userData, unsigned char* rgbData,
                  int imgHeight, int imgWidth,
                  unsigned long long timestamp);

#endif //ANALYSIS_SERVICE_FFMPEG_LINKER_H
