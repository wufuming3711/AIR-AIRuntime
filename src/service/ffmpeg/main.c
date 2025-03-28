//
// Created by Oboi on 2024/11/25.
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include "ffmpeg_decode.h"

void printUsage(const char* programName)
{
    fprintf(
        stderr,
        "用法: %s --tag <标签值> --fileName <文件名> --imageMaxHeight <最大高度> --imageMaxWidth <最大宽度> --frameStep <帧步长> --skipStep <跳过步长>\n",
        programName);
    fprintf(stderr, "  --tag             无符号整数，表示标签\n");
    fprintf(stderr, "  --fileName        文件名（字符串）\n");
    fprintf(stderr, "  --imageMaxHeight  最大图片高度（整数）\n");
    fprintf(stderr, "  --imageMaxWidth   最大图片宽度（整数）\n");
    fprintf(stderr, "  --frameStep       帧步长（无符号整数）\n");
    fprintf(stderr, "  --skipStep        跳过步长（无符号整数）\n");
}

int main(int argc, char* argv[])
{
    unsigned int tag = 0;
    const char* fileName = NULL;
    int imageMaxHeight = 0;
    int imageMaxWidth = 0;
    unsigned int frameStep = 0;
    unsigned int skipStep = 0;
    int paramDidSetCount = 0;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--tag") == 0 && i + 1 < argc)
        {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno != 0 || *end != '\0' || value < 0 || value > UINT_MAX)
            {
                fprintf(stderr, "Invalid value for --tag: %s\n", argv[i]);
                return 1;
            }
            tag = (unsigned int)value;
            paramDidSetCount++;
        }
        else if (strcmp(argv[i], "--fileName") == 0 && i + 1 < argc)
        {
            fileName = argv[++i];
            paramDidSetCount++;
        }
        else if (strcmp(argv[i], "--imageMaxHeight") == 0 && i + 1 < argc)
        {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno != 0 || *end != '\0' || value < INT_MIN || value > INT_MAX)
            {
                fprintf(stderr, "Invalid value for --imageMaxHeight: %s\n", argv[i]);
                return 1;
            }
            imageMaxHeight = (int)value;
            paramDidSetCount++;
        }
        else if (strcmp(argv[i], "--imageMaxWidth") == 0 && i + 1 < argc)
        {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno != 0 || *end != '\0' || value < INT_MIN || value > INT_MAX)
            {
                fprintf(stderr, "Invalid value for --imageMaxWidth: %s\n", argv[i]);
                return 1;
            }
            imageMaxWidth = (int)value;
            paramDidSetCount++;
        }
        else if (strcmp(argv[i], "--frameStep") == 0 && i + 1 < argc)
        {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno != 0 || *end != '\0' || value < 0 || value > UINT_MAX)
            {
                fprintf(stderr, "Invalid value for --frameStep: %s\n", argv[i]);
                return 1;
            }
            frameStep = (unsigned int)value;
            paramDidSetCount++;
        }
        else if (strcmp(argv[i], "--skipStep") == 0 && i + 1 < argc)
        {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno != 0 || *end != '\0' || value < 0 || value > UINT_MAX)
            {
                fprintf(stderr, "Invalid value for --skipStep: %s\n", argv[i]);
                return 1;
            }
            skipStep = (unsigned int)value;
            paramDidSetCount++;
        }
        else
        {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
        }
    }
    if (paramDidSetCount != 6)
    {
        printUsage(argv[0]);
        exit(-1);
    }
    ContextPrintLibrariesCodecsAndMutexes();
    void* ctx = ContextOpenWithSoftwareDecode(fileName, imageMaxHeight * imageMaxWidth * 3, tag);
    if (ctx == NULL)
    {
        sleep(1);
        exit(-1);
    }
    AVMainLoop(ctx, imageMaxHeight, imageMaxWidth, frameStep, skipStep);
    sleep(1);
}
