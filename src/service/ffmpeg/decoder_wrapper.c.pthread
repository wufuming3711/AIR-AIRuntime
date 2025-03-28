//
// Created by Oboi on 2024/11/25.
//

#include <malloc.h>
#include <mqueue.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/wait.h>
#include <pthread.h>  // 替换 <threads.h> 为 <pthread.h>
#include "decoder_wrapper.h"
#include "ipc.h"
#include <errno.h>
#include <stdatomic.h>

typedef struct SaidaIPCWrapper
{
    atomic_int waited;
    void* userData;
    void* context;
    int stdErrFD;
    unsigned int tag;
    mqd_t finUnique;
    mqd_t msgUnique;
    void* sharedMem;
    FOnLogLine* onLogLine;
    pid_t pid;
} SAIDA_IPC_WRAPPER;

int start(const DECODER_ARGS* args, SAIDA_IPC_WRAPPER* wrapper)
{
    int pipeFD[2];
    char errBuffer[1024];
    if (pipe(pipeFD) == -1)
    {
        snprintf(errBuffer, 1024, "无法创建管道\n");
        args->onLogLine(args->userData, errBuffer);
        return -1;
    }
    wrapper->pid = fork();
    if (wrapper->pid == -1)
    {
        snprintf(errBuffer, 1024, "无法fork进程\n");
        args->onLogLine(args->userData, errBuffer);
        close(pipeFD[0]);
        close(pipeFD[1]);
        return -1;
    }
    if (wrapper->pid == 0)
    {
        close(pipeFD[0]);
        if (dup2(pipeFD[1], STDERR_FILENO) == -1)
        {
            snprintf(errBuffer, 1024, "无法创建管道\n");
            args->onLogLine(args->userData, errBuffer);
            return -1;
        }
        char tagStr[10], frameStepStr[10], skipStepStr[10], heightStr[10], widthStr[10];
        snprintf(tagStr, sizeof(tagStr), "%u", wrapper->tag);
        if (args->keyFrameOnly)
        {
            snprintf(frameStepStr, sizeof(frameStepStr), "0");
            snprintf(skipStepStr, sizeof(skipStepStr), "0");
        }
        else
        {
            snprintf(frameStepStr, sizeof(frameStepStr), "%u", args->frameStep);
            snprintf(skipStepStr, sizeof(skipStepStr), "%u", args->skipStep);
        }
        snprintf(heightStr, sizeof(heightStr), "%d", args->imageMaxHeight);
        snprintf(widthStr, sizeof(widthStr), "%d", args->imageMaxWidth);
        char* execPath = strdup(args->execPath);
        char* filePath = strdup(args->inputFileName);
        char* command[] = {
            execPath,
            "--tag", tagStr,
            "--fileName", filePath,
            "--frameStep", frameStepStr,
            "--skipStep", skipStepStr,
            "--imageMaxHeight", heightStr,
            "--imageMaxWidth", widthStr,
            NULL
        };
        execvp(command[0], command);
        free(execPath);
        free(filePath);
    }
    else
    {
        close(pipeFD[1]);
        wrapper->stdErrFD = pipeFD[0];
    }
    return 0;
}

SAIDA_C_EXT void* saidaContextOpenWithSoftwareDecode(DECODER_ARGS* args)
{
    SAIDA_IPC_WRAPPER* rs = malloc(sizeof(SAIDA_IPC_WRAPPER));
    memset(rs, 0, sizeof(SAIDA_IPC_WRAPPER));
    rs->tag = saidaIPCGetTag();
    rs->userData = args->userData;
    rs->onLogLine = args->onLogLine;
    rs->waited = 0;
    rs->msgUnique = saidaCreateMessageUnique(rs->tag);
    if (rs->msgUnique < 0)
    {
        goto err0;
    }
    rs->finUnique = saidaCreateFinishedSignalUnique(rs->tag);
    if (rs->finUnique < 0)
    {
        goto err1;
    }
    rs->sharedMem = saidaMallocSharedMemory(rs->tag, args->imageMaxWidth * args->imageMaxHeight * 3);
    if (rs->sharedMem == NULL)
    {
        goto err2;
    }
    if (start(args, rs) != 0)
    {
        goto err3;
    }
    return rs;
err3:
    saidaFreeSharedMemory(rs->tag);
err2:
    saidaCloseFinishedSignalUnique(rs->tag, rs->finUnique);
err1:
    saidaCloseMessageUnique(rs->tag, rs->msgUnique);
err0:
    free(rs);
    return NULL;
}

void* stdErrorReadLoop(void* context)  // 线程函数返回类型改为 void*
{
    SAIDA_IPC_WRAPPER* wrapper = context;
    ssize_t n;
    char logLine[1024];
    size_t logLineLength = 0;
    while ((n = read(wrapper->stdErrFD, &logLine[logLineLength], sizeof(logLine) - logLineLength - 1)) > 0)
    {
        logLineLength += n;
        for (size_t i = 0; i < logLineLength; ++i)
        {
            if (logLine[i] == '\n')
            {
                logLine[i] = '\0';
                if (wrapper->onLogLine != NULL)
                {
                    wrapper->onLogLine(wrapper->userData, logLine);
                }
                logLineLength -= (i + 1);
                memmove(logLine, &logLine[i + 1], logLineLength);
                i = -1;
            }
        }
    }
    close(wrapper->stdErrFD);
    if (wrapper->finUnique > 0)
    {
        saidaCloseFinishedSignalUnique(wrapper->tag, wrapper->finUnique);
    }
    if (wrapper->msgUnique > 0)
    {
        saidaCloseMessageUnique(wrapper->tag, wrapper->msgUnique);
    }
    return NULL;  // 返回 NULL，符合 pthread 线程函数的要求
}

SAIDA_C_EXT int saidaAVMainLoop(void* context, FOnNewRGBData onDecoded)
{
    pthread_t thread;  // 使用 pthread_t 代替 thrd_t
    pthread_create(&thread, NULL, stdErrorReadLoop, context);  // 使用 pthread_create 代替 thrd_create
    SAIDA_IPC_WRAPPER* wrapper = context;
    size_t msgSize = sizeof(SAIDA_IPC_MESSAGE);
    ssize_t rs = 0;
    SAIDA_IPC_MESSAGE* message = malloc(msgSize);
    unsigned int msgPriority = 0;
    char buffer[256];
    snprintf(buffer, 256, "开始读取解码器进程信号\n");
    wrapper->onLogLine(wrapper->userData, buffer);
    unsigned int count = 0;
    struct timespec ts;
    while (true)
    {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 10;
        rs = mq_timedreceive(wrapper->msgUnique, (char*)message, msgSize, &msgPriority, &ts);
        if (rs != msgSize)
        {
            int err = errno;
            switch (err)
            {
            case 110:
                snprintf(buffer, 256, "解码器进程长时间没有返回结果\n");
                wrapper->onLogLine(wrapper->userData, buffer);
                break;
            case 9:
                break;
            default:
                snprintf(buffer, 256, "读取解码器进程的结果失败,错误:%zd,errno:%d\n", rs,errno);
                wrapper->onLogLine(wrapper->userData, buffer);
                break;
            }
            break;
        }
        onDecoded(wrapper->userData, wrapper->sharedMem,
                  message->imgHeight, message->imgWidth,
                  message->timestamp);
        count++;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 5;
        if (!saidaPostFinishedSignalToUnique(wrapper->finUnique, &ts))
        {
            snprintf(buffer, 256, "无法发送处理完成消息给解码器进程\n");
            wrapper->onLogLine(wrapper->userData, buffer);
            break;
        }
    }
    free(message);
    if (wrapper->finUnique > 0)
    {
        saidaCloseFinishedSignalUnique(wrapper->tag, wrapper->finUnique);
    }
    if (wrapper->msgUnique > 0)
    {
        saidaCloseMessageUnique(wrapper->tag, wrapper->msgUnique);
    }
    pthread_join(thread, NULL);  // 使用 pthread_join 代替 thrd_join
    if (wrapper->waited++ == 0)
    {
        int status;
        if (waitpid(wrapper->pid, &status, 0) == -1)
        {
            wrapper->onLogLine(wrapper->userData, "等待解码器进程终止失败\n");
        }
        else
        {
            snprintf(buffer, 256, "解码器进程%d以%d状态码退出\n", wrapper->pid, status);
            wrapper->onLogLine(wrapper->userData, buffer);
        }
    }
    snprintf(buffer, 256, "解码器进程Standard Error读取结束\n");
    wrapper->onLogLine(wrapper->userData, buffer);
    snprintf(buffer, 256, "解码器进程一共调用了%d次AI算法\n", count);
    wrapper->onLogLine(wrapper->userData, buffer);
    return 0;
}

SAIDA_C_EXT void saidaContextClose(void* context)
{
    SAIDA_IPC_WRAPPER* wrapper = context;
    char buffer[256];
    snprintf(buffer, 256, "准备清理资源\n");
    wrapper->onLogLine(wrapper->userData, buffer);
    if (wrapper->finUnique > 0)
    {
        saidaCloseFinishedSignalUnique(wrapper->tag, wrapper->finUnique);
    }
    if (wrapper->msgUnique > 0)
    {
        saidaCloseMessageUnique(wrapper->tag, wrapper->msgUnique);
    }
    snprintf(buffer, 256, "清理资源清理完成\n");
    wrapper->onLogLine(wrapper->userData, buffer);
}

SAIDA_C_EXT void saidaContextFree(void* context)
{
    if (context == NULL)
    {
        return;
    }
    SAIDA_IPC_WRAPPER* wrapper = context;
    if (wrapper->waited++ == 0)
    {
        int status;
        if (wrapper->pid != 0)
        {
            kill(wrapper->pid, SIGKILL);
        }
        if (waitpid(wrapper->pid, &status, 0) == -1)
        {
            wrapper->onLogLine(wrapper->userData, "等待解码器进程终止失败\n");
        }
        else
        {
            char buffer[256];
            snprintf(buffer, 256, "解码器进程%d以%d状态码退出\n", wrapper->pid, status);
            wrapper->onLogLine(wrapper->userData, buffer);
        }
    }
    if (wrapper->sharedMem != NULL)
    {
        saidaFreeSharedMemory(wrapper->tag);
    }
    free(context);
}

