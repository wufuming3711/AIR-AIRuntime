//
// Created by Oboi on 2024/11/25.
//

#ifndef ANALYSIS_SERVICE_IPC_H
#define ANALYSIS_SERVICE_IPC_H

#include <mqueue.h>

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
typedef struct SaidaIPCMessage
{
    int imgHeight;
    int imgWidth;
    unsigned long long timestamp;
} SAIDA_IPC_MESSAGE;

// saidaIPCGetTag 获取一个新的tag用于shared mem和Unique
SAIDA_C_EXT unsigned int saidaIPCGetTag();

// saidaMallocSharedMemory 父进程调用 创建并得到一段长度为length的共享内存
SAIDA_C_EXT void* saidaMallocSharedMemory(unsigned int tag, unsigned int length);

// saidaGetSharedMemory 子进程调用 得到一段长度为length的共享内存
SAIDA_C_EXT void* saidaGetSharedMemory(unsigned int tag, unsigned int length);

// saidaFreeSharedMemory 父进程调用 释放共享内存
SAIDA_C_EXT void saidaFreeSharedMemory(unsigned int tag);

// saidaCreateUnique 父进程调用 创建并得到一个Message Unique
SAIDA_C_EXT mqd_t saidaCreateMessageUnique(unsigned int tag);

// saidaCreateUnique 子进程调用 得到一个Message Unique
SAIDA_C_EXT mqd_t saidaGetMessageUnique(unsigned int tag);

// saidaCloseUnique 父进程调用 关闭一个Message Unique
SAIDA_C_EXT void saidaCloseMessageUnique(unsigned int tag, mqd_t mq);

// saidaCreateUnique 父进程调用 创建并得到一个FIN Unique
SAIDA_C_EXT mqd_t saidaCreateFinishedSignalUnique(unsigned int tag);

// saidaCreateUnique 子进程调用 得到一个FIN Unique
SAIDA_C_EXT mqd_t saidaGetFinishedSignalUnique(unsigned int tag);

// saidaCloseUnique 父进程调用 关闭一个FIN Unique
SAIDA_C_EXT void saidaCloseFinishedSignalUnique(unsigned int tag, mqd_t mq);

// saidaPostMessageToUnique 子进程调用,告诉父进程有新的数据写入了共享内存
SAIDA_C_EXT bool saidaPostMessageToUnique(const SAIDA_IPC_MESSAGE* message, mqd_t mq);

// saidaPostMessageToUnique 父进程调用,告诉子进程共享内存已经读取完毕,可以继续写入了,一个简单的信号,长度为4,随便什么都可以
bool saidaPostFinishedSignalToUnique(mqd_t mq, const struct timespec* timeout);

#endif //ANALYSIS_SERVICE_IPC_H
