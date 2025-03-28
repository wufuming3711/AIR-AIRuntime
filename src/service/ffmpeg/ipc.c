#include "ipc.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <mqueue.h>
#include <stdatomic.h>
#include <time.h>

#define SHM_PREFIX "/saida_shm_"
#define MQ_PREFIX "/saida_mq_unique_"
#define MQ_FINISH_PREFIX "/saida_finish_unique_"

unsigned int saidaIPCGetTag()
{
    static atomic_uint tag = 0;
    return tag++;
}

void* saidaMallocSharedMemory(unsigned int tag, unsigned int length)
{
    char shm_name[256];
    snprintf(shm_name, sizeof(shm_name), "%s%d", SHM_PREFIX, tag);
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1)
    {
        perror("shm_open");
        return NULL;
    }
    if (ftruncate(shm_fd, length) == -1)
    {
        perror("ftruncate");
        shm_unlink(shm_name);
        return NULL;
    }
    void* shm_ptr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED)
    {
        perror("mmap");
        shm_unlink(shm_name);
        return NULL;
    }
    close(shm_fd);
    return shm_ptr;
}

void* saidaGetSharedMemory(unsigned int tag, unsigned int length)
{
    char shm_name[256];
    snprintf(shm_name, sizeof(shm_name), "%s%d", SHM_PREFIX, tag);
    int shm_fd = shm_open(shm_name, O_RDWR, 0666);
    if (shm_fd == -1)
    {
        fprintf(stderr, "shm_open遇到错误\n");
        return NULL;
    }
    void* shm_ptr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED)
    {
        fprintf(stderr, "mmap遇到错误\n");
        close(shm_fd);
        return NULL;
    }
    close(shm_fd);
    return shm_ptr;
}

void saidaFreeSharedMemory(unsigned int tag)
{
    char shm_name[256];
    snprintf(shm_name, sizeof(shm_name), "%s%d", SHM_PREFIX, tag);
    shm_unlink(shm_name);
}

mqd_t saidaCreateMessageUnique(unsigned int tag)
{
    char mq_name[256];
    snprintf(mq_name, sizeof(mq_name), "%s%d", MQ_PREFIX, tag);
    struct mq_attr attr;
    attr.mq_maxmsg = 16;
    attr.mq_msgsize = sizeof(SAIDA_IPC_MESSAGE);
    return mq_open(mq_name, O_CREAT | O_RDWR, 0666, &attr);
}

mqd_t saidaGetMessageUnique(unsigned int tag)
{
    char mq_name[256];
    snprintf(mq_name, sizeof(mq_name), "%s%d", MQ_PREFIX, tag);
    return mq_open(mq_name, O_RDWR);
}

void saidaCloseMessageUnique(unsigned int tag, mqd_t mq)
{
    char mq_name[256];
    snprintf(mq_name, sizeof(mq_name), "%s%d", MQ_PREFIX, tag);
    mq_close(mq);
    mq_unlink(mq_name);
}

mqd_t saidaCreateFinishedSignalUnique(unsigned int tag)
{
    char mq_name[256];
    struct mq_attr attr;
    attr.mq_maxmsg = 16;
    attr.mq_msgsize = 4;
    snprintf(mq_name, sizeof(mq_name), "%s%d", MQ_FINISH_PREFIX, tag);
    return mq_open(mq_name, O_CREAT | O_RDWR, 0660, &attr);
}

mqd_t saidaGetFinishedSignalUnique(unsigned int tag)
{
    char mq_name[256];
    snprintf(mq_name, sizeof(mq_name), "%s%d", MQ_FINISH_PREFIX, tag);
    return mq_open(mq_name, O_RDWR);
}

const char finishMessage[4] = {0};
const char closeMessage[4] = {1, 1, 1, 1};

void saidaCloseFinishedSignalUnique(unsigned int tag, mqd_t mq)
{
    char mq_name[256];
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 1;
    for (int i = 0; i < 10; ++i)
    {
        if (mq_timedsend(mq, closeMessage, 4, 0, &ts) < 0)
        {
            break;
        }
    }
    snprintf(mq_name, sizeof(mq_name), "%s%d", MQ_FINISH_PREFIX, tag);
    mq_close(mq);
    mq_unlink(mq_name);
}

bool saidaPostMessageToUnique(const SAIDA_IPC_MESSAGE* message, mqd_t mq)
{
    return mq_send(mq, (const char*)message, sizeof(SAIDA_IPC_MESSAGE), 0) != -1;
}


bool saidaPostFinishedSignalToUnique(mqd_t mq, const struct timespec* timeout)
{
    return mq_timedsend(mq, finishMessage, 4, 0, timeout) != -1;
}
