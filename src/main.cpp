#include <iostream>
#include <sstream>
#include <cstring>
#include "conf/Config.h"
#include "service/Service.h"
#include "service/ai/Manager.h"

using namespace conf;
using namespace std;

int main(int argc, char* argv[])
{
    const char* confFileName = nullptr;
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc)
        {
            confFileName = argv[i + 1];
            break;
        }
    }
    if (confFileName == nullptr)
    {
        cout << "-c 携带配置文件路径" << endl;
        exit(-1);
    }
    if (!Config::getInstance().loadFormFile(string(confFileName)))
    {
        cout << "配置文件 " << confFileName << " 读取错误" << endl;
        exit(-1);
    }
    ai::Manager::getInstance().initHandlers(
        Config::getInstance().ai->gpuCount,
        Config::getInstance().algoConfig->logDir,
        Config::getInstance().algoConfig->configWorkflow,
        Config::getInstance().algoConfig->modelZoo
    );

    // 使用 std::ostringstream 拼接字符串
    stringstream addrStream;
    addrStream << Config::getInstance().gRPC->listenHost << ":"
        << Config::getInstance().gRPC->port;
    string addr = addrStream.str();
    auto s = grpcImpl::Service(addr);
    cout << "准备听" << addr << "的TCP" << endl;
    s.start();
    return 0;
}
