//
// Created by Oboi on 2024/11/5.
//

#include <fstream>
#include <sstream>
#include "File.h"

namespace tools
{
    using namespace std;

    string loadStringFromFile(const string& path)
    {
        ifstream file(path);
        if (!file.is_open())
        {
            throw runtime_error("无法打开文件");
        }
        stringstream basicStringStream;
        basicStringStream << file.rdbuf();
        return basicStringStream.str();
    }
} // tools
