#include <iostream>
#include <sstream>  // 引入 stringstream
#include "TrackableStatus.h"
#include <iomanip>
#include <chrono>
using namespace std::chrono;
namespace tools
{
    TrackableStatus::TrackableStatus(const string& fTag, const string& initialStatus)
    {
        tag = fTag;
        createdTime = chrono::system_clock::now();

        // 使用 ostringstream 代替 format
        auto timeDiff = duration_cast<chrono::milliseconds>(chrono::system_clock::now() - createdTime).count();
        strBuilder << "["
            << setw(16) << timeDiff
            << "ms|"
            << setw(16) << tag
            << "]     " << initialStatus << endl;
        parent = nullptr;
    }

    TrackableStatus* TrackableStatus::fork(const string& fTag, const string& initialStatus)
    {
        auto* rs = new TrackableStatus;
        rs->parent = this;
        rs->tag = fTag;
        rs->createdTime = createdTime;
        lock.lock();
        // 使用 ostringstream 代替 format
        auto timeDiff = duration_cast<chrono::milliseconds>(chrono::system_clock::now() - createdTime).count();
        strBuilder << "["
            << setw(16) << timeDiff
            << "ms|"
            << setw(16) << tag
            << "]     " << initialStatus << endl;
        lock.unlock();
        return rs;
    }

    void TrackableStatus::append(const string& newStatus)
    {
        if (parent != nullptr)
        {
            auto timeDiff = duration_cast<chrono::milliseconds>(
                chrono::system_clock::now() - parent->createdTime).count();
            parent->lock.lock();
            parent->strBuilder << "["
                << setw(16) << timeDiff
                << "ms|"
                << setw(16) << tag
                << "]     " << newStatus << endl;
            parent->lock.unlock();
        }
        else
        {
            auto timeDiff = duration_cast<chrono::milliseconds>(chrono::system_clock::now() - createdTime).count();
            lock.lock();
            strBuilder << "["
                << setw(16) << timeDiff
                << "ms|"
                << setw(16) << tag
                << "]     " << newStatus << endl;
            lock.unlock();
        }
    }

    string TrackableStatus::getString()
    {
        lock.lock();
        string rs = strBuilder.str();
        lock.unlock();
        return rs;
    }

    TrackableStatus::TrackableStatus() = default;
} // tools
