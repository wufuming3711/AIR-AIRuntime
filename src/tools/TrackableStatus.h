//
// Created by Oboi on 2024/11/6.
//

#ifndef ANALYSIS_SERVICE_TRACKABLESTATUS_H
#define ANALYSIS_SERVICE_TRACKABLESTATUS_H

#include <sstream>
#include <mutex>
#include <chrono>

namespace tools
{
    using namespace std;

    using time = chrono::time_point<chrono::system_clock, chrono::system_clock::duration>;

    class TrackableStatus
    {
    public:
        TrackableStatus(const string& tag, const string& initialStatus);

        TrackableStatus* fork(const string& tag, const string& initialStatus);

        void append(const string& newStatus);

        string getString();

    private:
        TrackableStatus();

        TrackableStatus* parent;
        ostringstream strBuilder;
        string tag;
        mutex lock;
        time createdTime;
    };
} // tools

#endif //ANALYSIS_SERVICE_TRACKABLESTATUS_H
