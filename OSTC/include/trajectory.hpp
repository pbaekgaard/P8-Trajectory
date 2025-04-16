#ifndef __TRAJECTORY_HPP__
#define __TRAJECTORY_HPP__

#include <cmath>
#include <iostream>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

struct Trajectory;
struct OSTCResult;

struct SamplePoint
{
    double longitude;  // longitude
    double latitude;   // latitude
    int timestamp;     // timestamp

    SamplePoint(double x, double y, int t): longitude(x), latitude(y), timestamp(t) {}

    bool operator==(const SamplePoint& other) const;
};

struct TimeCorrectionRecordEntry
{
    int point_index;
    int corrected_timestamp;
    TimeCorrectionRecordEntry(int idx, int ct): point_index(idx), corrected_timestamp(ct) {};
};

inline bool operator==(const TimeCorrectionRecordEntry& lhs, const TimeCorrectionRecordEntry& rhs) {
    return lhs.point_index == rhs.point_index &&
           lhs.corrected_timestamp == rhs.corrected_timestamp;
}

inline std::ostream& operator<<(std::ostream& os, const SamplePoint& point)
{
    os << std::fixed << std::setprecision(15) << "SamplePoint(Longitude: " << point.longitude
       << ", Latitude: " << point.latitude << ", Timestamp: " << point.timestamp << ")";
    return os;
}

struct ReferenceTrajectory
{
    uint32_t id;
    short start_index = -1;
    short end_index = -1;

    ReferenceTrajectory(uint32_t id, short start_index, short end_index);
    bool operator==(const ReferenceTrajectory& other) const;
    ReferenceTrajectory(const Trajectory& t);
};

struct Trajectory
{
    uint32_t id;
    std::vector<SamplePoint> points;
    int start_index = -1;
    int end_index = -1;

    Trajectory() : id(0), start_index(-1), end_index(-1) {}
    Trajectory(const uint32_t id, const std::vector<SamplePoint>& points);
    Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, int start_index, int end_index);

    Trajectory operator()(int start, int end);
    bool operator<(const Trajectory& other) const;

    Trajectory operator+(Trajectory other);

    bool operator==(const Trajectory& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Trajectory& traj)
    {
        os << "Trajectory ID: " << traj.id << "\n";
        os << "Points:\n";
        for (const auto& point : traj.points) {
            os << "  " << point << "\n";
        }
        return os;
    }

    std::unordered_map<Trajectory, std::vector<Trajectory>> MRTSearch(std::vector<Trajectory>& RefSet, double epsilon);
    OSTCResult OSTC(std::unordered_map<Trajectory, std::vector<Trajectory>> M, double tepsilon, double sepsilon);
};

template <>
struct std::hash<Trajectory>
{
    std::size_t operator()(const Trajectory& t) const noexcept;
};

struct OSTCResult
{
    std::vector<ReferenceTrajectory> references{};
    std::unordered_map<Trajectory, std::vector<TimeCorrectionRecordEntry>> time_corrections;

    OSTCResult(std::vector<ReferenceTrajectory> references,
               std::unordered_map<Trajectory, std::vector<TimeCorrectionRecordEntry>> time_corrections):
        references(std::move(references)), time_corrections(std::move(time_corrections)) {};
};

#endif
