#ifndef __TRAJECTORY_HPP__
#define __TRAJECTORY_HPP__

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <cstdint>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>

struct Trajectory;

struct SamplePoint
{
    double longitude;       // longitude
    double latitude;        // latitude
    unsigned int timestamp;  // timestamp

    SamplePoint(double x, double y, unsigned int t): longitude(x), latitude(y), timestamp(t) {}

    bool operator==(const SamplePoint& other) const;
};

struct TimeCorrectionRecordEntry
{
    int point_index;
    unsigned int corrected_timestamp;
    TimeCorrectionRecordEntry(int idx, unsigned int ct): point_index(idx), corrected_timestamp(ct) {
        std::cout << "i use the constructor:)"<<std::endl;
    }
};

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
    short start_index = -1;
    short end_index = -1;

    Trajectory(const uint32_t id, const std::vector<SamplePoint>& points);
    Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, short start_index, short end_index);

    Trajectory operator()(short start, short end);

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

    std::unordered_map<Trajectory, std::vector<Trajectory>> MRTSearch(std::vector<Trajectory>& RefSet,
                                                                               double epsilon);
    std::vector<ReferenceTrajectory> OSTC(std::unordered_map<Trajectory, std::vector<Trajectory>> M, double tepsilon);
};

template <>
struct std::hash<Trajectory>
{
    std::size_t operator()(const Trajectory& t) const noexcept;
};

#endif
