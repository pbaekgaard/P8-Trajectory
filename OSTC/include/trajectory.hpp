#ifndef __TRAJECTORY_HPP__
#define __TRAJECTORY_HPP__

#include <cmath>
#include <unordered_map>
#include <cstdint>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>
#include <functional>

struct Trajectory;
struct OSTCResult;

struct SamplePoint
{
    float longitude;  // longitude
    float latitude;   // latitude
    int timestamp;     // timestamp

    SamplePoint(float latitude, float longitude, int t): longitude(longitude), latitude(latitude), timestamp(t) {}

    bool operator==(const SamplePoint& other) const;
};

struct TimeCorrectionRecordEntry
{
    uint32_t trajectory_id;
    int point_index;
    int corrected_timestamp;
    TimeCorrectionRecordEntry(const uint32_t trajectory_id, const int idx, const int ct): trajectory_id(trajectory_id), point_index(idx), corrected_timestamp(ct) {};
};

inline bool operator==(const TimeCorrectionRecordEntry& lhs, const TimeCorrectionRecordEntry& rhs)
{
    return lhs.trajectory_id == rhs.trajectory_id && lhs.point_index == rhs.point_index && lhs.corrected_timestamp == rhs.corrected_timestamp;
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

    Trajectory(): id(0), start_index(-1), end_index(-1) {}
    Trajectory(const uint32_t id, const std::vector<SamplePoint>& points);
    Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, int start_index, int end_index);

    Trajectory operator()(int start, int end);
    Trajectory operator()(int start, int end) const;
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


    std::unordered_map<Trajectory, std::vector<Trajectory>> MRTSearch(std::vector<Trajectory>& RefSet, double epsilon, std::function<double(SamplePoint const& a,SamplePoint const & b)> distance_function);
    std::unordered_map<Trajectory, std::vector<Trajectory>> MRTSearchOptimized(std::vector<Trajectory>& RefSet, double epsilon, std::function<double(SamplePoint const& a,SamplePoint const & b)> distance_function);
    OSTCResult OSTC(std::unordered_map<Trajectory, std::vector<Trajectory>> M, double tepsilon, double sepsilon, std::function<double(SamplePoint const& a, SamplePoint const& b)> distance_function);
};

struct TrajectoryRemoval
{
    const Trajectory query_trajectory;
    Trajectory trajectory_to_remove;

    TrajectoryRemoval(const Trajectory query_traj, Trajectory trajectory_to_remove):
        query_trajectory(query_traj), trajectory_to_remove(trajectory_to_remove) {}
};

template <>
struct std::hash<Trajectory>
{
    std::size_t operator()(const Trajectory& t) const noexcept;
};

struct CompressedResultCorrection
{
    int point_id;
    int corrected_timestamp;

    CompressedResultCorrection(const int point_id, const int ct) : point_id(point_id), corrected_timestamp(ct) {};

    bool operator==(const CompressedResultCorrection& other) const
    {
        return point_id == other.point_id && corrected_timestamp == other.corrected_timestamp;
    };
};

struct CompressedResult
{
    uint32_t id;
    double latitude;
    double longitude;
    int timestamp;
    std::vector<CompressedResultCorrection> corrections;

    CompressedResult(const uint32_t id, const double latitude, const double longitude, const int timestamp, std::vector<CompressedResultCorrection> corrections)
        : id(id), latitude(latitude), longitude(longitude), timestamp(timestamp), corrections(std::move(corrections)) {};

    bool operator==(const CompressedResult& other) const
    {
        return id == other.id && latitude == other.latitude && longitude == other.longitude && timestamp == other.timestamp && corrections == other.corrections;
    };
};

struct OSTCResult
{
    std::vector<Trajectory> references{};
    std::unordered_map<Trajectory, std::vector<TimeCorrectionRecordEntry>> time_corrections;

    OSTCResult() = default;

    OSTCResult(std::vector<Trajectory> references,
               std::unordered_map<Trajectory, std::vector<TimeCorrectionRecordEntry>> time_corrections):
        references(std::move(references)), time_corrections(std::move(time_corrections)) {};
};

void convertCompressedTrajectoriesToPoints(std::vector<CompressedResult>& points, const Trajectory& trajectory_to_be_compressed, OSTCResult compressed);

#endif
