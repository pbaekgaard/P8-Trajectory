#include <cmath>
#include <vector>
#include "trajectory.hpp"
#include <cmath>
#if _WIN32
#include <cstdint>
#endif

SamplePoint::SamplePoint(const double x, const double y, const double t): x(x), y(y), t(t) {}

bool SamplePoint::operator==(const SamplePoint& other) const { return x == other.x && y == other.y && t == other.t; }
Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points): id(id), points(points) {}
Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, const int start_index,
                       const int end_index): id(id), points(points), start_index(start_index), end_index(end_index)
{}

Trajectory Trajectory::operator()(const int start, const int end)
{
    return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
}

bool Trajectory::operator==(const Trajectory& other) const
{
    return (id == other.id && start_index == other.start_index && end_index == other.end_index);
}

ReferenceTrajectory::ReferenceTrajectory(const uint32_t id, const short start_index, const short end_index):
    id(id), start_index(start_index), end_index(end_index)
{}

std::size_t std::hash<Trajectory>::operator()(const Trajectory& t) const noexcept
{
    return ((hash<uint32_t>()(t.id) ^ (hash<int>()(t.start_index) << 2) ^ (hash<int>()(t.end_index) >> 1) ^
             (hash<uint32_t>()(t.points.size()) << 1)) >>
            1);
}
