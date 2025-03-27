#ifndef __TRAJECTORY_HPP__
#define __TRAJECTORY_HPP__

#include <cmath>
#include <cstdint>
#include <vector>
#include <math.h>

struct SamplePoint
{
    double x;  // longitude
    double y;  // latitude
    double t;  // timestamp

    SamplePoint(double x, double y, double t);

    bool operator==(const SamplePoint& other) const;
};

struct Trajectory
{
    uint32_t id;
    std::vector<SamplePoint> points;
    int start_index = -1;
    int end_index = -1;

    Trajectory(uint32_t id, const std::vector<SamplePoint>& points);
    Trajectory(uint32_t id, const std::vector<SamplePoint>& points, int start_index, int end_index);

    Trajectory operator()(int start, int end);

    bool operator==(const Trajectory& other) const;
};

struct ReferenceTrajectory
{
    uint32_t id;
    short start_index = -1;
    short end_index = -1;

    ReferenceTrajectory(uint32_t id, short start_index, short end_index);
};

template <>
struct std::hash<Trajectory>
{
    std::size_t operator()(const Trajectory& t) const noexcept;
};

#endif
