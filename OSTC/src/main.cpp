#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <math.h>
#if _WIN32
#include <cstdint>
#endif

struct SamplePoint
{
    double x;  // longitude
    double y;  // latitude
    double t;  // timestamp

    SamplePoint(const double x, const double y, const double t): x(x), y(y), t(t) {}

    bool operator==(const SamplePoint& other) const { return x == other.x && y == other.y && t == other.t; }
};

struct Trajectory
{
    uint32_t id;
    std::vector<SamplePoint> points;
    int start_index = -1;
    int end_index = -1;

    explicit Trajectory(const uint32_t id, const std::vector<SamplePoint>& points): id(id), points(points) {}
    explicit Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, const int start_index,
                        const int end_index): id(id), points(points), start_index(start_index), end_index(end_index)
    {}

    Trajectory operator()(const int start, const int end)
    {
        return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
    }

    bool operator==(const Trajectory& other) const
    {
        return (id == other.id && start_index == other.start_index && end_index == other.end_index);
    }
};

struct ReferenceTrajectory
{
    uint32_t id;
    short start_index = -1;
    short end_index = -1;

    explicit ReferenceTrajectory(const uint32_t id, const short start_index, const short end_index):
        id(id), start_index(start_index), end_index(end_index)
    {}
};

template <>
struct std::hash<Trajectory>
{
    std::size_t operator()(const Trajectory& t) const noexcept
    {
        return ((hash<uint32_t>()(t.id) ^ (hash<int>()(t.start_index) << 2) ^ (hash<int>()(t.end_index) >> 1) ^
                 (hash<uint32_t>()(t.points.size()) << 1)) >>
                1);
    }
};

std::unordered_map<Trajectory, std::vector<Trajectory>> M;

const double euclideanDistance(SamplePoint a, SamplePoint b) { return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2)); }
const double maxDtw(Trajectory a, Trajectory b)
{
    if (a.points == b.points && b.points.empty()) {
        std::cout << "Is Empty!" << std::endl;
        return 0;
    } else if (a.points.empty() || b.points.empty()) {
        std::cout << "Is Empty!" << std::endl;
        return MAXFLOAT;
    }

    return std::max(euclideanDistance(a.points.back(), b.points.back()),
                    std::min(std::min(maxDtw(a(0, a.points.size() - 2), b(0, b.points.size() - 2)),
                                      maxDtw(a, b(0, b.points.size() - 2))),
                             maxDtw(a(0, a.points.size() - 2), b)));
}

int main()
{
    Trajectory t1(1, std::vector{SamplePoint(3, 15.5, 0), SamplePoint(5, 15.5, 1), SamplePoint(7, 15.5, 2),
                                 SamplePoint(8.5, 15.5, 3)});

    Trajectory sub_t1 = t1(0, 1);
    Trajectory sub_t2 = t1(2, 3);
    Trajectory sub_t3 = t1(2, 3);

    auto euc = euclideanDistance(sub_t2.points.back(), sub_t3.points.back());
    std::cout << "euc = " << euc << std::endl;

    auto test = maxDtw(sub_t1, sub_t3);
    std::cout << "MaxDTW = " << test << std::endl;

    return 0;
}
