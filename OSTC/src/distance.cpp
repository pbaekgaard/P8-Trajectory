#include "trajectory.hpp"
#include "distance.hpp"

double euclideanDistance(SamplePoint a, SamplePoint b)
{
    return sqrt(pow(a.longitude - b.longitude, 2) + pow(a.latitude - b.latitude, 2));
}

double maxDTW(Trajectory a, Trajectory b)
{
    if (a.points == b.points && b.points.empty()) {
        return 0;
    } else if (a.points.empty() || b.points.empty()) {
        return std::numeric_limits<double>::max();
    }

    return std::max(euclideanDistance(a.points.back(), b.points.back()),
                    std::min(std::min(maxDTW(a(0, static_cast<int>(a.points.size()) - 2),
                                             b(0, static_cast<int>(b.points.size()) - 2)),
                                      maxDTW(a, b(0, static_cast<int>(b.points.size()) - 2))),
                             maxDTW(a(0, static_cast<int>(a.points.size()) - 2), b)));
}
