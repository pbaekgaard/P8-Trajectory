#include "trajectory.hpp"
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include "distance.hpp"

double distance(const SamplePoint& p1, const SamplePoint& p2)
{
    return std::sqrt(std::pow(p1.longitude - p2.longitude, 2) + std::pow(p1.latitude - p2.latitude, 2));
}

double maxDTW(const Trajectory& T_a, const Trajectory& T_b);  // Forward declaration

double Q(const Trajectory& T_a, const Trajectory& T_b)
{
    int n = T_a.points.size();
    int m = T_b.points.size();

    double option1 = maxDTW(T_a(0, n - 2), T_b(0, m - 2));
    double option2 = maxDTW(T_a(0, n - 2), T_b(0, m - 1));
    double option3 = maxDTW(T_a(0, n - 1), T_b(0, m - 2));

    return std::min({option1, option2, option3});
}

double maxDTW(const Trajectory& T_a, const Trajectory& T_b)
{
    if (T_a.points.empty() && T_b.points.empty()) {
        return 0.0;
    }

    if (T_a.points.empty() || T_b.points.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    int n = T_a.points.size();
    int m = T_b.points.size();

    double d = distance(T_a.points[n - 1], T_b.points[m - 1]);
    double q = Q(T_a, T_b);

    return std::max(d, q);
}
