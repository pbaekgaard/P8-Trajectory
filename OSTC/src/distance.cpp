#include "trajectory.hpp"
#include "distance.hpp"

double euclideanDistance(SamplePoint a, SamplePoint b) { return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2)); }
double maxDtw(Trajectory a, Trajectory b)
{
    if (a.points == b.points && b.points.empty()) {
        std::cout << "Is Empty!" << std::endl;
        return 0;
    } else if (a.points.empty() || b.points.empty()) {
        std::cout << "Is Empty!" << std::endl;
        return MAXFLOAT;
    }

    return std::max(euclideanDistance(a.points.back(), b.points.back()),
                    std::min(std::min(maxDtw(a(0, static_cast<int>(a.points.size()) - 2),
                                             b(0, static_cast<int>(b.points.size()) - 2)),
                                      maxDtw(a, b(0, static_cast<int>(b.points.size()) - 2))),
                             maxDtw(a(0, static_cast<int>(a.points.size()) - 2), b)));
}
