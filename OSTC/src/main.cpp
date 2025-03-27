#include "trajectory.hpp"
#include "distance.hpp"
#include <unordered_map>
#include <iostream>
int main()
{
    std::unordered_map<Trajectory, std::vector<Trajectory>> M;
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
