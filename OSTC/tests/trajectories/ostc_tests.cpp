#include <gtest/gtest.h>
#include "trajectory.hpp"
#include "example_trajectories.hpp"

TEST(OSTC, outputs_correct_values)
{
    auto M = std::unordered_map<Trajectory, std::vector<Trajectory>>{
        {t(0, 0), {t(0, 0)}},      {t(1, 8), {t2(0, 7)}},   {t(9, 14), {t5(6, 11)}},
        {t(15, 16), {t6(12, 13)}}, {t(17, 19), {t7(7, 9)}},
    };


    OSTCResult T_prime = t.OSTC(M, 0.0, 0.9);

    std::cout << "henro " << std::endl;
}
