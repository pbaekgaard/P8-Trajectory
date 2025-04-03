#include <gtest/gtest.h>
#include <trajectory.hpp>

TEST(Trajectories, trajectories_are_equal)
{
    auto traj1 = Trajectory(1, std::vector<SamplePoint>{SamplePoint(1, 2, 0)});
    auto traj2 = Trajectory(2, std::vector<SamplePoint>{SamplePoint(3, 4, 0)});
    EXPECT_TRUE(true);
    EXPECT_EQ(traj1, traj1);
}
