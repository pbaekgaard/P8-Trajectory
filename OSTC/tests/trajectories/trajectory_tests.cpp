#include <gtest/gtest.h>
#include <trajectory.hpp>

TEST(Trajectories, trajectories_are_equal)
{
    auto traj1 = Trajectory(1, std::vector<SamplePoint>{SamplePoint(1, 2, "")});
    auto traj2 = Trajectory(2, std::vector<SamplePoint>{SamplePoint(3, 4, "")});
    auto traj3 = Trajectory(3, std::vector<SamplePoint>{SamplePoint(1, 2, "")});
    EXPECT_FALSE(traj1 == traj2);
    EXPECT_FALSE(traj1 == traj3);
    EXPECT_TRUE(traj1 == traj1);
    EXPECT_EQ(traj1, traj1);
}
