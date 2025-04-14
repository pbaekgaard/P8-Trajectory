#include <gtest/gtest.h>
#include "trajectory.hpp"  // Include the header for SamplePoint
#include "distance.hpp"    // Include the header for euclideanDistance function

// Test the EuclideanDistance function for correctness
TEST(EuclideanDistance, IsCorrectDistance)
{
    // Arrange: Create two SamplePoint objects
    auto point1 = SamplePoint(3.0, 16.0, 1);
    auto point2 = SamplePoint(1.0, 20.0, 2);

    // Act: Calculate the Euclidean distance
    auto distance = euclideanDistance(point1, point2);
    // Assert: Check if the calculated distance is correct
    double expectedDistance = std::sqrt(std::pow(3.0 - 1.0, 2) + std::pow(16.0 - 20.0, 2));  // Calculate manually
    EXPECT_EQ(distance, expectedDistance);  // Expect the distance to be within a small tolerance (1e-6)
}

TEST(MaxDTW, IsCorrectDistance)
{
    GTEST_SKIP() << "Skip until further notice" << std::endl;
    auto traj1 = Trajectory{1, {}};
    auto traj2 = Trajectory{2, {}};
    auto traj3 = Trajectory{3, std::vector<SamplePoint>{SamplePoint{1, 2, 3}}};
    auto traj4 = Trajectory{3, std::vector<SamplePoint>{SamplePoint{2, 3, 4}}};
    auto traj5 = Trajectory{3, std::vector<SamplePoint>{SamplePoint{6, 9, 5}}};

    // Test empty trajectories
    EXPECT_EQ(0, MaxDTW(traj1, traj2));

    // Test one empty
    EXPECT_EQ(std::numeric_limits<double>::max(), MaxDTW(traj1, traj3));
    EXPECT_EQ(std::numeric_limits<double>::max(), MaxDTW(traj3, traj1));
    EXPECT_GT(MaxDTW(traj3, traj5), MaxDTW(traj3, traj4));
    EXPECT_EQ(std::sqrt(2), MaxDTW(traj3, traj4));
    EXPECT_EQ(std::sqrt(74), MaxDTW(traj3, traj5));
}
