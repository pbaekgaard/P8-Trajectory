#include <gtest/gtest.h>
#include "trajectory.hpp"

TEST(SamplePoint, isEqual)
{
    auto samePoint1 = SamplePoint(1, 2, 1);
    auto samePoint2 = SamplePoint(1, 2, 2);
    auto differentY = SamplePoint(1, 3, 3);
    auto differentX = SamplePoint(2, 2, 4);
    auto differentTimestamp = SamplePoint(1, 2, 4);

    EXPECT_EQ(samePoint1, samePoint2);
    EXPECT_TRUE(samePoint1 == samePoint2);
    EXPECT_FALSE(samePoint1 == differentY);
    EXPECT_FALSE(samePoint1 == differentX);
    EXPECT_FALSE(samePoint1 == differentTimestamp);
    EXPECT_FALSE(samePoint2 == differentY);
    EXPECT_FALSE(samePoint2 == differentX);
    EXPECT_FALSE(samePoint2 == differentTimestamp);
}
