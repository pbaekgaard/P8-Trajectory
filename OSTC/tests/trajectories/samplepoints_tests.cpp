#include <gtest/gtest.h>
#include "trajectory.hpp"

TEST(SamplePoint, isEqual)
{
    auto samePoint1 = SamplePoint(1, 2, "ts");
    auto samePoint2 = SamplePoint(1, 2, "ts");
    auto differentY = SamplePoint(1, 3, "ts");
    auto differentX = SamplePoint(2, 2, "ts");
    auto differentTimestamp = SamplePoint(1, 2, "ts!");

    EXPECT_EQ(samePoint1, samePoint2);
    EXPECT_TRUE(samePoint1 == samePoint2);
    EXPECT_FALSE(samePoint1 == differentY);
    EXPECT_FALSE(samePoint1 == differentX);
    EXPECT_FALSE(samePoint1 == differentTimestamp);
    EXPECT_FALSE(samePoint2 == differentY);
    EXPECT_FALSE(samePoint2 == differentX);
    EXPECT_FALSE(samePoint2 == differentTimestamp);
}
