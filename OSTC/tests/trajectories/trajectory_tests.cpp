#include "gtest/gtest.h"
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

TEST(Trajectories, subtrajectories_are_subsets_of_parent_trajectory)
{
    auto parent = Trajectory{1, std::vector<SamplePoint>{SamplePoint(1, 2, ""), SamplePoint(2, 3, ""),
                                                         SamplePoint(4, 7, ""), SamplePoint(11, 12, "")}};
    auto subtrajectory = parent(2, 3);
    auto expected_subtrajectory =
        Trajectory{1, std::vector<SamplePoint>{SamplePoint(4, 7, ""), SamplePoint(11, 12, "")}, 2, 3};
    EXPECT_EQ(subtrajectory, expected_subtrajectory);
}

TEST(Trajectories, mrtset_is_correct_format)
{
    // SKIPPED: Skipped until mrtsearch is done.
    GTEST_SKIP_("SKIP UNTIL MRTSEARCH IS DONE");
    auto T1 =
        Trajectory(1, std::vector<SamplePoint>{SamplePoint(2.0, 2.5), SamplePoint(1.5, 3.0), SamplePoint(1.5, 4.0),
                                               SamplePoint(1.5, 5.5), SamplePoint(1.5, 7.0), SamplePoint(1.5, 8.5),
                                               SamplePoint(1.5, 9.5), SamplePoint(1.5, 10.5), SamplePoint(1.5, 12.0),
                                               SamplePoint(1.5, 13.0), SamplePoint(2, 14.0), SamplePoint(3, 14.5),
                                               SamplePoint(5, 14.5), SamplePoint(6.5, 14.5)});
    auto T2 = Trajectory(
        2, std::vector<SamplePoint>{SamplePoint(5.0, 16.0), SamplePoint(7.5, 16.0), SamplePoint(8.5, 16.0),
                                    SamplePoint(9.5, 16.0), SamplePoint(12, 15.5), SamplePoint(12.5, 14.5),
                                    SamplePoint(12.5, 13.5), SamplePoint(12.5, 12.0), SamplePoint(14.0, 12.5),
                                    SamplePoint(15.0, 11.0), SamplePoint(16.5, 11.0), SamplePoint(17.5, 11.0),
                                    SamplePoint(18.5, 11.0), SamplePoint(19.5, 11.5), SamplePoint(19.5, 12.0),
                                    SamplePoint(19.5, 13.5), SamplePoint(19.5, 14.5)});

    auto T3 = Trajectory(
        3, std::vector<SamplePoint>{SamplePoint(5.5, 14.0), SamplePoint(7.0, 14.0), SamplePoint(8, 14.5),
                                    SamplePoint(10.0, 14.5), SamplePoint(11.0, 14.0), SamplePoint(11.5, 12.0),
                                    SamplePoint(9.5, 12.0), SamplePoint(8.0, 12.0), SamplePoint(6.5, 12.0),
                                    SamplePoint(4.5, 12.0), SamplePoint(3.5, 12.0), SamplePoint(2.5, 11.0),
                                    SamplePoint(2, 10.5), SamplePoint(2, 9.0), SamplePoint(2, 8.0)});

    auto T4 = Trajectory(
        4, std::vector<SamplePoint>{
               SamplePoint(5.5, 14.0), SamplePoint(6.5, 14.0),  SamplePoint(7.5, 14.5),  SamplePoint(8.5, 14.5),
               SamplePoint(9.5, 14.0), SamplePoint(11.5, 12.0), SamplePoint(11.5, 10.0), SamplePoint(11.5, 8.5),
               SamplePoint(11.5, 8.0), SamplePoint(11.5, 6.0),  SamplePoint(11.5, 5.0),  SamplePoint(12.5, 4.0),
               SamplePoint(13.0, 3.5), SamplePoint(13.5, 3.0),  SamplePoint(13.5, 2.0),  SamplePoint(13.5, 1.0),
               SamplePoint(12.5, 1.0), SamplePoint(10.5, 1.0),  SamplePoint(8.5, 1.0),

           });

    auto T5 = Trajectory(
        5, std::vector<SamplePoint>{SamplePoint(20.5, 13.0), SamplePoint(19.0, 13.0), SamplePoint(17.5, 13.0),
                                    SamplePoint(16.0, 13.0), SamplePoint(14.5, 13.0), SamplePoint(13.5, 12.5),
                                    SamplePoint(12.5, 11.0), SamplePoint(12.5, 10.0), SamplePoint(12.5, 8.0),
                                    SamplePoint(12.5, 6.0), SamplePoint(13.5, 4.5), SamplePoint(14.5, 3.5)});

    auto T6 =
        Trajectory(6, std::vector<SamplePoint>{SamplePoint(2.0, 6.0), SamplePoint(2.0, 4.5), SamplePoint(2.0, 3.5),
                                               SamplePoint(2.5, 2.5), SamplePoint(3.5, 2.5), SamplePoint(4.5, 2.5),
                                               SamplePoint(5.5, 2.5), SamplePoint(6.5, 2.5), SamplePoint(7.5, 2.5),
                                               SamplePoint(9.0, 2.5), SamplePoint(11.0, 2.5), SamplePoint(12.0, 2.5),
                                               SamplePoint(14.5, 2.5), SamplePoint(16.0, 2.5)});

    auto T7 =
        Trajectory(7, std::vector<SamplePoint>{SamplePoint(15.5, 6.5), SamplePoint(15.5, 6.0), SamplePoint(15.5, 5.0),
                                               SamplePoint(16.5, 5.0), SamplePoint(17.5, 5.0), SamplePoint(17.5, 4.0),
                                               SamplePoint(18.0, 3.0), SamplePoint(18.0, 2.5), SamplePoint(20.0, 2.5),
                                               SamplePoint(22.0, 2.5)});

    auto T8 = Trajectory(8, std::vector<SamplePoint>{SamplePoint(12.0, 11.5), SamplePoint(12.0, 10.0),
                                                     SamplePoint(12.0, 8.5), SamplePoint(12.0, 5.5)});

    auto references = std::vector<Trajectory>{T1, T2, T3, T4, T5, T6, T7, T8};

    auto T = Trajectory(
        9, std::vector<SamplePoint>{
               SamplePoint(3.0, 15.5),  SamplePoint(5.0, 15.5),  SamplePoint(7.0, 15.5),  SamplePoint(8.5, 15.5),
               SamplePoint(9.5, 15.5),  SamplePoint(10.0, 15.5), SamplePoint(11.5, 15.5), SamplePoint(12.0, 14.0),
               SamplePoint(12.0, 12.0), SamplePoint(12.0, 11.0), SamplePoint(12.0, 10.0), SamplePoint(12.0, 8.0),
               SamplePoint(12.0, 5.5),  SamplePoint(13.0, 4.0),  SamplePoint(14.0, 3.0),  SamplePoint(14.0, 2.0),
               SamplePoint(16.0, 2.0),  SamplePoint(18.5, 2.0),  SamplePoint(20.5, 2.0),  SamplePoint(21.5, 2.0),
           });
    auto M = T.MRTSearch(references, 0.9);
    auto expected_M1 = std::vector<ReferenceTrajectory>{T2(1, 9)};
    auto expected_M2 = std::vector<ReferenceTrajectory>{T4(6, 14), T5(7, 12)};
    EXPECT_EQ(M[T(2, 9)], expected_M1);
    EXPECT_EQ(M[T(10, 15)], expected_M2);
}
