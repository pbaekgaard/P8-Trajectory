#include <gtest/gtest.h>
#include "trajectory.hpp"
#include "example_trajectories.hpp"

TEST(OSTC, outputs_correct_values)
{
    const auto M_opt = std::unordered_map<Trajectory, std::vector<Trajectory>>{
        {t(0, 0), {t(0, 0)}},      {t(1, 8), {t2(0, 7)}},   {t(9, 14), {t5(6, 11)}},
        {t(15, 16), {t6(12, 13)}}, {t(17, 19), {t7(7, 9)}},
    };

    const OSTCResult compressed = t.OSTC(M_opt, 0.5, 0.9);

    const auto expected_references = std::vector<Trajectory>{t(0, 0), t2(0, 7), t5(6, 11), t6(12, 13), t7(7, 9)};

    EXPECT_EQ(compressed.references, expected_references);

    const auto expected_time_corrections = std::unordered_map<Trajectory, std::vector<TimeCorrectionRecordEntry>>{
        {t7(7, 9), {TimeCorrectionRecordEntry(0, 18)}},
        {t6(12, 13), {TimeCorrectionRecordEntry(0, 16)}},
        {t5(6, 11), {TimeCorrectionRecordEntry(0, 10)}},
        {t2(0, 7), {TimeCorrectionRecordEntry(3, 6), TimeCorrectionRecordEntry(6, 8)}},
    };

    EXPECT_EQ(compressed.time_corrections, expected_time_corrections);
}
