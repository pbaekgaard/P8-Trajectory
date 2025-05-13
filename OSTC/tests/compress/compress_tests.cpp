#include <gtest/gtest.h>
#include <compress.hpp>
#include <pybind11/embed.h>

// TODO: Check at ndarraytotrajectories virker med rigtigt data. At timestamps ikke bliver lavet om længere.

auto create_mock_trajectory_data() {
    py::list data;

    // Each row: id, timestamp, lon, lat, extra -> we'll ignore extra
    std::vector<std::tuple<int, int, float, float>> rows = {
        {2, 1201959232, 116.36422f, 39.88781f},
        {2, 1201959436, 116.37481f, 39.88782f},
        {2, 1201959533, 116.37677f, 39.88791f},
        {2, 1201959738, 116.38033f, 39.88795f},
        {2, 1201959835, 116.39392f, 39.89014f},
        {2, 1201960040, 116.41149f, 39.89152f},
        {2, 1201960137, 116.42105f, 39.89194f},
        {2, 1201960341, 116.4215f, 39.89823f},
        {2, 1201960438, 116.4215f, 39.89902f},
        {2, 1201960643, 116.42972f, 39.90726f},
        {3, 1201959548, 116.35743f, 39.88957f},
        {3, 1201959848, 116.35732f, 39.89726f},
        {3, 1201960149, 116.3506f,  39.90712f},
        {3, 1201960449, 116.35171f, 39.91145f},
        {3, 1201960748, 116.34366f, 39.89655f},
        {3, 1201961048, 116.34347f, 39.87605f},
        {3, 1201961348, 116.35298f, 39.87643f},
        {3, 1201961648, 116.36223f, 39.88766f},
        {3, 1201961948, 116.37556f, 39.88786f},
        {3, 1201962248, 116.37818f, 39.88815f},
        {4, 1201965304, 116.47002f, 39.90666f},
        {4, 1201965904, 116.44422f, 39.92078f},
        {4, 1201966504, 116.4344f,  39.92296f},
        {4, 1201967103, 116.43988f, 39.92189f},
        {4, 1201967704, 116.47187f, 39.91212f},
        {4, 1201968305, 116.47448f, 39.90664f},
        {4, 1201968904, 116.47333f, 39.91435f},
        {4, 1201969505, 116.46395f, 39.92896f},
        {4, 1201970105, 116.47235f, 39.92161f},
        {4, 1201970704, 116.45444f, 39.91262f},
        {5, 1201959784, 116.62934f, 39.82726f},
        {5, 1201960384, 116.62934f, 39.82725f},
        {5, 1201960984, 116.62933f, 39.82725f},
        {5, 1201961584, 116.61582f, 39.82817f},
        {5, 1201962188, 116.59818f, 39.82816f},
        {5, 1201962784, 116.62944f, 39.82734f},
        {5, 1201963384, 116.62946f, 39.82743f},
        {5, 1201963984, 116.62945f, 39.82746f},
        {5, 1201964584, 116.62944f, 39.82746f},
        {5, 1201965184, 116.62949f, 39.82738f},
        {7, 1201965026, 116.76038f, 39.79758f},
        {7, 1201965349, 116.7666f,  39.8027f},
        {7, 1201981618, 116.7666f,  39.8027f},
        {7, 1201981978, 116.7522f,  39.80078f},
        {7, 1201982338, 116.72105f, 39.81482f},
        {7, 1202022778, 116.70168f, 39.8301f},
        {7, 1202022785, 116.70168f, 39.8301f},
        {7, 1202024105, 116.6892f,  39.8266f},
        {7, 1202024825, 116.71878f, 39.82333f},
        {7, 1202025185, 116.73115f, 39.79662f},
        {9, 1201959522, 116.37412f, 39.99295f},
        {9, 1201959882, 116.36432f, 39.99978f},
        {9, 1201960242, 116.3611f,  39.99237f},
        {9, 1201960602, 116.3638f,  39.97307f},
        {9, 1201963297, 116.35667f, 39.98615f},
        {9, 1201964017, 116.44587f, 40.00275f},
        {9, 1201964392, 116.46187f, 40.00365f},
        {9, 1201964752, 116.45647f, 39.98852f},
        {9, 1201965112, 116.46242f, 39.98458f},
        {9, 1201965475, 116.478f,   39.98205f},
        {10, 1201959123, 116.44457f, 39.92157f},
        {10, 1201959238, 116.44043f, 39.9219f},
        {10, 1201959265, 116.4404f,  39.92192f},
        {10, 1201959308, 116.43528f, 39.9228f},
        {10, 1201959363, 116.43523f, 39.92287f},
        {10, 1201959418, 116.42965f, 39.92307f},
        {10, 1201959482, 116.42955f, 39.92313f},
        {10, 1201959500, 116.42842f, 39.92335f},
        {10, 1201959507, 116.42767f, 39.92328f},
        {10, 1201959515, 116.42667f, 39.92317f},
    };

    for (const auto& [id, ts, lon, lat] : rows) {
        py::list row;
        row.append(id);
        row.append(ts);
        row.append(lon);
        row.append(lat);
        data.append(row);
    }

    return py::object(data);
}

auto create_mock_reference_trajectory_data() {
    py::list data;

    // Each row: id, timestamp, lon, lat (ignoring extras)
    std::vector<std::tuple<int, int, float, float>> rows = {
        // Trajectory ID 1
        {1, 1201956968, 116.51172f, 39.92123f},
        {1, 1201961168, 116.51135f, 39.93883f},
        {1, 1201961168, 116.51135f, 39.93883f},
        {1, 1201965368, 116.51627f, 39.91034f},
        {1, 1201969568, 116.47186f, 39.91248f},
        {1, 1201973768, 116.47217f, 39.92498f},
        {1, 1201977968, 116.47179f, 39.90718f},
        {1, 1201982168, 116.45617f, 39.90531f},
        {1, 1201983624, 116.47191f, 39.90577f},
        {1, 1201984224, 116.50661f, 39.91450f},

        // Updated Trajectory ID 6
        {6, 1201957112, 116.43453f, 39.92262f},
        {6, 1201957472, 116.44723f, 39.92572f},
        {6, 1201957832, 116.4493f,  39.93550f},
        {6, 1201958192, 116.44238f, 39.94755f},
        {6, 1201958552, 116.45507f, 39.94780f},
        {6, 1201958912, 116.4438f,  39.95938f},
        {6, 1201959272, 116.42597f, 39.96815f},
        {6, 1201959632, 116.41008f, 39.96773f},
        {6, 1201959992, 116.3752f,  39.97650f},
        {6, 1201960352, 116.38145f, 39.98073f},

        // Trajectory ID 8
        {8, 1202108473, 116.36606f, 39.68410f},
        {8, 1202194895, 116.36606f, 39.68409f},
        {8, 1202281320, 116.36605f, 39.68409f},
        {8, 1202368965, 116.36611f, 39.68408f},
        {8, 1202457637, 116.36615f, 39.68413f}
    };

    for (const auto& [id, ts, lon, lat] : rows) {
        py::list row;
        row.append(id);
        row.append(ts);
        row.append(lon);
        row.append(lat);
        data.append(row);
    }

    return py::object(data);
}


size_t lookup_points_size(const std::vector<Trajectory>& trajectories, const int trajectory_id)
{
    const auto found_traj = std::ranges::find_if(trajectories,
              [&](const Trajectory& t) {
                  return t.id == trajectory_id;
              }
    );
    return found_traj->points.size();
}

py::scoped_interpreter guard{};

TEST(ndarrayToTrajectories, check_trajectory_size)
{
    const auto trajectories = ndarrayToTrajectories(create_mock_trajectory_data());
    const auto refTrajectories = ndarrayToTrajectories(create_mock_reference_trajectory_data());

    EXPECT_EQ(trajectories.size(), 7);
    EXPECT_EQ(refTrajectories.size(), 3);
}

TEST(ndarrayToTrajectories, check_trajectory_point_sizes)
{
    const auto trajectories = ndarrayToTrajectories(create_mock_trajectory_data());
    const auto refTrajectories = ndarrayToTrajectories(create_mock_reference_trajectory_data());

    EXPECT_EQ(lookup_points_size(trajectories, 2), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 3), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 4), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 5), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 7), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 9), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 10), 10);

    EXPECT_EQ(lookup_points_size(refTrajectories, 1), 10);
    EXPECT_EQ(lookup_points_size(refTrajectories, 6), 10);
    EXPECT_EQ(lookup_points_size(refTrajectories, 8), 5);
}

TEST(ndarrayToTrajectories, check_trajectory_point)
{
    const auto trajectories = ndarrayToTrajectories(create_mock_trajectory_data());

    EXPECT_EQ(trajectories[6].id, 2);
    EXPECT_FLOAT_EQ(trajectories[6].points[0].longitude, 116.36422f);
    EXPECT_FLOAT_EQ(trajectories[6].points[0].latitude, 39.88781f);
    EXPECT_EQ(trajectories[6].points[0].timestamp, 1201959232);
}


TEST(Compress, MRTSearch)
{
    auto trajectories = ndarrayToTrajectories(create_mock_trajectory_data());
    auto refTrajectories = ndarrayToTrajectories(create_mock_reference_trajectory_data());

    constexpr auto temporal_deviation_threshold = 60;
    constexpr auto spatial_deviation_threshold = 200;
    auto distance_function = haversine_distance;

    std::unordered_map<uint32_t, OSTCResult> all_compressed_results;

    // small deviation threshold

    for (auto trajectory : trajectories) {
        const auto M = trajectory.MRTSearch(refTrajectories, spatial_deviation_threshold, distance_function); //TODO: uncomment when done testing
        OSTCResult compressed = trajectory.OSTC(M, temporal_deviation_threshold, spatial_deviation_threshold, distance_function);

        EXPECT_TRUE(compressed.references.size() > 0);
        EXPECT_TRUE(compressed.time_corrections.size() == 0);

        all_compressed_results[trajectory.id] = compressed;
    }

    for (auto compressedResult : all_compressed_results) {
        EXPECT_EQ(compressedResult.second.references.size(), 1);
    }

    all_compressed_results.clear();


    // large deviation threshold

    auto M = trajectories[0].MRTSearch(refTrajectories, 20000, distance_function); //TODO: uncomment when done testing
    OSTCResult compressed = trajectories[0].OSTC(M, temporal_deviation_threshold, 20000, distance_function);

    EXPECT_EQ(compressed.references[0].id, 6);
}

TEST(MRT_VS_MRTOptimized, threshold_200)
{
    auto trajectories = ndarrayToTrajectories(create_mock_trajectory_data());
    auto refTrajectories = ndarrayToTrajectories(create_mock_reference_trajectory_data());

    constexpr auto temporal_deviation_threshold = 60;
    constexpr auto spatial_deviation_threshold = 200;
    auto distance_function = haversine_distance;

    float duration_MRTSearch = 0;
    float duration_MRTSearchOptimized = 0;


    for (auto trajectory : trajectories) {
        auto start_MRTSearch = std::chrono::high_resolution_clock::now();
        const auto M = trajectory.MRTSearch(refTrajectories, spatial_deviation_threshold, distance_function);
        duration_MRTSearch += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_MRTSearch).count();

        auto start_MRTSearchOptimized = std::chrono::high_resolution_clock::now();
        const auto M_Optimized = trajectory.MRTSearchOptimized(refTrajectories, spatial_deviation_threshold, distance_function);
        duration_MRTSearchOptimized += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_MRTSearchOptimized).count();

        EXPECT_EQ(M, M_Optimized);
    }

    std::cout << "duration_MRT: " << duration_MRTSearch << ", duration_MRT_Optimized: " << duration_MRTSearchOptimized << std::endl;
    EXPECT_TRUE(duration_MRTSearch - duration_MRTSearchOptimized > -3);

}

TEST(MRT_VS_MRTOptimized, threshold_20000)
{
    auto trajectories = ndarrayToTrajectories(create_mock_trajectory_data());
    auto refTrajectories = ndarrayToTrajectories(create_mock_reference_trajectory_data());

    constexpr auto temporal_deviation_threshold = 60;
    constexpr auto spatial_deviation_threshold = 20000;
    auto distance_function = haversine_distance;

    float duration_MRTSearch = 0;
    float duration_MRTSearchOptimized = 0;


    for (auto trajectory : trajectories) {
        auto start_MRTSearch = std::chrono::high_resolution_clock::now();
        const auto M = trajectory.MRTSearch(refTrajectories, spatial_deviation_threshold, distance_function);
        duration_MRTSearch += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_MRTSearch).count();

        auto start_MRTSearchOptimized = std::chrono::high_resolution_clock::now();
        const auto M_Optimized = trajectory.MRTSearchOptimized(refTrajectories, spatial_deviation_threshold, distance_function);
        duration_MRTSearchOptimized += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_MRTSearchOptimized).count();

        EXPECT_EQ(M, M_Optimized);
    }

    std::cout << "duration_MRT: " << duration_MRTSearch << ", duration_MRT_Optimized: " << duration_MRTSearchOptimized << std::endl;
    EXPECT_TRUE(duration_MRTSearch - duration_MRTSearchOptimized > -3);

}