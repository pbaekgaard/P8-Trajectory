#include <gtest/gtest.h>
#include <compress.hpp>
#include <pybind11/embed.h>

// TODO: Check at ndarraytotrajectories virker med rigtigt data. At timestamps ikke bliver lavet om længere.

py::object create_mock_trajectory_data() {
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

    EXPECT_EQ(trajectories.size(), 7);
}

TEST(ndarrayToTrajectories, check_trajectory_point_sizes)
{
    const auto trajectories = ndarrayToTrajectories(create_mock_trajectory_data());

    EXPECT_EQ(lookup_points_size(trajectories, 2), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 3), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 4), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 5), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 7), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 9), 10);
    EXPECT_EQ(lookup_points_size(trajectories, 10), 10);
}

TEST(ndarrayToTrajectories, check_trajectory_point)
{
    const auto trajectories = ndarrayToTrajectories(create_mock_trajectory_data());

    EXPECT_EQ(trajectories[6].id, 2);
    EXPECT_FLOAT_EQ(trajectories[6].points[0].longitude, 116.36422f);
    EXPECT_FLOAT_EQ(trajectories[6].points[0].latitude, 39.88781f);
    EXPECT_EQ(trajectories[6].points[0].timestamp, 1201959232);
}

TEST(compress_test, check_compressed_results) {
    auto mock = create_mock_trajectory_data();
    auto compressed_result = compress(mock, mock);
    EXPECT_EQ(compressed_result.size(), 0);
}