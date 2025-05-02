#pragma once
#include <trajectory.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::vector<Trajectory> ndarrayToTrajectories(py::object array)
{
    // Convert the Python object to a list of lists
    auto py_list = array.cast<py::list>();
    std::vector<Trajectory> trajectories;
    std::unordered_map<int, std::vector<SamplePoint>> traject_dict;
    for (const auto& row_handle : py_list) {
        auto row = row_handle.cast<py::list>();  // Cast to py::list
        int id = row[0].cast<float>();
        auto timestamp = row[1].cast<float>();
        auto longitude = row[2].cast<float>();
        auto latitude = row[3].cast<float>();
        auto point = SamplePoint(latitude, longitude, timestamp);

        traject_dict[id].push_back(point);
    }
    for (const auto& [id, points] : traject_dict) {
        trajectories.push_back(Trajectory(id, points));
    }

    return trajectories;
}