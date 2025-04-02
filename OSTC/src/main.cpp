#include "trajectory.hpp"
#include "distance.hpp"
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <iomanip>

namespace py = pybind11;
// Custom format descriptor for std::tuple<int, std::string, double, double>
void say_hello() { std::cout << "Hello From C++!" << std::endl; }

std::vector<Trajectory> ndarrayToTrajectories(py::object array)
{
    // Convert the Python object to a list of lists
    auto py_list = array.cast<py::list>();
    std::vector<Trajectory> trajectories;
    std::unordered_map<int, std::vector<SamplePoint>> traject_dict;
    for (const auto& row_handle : py_list) {
        auto row = row_handle.cast<py::list>();  // Cast to py::list
        int id = row[0].cast<int>();
        auto timestamp = row[1].cast<std::string>();
        auto latitude = row[2].cast<float>();
        auto longitude = row[3].cast<float>();
        auto point = SamplePoint(latitude, longitude, timestamp);
        traject_dict[id].push_back(point);
    }
    for (const auto& [id, points] : traject_dict) {
        trajectories.push_back(Trajectory(id, points));
    }
    return trajectories;
}

void print_numpy(py::object array)
{
    // Convert the Python object to a list of lists
    std::vector<Trajectory> traj = ndarrayToTrajectories(array);
    std::cout << traj[0].points[0] << std::endl;
}

// This function is to demonstrate how you could expose C++ logic to Python
void run_example()
{
    std::unordered_map<Trajectory, std::vector<Trajectory>> M;

    Trajectory t1(1, std::vector{SamplePoint(3, 15.5, "ts"), SamplePoint(5, 15.5, "ts"), SamplePoint(7, 15.5, "ts"),
                                 SamplePoint(8.5, 15.5, "ts")});

    Trajectory sub_t1 = t1(0, 1);
    Trajectory sub_t2 = t1(2, 3);
    Trajectory sub_t3 = t1(2, 3);

    auto euc = euclideanDistance(sub_t2.points.back(), sub_t3.points.back());
    std::cout << "euc = " << euc << std::endl;

    auto test = maxDtw(sub_t1, sub_t3);
    std::cout << "MaxDTW = " << test << std::endl;
}

// Binding the functions to Python

PYBIND11_MODULE(ostc, m)
{
    m.doc() = R"pbdoc(
        OSTC Module
        -----------------------

        .. currentmodule:: OSTC

        .. autosummary::
           :toctree: _generate

           say_hello
           run_example
    )pbdoc";

    m.def("say_hello", &say_hello, R"pbdoc(
        Print Hello

        Some other explanation about the say_hello function.
    )pbdoc");
    m.def("print_ndarray", &print_numpy, R"pbdoc(
        run_example function

        Some other explanation about the run_example function.
    )pbdoc");

    m.def("run_example", &run_example, R"pbdoc(
        run_example function

        Some other explanation about the run_example function.
    )pbdoc");
}
