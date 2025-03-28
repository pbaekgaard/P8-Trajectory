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
    Trajectory t1(1, std::vector{SamplePoint(3.0, 16.0, 0), SamplePoint(5.2, 17.5, 1), SamplePoint(6.7, 17.5, 2),
                                 SamplePoint(9.0, 16.0, 3)});
    Trajectory t2(2, std::vector{SamplePoint(4.0, 15.0, 0), SamplePoint(5.0, 15.0, 1), SamplePoint(6.3, 17.4, 2),
                                 SamplePoint(7.8, 17.4, 3)});
    Trajectory t3(3, std::vector{SamplePoint(2.0, 14.0, 0), SamplePoint(4.0, 14.0, 1), SamplePoint(6.0, 14.0, 2),
                                 SamplePoint(7.0, 14.0, 3)});
    Trajectory t4(4, std::vector{SamplePoint(5.0, 17.0, 0), SamplePoint(6.5, 17.0, 1), SamplePoint(8.0, 17.0, 2),
                                 SamplePoint(9.5, 17.0, 3)});
    Trajectory t5(5, std::vector{SamplePoint(3.5, 19.0, 0), SamplePoint(5.5, 19.0, 1), SamplePoint(7.0, 19.0, 2),
                                 SamplePoint(8.0, 19.0, 3)});
    std::vector<Trajectory> Trajectories{t1, t2, t3, t4, t5};
    std::vector<uint32_t> RefSet{0, 1, 1, 3, 3};
    std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> M = t4.MRTSearch(Trajectories, RefSet, 0.9);
    try {
        std::vector<ReferenceTrajectory> T_prime = t4.OSTC(M);
        std::cout << "Compressed trajectory T':\n";
        for (const auto& mrt : T_prime) {
            std::cout << "MRT: (id=" << mrt.id << ", start=" << mrt.start_index << ", end=" << mrt.end_index << ")\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

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
