#include "trajectory.hpp"
#include <unordered_map>
#include <iostream>
#include "example_trajectories.hpp"

void run_example()
{
    std::vector<Trajectory> RefSet{t1, t2, t3(2, 14), t4, t5, t6, t7};
    auto M = t.MRTSearch(RefSet, 0.9);
    OSTCResult ostc_res = t.OSTC(M, 15, 0.9);
    std::cout << M.size() << std::endl;
    // try {
    //     std::vector<ReferenceTrajectory> T_prime = t.OSTC(M);
    //     std::cout << "Compressed trajectory T':\n";
    //     for (const auto& mrt : T_prime) {
    //         std::cout << "MRT: (id=" << mrt.id << ", start=" << mrt.start_index << ", end=" << mrt.end_index <<
    //         ")\n";
    //     }
    // } catch (const std::exception& e) {
    //     std::cerr << "Error: " << e.what() << "\n";
    // }
}
#ifndef Debug
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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

#endif

#ifdef Debug
int main()
{
    std::cout << "Hello from Debug Main" << std::endl;
    run_example();
    return 0;
}
#endif
