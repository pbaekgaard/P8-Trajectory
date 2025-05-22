#include <distance.hpp>

#include "trajectory.hpp"
#include <unordered_map>
#include <iostream>
#include "example_trajectories.hpp"
#include <vector>
#ifdef Debug




void run_example()
{
    constexpr auto spatial_deviation_threshold = 1.0;
    constexpr auto temporal_deviation_threshold = 0.5;
    std::vector<Trajectory> RefSet{t1, t2, t3(2, 14), t4, t5, t6, t7};
    auto distance_function = euclideanDistance;

    const auto M = t.MRTSearch(RefSet, spatial_deviation_threshold, euclideanDistance);
    OSTCResult compressed = t.OSTC(M,temporal_deviation_threshold, spatial_deviation_threshold, euclideanDistance);
}
#endif
#ifndef Debug
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <compress.hpp>

namespace py = pybind11;
// Custom format descriptor for std::tuple<int, std::string, double, double>
void say_hello() { std::cout << "Hello From C++!" << std::endl; }

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

    m.def("compress", &compress, R"pbdoc(
        Compress a list of trajectories to the reference set

        Some other explanation about the run_example function.
    )pbdoc");

    /*
    m.def("test_compression_to_pandas", &test_compression_to_pandas, R"pbdoc(
        test_compression_to_pandas function

        Boi.
    )pbdoc");
    */
}

#endif


#ifdef Debug
int main()
{
    run_example();
    return 0;
}
#endif
