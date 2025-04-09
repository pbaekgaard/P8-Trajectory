#include "trajectory.hpp"
#include <unordered_map>
#include <iostream>
#include <string>

void run_example()
{
    Trajectory t(0, std::vector<SamplePoint>{
                        SamplePoint(3, 15.5, "ts"),    SamplePoint(5, 15.5, "ts"),   SamplePoint(7, 15.5, "ts"),
                        SamplePoint(8.5, 15.5, "ts"),  SamplePoint(9.5, 15.5, "ts"), SamplePoint(10, 15.5, "ts"),
                        SamplePoint(11.5, 15.5, "ts"), SamplePoint(12, 14, "ts"),    SamplePoint(12, 12, "ts"),
                        SamplePoint(12, 11, "ts"),     SamplePoint(12, 10, "ts"),    SamplePoint(12, 8, "ts"),
                        SamplePoint(12, 5.5, "ts"),    SamplePoint(13, 4, "ts"),     SamplePoint(14, 3, "ts"),
                        SamplePoint(14, 2, "ts"),      SamplePoint(16, 2, "ts"),     SamplePoint(18.5, 2, "ts"),
                        SamplePoint(20.5, 2, "ts"),    SamplePoint(21.5, 2, "ts")});

    Trajectory t1(1, std::vector<SamplePoint>{
                         SamplePoint(2, 2.5, "ts"), SamplePoint(1.5, 3, "ts"), SamplePoint(1.5, 4, "ts"),
                         SamplePoint(1.5, 5.5, "ts"), SamplePoint(1.5, 7, "ts"), SamplePoint(1.5, 8.5, "ts"),
                         SamplePoint(1.5, 9.5, "ts"), SamplePoint(1.5, 10.5, "ts"), SamplePoint(1.5, 12, "ts"),
                         SamplePoint(1.5, 13, "ts"), SamplePoint(2, 14, "ts"), SamplePoint(3, 14.55, "ts"),
                         SamplePoint(5, 14.55, "ts"), SamplePoint(7, 14.55, "ts")});

    Trajectory t2(2, std::vector<SamplePoint>{
                         SamplePoint(5, 16, "ts"),
                         SamplePoint(7.5, 16, "ts"),
                         SamplePoint(8.5, 16, "ts"),
                         SamplePoint(9.5, 16, "ts"),
                         SamplePoint(12, 15.5, "ts"),
                         SamplePoint(12.5, 14.5, "ts"),
                         SamplePoint(12.5, 13.5, "ts"),
                         SamplePoint(12.5, 12, "ts"),
                         SamplePoint(14, 11.5, "ts"),
                         SamplePoint(15, 11, "ts"),
                         SamplePoint(16.5, 11, "ts"),
                         SamplePoint(17.5, 11, "ts"),
                         SamplePoint(18.5, 11, "ts"),
                         SamplePoint(19.5, 11.5, "ts"),
                         SamplePoint(19.5, 12, "ts"),
                         SamplePoint(19.5, 13.5, "ts"),
                         SamplePoint(19.5, 15, "ts"),
                     });

    Trajectory t3(3, std::vector<SamplePoint>{
                         SamplePoint(5.5, 14, "ts"),
                         SamplePoint(7, 14, "ts"),
                         SamplePoint(8, 14.5, "ts"),
                         SamplePoint(10, 14.5, "ts"),
                         SamplePoint(11, 14, "ts"),
                         SamplePoint(11.5, 12, "ts"),
                         SamplePoint(9.5, 12, "ts"),
                         SamplePoint(8, 12, "ts"),
                         SamplePoint(6, 12, "ts"),
                         SamplePoint(4.5, 12, "ts"),
                         SamplePoint(3.5, 12, "ts"),
                         SamplePoint(2.5, 11, "ts"),
                         SamplePoint(2, 10.5, "ts"),
                         SamplePoint(2, 9, "ts"),
                         SamplePoint(2, 8, "ts"),
                     });

    Trajectory t4(4, std::vector<SamplePoint>{
                         SamplePoint(5.58, 11, "ts"), SamplePoint(6.58, 11, "ts"),   SamplePoint(7.57, 11, "ts"),
                         SamplePoint(8.58, 11, "ts"), SamplePoint(9.55, 11, "ts"),   SamplePoint(11.5, 11, "ts"),
                         SamplePoint(11.5, 10, "ts"), SamplePoint(11.5, 8.5, "ts"),  SamplePoint(11.5, 8, "ts"),
                         SamplePoint(11.5, 6, "ts"),  SamplePoint(11.5, 5.11, "ts"), SamplePoint(12.5, 4.1, "ts"),
                         SamplePoint(13, 3.5, "ts"),  SamplePoint(13.5, 3, "ts"),    SamplePoint(13.5, 2, "ts"),
                         SamplePoint(13.5, 1, "ts"),  SamplePoint(12.5, 1, "ts"),    SamplePoint(10.5, 1, "ts"),
                         SamplePoint(8.5, 1, "ts"),
                     });

    Trajectory t5(5, std::vector<SamplePoint>{
                         SamplePoint(20.5, 13, "ts"), SamplePoint(19, 13, "ts"), SamplePoint(17.5, 13, "ts"),
                         SamplePoint(16, 13, "ts"), SamplePoint(15, 13, "ts"), SamplePoint(14, 12.5, "ts"),
                         SamplePoint(12.5, 11, "ts"), SamplePoint(12.5, 10, "ts"), SamplePoint(12.5, 8, "ts"),
                         SamplePoint(12.5, 6, "ts"), SamplePoint(13.5, 4.5, "ts"), SamplePoint(14.5, 3.5, "ts")});

    Trajectory t6(6, std::vector<SamplePoint>{
                         SamplePoint(2, 6, "ts"), SamplePoint(2, 4.5, "ts"), SamplePoint(2, 3.5, "ts"),
                         SamplePoint(2.5, 2.5, "ts"), SamplePoint(3.5, 2.5, "ts"), SamplePoint(4.5, 2.5, "ts"),
                         SamplePoint(5.5, 2.5, "ts"), SamplePoint(6.5, 2.5, "ts"), SamplePoint(7.5, 2.5, "ts"),
                         SamplePoint(9, 2.5, "ts"), SamplePoint(11, 2.5, "ts"), SamplePoint(12, 2.5, "ts"),
                         SamplePoint(14.5, 2.5, "ts"), SamplePoint(16, 2.5, "ts")});

    Trajectory t7(
        7, std::vector<SamplePoint>{SamplePoint(15.5, 6.5, "ts"), SamplePoint(15.5, 6, "ts"),
                                    SamplePoint(15.5, 5, "ts"), SamplePoint(16.5, 5, "ts"), SamplePoint(17.5, 5, "ts"),
                                    SamplePoint(17.5, 4, "ts"), SamplePoint(18, 3, "ts"), SamplePoint(18, 2.5, "ts"),
                                    SamplePoint(20, 2.5, "ts"), SamplePoint(22, 2.5, "ts")});
    std::vector<Trajectory> RefSet{t1, t2, t3(2, 14), t4, t5, t6, t7};
    std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> M = t.MRTSearch(RefSet, 1);
    t.OSTC(M);
    std::cout << M.size() << std::endl;
    // try {
    //     std::vector<ReferenceTrajectory> T_prime = t.OSTC(M);
    //     std::cout << "Compressed trajectory T':\n";
    //     for (const auto& mrt : T_prime) {
    //         std::cout << "MRT: (id=" << mrt.id << ", start=" << mrt.start_index << ", end=" << mrt.end_index << ")\n";
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
