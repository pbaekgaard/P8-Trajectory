#include "trajectory.hpp"
#include <unordered_map>
#include <iostream>
#include <string>

void run_example()
{
    Trajectory t(0, std::vector<SamplePoint>{
        SamplePoint(3, 15.5, 0),    SamplePoint(5, 15.5, 1),   SamplePoint(7, 15.5, 2),
        SamplePoint(8.5, 15.5, 3),  SamplePoint(9.5, 15.5, 4), SamplePoint(10, 15.5, 5),
        SamplePoint(11.5, 15.5, 6), SamplePoint(12, 14, 7),    SamplePoint(12, 12, 8),
        SamplePoint(12, 11, 9),     SamplePoint(12, 10, 10),   SamplePoint(12, 8, 11),
        SamplePoint(12, 5.5, 12),   SamplePoint(13, 4, 13),    SamplePoint(14, 3, 14),
        SamplePoint(14, 2, 15),     SamplePoint(16, 2, 16),    SamplePoint(18.5, 2, 17),
        SamplePoint(20.5, 2, 18),   SamplePoint(21.5, 2, 19)
    });

    Trajectory t1(1, std::vector<SamplePoint>{
        SamplePoint(2, 2.5, 10),    SamplePoint(1.5, 3, 11),   SamplePoint(1.5, 4, 12),
        SamplePoint(1.5, 5.5, 13),  SamplePoint(1.5, 7, 14),   SamplePoint(1.5, 8.5, 15),
        SamplePoint(1.5, 9.5, 16),  SamplePoint(1.5, 10.5, 17), SamplePoint(1.5, 12, 18),
        SamplePoint(1.5, 13, 19),   SamplePoint(2, 14, 20),    SamplePoint(3, 14.55, 21),
        SamplePoint(5, 14.55, 22),  SamplePoint(7, 14.55, 23)
    });


    Trajectory t2(2, std::vector<SamplePoint>{
        SamplePoint(5, 16, 20),     SamplePoint(7.5, 16, 21),  SamplePoint(8.5, 16, 22),
        SamplePoint(9.5, 16, 23),   SamplePoint(12, 15.5, 24), SamplePoint(12.5, 14.5, 25),
        SamplePoint(12.5, 13.5, 26), SamplePoint(12.5, 12, 27), SamplePoint(14, 11.5, 28),
        SamplePoint(15, 11, 29),    SamplePoint(16.5, 11, 30), SamplePoint(17.5, 11, 31),
        SamplePoint(18.5, 11, 32),  SamplePoint(19.5, 11.5, 33), SamplePoint(19.5, 12, 34),
        SamplePoint(19.5, 13.5, 35), SamplePoint(19.5, 15, 36)
    });

    Trajectory t3(3, std::vector<SamplePoint>{
        SamplePoint(5.5, 14, 30),   SamplePoint(7, 14, 31),    SamplePoint(8, 14.5, 32),
        SamplePoint(10, 14.5, 33),  SamplePoint(11, 14, 34),   SamplePoint(11.5, 12, 35),
        SamplePoint(9.5, 12, 36),   SamplePoint(8, 12, 37),    SamplePoint(6, 12, 38),
        SamplePoint(4.5, 12, 39),   SamplePoint(3.5, 12, 40),  SamplePoint(2.5, 11, 41),
        SamplePoint(2, 10.5, 42),   SamplePoint(2, 9, 43),     SamplePoint(2, 8, 44)
    });

    Trajectory t4(4, std::vector<SamplePoint>{
        SamplePoint(5.58, 11, 40),  SamplePoint(6.58, 11, 41), SamplePoint(7.57, 11, 42),
        SamplePoint(8.58, 11, 43),  SamplePoint(9.55, 11, 44), SamplePoint(11.5, 11, 45),
        SamplePoint(11.5, 10, 46),  SamplePoint(11.5, 8.5, 47), SamplePoint(11.5, 8, 48),
        SamplePoint(11.5, 6, 49),   SamplePoint(11.5, 5.11, 50), SamplePoint(12.5, 4.1, 51),
        SamplePoint(13, 3.5, 52),   SamplePoint(13.5, 3, 53),  SamplePoint(13.5, 2, 54),
        SamplePoint(13.5, 1, 55),   SamplePoint(12.5, 1, 56),  SamplePoint(10.5, 1, 57),
        SamplePoint(8.5, 1, 58)
    });

    Trajectory t5(5, std::vector<SamplePoint>{
        SamplePoint(20.5, 13, 50),  SamplePoint(19, 13, 51),   SamplePoint(17.5, 13, 52),
        SamplePoint(16, 13, 53),    SamplePoint(15, 13, 54),   SamplePoint(14, 12.5, 55),
        SamplePoint(12.5, 11, 56),  SamplePoint(12.5, 10, 57), SamplePoint(12.5, 8, 58),
        SamplePoint(12.5, 6, 59),   SamplePoint(13.5, 4.5, 60), SamplePoint(14.5, 3.5, 61)
    });

    Trajectory t6(6, std::vector<SamplePoint>{
        SamplePoint(2, 6, 60),      SamplePoint(2, 4.5, 61),   SamplePoint(2, 3.5, 62),
        SamplePoint(2.5, 2.5, 63),  SamplePoint(3.5, 2.5, 64), SamplePoint(4.5, 2.5, 65),
        SamplePoint(5.5, 2.5, 66),  SamplePoint(6.5, 2.5, 67), SamplePoint(7.5, 2.5, 68),
        SamplePoint(9, 2.5, 69),    SamplePoint(11, 2.5, 70),  SamplePoint(12, 2.5, 71),
        SamplePoint(14.5, 2.5, 72), SamplePoint(16, 2.5, 73)
    });

    Trajectory t7(7, std::vector<SamplePoint>{
        SamplePoint(15.5, 6.5, 70), SamplePoint(15.5, 6, 71),  SamplePoint(15.5, 5, 72),
        SamplePoint(16.5, 5, 73),   SamplePoint(17.5, 5, 74),  SamplePoint(17.5, 4, 75),
        SamplePoint(18, 3, 76),     SamplePoint(18, 2.5, 77),  SamplePoint(20, 2.5, 78),
        SamplePoint(22, 2.5, 79)
    });
    std::vector<Trajectory> RefSet{t1, t2, t3(2, 14), t4, t5, t6, t7};
    auto M = t.MRTSearch(RefSet, 0.9);
    t.OSTC(M, 60);
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
