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

py::list corrections_to_dict(const std::vector<CompressedResultCorrection> &corrections) {
    py::list result;

    for (const auto &c : corrections) {
        py::dict d;
        d[py::int_(c.id)] = c.corrected_timestamp;
        result.append(d);
    }

    return result;
}
py::object concat_dfs(const std::vector<py::object> &dfs)
{
    py::module_ pd = py::module_::import("pandas");
    py::object concat = pd.attr("concat");
    return concat(dfs);
}

py::object compressed_trajectory_to_dataframe(const std::vector<CompressedResult> &compressed_points)
{
    py::list ids, lats, lons, timestamps, corrections;

    for (const auto &point : compressed_points) {
        ids.append(point.id);
        lats.append(point.latitude);
        lons.append(point.longitude);
        timestamps.append(point.timestamp);
        corrections.append(corrections_to_dict(point.corrections));  // nested list of dicts
    }

    py::dict data;
    data["trajectory_id"] = ids;
    data["latitude"] = lats;
    data["longitude"] = lons;
    data["timestamp"] = timestamps;
    data["timestamp_corrected"] = corrections;

    py::module_ pd = py::module_::import("pandas");
    return pd.attr("DataFrame")(data);
}

void test_compression_to_pandas()
{
    const auto M_opt = std::unordered_map<Trajectory, std::vector<Trajectory>>{
                {t(0, 0), {t(0, 0)}}, {t(1, 8), {t2(0, 7)}}, {t(9, 14), {t5(6, 11)}},
                {t(15, 16), {t6(12, 13)}}, {t(17, 19), {t7(7, 9)}},
            };

    const auto raw_trajectories = std::vector<Trajectory>{t_copy};

    auto points = std::vector<CompressedResult>{};
    for (const auto &raw_traj : raw_trajectories) {
        const OSTCResult compressed = t.OSTC(M_opt, 0.5, 0.9, euclideanDistance);
        convertCompressedTrajectoriesToPoints(points, raw_traj, compressed);
    }
}

std::tuple<py::dict, py::object> build_triples_and_unreferenced_df(std::unordered_map<int, OSTCResult>& compressed_results, py::object& numpy_reference_set)
{
    // returns all triples and full unreferenced_df. Takes input of compressed results (triples before they are made into python syntax). Also updates numpy reference_set with time corrections
    py::dict data;
    py::list ids, lats, lons, timestamps, corrections;
    py::dict all_triples;

    for (const auto& [id, compressed_result]: compressed_results) {
        int counter = 0;
        py::list triples;

        for (const auto &triple : compressed_result.references) {
            if (triple.id == id)
            {
                const auto new_end_index = counter + (triple.end_index - triple.start_index);
                const auto new_start_index = counter;

                for (auto i = new_start_index; i <= new_end_index; i++)
                {
                    ids.append(id);
                    lats.append(triple.points[i].latitude);
                    lons.append(triple.points[i].longitude);
                    timestamps.append(triple.points[i].timestamp);// nested list of dicts
                    corrections.append(py::none());
                    counter++;
                }

                py::list new_triple;
                new_triple.append(py::int_(id));
                new_triple.append(py::int_(new_start_index));
                new_triple.append(py::int_(new_end_index));
                triples.append(new_triple);
            }
            else {

            }
        }
        all_triples[py::int_(id)] = triples;
    }

    data["trajectory_id"] = ids;
    data["latitude"] = lats;
    data["longitude"] = lons;
    data["timestamp"] = timestamps;
    data["timestamp_corrected"] = corrections;

    py::module_ pd = py::module_::import("pandas");
    auto unreferenced_df = pd.attr("DataFrame")(data);

}

py::object build_ref_set_df(std::vector<Trajectory> &ref_trajectories,
std::unordered_map<int, std::unordered_map<int, std::unordered_map<uint32_t, int>>> &used_points_from_ref_set)
{
    py::list ids, lats, lons, timestamps, corrections;

    std::cout << "u_p_f_r_s size : " << used_points_from_ref_set.size() << std::endl;

    std::cout << "u_p_f_r_s : " << std::endl;
    for (auto [id, points_and_corrections] : used_points_from_ref_set) {
        std::cout << "outer id: " << id << std::endl;
        std::cout << "points_and_corrections.size: " << points_and_corrections.size() << std::endl;
        for ( auto [point_id, point_corrections] : points_and_corrections ) {
            std::cout << "point id: " << point_id << std::endl;
        }
    }

    for (const auto &[traj_id, points_and_corrections_map] : used_points_from_ref_set) {
        for (const auto &[point_id, point_corrections] : points_and_corrections_map){

            auto found_ref_traj = std::ranges::find_if(ref_trajectories, [&](const Trajectory &t) {
                return t.id == traj_id;
            });

            if (found_ref_traj != ref_trajectories.end()) {
                ids.append(traj_id);

                auto point = found_ref_traj->points[point_id];
                lats.append(point.latitude);
                lons.append(point.longitude);
                timestamps.append(point.timestamp);

                corrections.append(point_corrections);
            }
            else {
                std::cout << "cannot find for " + std::to_string(traj_id) << " ";
            }
        }
    }

    py::dict data;
    data["trajectory_id"] = ids;
    data["latitude"] = lats;
    data["longitude"] = lons;
    data["timestamp"] = timestamps;
    data["timestamp_corrected"] = corrections;



    py::module_ pd = py::module_::import("pandas");
    return pd.attr("DataFrame")(data);
}
int compute_missing_ids_till_now(int startindex, std::unordered_map<int, std::unordered_map<uint32_t, int>> point_correction_map){
    int count = 0;
    for (int i = 0; i < startindex; i++){
        // if point not found: count++
        auto point_not_found = point_correction_map.find(i) == point_correction_map.end();
        if (point_not_found){
            count++;
        }
    }
    return count;
}


py::object merge_uncompressed_and_ref_set(py::object uncompressed_trajectories_df, py::object ref_set_df) //TODO: maybe needed later, if so it needs to be changed as well
{
    py::module_ pd = py::module_::import("pandas");
    py::object concat = pd.attr("concat");
    return concat(uncompressed_trajectories_df, ref_set_df);
}



py::tuple compress(py::array rawTrajectoryArray, py::array refTrajectoryArray)
{

    //TODO: Delete when work/done testing:)
    /*const auto M = std::unordered_map<Trajectory, std::vector<Trajectory>>{
                    {t(0, 0), {t(0, 0)}}, {t(1, 8), {t2(0, 7)}}, {t(9, 14), {t5(6, 11)}},
                    {t(15, 16), {t6(12, 13)}}, {t(17, 19), {t7(7, 9)}},
                };*/

    // std::vector<Trajectory> rawTrajs {t};
    //Delete to here

    std::vector<Trajectory> rawTrajs = ndarrayToTrajectories(rawTrajectoryArray); // TODO: uncomment this shit when done testing
    std::vector<Trajectory> refTrajs = ndarrayToTrajectories(refTrajectoryArray);

    std::vector<py::object> uncompressed_trajectories_dfs;
    std::vector<OSTCResult> compressedTrajectories{};
    std::vector<py::object> trajectory_dfs{};
    std::unordered_map<int, OSTCResult> all_compressed_results;
    py::dict triples_dict;
    constexpr auto temporal_deviation_threshold = 60;
    auto distance_function = haversine_distance; //TODO: uncomment this shit when done testing
    constexpr auto spatial_deviation_threshold = 20000;
    //auto distance_function = euclideanDistance;

    for (auto t : rawTrajs) {
        std::cout << "compressing Trajectory " << t.id << std::endl;
        std::cout << "performing MRT search" << std::endl;
        const auto M = t.MRTSearch(refTrajs, spatial_deviation_threshold, distance_function); //TODO: uncomment when done testing
        std::cout << "MRT search done" << std::endl;
        std::cout << "performing OSTC" << std::endl;
        OSTCResult compressed = t.OSTC(M, temporal_deviation_threshold, spatial_deviation_threshold, distance_function);
        std::cout << "OSTC done" << std::endl;
        std::cout << "" << std::endl;

        all_compressed_results[t.id] = compressed;
    }

    triples_dict = build_triples_and_unreferenced_df(all_compressed_results, refTrajectoryArray);

    py::object uncompressed_trajectories_df = concat_dfs(uncompressed_trajectories_dfs);
                                                            // TODO: join/merge/concat uncompressed_trajectories med refTrajectoryArray, som er et ndarray. Burde kunne lade sig gøre, fordi uncompressed er en pd.df. kan laves til et ndarray i stedet for speed.
    std::cout << "concat done" << std::endl;
    std::cout << "uncompressed: " << uncompressed_trajectories_df << std::endl;

    auto ref_set_df = build_ref_set_df(refTrajs, used_points_from_ref_set);
    std::cout << "ref set: " << ref_set_df << std::endl;
    std::cout << "build_ref_set_df done" << std::endl;
    triples_dict = build_triples(all_references, used_points_from_ref_set);
    std::cout << "triples: " << triples_dict << std::endl;
    std::cout << "build_triples done" << std::endl;
    std::vector<py::object> merged_dfs = {uncompressed_trajectories_df, ref_set_df};
    py::object merged_df = concat_dfs(merged_dfs);
    std::cout << "concat_dfs done" << std::endl;
    // Update: My query fellas, say concat is good, so we use that
    // If not, we need to change  merge_uncompressed_and_ref_set, and use that one instead.

    return py::make_tuple(triples_dict, ref_set_df);
}


// This function is to demonstrate how you could expose C++ logic to Python

// Binding the functions to Python

PYBIND11_MODULE(ostc, m)
{
    py::class_<CompressedResultCorrection>(m, "CompressedResultCorrection")
            .def(py::init<uint32_t, int>())
            .def_readwrite("id", &CompressedResultCorrection::id)
            .def_readwrite("corrected_timestamp", &CompressedResultCorrection::corrected_timestamp);

    py::class_<CompressedResult>(m, "CompressedResult")
        .def(py::init<uint32_t, double, double, int, std::vector<CompressedResultCorrection>>())
        .def_readwrite("id", &CompressedResult::id)
        .def_readwrite("latitude", &CompressedResult::latitude)
        .def_readwrite("longitude", &CompressedResult::longitude)
        .def_readwrite("timestamp", &CompressedResult::timestamp)
        .def_readwrite("corrections", &CompressedResult::corrections);

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

    m.def("test_compression_to_pandas", &test_compression_to_pandas, R"pbdoc(
        test_compression_to_pandas function

        Boi.
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
