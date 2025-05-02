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
    std::cout << "trajectory output from ndarray" << std::endl;
    for (const auto trajectory  : trajectories) {
        std:: cout << trajectory.id << std::endl;
        for (const auto& point : trajectory.points) {
            std::cout << point << std::endl;
        }
    }

    return trajectories;
}

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

py::object find_uncompressed_trajectory(std::unordered_map<Trajectory, std::vector<TimeCorrectionRecordEntry>> time_cors,
                                        std::vector<Trajectory>& T_prime,
                                        uint32_t id,
                                        std::unordered_map<int, std::unordered_map<int, std::unordered_map<uint32_t, int>>> &used_points_from_ref_set)
{
    py::list ids, lats, lons, timestamps, corrections;
    int counter = 0;
    std::cout << std::endl << std::endl << std::endl;
    std::cout << id << std::endl;


    for (Trajectory &triple : T_prime)
    {
        std::cout << "triple: " <<triple << std::endl;
        if (triple.id == id)
        {
            std::cout << "we in" << std::endl;
            triple.end_index = counter + (triple.end_index - triple.start_index);
            triple.start_index = counter;

            for (auto i = 0; i <= triple.end_index; i++)
            {
                ids.append(id);
                lats.append(triple.points[i].latitude);
                lons.append(triple.points[i].longitude);
                timestamps.append(triple.points[i].timestamp);// nested list of dicts
                corrections.append(py::none());
                counter++;
            }
        }
        else
        {
            auto time_correction = time_cors.find(triple);

            auto found_trajectory_id = used_points_from_ref_set.find(triple.id);
            if (found_trajectory_id != used_points_from_ref_set.end())
            {
                for (auto i = triple.start_index; i <= triple.end_index; i++)
                {
                    // If point already exists only add time_correction
                    auto &traj = found_trajectory_id->second;
                    auto point = traj.find(i);

                    if (point != traj.end()) {
                        if (time_correction != time_cors.end()) {
                            auto time_correction_entries = time_correction->second;
                            auto found_time_correction = std::ranges::find_if(time_correction_entries,
                              [&](const TimeCorrectionRecordEntry &t) {
                                  return t.point_index == i;
                              }
                            );

                            if (found_time_correction != time_correction_entries.end()) {
                                //point->second.push_back(std::make_pair(id, found_time_correction->corrected_timestamp));
                                point->second[id] = found_time_correction->corrected_timestamp;
                            }
                        }
                    }

                    else {
                        // If point does not already exist add entire point and time_correction
                        if (time_correction != time_cors.end()) {
                            auto time_correction_entries = time_correction->second;
                            auto found_time_correction = std::ranges::find_if(time_correction_entries,
                              [&](const TimeCorrectionRecordEntry &t) {
                                  return t.point_index == i;
                              }
                            );

                            if (found_time_correction != time_correction_entries.end()) {
                                //traj[i].push_back(std::make_pair(id, found_time_correction->corrected_timestamp));
                                traj[i][id] = found_time_correction->corrected_timestamp;
                            }
                        }
                    }
                }
            }
            else
            {
                std::unordered_map<int, std::unordered_map<uint32_t, int>> new_traj;
                for (auto i = triple.start_index; i <= triple.end_index; i++)
                {
                    if (time_correction != time_cors.end()) {
                        auto time_correction_entries = time_correction->second;
                        auto found_time_correction = std::ranges::find_if(time_correction_entries,
                          [&](const TimeCorrectionRecordEntry &t) {
                              return t.point_index == i;
                          }
                        );

                        if (found_time_correction != time_correction_entries.end()) {
                            //new_traj[i].push_back(std::make_pair(id, found_time_correction->corrected_timestamp));
                            new_traj[i][id] = found_time_correction->corrected_timestamp;
                        }
                        else {
                            new_traj[i][id] = {};
                        }
                    }
                }
                used_points_from_ref_set[triple.id] = new_traj;
            }
        }
    }
    py::dict data;

    std::cout << "size ids: " << ids.size() << std::endl;
    data["trajectory_id"] = ids;
    data["latitude"] = lats;
    data["longitude"] = lons;
    data["timestamp"] = timestamps;
    data["timestamp_corrected"] = corrections;

    py::module_ pd = py::module_::import("pandas");
    return pd.attr("DataFrame")(data);
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
py::dict build_triples(std::unordered_map<int, std::vector<Trajectory>> all_compressed,
                       std::unordered_map<int, std::unordered_map<int, std::unordered_map<uint32_t, int>>> used_points_from_ref_set) {
    py::dict triples_dict;
    for(const auto &[traj_id, compressed] : all_compressed) {
        py::list triple_list;
        for (const auto &compressed_traj : compressed)
        {
            auto missing_ids_till_now = compute_missing_ids_till_now(compressed_traj.start_index, used_points_from_ref_set[traj_id]);
            triple_list.append(py::make_tuple(compressed_traj.id,compressed_traj.start_index - missing_ids_till_now, compressed_traj.end_index - missing_ids_till_now));
        }
        triples_dict[py::int_(traj_id)] = triple_list;
    }

    return triples_dict;
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
    std::cout << "we not done it       yet" << std::endl;
    std::vector<Trajectory> rawTrajs = ndarrayToTrajectories(rawTrajectoryArray); // TODO: uncomment this shit when done testing
    std::vector<Trajectory> refTrajs = ndarrayToTrajectories(refTrajectoryArray);
    std::cout << "rawTrajs size: " << rawTrajs.size() << std::endl;
    std::cout << "refTrajs size: " << refTrajs.size() << std::endl;
    std::vector<py::object> uncompressed_trajectories_dfs;
    std::vector<OSTCResult> compressedTrajectories{};
    std::vector<py::object> trajectory_dfs{};
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<uint32_t, int>>> used_points_from_ref_set{};
    std::unordered_map<int, std::vector<Trajectory>> all_references;
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

        all_references[t.id] = compressed.references;

        for (auto &traj : compressed.references) {
            std::cout << "adding Trajectory " << traj.id << std::endl;
        }

        py::object uncompressed_trajectory = find_uncompressed_trajectory(compressed.time_corrections, compressed.references, t.id, used_points_from_ref_set);
        std::cout << "uncompressed: " << uncompressed_trajectory << std::endl;
        uncompressed_trajectories_dfs.push_back(uncompressed_trajectory);
    }
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
