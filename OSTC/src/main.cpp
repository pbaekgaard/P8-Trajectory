#include <distance.hpp>

#include "trajectory.hpp"
#include <unordered_map>
#include <iostream>
#include "example_trajectories.hpp"
#ifdef Debug




void run_example()
{
    constexpr auto spatial_deviation_threshold = 0.9;
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
    for (const auto &row_handle : py_list) {
        auto row = row_handle.cast<py::list>();  // Cast to py::list
        int id = row[0].cast<int>();
        auto timestamp = row[1].cast<std::string>();
        auto latitude = row[2].cast<float>();
        auto longitude = row[3].cast<float>();
        auto point = SamplePoint(latitude, longitude, 0);
        traject_dict[id].push_back(point);
    }
    for (const auto &[id, points] : traject_dict) {
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
                                        std::unordered_map<int, std::unordered_map<int, std::vector<std::pair<uint32_t, int>>>> &used_points_from_ref_set)
{
    py::list ids, lats, lons, timestamps, corrections;
    int counter = 0;

    for (Trajectory &triple : T_prime)
    {
        if (triple.id == id)
        {
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
                for (auto i = 0; i <= triple.end_index; i++)
                {
                    // TODO: find out if point exists in this traj and add entire point or just time_correction
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
                                point->second.push_back(std::make_pair(id, found_time_correction->corrected_timestamp));
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
                                traj[i].push_back(std::make_pair(id, found_time_correction->corrected_timestamp));
                            }
                        }
                    }
                }
            }
            else
            {
                std::unordered_map<int, std::vector<std::pair<uint32_t, int>>> new_traj;
                for (auto i = 0; i <= triple.end_index; i++)
                {
                    if (time_correction != time_cors.end()) {
                        auto time_correction_entries = time_correction->second;
                        auto found_time_correction = std::ranges::find_if(time_correction_entries,
                          [&](const TimeCorrectionRecordEntry &t) {
                              return t.point_index == i;
                          }
                        );

                        if (found_time_correction != time_correction_entries.end()) {
                            new_traj[i].push_back(std::make_pair(id, found_time_correction->corrected_timestamp));
                        }
                    }
                }
                used_points_from_ref_set[triple.id] = new_traj;
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

py::object build_ref_set_df(std::unordered_map<int, std::vector<Trajectory>> &all_references,
                            std::unordered_map<int, std::unordered_map<int, std::vector<std::pair<uint32_t, int>>>> &used_points_from_ref_set)
{
    py::list ids, lats, lons, timestamps, corrections;

    for (const auto &[traj_id, points_and_corrections_map] : used_points_from_ref_set) {
        auto ref_traj_vector = all_references.find(traj_id);
        if (ref_traj_vector != all_references.end())
        {
            for (const auto &traj: ref_traj_vector->second)
            {
                for (int i = 0; i < traj.points.size(); i++)
                {
                    py::dict correction_dict;
                    ids.append(traj_id);
                    lats.append(traj.points[i].latitude);
                    lons.append(traj.points[i].longitude);
                    timestamps.append(traj.points[i].timestamp);

                    auto point_iter = points_and_corrections_map.find(i);
                    if (point_iter != points_and_corrections_map.end()) {
                        auto time_corrections = point_iter->second;

                        for (const auto &correction_pair : time_corrections) {
                            correction_dict[py::int_(correction_pair.first)] = py::int_(correction_pair.second);
                        }
                        corrections.append(correction_dict);
                    }
                    else {
                        corrections.append(py::none());
                    }
                }
            }
        }
        else {
            throw std::invalid_argument("Cannot find reference trajectories for " + std::to_string(traj_id));
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
void build_list_of_triples(uint32_t id, std::vector<Trajectory> references, py::dict &triples_dict) {
    py::list triple_list;
    for (const auto &ref_traj : references)
    {
        triple_list.append(py::make_tuple(ref_traj.id,ref_traj.start_index, ref_traj.end_index));
    }
    triples_dict[py::int_(id)] = triple_list;
}
py::object merge_uncompressed_and_ref_set(py::object uncompressed_trajectories_df, py::object ref_set_df)
{

}
py::tuple compress(py::object rawTrajectoryArray, py::object refTrajectoryArray)
{

    //TODO: Delete when work :)
    const auto M = std::unordered_map<Trajectory, std::vector<Trajectory>>{
                    {t(0, 0), {t(0, 0)}}, {t(1, 8), {t2(0, 7)}}, {t(9, 14), {t5(6, 11)}},
                    {t(15, 16), {t6(12, 13)}}, {t(17, 19), {t7(7, 9)}},
                };

    std::vector<Trajectory> rawTrajs {t};
    //Delete to here
    //TODO: Uncomment below when done testing
    // std::vector<Trajectory> rawTrajs = ndarrayToTrajectories(rawTrajectoryArray);
    //std::vector<Trajectory> refTrajs = ndarrayToTrajectories(refTrajectoryArray);
    std::vector<py::object> uncompressed_trajectories_dfs;
    std::vector<OSTCResult> compressedTrajectories{};
    std::vector<py::object> trajectory_dfs{};
    std::unordered_map<int, std::unordered_map<int, std::vector<std::pair<uint32_t, int>>>> used_points_from_ref_set{};
    std::unordered_map<int, std::vector<Trajectory>> all_references;
    py::dict triples_dict;
    constexpr auto spatial_deviation_threshold = 0.9;
    constexpr auto temporal_deviation_threshold = 0.5;
    // auto distance_function = haversine_distance; //TODO: uncomment this shit when done testing
    auto distance_function = euclideanDistance;

    // TODO: return tuple of dfs. <compressed results, df2>. compressed results is the alle the trajectories compressed. can be df or vector of tuples. df2 is the merged df of the original df and the reference set df.
    for (auto t : rawTrajs) {
        std::cout << "compressing Trajectory " << t.id << std::endl;
        std::cout << "performing MRT search" << std::endl;
        // const auto M = t.MRTSearch(refTrajs, spatial_deviation_threshold, distance_function); //TODO: uncomment when done testing
        std::cout << "MRT search done" << std::endl;
        std::cout << "performing OSTC" << std::endl;
        OSTCResult compressed = t.OSTC(M, temporal_deviation_threshold, spatial_deviation_threshold, distance_function);
        std::cout << "OSTC done" << std::endl;

        py::object uncompressed_trajectory = find_uncompressed_trajectory(compressed.time_corrections, compressed.references, t.id, used_points_from_ref_set);
        uncompressed_trajectories_dfs.push_back(uncompressed_trajectory);
        all_references[t.id] = compressed.references;
        build_list_of_triples(t.id, compressed.references, triples_dict);

    }
    py::object uncompressed_trajectories_df = concat_dfs(uncompressed_trajectories_dfs); // TODO: check uncompressed_trajectories er rigtige.
                                                            // TODO: join/merge/concat uncompressed_trajectories med refTrajectoryArray, som er et ndarray. Burde kunne lade sig gøre, fordi uncompressed er en pd.df. kan laves til et ndarray i stedet for speed.



    auto ref_set_df = build_ref_set_df(all_references, used_points_from_ref_set);

    py::object merged_df = merge_uncompressed_and_ref_set(uncompressed_trajectories_df, ref_set_df); // TODO: merge those bitches, somehow


    return py::make_tuple(triples_dict, ref_set_df);   // TODO: return en liste af compressed trajectories og så den mergede df
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
