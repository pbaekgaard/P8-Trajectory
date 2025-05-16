#ifndef __COMPRESS_HPP__
#define __COMPRESS_HPP__

#include <distance.hpp>
#include <example_trajectories.hpp>
#include <chrono>
#include <trajectory.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<Trajectory> ndarrayToTrajectories(py::object array)
{
    // Convert the Python object to a list of lists
    auto py_list = array.cast<py::list>();
    std::vector<Trajectory> trajectories;
    std::unordered_map<int, std::vector<SamplePoint>> traject_dict;
    for (const auto& row_handle : py_list) {
        auto row = row_handle.cast<py::list>();  // Cast to py::list
        int id = row[0].cast<int>();
        auto timestamp = row[1].cast<int>();
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

std::unordered_map<int, int> pyDictToIds(py::object dict)
{
    py::dict ids = dict.cast<py::dict>();

    std::unordered_map<int, int> result;

    for (const auto& item : ids) {
        int key = item.first.cast<int>();
        int value = item.second.cast<int>();
        result[key] = value;
    }

    return result;
}


struct ReferenceSetMapKey
{
    uint32_t trajectory_id;
    int point_index;

    bool operator==(const ReferenceSetMapKey& other) const
    {
        return trajectory_id == other.trajectory_id && point_index == other.point_index;
    }

    ReferenceSetMapKey(const uint32_t trajectory_id, const int point_index) : trajectory_id(trajectory_id), point_index(point_index) {}
};

template <>
struct std::hash<ReferenceSetMapKey>
{
    std::size_t operator()(const ReferenceSetMapKey& t) const noexcept;
};

std::size_t std::hash<ReferenceSetMapKey>::operator()(const ReferenceSetMapKey& r) const noexcept
{
    return (hash<uint32_t>()(r.trajectory_id) ^ (hash<int>()(r.point_index) << 2));
}

using reference_set_map_t = std::unordered_map<ReferenceSetMapKey, std::tuple<Trajectory, std::vector<TimeCorrectionRecordEntry>>>;

reference_set_map_t build_reference_set_map(const std::vector<Trajectory>& reference_set)
{
    reference_set_map_t reference_set_map;

    for (const auto &ref_set : reference_set) {
        for (int i = 0; i < ref_set.points.size(); i++) {
            reference_set_map[{ref_set.id, i}] = {ref_set, {}};
        }
    }

    return reference_set_map;
}

py::object reference_map_to_df(reference_set_map_t& reference_set_map)
{
    py::dict result;
    py::list ids, lats, lons, timestamps, corrections;

    for (const auto& [ref_key, ref_data] : reference_set_map) {
        auto trajectory = std::get<0>(ref_data);
        auto time_corrections = std::get<1>(ref_data);

        auto point = trajectory.points[ref_key.point_index];

        ids.append(ref_key.trajectory_id);
        lats.append(point.latitude);
        lons.append(point.longitude);
        timestamps.append(point.timestamp);

        auto correction = std::ranges::find_if(time_corrections,
            [&](const TimeCorrectionRecordEntry& correction_record) {
                return correction_record.point_index == ref_key.point_index;
            }
        );

        py::dict corrections_dict = {};
        if (correction != time_corrections.end()) {
            corrections_dict[py::int_(correction->trajectory_id)] = py::int_(correction->corrected_timestamp);
        }
        corrections.append(corrections_dict);
    }

    result["trajectory_id"] = ids;
    result["latitude"] = lats;
    result["longitude"] = lons;
    result["timestamp"] = timestamps;
    result["timestamp_corrected"] = corrections;

    py::module_ pd = py::module_::import("pandas");
    auto reference_set_df = pd.attr("DataFrame")(result);

    return reference_set_df;
}

std::tuple<py::dict, py::object> build_triples_and_unreferenced_df(std::unordered_map<uint32_t, OSTCResult>& compressed_results, reference_set_map_t& reference_set_map)
{
    py::dict unreferenced_dict;
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

                for (auto i = 0; i <= (triple.end_index - triple.start_index); i++)
                {
                    ids.append(py::int_(id));
                    lats.append(py::float_(triple.points[i].latitude));
                    lons.append(py::float_(triple.points[i].longitude));
                    timestamps.append(py::int_(triple.points[i].timestamp));// nested list of dicts
                    corrections.append(py::dict{});
                    counter++;
                }

                py::list new_triple;
                new_triple.append(py::int_(id));
                new_triple.append(py::int_(new_start_index));
                new_triple.append(py::int_(new_end_index));
                triples.append(new_triple);
            }
            else {
                auto has_time_correction = compressed_result.time_corrections.find(triple);

                if (has_time_correction != compressed_result.time_corrections.end()) {
                    auto time_corrections = has_time_correction->second;
                    for (const auto& time_correction : time_corrections) {
                        auto has_reference = reference_set_map.find({triple.id, time_correction.point_index});
                        if (has_reference != reference_set_map.end()) {
                            auto& reference = has_reference->second;
                            auto& correction_vector = std::get<1>(reference);

                            correction_vector.emplace_back(time_correction);
                        }
                    }
                }

                py::list new_triple;
                new_triple.append(py::int_(triple.id));
                new_triple.append(py::int_(triple.start_index));
                new_triple.append(py::int_(triple.end_index));
                triples.append(new_triple);
            }
        }
        all_triples[py::int_(id)] = triples;
    }

    unreferenced_dict["trajectory_id"] = ids;
    unreferenced_dict["latitude"] = lats;
    unreferenced_dict["longitude"] = lons;
    unreferenced_dict["timestamp"] = timestamps;
    unreferenced_dict["timestamp_corrected"] = corrections;

    py::module_ pd = py::module_::import("pandas");
    auto unreferenced_df = pd.attr("DataFrame")(unreferenced_dict);

    return {all_triples, unreferenced_df};
}

py::object concat_dfs(const std::vector<py::object> &dfs)
{
    py::module_ pd = py::module_::import("pandas");
    py::object concat = pd.attr("concat");
    return concat(dfs);
}

py::tuple compress(py::array rawTrajectoryArray, py::array refTrajectoryArray, py::dict refIds)
{
    //TODO: Delete when work/done testing:)
    /*const auto M = std::unordered_map<Trajectory, std::vector<Trajectory>>{
                    {t(0, 0), {t(0, 0)}}, {t(1, 8), {t2(0, 7)}}, {t(9, 14), {t5(6, 11)}},
                    {t(15, 16), {t6(12, 13)}}, {t(17, 19), {t7(7, 9)}},
                };

    std::vector<Trajectory> rawTrajs {t};
    std::vector<Trajectory> refTrajs {t1,t2,t3,t4,t5,t6,t7};
    */
    //Delete to here

    std::vector<Trajectory> rawTrajs = ndarrayToTrajectories(rawTrajectoryArray); // TODO: uncomment this shit when done testing
    std::vector<Trajectory> refTrajs = ndarrayToTrajectories(refTrajectoryArray);
    std::unordered_map<int, int> ref_ids = pyDictToIds(refIds);

    std::vector<py::object> uncompressed_trajectories_dfs;
    std::vector<OSTCResult> compressedTrajectories{};
    std::vector<py::object> trajectory_dfs{};
    std::unordered_map<uint32_t, OSTCResult> all_compressed_results;
    constexpr auto temporal_deviation_threshold = 60;
    //constexpr auto temporal_deviation_threshold = 0.5;
    auto distance_function = haversine_distance; //TODO: uncomment this shit when done testing
    constexpr auto spatial_deviation_threshold = 200; // TODO: Change to 200 when running actual compression
    //constexpr auto spatial_deviation_threshold = 0.9;
    //auto distance_function = euclideanDistance;
    float duration_MRTSearch = 0;
    float duration_OSTC = 0;


    for (auto t : rawTrajs) {
        auto ref_trajectory_id = ref_ids[t.id];
        auto ref_trajectory = std::ranges::find_if(refTrajs, [&](const Trajectory &ref_traj) {
            return ref_trajectory_id == ref_traj.id;
        });
        std::vector<Trajectory> ref_trajectories = std::vector<Trajectory>{*ref_trajectory};
        auto start_MRTSearch = std::chrono::high_resolution_clock::now();
        const auto M = t.MRTSearchOptimized(ref_trajectories, spatial_deviation_threshold, distance_function);
        duration_MRTSearch += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_MRTSearch).count();
        auto start_OSTC = std::chrono::high_resolution_clock::now();
        OSTCResult compressed = t.OSTC(M, temporal_deviation_threshold, spatial_deviation_threshold, distance_function);
        duration_OSTC += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_OSTC).count();

        all_compressed_results[t.id] = compressed;
    }

    auto reference_set_map = build_reference_set_map(refTrajs);
    auto [triples_dict, unreferenced_df] = build_triples_and_unreferenced_df(all_compressed_results, reference_set_map);

    auto reference_map_df = reference_map_to_df(reference_set_map);
    std::vector<py::object> merged_dfs = {reference_map_df, unreferenced_df};
    py::object merged_df = concat_dfs(merged_dfs);
    // Duration_MRTSearch and duration_OSTC is returned in milliseconds
    return py::make_tuple(triples_dict, merged_df, py::float_(duration_MRTSearch), py::float_(duration_OSTC / 1000));
}
#endif