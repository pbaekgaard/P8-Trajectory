#include <algorithm>
#include <cmath>
#include <vector>
#include "trajectory.hpp"
#include <ranges>

#include "distance.hpp"
#include <cmath>
#include <unordered_map>
#include <iostream>
#if _WIN32
#include <cstdint>
#endif

bool SamplePoint::operator==(const SamplePoint& other) const
{
    return longitude == other.longitude && latitude == other.latitude && timestamp == other.timestamp;
}

Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points): id(id), points(points), start_index(0), end_index(points.size() - 1) {}
Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, const short start_index,
                       const short end_index): id(id), points(points), start_index(start_index), end_index(end_index)
{}

Trajectory Trajectory::operator()(const short start, const short end) const
{
    if (end + 1 > points.size()) {
        return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end), start, end);
    }

    return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
}

Trajectory Trajectory::operator+(const Trajectory& other) const {
    std::vector<SamplePoint> mergedPoints = points;
    mergedPoints.insert(mergedPoints.end(), other.points.begin(), other.points.end());

    // Calculate the correct start and end indices.
    short newStartIndex = start_index;
    short newEndIndex = std::max(end_index, other.end_index);

    return Trajectory(id, mergedPoints, newStartIndex, newEndIndex);
}

bool Trajectory::operator==(const Trajectory& other) const
{
    return (id == other.id && start_index == other.start_index && end_index == other.end_index);
}

bool ReferenceTrajectory::operator==(const ReferenceTrajectory& other) const
{
    return (id == other.id && start_index == other.start_index && end_index == other.end_index);
}

std::size_t std::hash<Trajectory>::operator()(const Trajectory& t) const noexcept
{
    return ((hash<uint32_t>()(t.id) ^ (hash<short>()(t.start_index) << 2) ^ (hash<short>()(t.end_index) >> 1) ^
             (hash<uint32_t>()(t.points.size()) << 1)) >>
            1);
}

ReferenceTrajectory::ReferenceTrajectory(const uint32_t id, const short start_index, const short end_index):
    id(id), start_index(start_index), end_index(end_index)
{}

ReferenceTrajectory::ReferenceTrajectory(const Trajectory& t):
    id(t.id), start_index(t.start_index), end_index(t.end_index)
{}

std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> Trajectory::MRTSearch(std::vector<Trajectory>& RefSet,
                                                                                       const double epsilon)
{
    std::cout << "Running MRTSearch with Epsilon = " << epsilon << std::endl;
    std::unordered_map<Trajectory, std::vector<Trajectory>> M;

    for (auto i = 0; i < points.size() - 1; i++) {
        auto j = i + 1;
        auto current_sub_traj = (*this)(i, j);
        for (auto& ref_trajectory : RefSet) {
            for (size_t length = 2; length <= ref_trajectory.points.size(); ++length) {
                for (size_t k = 0; k <= ref_trajectory.points.size() - length; ++k) {
                    size_t l = k + length - 1;  // Calculate the end index

                    auto ref_sub_traj = ref_trajectory(k, l);

                    if (maxDTW(current_sub_traj, ref_sub_traj) <= epsilon) {
                        std::cout << "Added T" << ref_sub_traj.id << "(" << ref_sub_traj.start_index << "," << ref_sub_traj.end_index << ") to T(" << current_sub_traj.start_index << "," << current_sub_traj.end_index << ")" << std::endl;

                        M[current_sub_traj].push_back(ref_sub_traj);
                    }
                }
            }
        }
    }

    for (auto n = 2; n < points.size() - 1; n++) {
        for (auto k = 0; k + n < points.size() - 1; k++) {
            auto lengthNSubtrajectory = (*this)(k, k + n);
            std::cout << "lengthNSubTrajectory: points:" << lengthNSubtrajectory.points.size() << " start_index:" << lengthNSubtrajectory.start_index << " end_index:" << lengthNSubtrajectory.end_index << std::endl;

            auto T_a_entry = M.find((*this)(k, k+n - 1));
            auto T_b_entry = M.find((*this)(k + n -1, k+n));

            auto T_a_vector = T_a_entry != M.end() ? T_a_entry->second : std::vector<Trajectory>();
            auto T_b_vector = T_b_entry != M.end() ? T_b_entry->second : std::vector<Trajectory>();

            for (auto& T_a : T_a_vector) {
                for (const auto& T_b : T_b_vector) {
                    if (maxDTW(lengthNSubtrajectory, T_a) <= epsilon) {
                        M[lengthNSubtrajectory].push_back(T_a);
                    }

                    if (maxDTW(lengthNSubtrajectory, T_b) <= epsilon) {
                        M[lengthNSubtrajectory].push_back(T_b);
                    }

                    if (T_a.id == T_b.id && T_a.end_index == T_b.start_index) {
                        std::cout << "samesies" << std::endl;
                        M[lengthNSubtrajectory].push_back(T_a + T_b);
                    }
                }
            }
        }
    }
    std::cout << "M.size(): " << M.size() << std::endl;

    std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> M1;
    for (const auto& pair : M) {
        std::vector<ReferenceTrajectory> refTrajectories;
        refTrajectories.reserve(pair.second.size());

        for (const auto& ref_trajectory : pair.second) {
            refTrajectories.emplace_back(ref_trajectory);
        }

        M1[pair.first] = refTrajectories;
    }

    std::vector<std::pair<Trajectory, std::vector<Trajectory>>> vec(M.begin(), M.end());

    // Sort by the Trajectory's id (or whatever other logic you'd like)
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.first.start_index < b.first.start_index;  // or any other sort criteria
    });

    auto test = M.find((*this)(1, 8));
    if (test != M.end()) {
        std::cout << "Found 1.3" << std::endl;
        for (const auto& t : test->second) {  // Access test->second
            std::cout << "T" << t.id << "(" << t.start_index << "," << t.end_index << ")" << std::endl;
        }
    } else {
        std::cout << "didnt find" << std::endl;
    }

    return M1;
}

std::vector<ReferenceTrajectory> Trajectory::OSTC(std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> M)
{
    std::vector<uint16_t> Ft{0};
    std::vector<ReferenceTrajectory> T_prime;

    return T_prime;
}
