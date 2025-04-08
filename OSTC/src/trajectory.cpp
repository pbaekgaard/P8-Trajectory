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

Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points): id(id), points(points) {}
Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, const short start_index,
                       const short end_index): id(id), points(points), start_index(start_index), end_index(end_index)
{}

Trajectory Trajectory::operator()(const short start, const short end)
{
    if (end + 1 > points.size()) {
        return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end), start, end);
    }

    return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
}

Trajectory Trajectory::operator+(Trajectory other)
{
    auto mergedPoints = points;
    std::copy(other.points.begin()+1, other.points.end(), std::back_inserter(mergedPoints));

    return Trajectory(id, mergedPoints, start_index, other.end_index);
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
    std::unordered_map<Trajectory, std::vector<Trajectory>> M;

    for (auto i = 0; i < points.size()-1; i++) {
        auto j = i + 1;
        auto current_sub_traj = (*this)(i, j);
        for (auto& ref_trajectory : RefSet) {
            for (auto k = 0; k < ref_trajectory.points.size()-1; k++) {
                auto l = k + 1;
                auto ref_sub_traj = ref_trajectory(k, l);

                if (maxDTW(current_sub_traj, ref_sub_traj) < epsilon) {
                    M[current_sub_traj].push_back(ref_sub_traj);
                }
            }
        }
    }

    for (auto n = 3; n < points.size(); n++) {
        for (auto k = 0; k + n - 1 < points.size(); k++) {
            auto lengthNSubtrajectory = (*this)(k, k + n - 1);

            auto T_a_entry = M.find((*this)(lengthNSubtrajectory.start_index, lengthNSubtrajectory.end_index - 1));
            auto T_b_entry = M.find((*this)(lengthNSubtrajectory.end_index - 1, lengthNSubtrajectory.end_index));


            auto T_a_vector = T_a_entry != M.end() ? T_a_entry->second : std::vector<Trajectory>();
            auto T_b_vector = T_b_entry != M.end() ? T_b_entry->second : std::vector<Trajectory>();

            for (const auto& T_a : T_a_vector) {
                if (maxDTW(lengthNSubtrajectory, T_a) <= epsilon) {
                    M[lengthNSubtrajectory].push_back(T_a);
                }
            }

            for (const auto& T_b : T_b_vector) {
                if (maxDTW(lengthNSubtrajectory, T_b) <= epsilon) {
                    M[lengthNSubtrajectory].push_back(T_b);
                }
            }

            for (auto& T_a : T_a_vector) {
                for (const auto& T_b : T_b_vector) {
                    if (T_a.id == T_b.id && T_a.end_index == T_b.start_index) {
                        M[lengthNSubtrajectory].push_back(T_a + T_b);
                    }
                }
            }
        }
    }

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
    auto found = M1.find((*this)(2,9));
    if (found != M1.end() && !found->second.empty()) {
        for (const auto f : found->second) {
            std::cout<< "id: " << f.id << " Start: " <<f.start_index << " end: " << f.end_index<< std::endl;
        }
    }
    return M1;
}

std::vector<ReferenceTrajectory> Trajectory::OSTC(std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> M)
{
    std::vector<int> Ft(points.size(), 0);
    std::vector<int> pre(points.size(), -1);
    std::vector<ReferenceTrajectory> T_prime {};
    for (short i=0; i<points.size(); i++) {
        //i og j skal måske være 1 før det virker, pga Ft[0] = 0 i pseudo koden.
        //Men så skal trajectories 0-indexeres, og så skal fx mrt search ændres også
        int min = 8*points.size(); //obs på om denne narrowing giver fejl
        for (short j=0; j<i; j++) { // maybe stop condition giver fejl sowwy
            auto key = (*this)(j, i);
            auto it = M.find(key);

            if (it != M.end() && !it -> second.empty()) {
                if (j>0 && Ft[j-1] + 8 < min) {
                    min = Ft[j-1] + 8;
                    pre[i] = j - 1;
                }
            }
        }
        Ft[i] = min;
    }
    auto i = points.size();
    while (i>0 && i<= points.size()) {

    }
    return T_prime;
}

