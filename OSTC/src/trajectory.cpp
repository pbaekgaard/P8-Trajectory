#include <algorithm>
#include <cmath>
#include <vector>
#include "trajectory.hpp"
#include <ranges>
#include "distance.hpp"

#include <omp.h>
#include <functional>
#include <unordered_map>
#include <map>
#include <ranges>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
#if _WIN32
#include <cstdint>
#endif


bool SamplePoint::operator==(const SamplePoint& other) const
{
    return longitude == other.longitude && latitude == other.latitude && timestamp == other.timestamp;
}

Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points): id(id), points(points), start_index(0), end_index(points.size()-1) {}

Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, const int start_index,
                       const int end_index): id(id), points(points), start_index(start_index), end_index(end_index)
{}

bool Trajectory::operator<(const Trajectory& other) const
{
    if (id == other.id) {
        return start_index < other.start_index;
    }
    return id < other.id;

}


Trajectory Trajectory::operator()(const int start, const int end)
{
    if (end + 1 > points.size()) {
        return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end), start, end);
    }

    return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
}

Trajectory Trajectory::operator()(const int start, const int end) const
{
    if (end + 1 > points.size()) {
        return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end), start, end);
    }

    return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
}

Trajectory Trajectory::operator+(Trajectory other)
{
    auto mergedPoints = points;
    std::copy(other.points.begin() + 1, other.points.end(), std::back_inserter(mergedPoints));

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
    return ((hash<uint32_t>()(t.id) ^ (hash<int>()(t.start_index) << 2) ^ (hash<int>()(t.end_index) >> 1) ^
             (hash<uint32_t>()(t.points.size()) << 1)) >>
            1);
}

ReferenceTrajectory::ReferenceTrajectory(const uint32_t id, const short start_index, const short end_index):
    id(id), start_index(start_index), end_index(end_index)
{}

ReferenceTrajectory::ReferenceTrajectory(const Trajectory& t):
    id(t.id), start_index(t.start_index), end_index(t.end_index)
{}

std::unordered_map<Trajectory, std::vector<Trajectory>> Trajectory::MRTSearch(std::vector<Trajectory>& RefSet,
                                                                             const double epsilon,
                                                                             std::function<double(SamplePoint const& a, SamplePoint const& b)> distance_function)
{

    std::cout << "loop 1. doubles" << std::endl;
     std::unordered_map<Trajectory, std::vector<Trajectory>> M;
    for (int i = 0; i < points.size() - 1; i++) {
        Trajectory subtraj = (*this)(i, i + 1);

        for (auto refTraj : RefSet) {
            for (int j = 0; j < refTraj.points.size() - 1; j++) {
                for (int k = j + 1; k < refTraj.points.size(); k++) {
                    Trajectory subRefTraj = refTraj(j, k);
                    if (MaxDTW(subtraj, subRefTraj, distance_function) <= epsilon) {
                        subRefTraj.start_index = subRefTraj.start_index + refTraj.start_index;
                        subRefTraj.end_index = subRefTraj.end_index + refTraj.start_index;
                        M[subtraj].emplace_back(subRefTraj);
                    }
                }

            }
        }
    }
    std::cout << "loop 2. n-tuples" << std::endl;
    for (int n = 2; n < points.size(); n++) {
        std::cout << "loop 3. n.tuples. n is " << n << std::endl;
        auto found = false;
        for (int i = 0, j = i + n; j <= points.size() - 1; i++, j++) {

            Trajectory sub_left = (*this)(i, j - 1);
            Trajectory sub_right = (*this)(j - 1, j);

            auto T_a_vec = M.find(sub_left);
            auto T_b_vec = M.find(sub_right);

            if (T_a_vec != M.end() && T_b_vec != M.end()) {
                const std::vector<Trajectory>& T_as = T_a_vec->second;
                const std::vector<Trajectory>& T_bs = T_b_vec->second;

                for (Trajectory a : T_as) {

                    for (auto& b : T_bs) {
                        if (a.id == b.id && a.end_index == b.start_index) {
                            M[(*this)(i,j)].emplace_back(a + b);
                            found = true;
                        }
                    }
                }
            }
            if (T_a_vec != M.end()) {
                auto T_as = T_a_vec->second;
                for (auto& a : T_as) {
                    if (MaxDTW((*this)(i,j), a, distance_function) <= epsilon) {
                        M[(*this)(i,j)].emplace_back(a);
                        found = true;
                    }
                }
            }
            if (T_b_vec != M.end()) {
                auto T_bs = T_b_vec->second;
                for (auto& b : T_bs) {
                    if (MaxDTW((*this)(i,j), b, distance_function) <= epsilon) {
                        M[(*this)(i,j)].emplace_back(b);
                        found = true;
                    }
                }
            }
        }

        if (!found) break;
    }

    std::vector<TrajectoryRemoval> to_remove;

    std::cout << "loop 3. ref trajectories." << std::endl;

    for (auto& [query_traj, ref_trajs] : M) {
        auto query_start_index = query_traj.start_index;
        auto query_end_index = query_traj.end_index;

        for (auto i = query_start_index; i <= query_end_index - 1; i++) {
            for (auto j = i + 1; j <= query_end_index; j++) {
                if (i == query_start_index && j == query_end_index) {
                    continue;
                }

                auto ref_iterator = M.find((*this)(i, j));
                if (ref_iterator != M.end()) {
                    auto& ref_trajectories = ref_iterator->second;
                    for (auto& ref_trajectory : ref_trajectories) {
                        auto is_sub_trajectory = std::ranges::find_if(ref_trajs,
                            [&](const Trajectory& ref_traj) {
                            return ref_trajectory.id == ref_traj.id && ref_trajectory.start_index >= ref_traj.start_index && ref_trajectory.end_index <= ref_traj.end_index;
                        }) != ref_trajs.end();

                        if (is_sub_trajectory) {
                            to_remove.push_back(TrajectoryRemoval{(*this)(i, j), ref_trajectory});
                        }
                    }

                }
            }
        }
    }
    std::cout << "loop 4. removals" << std::endl;
    for (auto& removal : to_remove) {
        auto& ref_trajectory_to_remove = removal.trajectory_to_remove;
        auto iter = M.find(removal.query_trajectory);
        if (iter == M.end()) {
            continue;
        }

        auto& ref_trajectories = iter->second;
        std::erase_if(ref_trajectories, [&](const Trajectory& ref_traj) { return ref_traj == ref_trajectory_to_remove; });

        if (ref_trajectories.size() == 0) {
            M.erase(removal.query_trajectory);
        }
    }

    std::cout << "loop 5. M." << std::endl;
    for (auto& [query_traj, ref_trajs] : M) {
        std::unordered_set<Trajectory> seen;
        ref_trajs.erase(std::remove_if(ref_trajs.begin(), ref_trajs.end(),
                       [&seen](Trajectory x){
                           return !seen.insert(x).second;   // true  ⇒ duplicate
                       }),
        ref_trajs.end());
    }

    return M;
}


std::unordered_map<Trajectory, std::vector<Trajectory>> Trajectory::MRTSearchOptimized(std::vector<Trajectory>& RefSet,
                                                                             const double epsilon,
                                                                             std::function<double(SamplePoint const& a, SamplePoint const& b)> distance_function)
{
    std::cout << "loop 1. doubles" << std::endl;
    std::unordered_map<Trajectory, std::vector<Trajectory>> M;
    std::cout << "Size of RefSet: " << RefSet.size() << std::endl;
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < points.size() - 1; i++) {
            Trajectory subtraj = (*this)(i, i + 1);

            for (auto& refTraj : RefSet) {
                for (int j = 0; j < refTraj.points.size() - 1; j++) {
                    for (int k = j + 1; k < refTraj.points.size(); k++) {
                        Trajectory subRefTraj = refTraj(j, k);
                        if (MaxDTW(subtraj, subRefTraj, distance_function) <= epsilon) {
                            subRefTraj.start_index += refTraj.start_index;
                            subRefTraj.end_index += refTraj.start_index;

                            // Protect shared map with critical section
                            #pragma omp critical
                            {
                                M[subtraj].emplace_back(subRefTraj);
                            }
                        }
                    }
                }
            }
        }
    }
    // #pragma omp parallel for schedule(dynamic)
    // for (int i = 0; i < points.size() - 1; i++) {
    //     Trajectory subtraj = (*this)(i, i + 1);
    //
    //     for (auto& refTraj : RefSet) {
    //         for (int j = 0; j < refTraj.points.size() - 1; j++) {
    //             for (int k = j + 1; k < refTraj.points.size(); k++) {
    //                 Trajectory subRefTraj = refTraj(j, k);
    //                 if (MaxDTW(subtraj, subRefTraj, distance_function) <= epsilon) {
    //                     subRefTraj.start_index += refTraj.start_index;
    //                     subRefTraj.end_index += refTraj.start_index;
    //
    //                     // Protect shared map with critical section
    //                     #pragma omp critical
    //                     {
    //                         M[subtraj].emplace_back(subRefTraj);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    std::cout << "loop 2. n-tuples" << std::endl;
    for (int n = 2; n < points.size(); n++) {
        auto found = false;
        std::cout << "loop 3. n.tuples. n is " << n << std::endl;
        for (int i = 0, j = i + n; j <= points.size() - 1; i++, j++) {
            Trajectory sub_left = (*this)(i, j - 1);
            Trajectory sub_right = (*this)(j - 1, j);

            auto T_a_vec = M.find(sub_left);
            auto T_b_vec = M.find(sub_right);
            std::unordered_set<Trajectory> seen;

            if (T_a_vec != M.end() && T_b_vec != M.end()) {
                const std::vector<Trajectory>& T_as = T_a_vec->second;
                const std::vector<Trajectory>& T_bs = T_b_vec->second;

                for (Trajectory a : T_as) {
                    for (auto& b : T_bs) {
                        if (a.id == b.id && a.end_index == b.start_index) {
                            if (seen.find(a+b) == seen.end()) {
                                seen.insert(a+b);
                                M[(*this)(i,j)].emplace_back(a + b);
                            }
                            found = true;
                        }
                    }
                }
            }
            if (T_a_vec != M.end()) {
                auto T_as = T_a_vec->second;
                for (auto& a : T_as) {
                    if (MaxDTW((*this)(i,j), a, distance_function) <= epsilon) {
                        if (seen.find(a) == seen.end()) {
                            seen.insert(a);
                            M[(*this)(i,j)].emplace_back(a);
                        }
                        found = true;
                    }
                }
            }
            if (T_b_vec != M.end()) {
                auto T_bs = T_b_vec->second;
                for (auto& b : T_bs) {
                    if (MaxDTW((*this)(i,j), b, distance_function) <= epsilon) {
                        if (seen.find(b) == seen.end()) {
                            seen.insert(b);
                            M[(*this)(i,j)].emplace_back(b);
                        }
                        found = true;
                    }
                }
            }
        }

        if (!found) break;
    }

    std::vector<TrajectoryRemoval> to_remove;

    std::cout << "loop 4. ref trajectories." << std::endl;

    for (auto& [query_traj, ref_trajs] : M) {
        auto query_start_index = query_traj.start_index;
        auto query_end_index = query_traj.end_index;

        for (auto i = query_start_index; i <= query_end_index - 1; i++) {
            for (auto j = i + 1; j <= query_end_index; j++) {
                if (i == query_start_index && j == query_end_index) {
                    continue;
                }

                auto ref_iterator = M.find((*this)(i, j));
                if (ref_iterator != M.end()) {
                    auto& ref_trajectories = ref_iterator->second;
                    for (auto& ref_trajectory : ref_trajectories) {
                        auto is_sub_trajectory = std::ranges::find_if(ref_trajs,
                            [&](const Trajectory& ref_traj) {
                            return ref_trajectory.id == ref_traj.id && ref_trajectory.start_index >= ref_traj.start_index && ref_trajectory.end_index <= ref_traj.end_index;
                        }) != ref_trajs.end();

                        if (is_sub_trajectory) {
                            to_remove.push_back(TrajectoryRemoval{(*this)(i, j), ref_trajectory});
                        }
                    }

                }
            }
        }
    }

    std::cout << "loop 5. removals" << std::endl;
    for (auto& removal : to_remove) {
        auto& ref_trajectory_to_remove = removal.trajectory_to_remove;
        auto iter = M.find(removal.query_trajectory);
        if (iter == M.end()) {
            continue;
        }

        auto& ref_trajectories = iter->second;
        std::erase_if(ref_trajectories, [&](const Trajectory& ref_traj) { return ref_traj == ref_trajectory_to_remove; });

        if (ref_trajectories.size() == 0) {
            M.erase(removal.query_trajectory);
        }
    }

    std::cout << "loop 6. M." << std::endl;
    for (auto& [query_traj, ref_trajs] : M) {
        std::unordered_set<Trajectory> seen;
        ref_trajs.erase(std::remove_if(ref_trajs.begin(), ref_trajs.end(),
                       [&seen](Trajectory x){
                           return !seen.insert(x).second;   // true  ⇒ duplicate
                       }),
        ref_trajs.end());
    }

    return M;
}


OSTCResult Trajectory::OSTC(std::unordered_map<Trajectory, std::vector<Trajectory>> M, const double tepsilon, const double sepsilon, std::function<double(SamplePoint const& a, SamplePoint const& b)> distance_function)
{
    std::unordered_map<Trajectory, int> time_correction_cost{};
    std::unordered_map<Trajectory, std::vector<TimeCorrectionRecordEntry>> time_correction_record{};
    auto c = 4;

    for (auto i = 0; i < points.size(); i++) {
        auto subtraj = (*this)(i,i);
        auto TT = M.find(subtraj);
        if (TT == M.end()) {
            M[subtraj].emplace_back(subtraj);
        }
    }

    for (auto& MRT : M) {
        auto ref = MRT.second[0];
        auto a = MRT.first.points;
        auto b = ref.points;

        signed int t = 0;
        time_correction_cost[ref] = 0;

        for (int i = 0; i <= b.size() - 1; i++) {
            auto a_i = a[i];
            auto b_i = b[i];

            if (i+1 < a.size() && distance_function(b_i, a[i+1]) < sepsilon)
                a_i = a[i+1];

            auto previousTimeStamp = i == 0 ? 0 : b[i-1].timestamp;
            signed int leftside = abs(t + b_i.timestamp - previousTimeStamp - a_i.timestamp);
            if (leftside <= tepsilon) {
                t = t + b_i.timestamp - previousTimeStamp;
            } else {
                t = std::max(a_i.timestamp, previousTimeStamp);
                time_correction_cost[ref] += c;
                time_correction_record[ref].emplace_back(MRT.first.id, ref.start_index + i, t);
            }
        }
    }
    std::vector<int> Ft(points.size() + 1, 0);      // +1 for F_T[0] = 0
    std::vector<int> pre(points.size() + 1, 0);  // -1 indicates no predecessor
    std::vector<Trajectory> T_prime;

    for (size_t i = 1; i <= points.size(); ++i) {
        int min_cost = Ft[i - 1] + 12;

        for (size_t j = 1; j <= i; ++j) {
            Trajectory sub_traj = (*this)(j - 1, i - 1);
            auto it = M.find(sub_traj);
            if (it != M.end() && !it->second.empty()) {
                auto time_correction_cost_lookup = time_correction_cost.find(it->second[0])->second;
                int cost = std::min(Ft[i - 1] + 12, Ft[j - 1] + time_correction_cost_lookup + 8);

                if (cost < min_cost) {
                    min_cost = cost;
                    pre[i] = j - 1;
                }
            }
        }
        Ft[i] = min_cost;
    }

    int i = points.size();
    while (i > 0) {
        if (pre[i] == i - 1) {
            T_prime.emplace_back((*this)(i - 1, i - 1));
            --i;
        } else {
            Trajectory sub_traj = (*this)(pre[i], i - 1);
            auto it = M.find(sub_traj);
            if (it != M.end() && !it->second.empty()) {
                T_prime.emplace_back(it->second[0]);
            }
            i = pre[i];
        }
    }
    std::reverse(T_prime.begin(), T_prime.end());

    for (auto t = T_prime.begin(); t != T_prime.end() - 1; ++t) {
        auto next_trajectory = (t+1);
        if (t->id == next_trajectory->id && t->end_index+1 == next_trajectory->start_index) {
            auto concatted_points = std::vector<SamplePoint>();
            concatted_points.reserve(t->points.size() + next_trajectory->points.size());
            concatted_points.insert(concatted_points.end(), t->points.begin(), t->points.end());
            concatted_points.insert(concatted_points.end(), next_trajectory->points.begin(), next_trajectory->points.end());

            next_trajectory->points = concatted_points;
            next_trajectory->start_index = t->start_index;

            T_prime.erase(t);
            // We need to adjust i because we remove t, which results in a shift to the left of all elements of T_prime
            --t;
        }
    }

    return {T_prime, time_correction_record};
}




void convertCompressedTrajectoriesToPoints(std::vector<CompressedResult>& points, const Trajectory& trajectory_to_be_compressed, OSTCResult compressed)
{
    for (const auto& traj : compressed.references) {
        auto correction = compressed.time_corrections.find(traj);

        for (int i = 0; i < traj.points.size(); i++) {
            auto point = traj.points[i];
            auto corrections = std::vector<CompressedResultCorrection> {};

            auto existing_point = std::ranges::find_if(points,
               [&](const CompressedResult& p) {
                   return p.id == traj.id &&
                          p.latitude == point.latitude &&
                          p.longitude == point.longitude &&
                          p.timestamp == point.timestamp;
               }
            );

            const auto does_point_exist = existing_point != points.end();

            if (correction != compressed.time_corrections.end()) {
                for (const auto& correction_entry : correction->second) {
                    if (correction_entry.point_index == i) {
                        corrections.emplace_back(
                            trajectory_to_be_compressed.id,
                            correction_entry.corrected_timestamp
                        );
                    }
                }
            }

            if (does_point_exist) {
                existing_point->corrections.insert(existing_point->corrections.end(), corrections.begin(), corrections.end());
            }
            else {
                points.emplace_back(
                    traj.id,
                    point.latitude,
                    point.longitude,
                    point.timestamp,
                    corrections
                );
            }
        }
    }
}