#include <algorithm>
#include <cmath>
#include <vector>
#include "trajectory.hpp"
#include <ranges>

#include "distance.hpp"
#include <unordered_map>
#include <iostream>
#include <map>
#include <set>
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
                                                                             const double epsilon)
{
    std::unordered_map<Trajectory, std::vector<Trajectory>> M;
    
    // Step 1: Process length-2 subtrajectories
    for (auto i = 0; i < points.size() - 1; i++) {
        auto j = i + 1;
        auto current_sub_traj = (*this)(i, j);
        
        for (auto& ref_trajectory : RefSet) {
            for (auto k = 0; k < ref_trajectory.points.size() - 1; k++) {
                auto l = k + 1;
                auto ref_sub_traj = ref_trajectory(k, l);

                if (MaxDTW(current_sub_traj, ref_sub_traj) < epsilon) {
                    // Check if this reference is already covered by a longer one
                    bool already_covered = false;
                    for (auto& [existing_query, existing_refs] : M) {
                        if (existing_query.start_index <= current_sub_traj.start_index && 
                            existing_query.end_index >= current_sub_traj.end_index) {
                            for (auto& existing_ref : existing_refs) {
                                if (existing_ref.id == ref_sub_traj.id &&
                                    existing_ref.start_index <= ref_sub_traj.start_index &&
                                    existing_ref.end_index >= ref_sub_traj.end_index) {
                                    already_covered = true;
                                    break;
                                }
                            }
                            if (already_covered) break;
                        }
                    }
                    if (!already_covered) {
                        M[current_sub_traj].push_back(ref_sub_traj);
                    }
                }
            }
        }
    }
    
    // Keep track of processed lengths to avoid redundancy
    std::set<int, std::less<int>> processed_lengths = {2};
    
    // Step 2: Iteratively find longer matching subtrajectories
    for (auto n = 3; n <= points.size(); n++) {
        bool found = false;
        
        // First try to find direct matches of length n
        for (auto k = 0; k + n - 1 < points.size(); k++) {
            auto current_n_subtraj = (*this)(k, k + n - 1);
            
            // Check all reference trajectories for direct matches
            for (auto& ref_trajectory : RefSet) {
                if (ref_trajectory.points.size() >= n) {
                    for (auto r = 0; r + n - 1 < ref_trajectory.points.size(); r++) {
                        auto ref_n_subtraj = ref_trajectory(r, r + n - 1);
                        if (MaxDTW(current_n_subtraj, ref_n_subtraj) <= epsilon) {
                            // Check if this reference is not already covered
                            bool already_covered = false;
                            for (auto& [existing_query, existing_refs] : M) {
                                if (existing_query.start_index <= k && existing_query.end_index >= k + n - 1) {
                                    already_covered = true;
                                    break;
                                }
                            }
                            
                            if (!already_covered) {
                                M[current_n_subtraj].push_back(ref_n_subtraj);
                                found = true;
                            }
                        }
                    }
                }
            }
            
            // If no direct match, try combining shorter matches
            if (M.find(current_n_subtraj) == M.end()) {
                for (int split = k + 1; split < k + n - 1; split++) {
                    auto left = (*this)(k, split);
                    auto right = (*this)(split, k + n - 1);
                    
                    if (M.find(left) != M.end() && M.find(right) != M.end()) {
                        for (auto& ref_left : M[left]) {
                            for (auto& ref_right : M[right]) {
                                if (ref_left.id == ref_right.id && 
                                    ref_left.end_index == ref_right.start_index) {
                                    
                                    auto merged = ref_left + ref_right;
                                    if (MaxDTW(current_n_subtraj, merged) <= epsilon) {
                                        M[current_n_subtraj].push_back(merged);
                                        found = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // If nothing found at this length, stop
        if (!found) {
            break;
        }
        
        // Prune shorter trajectories that are fully covered by longer ones
        if (found) {
            std::vector<Trajectory> to_remove;
            
            for (auto& [query_traj, refs] : M) {
                if (query_traj.end_index - query_traj.start_index + 1 < n) {
                    // Check if this is fully covered by a longer trajectory
                    for (auto& [longer_query, longer_refs] : M) {
                        if (longer_query.end_index - longer_query.start_index + 1 == n &&
                            longer_query.start_index <= query_traj.start_index &&
                            longer_query.end_index >= query_traj.end_index) {
                            to_remove.push_back(query_traj);
                            break;
                        }
                    }
                }
            }
            
            // Remove covered trajectories
            for (auto& traj : to_remove) {
                M.erase(traj);
            }
        }
    }
    
    // Final optimization: keep only the longest subtrajectory for each start point
    std::map<int, Trajectory> best_for_start;
    
    for (auto& [query_traj, refs] : M) {
        int start = query_traj.start_index;
        if (best_for_start.find(start) == best_for_start.end() || 
            query_traj.end_index > best_for_start[start].end_index) {
            best_for_start[start] = query_traj;
        }
    }
    
    // Create optimized map with only the best matches
    std::unordered_map<Trajectory, std::vector<Trajectory>> optimized_M;
    for (auto& [start, query_traj] : best_for_start) {
        optimized_M[query_traj] = M[query_traj];
    }
    
    // Add debug output
    std::cout << "MRTSearch found " << M.size() << " matching trajectories\n";
    std::cout << "After optimization: " << optimized_M.size() << " trajectories\n";
    
    for (auto& [query, refs] : optimized_M) {
        std::cout << "Query (" << query.start_index << "," << query.end_index << ") matches " 
                  << refs.size() << " refs\n";
    }
    
    return optimized_M;
}

OSTCResult Trajectory::OSTC(std::unordered_map<Trajectory, std::vector<Trajectory>> M, const double tepsilon, const double sepsilon)
{
    // Ensure we only keep the first reference for each query
    std::unordered_map<Trajectory, std::vector<Trajectory>> simplified_M;
    for (auto& [query_traj, ref_trajs] : M) {
        if (!ref_trajs.empty()) {
            simplified_M[query_traj] = {ref_trajs[0]};
        }
    }
    M = simplified_M;

    std::unordered_map<Trajectory, int> time_correction_cost{};
    std::unordered_map<Trajectory, std::vector<TimeCorrectionRecordEntry>> time_correction_record{};
    auto c = 4;

    for (auto& MRT : M) {
        auto ref = MRT.second[0];
        auto a = MRT.first.points;
        auto b = ref.points;

        signed int t = 0;
        time_correction_cost[ref] = 0;

        for (int i = 0; i <= b.size() - 1; i++) {
            auto a_i = a[i];
            auto b_i = b[i];

            if (i+1 < a.size() && euclideanDistance(b_i, a[i+1]) < sepsilon)
                a_i = a[i+1];

            auto previousTimeStamp = i == 0 ? 0 : b[i-1].timestamp;
            signed int leftside = abs(t + b_i.timestamp - previousTimeStamp - a_i.timestamp);
            if (leftside <= tepsilon) {
                t = t + b_i.timestamp - previousTimeStamp;
            } else {
                t = std::max(a_i.timestamp, previousTimeStamp);
                time_correction_cost[ref] += c;
                time_correction_record[ref].emplace_back(i, t);
            }
        }
    }
    std::vector<int> Ft(points.size() + 1, 0);      // +1 for F_T[0] = 0
    std::vector<int> pre(points.size() + 1, 0);  // -1 indicates no predecessor
    std::vector<ReferenceTrajectory> T_prime;

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

    return {T_prime, time_correction_record};
}
