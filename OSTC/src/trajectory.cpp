#include <algorithm>
#include <cmath>
#include <vector>
#include "trajectory.hpp"
#include <ranges>

#include "distance.hpp"
#include <unordered_map>
#include <ranges>
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

Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points):
    id(id), points(points), start_index(0), end_index(points.size() - 1)
{}

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
    for (int i = 0; i < points.size() - 1; i++) {
        Trajectory subtraj = (*this)(i, i + 1);
        M[subtraj] = std::vector<Trajectory>{};
        for (auto refTraj : RefSet) {
            for (int j = 0; j < refTraj.points.size() - 1; j++) {
                Trajectory subRefTraj = refTraj(j, j + 1);
                if (MaxDTW(subtraj, subRefTraj) <= epsilon) {
                    M[subtraj].push_back(subRefTraj);
                }
            }
        }
    }

    for (int n = 2; n < points.size(); n++) {
        for (int i = 0, j = i + n; j <= points.size() - 1; i++, j++) {
            Trajectory sub_left = (*this)(i, j - 1);
            Trajectory sub_right = (*this)(j - 1, j);

            auto T_a_vec = M.find(sub_left);
            auto T_b_vec = M.find(sub_right);
            if (T_a_vec != M.end() && T_b_vec != M.end()) {
                const std::vector<Trajectory>& T_as = T_a_vec->second;
                const std::vector<Trajectory>& T_bs = T_b_vec->second;
                for (const auto& [T_a, T_b] : std::views::zip(T_a_vec, T_b_vec) {

                }
            }
        }
    }
}

OSTCResult Trajectory::OSTC(std::unordered_map<Trajectory, std::vector<Trajectory>> M, const double tepsilon,
                            const double sepsilon)
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
    auto j = 1;

    for (auto& MRT : M) {
        auto ref = MRT.second[0];
        auto a = MRT.first.points;
        auto b = ref.points;

        signed int t = 0;
        time_correction_cost[ref] = 0;

        for (int i = 0; i <= b.size() - 1; i++) {
            auto a_i = a[i];
            auto b_i = b[i];

            if (i + 1 < a.size() && euclideanDistance(b_i, a[i + 1]) < sepsilon)
                a_i = a[i + 1];

            auto previousTimeStamp = i == 0 ? 0 : b[i - 1].timestamp;
            signed int leftside = abs(t + b_i.timestamp - previousTimeStamp - a_i.timestamp);
            if (leftside <= tepsilon) {
                t = t + b_i.timestamp - previousTimeStamp;
            } else {
                t = std::max(a_i.timestamp, previousTimeStamp);
                time_correction_cost[ref] += c;
                time_correction_record[ref].emplace_back(i, t);
            }
            j++;
        }
    }
    std::vector<int> Ft(points.size() + 1, 0);   // +1 for F_T[0] = 0
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

    return {T_prime, time_correction_record};
}
