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
    return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
}

bool Trajectory::operator==(const Trajectory& other) const
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

std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> Trajectory::MRTSearch(
    std::vector<Trajectory>& RefSet, const double epsilon)
{
    std::unordered_map<Trajectory, std::vector<Trajectory>> M;

    for (auto i = 0; i < points.size(); ++i) {
        auto j = i+1;
        auto current_sub_traj = (*this)(i,j);
        for (auto& ref_trajectory : RefSet) {
            for (auto k = 0; k < ref_trajectory.points.size(); ++k) {
                auto l = k+1;
                auto ref_sub_traj = ref_trajectory(k, l);

                if (maxDTW(current_sub_traj, ref_sub_traj) < epsilon) {
                    M[current_sub_traj].push_back(ref_sub_traj);
                }
            }
        }
    }

    for (auto n = 3; n < points.size(); ++n) {
        for (auto k = 0; k+n-1 < points.size(); ++k) {
            auto lengthNSubtrajectory = (*this)(k, k+n-1);

            auto T_a_vector = M[(*this)(lengthNSubtrajectory.start_index, lengthNSubtrajectory.end_index - 1)];
            auto T_b_vector = M[(*this)(lengthNSubtrajectory.end_index - 1, lengthNSubtrajectory.end_index)];

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
                        M[lengthNSubtrajectory].push_back(T_a(T_a.start_index, T_b.end_index));
                    }
                }
            }
        }
    }

    std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> M1;
    // TODO: Convert M values to ReferenceTrajectory

    return M1;

}

std::vector<ReferenceTrajectory> Trajectory::OSTC(std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> M)
{
    std::vector<int64_t> FT(points.size() + 1, std::numeric_limits<int64_t>::max());
    FT[0] = 0;
    std::vector<int> prev(points.size() + 1, -1);

    for (int i = 1; i <= static_cast<int>(points.size()); ++i) {
        int64_t minCost = FT[i - 1] + 8;  // Cost of storing the i-th point as an original sample
        int bestJ = i;                    // Default to storing the point directly

        for (int j = 1; j <= i; ++j) {
            Trajectory subTraj = (*this)(j - 1, i - 1);
            auto it = M.find(subTraj);
            if (it != M.end() && !it->second.empty()) {
                int64_t cost = FT[j - 1] + 8;  // Cost of using an MRT
                if (cost < minCost) {
                    minCost = cost;
                    bestJ = j;
                }
            }
        }

        FT[i] = minCost;
        prev[i] = bestJ;
    }

    // Debug: Print FT and prev
    std::cout << "FT: ";
    for (int i = 0; i <= points.size(); ++i) {
        std::cout << FT[i] << " ";
    }
    std::cout << "\nprev: ";
    for (int i = 0; i <= points.size(); ++i) {
        std::cout << prev[i] << " ";
    }
    std::cout << "\n";

    std::vector<ReferenceTrajectory> T_prime;
    int i = points.size();
    while (i > 0) {
        int j = prev[i];
        if (j == -1) {
            break;
        }
        if (j == i) {  // Store the original point
            // Note: Adjust this based on how you want to represent original points in T_prime
            i--;
        } else {  // Use an MRT
            Trajectory subTraj = (*this)(j - 1, i - 1);
            auto it = M.find(subTraj);
            if (it != M.end() && !it->second.empty()) {
                T_prime.push_back(it->second[0]);
            }
            i = j - 1;
        }
    }
    std::ranges::reverse(T_prime.begin(), T_prime.end());
    return T_prime;
}
