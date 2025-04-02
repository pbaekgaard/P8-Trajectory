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
Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint>& points, const int start_index,
                       const int end_index): id(id), points(points), start_index(start_index), end_index(end_index)
{}

Trajectory Trajectory::operator()(const int start, const int end)
{
    return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
}

bool Trajectory::operator==(const Trajectory& other) const
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

std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> Trajectory::MRTSearch(
    std::vector<Trajectory>& Trajectories, std::vector<uint32_t> RefSet, const double epsilon)
{
    std::unordered_map<Trajectory, std::vector<ReferenceTrajectory>> M;

    // Step 1: Compute MRT sets for all length-2 sub-trajectories T^(i,i+1)
    for (int i = 0; i < static_cast<int>(points.size()) - 1; ++i) {
        Trajectory subTraj = (*this)(i, i + 1);

        // Store MRTs as ReferenceTrajectory objects
        std::vector<ReferenceTrajectory> mrtSet;

        // Compare against all reference sub-trajectories in R
        for (uint32_t refIdx : RefSet) {
            Trajectory& refTraj = Trajectories[refIdx];
            for (int k = 0; k < static_cast<int>(refTraj.points.size()) - 1; ++k) {
                Trajectory refSubTraj = refTraj(k, k + 1);
                if (maxDTW(subTraj, refSubTraj) <= epsilon) {
                    // Store the (id, start, end) triplet instead of the full Trajectory
                    mrtSet.emplace_back(refSubTraj.id, refSubTraj.start_index, refSubTraj.end_index);
                }
            }
        }

        // Store the MRT set in M
        M[subTraj] = mrtSet;
    }

    // Step 2: Compute MRT sets for sub-trajectories of length 3 to |T|
    for (int n = 3; n <= static_cast<int>(points.size()); ++n) {
        bool hasMRT = false;

        // For each sub-trajectory T^(i,j) of length n
        for (int i = 0; i <= static_cast<int>(points.size()) - n; ++i) {
            int j = i + n - 1;
            Trajectory subTraj = (*this)(i, j);

            // Store MRTs as ReferenceTrajectory objects
            std::vector<ReferenceTrajectory> mrtSet;

            // Get the MRT sets of T^(i,j-1) and T^(j-1,j)
            Trajectory subTraj1 = (*this)(i, j - 1);  // T^(i,j-1)
            Trajectory subTraj2 = (*this)(j - 1, j);  // T^(j-1,j)
            auto it1 = M.find(subTraj1);
            auto it2 = M.find(subTraj2);
            if (it1 == M.end() || it2 == M.end())
                continue;

            const std::vector<ReferenceTrajectory>& mrtSet1 = it1->second;  // M(T^(i,j-1))
            const std::vector<ReferenceTrajectory>& mrtSet2 = it2->second;  // M(T^(j-1,j))

            // Step 3: Check if MRTs of T^(i,j-1) can be extended to T^(i,j)
            for (const ReferenceTrajectory& Tmn : mrtSet1) {
                // Reconstruct the Trajectory object for Tmn to compute MaxDTW
                Trajectory TmnTraj = Trajectories[Tmn.id](Tmn.start_index, Tmn.end_index);
                if (maxDTW(subTraj, TmnTraj) <= epsilon) {
                    mrtSet.push_back(Tmn);
                }
            }

            // Step 4: Join MRTs of T^(i,j-1) and T^(j-1,j) if they are contiguous
            for (const ReferenceTrajectory& Tmn : mrtSet1) {
                for (const ReferenceTrajectory& Tst : mrtSet2) {
                    // Check if Tmn and Tst are from the same trajectory and contiguous
                    if (Tmn.id == Tst.id && Tmn.end_index + 1 == Tst.start_index) {
                        // Create the joined sub-trajectory T_a^(m,t) as a ReferenceTrajectory
                        ReferenceTrajectory joinedRefTraj(Tmn.id, Tmn.start_index, Tst.end_index);
                        // Reconstruct the Trajectory object to compute MaxDTW
                        Trajectory joinedTraj = Trajectories[Tmn.id](Tmn.start_index, Tst.end_index);
                        if (maxDTW(subTraj, joinedTraj) <= epsilon) {
                            mrtSet.push_back(joinedRefTraj);
                        }
                    }
                }
            }

            // Store the MRT set in M
            if (!mrtSet.empty()) {
                hasMRT = true;
                M[subTraj] = mrtSet;
            }
        }

        // Step 5: Early termination if no sub-trajectories of length n have MRTs
        if (!hasMRT) {
            break;
        }
    }

    return M;
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
