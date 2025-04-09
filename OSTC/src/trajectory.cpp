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

bool SamplePoint::operator==(const SamplePoint &other) const {
    return longitude == other.longitude && latitude == other.latitude && timestamp == other.timestamp;
}

Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint> &points): id(id), points(points) {
}

Trajectory::Trajectory(const uint32_t id, const std::vector<SamplePoint> &points, const short start_index,
                       const short end_index): id(id), points(points), start_index(start_index), end_index(end_index) {
}

Trajectory Trajectory::operator()(const short start, const short end) {
    if (end + 1 > points.size()) {
        return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end), start, end);
    }

    return Trajectory(id, std::vector<SamplePoint>(points.begin() + start, points.begin() + end + 1), start, end);
}

Trajectory Trajectory::operator+(Trajectory other) {
    auto mergedPoints = points;
    std::copy(other.points.begin() + 1, other.points.end(), std::back_inserter(mergedPoints));

    return Trajectory(id, mergedPoints, start_index, other.end_index);
}

bool Trajectory::operator==(const Trajectory &other) const {
    return (id == other.id && start_index == other.start_index && end_index == other.end_index);
}

bool ReferenceTrajectory::operator==(const ReferenceTrajectory &other) const {
    return (id == other.id && start_index == other.start_index && end_index == other.end_index);
}

std::size_t std::hash<Trajectory>::operator()(const Trajectory &t) const noexcept {
    return ((hash<uint32_t>()(t.id) ^ (hash<short>()(t.start_index) << 2) ^ (hash<short>()(t.end_index) >> 1) ^
             (hash<uint32_t>()(t.points.size()) << 1)) >>
            1);
}

ReferenceTrajectory::ReferenceTrajectory(const uint32_t id, const short start_index,
                                         const short end_index): id(id), start_index(start_index),
                                                                 end_index(end_index) {
}

ReferenceTrajectory::ReferenceTrajectory(const Trajectory &t): id(t.id), start_index(t.start_index),
                                                               end_index(t.end_index) {
}

std::unordered_map<Trajectory, std::vector<ReferenceTrajectory> > Trajectory::MRTSearch(std::vector<Trajectory> &RefSet,
    const double epsilon) {
    std::unordered_map<Trajectory, std::vector<Trajectory> > M;

    for (auto i = 0; i < points.size() - 1; i++) {
        auto j = i + 1;
        auto current_sub_traj = (*this)(i, j);
        for (auto &ref_trajectory: RefSet) {
            for (auto k = 0; k < ref_trajectory.points.size() - 1; k++) {
                auto l = k + 1;
                auto ref_sub_traj = ref_trajectory(k, l);

                if (MaxDTW(current_sub_traj, ref_sub_traj) < epsilon) {
                    M[current_sub_traj].push_back(ref_sub_traj);
                }
            }
        }
    }

    for (auto n = 3; n < points.size(); n++) {
        bool found = false;
        for (auto k = 0; k + n - 1 < points.size(); k++) {
            auto lengthNSubtrajectory = (*this)(k, k + n - 1);

            auto T_a_entry = M.find((*this)(lengthNSubtrajectory.start_index, lengthNSubtrajectory.end_index - 1));
            auto T_b_entry = M.find((*this)(lengthNSubtrajectory.end_index - 1, lengthNSubtrajectory.end_index));


            auto T_a_vector = T_a_entry != M.end() ? T_a_entry->second : std::vector<Trajectory>();
            auto T_b_vector = T_b_entry != M.end() ? T_b_entry->second : std::vector<Trajectory>();

            for (auto &T_a: T_a_vector) {
                if (MaxDTW(lengthNSubtrajectory, T_a) <= epsilon) {
                    found = true;
                    M[lengthNSubtrajectory].push_back(T_a);
                }
                for (const auto &T_b: T_b_vector) {
                    if (MaxDTW(lengthNSubtrajectory, T_b) <= epsilon) {
                        found = true;
                        M[lengthNSubtrajectory].push_back(T_b);
                    }
                    if (T_a.id == T_b.id && T_a.end_index == T_b.start_index) {
                        found = true;
                        M[lengthNSubtrajectory].push_back(T_a + T_b);
                    }
                }
            }
        }
        if (!found) { break; }
    }

    std::unordered_map<Trajectory, std::vector<ReferenceTrajectory> > M1;
    for (const auto &pair: M) {
        std::vector<ReferenceTrajectory> refTrajectories;
        refTrajectories.reserve(pair.second.size());

        for (const auto &ref_trajectory: pair.second) {
            refTrajectories.emplace_back(ref_trajectory);
        }

        M1[pair.first] = refTrajectories;
    }

    std::vector<std::pair<Trajectory, std::vector<Trajectory> > > vec(M.begin(), M.end());

    // Sort by the Trajectory's id (or whatever other logic you'd like)
    std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
        return a.first.start_index < b.first.start_index; // or any other sort criteria
    });
    auto found = M1.find((*this)(9, 14));
    if (found != M1.end() && !found->second.empty()) {
        for (const auto f: found->second) {
            std::cout << "id: " << f.id << " Start: " << f.start_index << " end: " << f.end_index << std::endl;
        }
    }
    return M1;
}

std::vector<ReferenceTrajectory> Trajectory::OSTC(std::unordered_map<Trajectory, std::vector<ReferenceTrajectory> > M) {
    std::vector<int> Ft(points.size() + 1, 0); // +1 for F_T[0] = 0
    std::vector<short> pre(points.size() + 1, -1); // -1 indicates no predecessor
    std::vector<ReferenceTrajectory> T_prime;

    for (size_t i = 1; i <= points.size(); ++i) {
        int min_cost = Ft[i - 1] + 8; // Fallback: use original point
        pre[i] = i - 1; // Default to previous point
        for (size_t j = 1; j <= i; ++j) {
            Trajectory sub_traj = (*this)(j - 1, i - 1); // 0-based sub-trajectory
            auto it = M.find(sub_traj);
            if (it != M.end() && !it->second.empty()) {
                int cost = Ft[j - 1] + 8; // 8 bytes for MRT
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
        if (pre[i] == i - 1) { // Use original point
            T_prime.push_back((*this)(i - 1, i - 1)); // Single point as ReferenceTrajectory
            --i;
        } else { // Use MRT
            Trajectory sub_traj = (*this)(pre[i], i - 1);
            auto it = M.find(sub_traj);
            if (it != M.end() && !it->second.empty()) {
                T_prime.push_back(it->second[0]); // Arbitrary MRT (could optimize for longest)
            }
            i = pre[i];
        }
    }

    std::reverse(T_prime.begin(), T_prime.end());
    return T_prime;
}
