#ifndef __DISTANCE_HPP__
#define __DISTANCE_HPP__

#include <functional>
#include "trajectory.hpp"
#include "haversine.hpp"

double euclideanDistance(SamplePoint const& a, SamplePoint const& b);
double MaxDTW(const Trajectory& A, const Trajectory& B, std::function<double(SamplePoint const& a, SamplePoint const& b)> distance);
auto haversine_distance(SamplePoint const& a, SamplePoint const& b) -> meters_t;
#endif