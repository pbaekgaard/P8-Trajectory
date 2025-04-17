#ifndef __DISTANCE_HPP__
#define __DISTANCE_HPP__

#include <functional>
#include "trajectory.hpp"
#include "haversine.hpp"

double euclideanDistance(const SamplePoint& a, const SamplePoint& b);
double MaxDTW(const Trajectory& A, const Trajectory& B, std::function<double(const SamplePoint& a, const SamplePoint& b)> distance);
auto haversine_distance(const SamplePoint& a, const SamplePoint& b) -> meters_t;
#endif