#ifndef __DISTANCE_HPP__
#define __DISTANCE_HPP__

#include "trajectory.hpp"
double euclideanDistance(const SamplePoint& a, const SamplePoint& b);
double MaxDTW(const Trajectory& A, const Trajectory& B);
#endif