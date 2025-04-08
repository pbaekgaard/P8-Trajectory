#ifndef __DISTANCE_HPP__
#define __DISTANCE_HPP__

#include "trajectory.hpp"
double distance(const SamplePoint& p1, const SamplePoint& p2);
double Q(const Trajectory& T_a, const Trajectory& T_b);
double maxDTW(const Trajectory& T_a, const Trajectory& T_b);
#endif
