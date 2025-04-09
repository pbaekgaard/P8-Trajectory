#ifndef __DISTANCE_HPP__
#define __DISTANCE_HPP__

#include "trajectory.hpp"

double euclideanDistance(SamplePoint a, SamplePoint b);
double maxDTW(Trajectory a, Trajectory b);
#endif
