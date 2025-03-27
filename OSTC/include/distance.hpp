#ifndef __DISTANCE_HPP__
#define __DISTANCE_HPP__
#include "trajectory.hpp"
double euclideanDistance(SamplePoint a, SamplePoint b);
double maxDtw(Trajectory a, Trajectory b);
#endif
