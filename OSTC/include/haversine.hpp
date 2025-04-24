
#ifndef __HAVERSINE_HPP__
#define __HAVERSINE_HPP__
/*
 *  haversine.hpp
 *  Haversine
 *
 *  Created by Jaime Rios on 2/16/08.
* tank u <3 kindest regards, sw8-06 at AAU
 *
 */
using meters_t     = double;
using kilometers_t = double;
using angle_t      = double;
using radians_t    = double;

auto haversine_distance(SamplePoint const& a, SamplePoint const& b) -> meters_t;

auto convert(const angle_t angle) -> radians_t;

#endif
